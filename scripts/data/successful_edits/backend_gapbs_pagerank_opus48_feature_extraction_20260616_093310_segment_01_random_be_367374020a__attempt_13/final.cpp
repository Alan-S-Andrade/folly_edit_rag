/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <folly/Random.h>

#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <glog/logging.h>

#include <folly/Benchmark.h>
#include <folly/Portability.h>
#if FOLLY_HAVE_EXTRANDOM_SFMT19937
#include <ext/random>
#endif

#if FOLLY_X64
#endif

using namespace folly;

BENCHMARK(minstdrand, n) {
  BenchmarkSuspender braces;
  std::random_device rd;
  std::minstd_rand rng(rd());

  braces.dismiss();

  for (unsigned i = 0; i < n; i++) {
    doNotOptimizeAway(rng());
  }
}

BENCHMARK(ranlux24_base, n) {
  BenchmarkSuspender braces;
  std::random_device rd;
  std::ranlux24_base rng(rd());

  braces.dismiss();

  for (unsigned i = 0; i < n; i++) {
    doNotOptimizeAway(rng());
  }
}

namespace {

// Pointer-chase infrastructure for the ranlux24_base_chasewalk variant.
// A large permuted chain (>64 KB) creates dependent loads that stall the
// backend, while three 256-case switch functions selected by load values add
// frontend pressure and unpredictable indirect/branch behavior.
struct ChainNode {
  ChainNode* next;
  char pad[48];
};

constexpr size_t kHotLen = 16384; // 1 MB working set, exceeds L1d.
constexpr size_t kColdLen = 2u << 20; // 128 MB working set, exceeds LLC.

ChainNode g_hotNodes[kHotLen];
ChainNode* g_coldNodes = nullptr;

void initChain(ChainNode* nodes, size_t len) {
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), 0u);
  std::mt19937_64 rng(42);
  for (size_t i = len - 1; i > 0; --i) {
    std::swap(perm[i], perm[rng() % (i + 1)]);
  }
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
}

#define CHASE_OP(n, s)                            \
  case (n):                                       \
    v += 0x1234u * (static_cast<uint64_t>(n) + 1) + (s); \
    v ^= 0x5678u + static_cast<uint64_t>(n);      \
    v = (v << 1) | (v >> 63);                     \
    break;
#define CHASE_OP16(b, s)                                                     \
  CHASE_OP(b + 0, s) CHASE_OP(b + 1, s) CHASE_OP(b + 2, s) CHASE_OP(b + 3, s) \
  CHASE_OP(b + 4, s) CHASE_OP(b + 5, s) CHASE_OP(b + 6, s) CHASE_OP(b + 7, s) \
  CHASE_OP(b + 8, s) CHASE_OP(b + 9, s) CHASE_OP(b + 10, s)                  \
  CHASE_OP(b + 11, s) CHASE_OP(b + 12, s) CHASE_OP(b + 13, s)               \
  CHASE_OP(b + 14, s) CHASE_OP(b + 15, s)
#define CHASE_OP256(s)                                                       \
  CHASE_OP16(0, s) CHASE_OP16(16, s) CHASE_OP16(32, s) CHASE_OP16(48, s)     \
  CHASE_OP16(64, s) CHASE_OP16(80, s) CHASE_OP16(96, s) CHASE_OP16(112, s)   \
  CHASE_OP16(128, s) CHASE_OP16(144, s) CHASE_OP16(160, s)                   \
  CHASE_OP16(176, s) CHASE_OP16(192, s) CHASE_OP16(208, s)                  \
  CHASE_OP16(224, s) CHASE_OP16(240, s)

__attribute__((noinline)) uint64_t chaseSwitchA(uint64_t v, int c) {
  switch (c) { CHASE_OP256(0x11u) }
  return v;
}
__attribute__((noinline)) uint64_t chaseSwitchB(uint64_t v, int c) {
  switch (c) { CHASE_OP256(0x22u) }
  return v;
}
__attribute__((noinline)) uint64_t chaseSwitchC(uint64_t v, int c) {
  switch (c) { CHASE_OP256(0x33u) }
  return v;
}

#undef CHASE_OP256
#undef CHASE_OP16
#undef CHASE_OP

} // namespace

BENCHMARK(ranlux24_base_chasewalk, n) {
  BenchmarkSuspender braces;
  std::random_device rd;
  std::ranlux24_base rng(rd());

  if (g_coldNodes == nullptr) {
    g_coldNodes = new ChainNode[kColdLen];
    initChain(g_hotNodes, kHotLen);
    initChain(g_coldNodes, kColdLen);
  }
  ChainNode* hotPos = &g_hotNodes[0];
  ChainNode* coldPos = &g_coldNodes[0];
  uint64_t acc = 0;

  braces.dismiss();

  for (unsigned j = 0; j < n; j++) {
    // Long dependent pointer-chase: each hop is an L1d miss that hits L2,
    // serializing the backend so IPC drops and L1d-load-miss MPKI rises.
    // Eight strictly dependent hops per iteration multiply the L1d miss
    // density per retired instruction (the primary lever this pass).
    hotPos = hotPos->next;
    hotPos = hotPos->next;
    hotPos = hotPos->next;
    hotPos = hotPos->next;
    hotPos = hotPos->next;
    hotPos = hotPos->next;
    hotPos = hotPos->next;
    hotPos = hotPos->next;
    uint64_t payload = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(hotPos)) ^
        static_cast<uint64_t>(rng());
    switch (j % 3) {
      case 0:
        acc = chaseSwitchA(acc, static_cast<int>(payload & 0xFF));
        break;
      case 1:
        acc = chaseSwitchB(acc, static_cast<int>(payload & 0xFF));
        break;
      default:
        acc = chaseSwitchC(acc, static_cast<int>(payload & 0xFF));
        break;
    }
    // Branchless cold-chain advance every 8th step for LLC misses.
    uint64_t coldMask = -static_cast<uint64_t>((j % 8) == 0);
    coldPos = reinterpret_cast<ChainNode*>(
        (reinterpret_cast<uintptr_t>(coldPos->next) & coldMask) |
        (reinterpret_cast<uintptr_t>(coldPos) & ~coldMask));
    doNotOptimizeAway(coldPos);
  }
  doNotOptimizeAway(acc);
}

BENCHMARK(mt19937, n) {
  BenchmarkSuspender braces;
  std::random_device rd;
  std::mt19937 rng(rd());

  braces.dismiss();

  for (unsigned i = 0; i < n; i++) {
    doNotOptimizeAway(rng());
  }
}

BENCHMARK(mt19937_64, n) {
  BenchmarkSuspender braces;
  std::random_device rd;
  std::mt19937 rng(rd());

  braces.dismiss();

  for (unsigned i = 0; i < n; i++) {
    doNotOptimizeAway(rng());
  }
}

#if FOLLY_HAVE_EXTRANDOM_SFMT19937
BENCHMARK(sfmt19937, n) {
  BenchmarkSuspender braces;
  std::random_device rd;
  __gnu_cxx::sfmt19937 rng(rd());

  braces.dismiss();

  for (unsigned i = 0; i < n; i++) {
    doNotOptimizeAway(rng());
  }
}

BENCHMARK(sfmt19937_64, n) {
  BenchmarkSuspender braces;
  std::random_device rd;
  __gnu_cxx::sfmt19937_64 rng(rd());

  braces.dismiss();

  for (unsigned i = 0; i < n; i++) {
    doNotOptimizeAway(rng());
  }
}
#endif

BENCHMARK(xoshiro256, n) {
  BenchmarkSuspender braces;
  std::random_device rd;
  folly::xoshiro256pp_32 rng(rd());

  braces.dismiss();

  for (unsigned i = 0; i < n; i++) {
    doNotOptimizeAway(rng());
  }
}

BENCHMARK(xoshiro256_64, n) {
  BenchmarkSuspender braces;
  std::random_device rd;
  folly::xoshiro256pp_64 rng(rd());

  braces.dismiss();

  for (unsigned i = 0; i < n; i++) {
    doNotOptimizeAway(rng());
  }
}

BENCHMARK(threadprng, n) {
  BenchmarkSuspender braces;
  ThreadLocalPRNG tprng;
  tprng();

  braces.dismiss();

  for (unsigned i = 0; i < n; i++) {
    doNotOptimizeAway(tprng());
  }
}

BENCHMARK(RandomDouble) {
  doNotOptimizeAway(Random::randDouble01());
}
BENCHMARK(Random32) {
  doNotOptimizeAway(Random::rand32());
}
BENCHMARK(Random32Num) {
  doNotOptimizeAway(Random::rand32(100));
}
BENCHMARK(Random64) {
  doNotOptimizeAway(Random::rand64());
}
BENCHMARK(Random64Num) {
  doNotOptimizeAway(Random::rand64(100ull << 32));
}
BENCHMARK(Random64OneIn) {
  doNotOptimizeAway(Random::oneIn(100));
}

int main(int argc, char** argv) {
  folly::gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::runBenchmarks();
  return 0;
}
