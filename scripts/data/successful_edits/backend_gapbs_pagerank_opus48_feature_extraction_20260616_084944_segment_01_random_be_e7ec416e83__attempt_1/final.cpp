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

// Pointer-chase + data-dependent-branch infrastructure for the
// ranlux24_base_chase variant. This variant intentionally trades the
// compute-bound character of ranlux24_base for a memory/branch-bound hot
// path: a large permuted pointer-chase array (>64 KB) generates dependent
// load misses, while a rotation among three large switch functions gated on
// loaded payload bytes generates frontend / branch-mispredict pressure.
struct ChainNode {
  ChainNode* next;
  char pad[48];
};

// Hot chain: 16384 nodes x 64 bytes = 1 MB. Exceeds L1d (48 KB), fits L2.
constexpr size_t kHotLen = 16384;
ChainNode g_hotNodes[kHotLen];

void initChain(ChainNode* nodes, size_t len) {
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), size_t{0});
  std::mt19937_64 rng(42);
  for (size_t i = len - 1; i > 0; --i) {
    std::swap(perm[i], perm[rng() % (i + 1)]);
  }
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
}

#define CHASE_CASES(BASE)                                       \
  case 0: v += (BASE) ^ 0x1234u; v ^= (v << 1); break;          \
  case 1: v -= (BASE) + 0x5678u; v ^= (v >> 2); break;          \
  case 2: v += (BASE) * 3u; v ^= (v << 3); break;               \
  case 3: v ^= (BASE) | 0x9abcu; v += (v >> 1); break;          \
  default: v += (BASE) + uint64_t(c) * 0x1000193u; v ^= (v << 2); break;

__attribute__((noinline)) uint64_t switchA(uint64_t v, int c) {
  switch (c & 0x7) {
    CHASE_CASES(0xa5a5a5a5u)
  }
  return v;
}

__attribute__((noinline)) uint64_t switchB(uint64_t v, int c) {
  switch (c & 0x7) {
    CHASE_CASES(0x3c3c3c3cu)
  }
  return v;
}

__attribute__((noinline)) uint64_t switchC(uint64_t v, int c) {
  switch (c & 0x7) {
    CHASE_CASES(0xf0f0f0f0u)
  }
  return v;
}

#undef CHASE_CASES

} // namespace

BENCHMARK(ranlux24_base_chase, n) {
  BenchmarkSuspender braces;
  std::random_device rd;
  std::ranlux24_base rng(rd());
  initChain(g_hotNodes, kHotLen);

  // Three independent walkers multiply L1d pressure without growing the
  // working set beyond L2.
  ChainNode* w0 = &g_hotNodes[0];
  ChainNode* w1 = &g_hotNodes[kHotLen / 3];
  ChainNode* w2 = &g_hotNodes[(2 * kHotLen) / 3];
  uint64_t acc = 0;

  braces.dismiss();

  for (unsigned i = 0; i < n; i++) {
    acc += rng();

    w0 = w0->next; // dependent L1d-missing load
    w1 = w1->next;
    w2 = w2->next;

    uint64_t payload = uint64_t(uintptr_t(w0)) ^ (uint64_t(uintptr_t(w1)) >> 3) ^
        (uint64_t(uintptr_t(w2)) << 1);
    int sel = int(payload & 0xFF);

    switch (i % 3) {
      case 0:
        acc = switchA(acc, sel);
        break;
      case 1:
        acc = switchB(acc, sel);
        break;
      default:
        acc = switchC(acc, sel);
        break;
    }

    doNotOptimizeAway(acc);
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
