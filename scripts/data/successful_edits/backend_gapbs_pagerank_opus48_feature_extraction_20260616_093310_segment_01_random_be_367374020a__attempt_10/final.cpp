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
// A permuted hot chain that fits in L2 but not L1d is walked by several
// independent walkers per iteration. Each walker issues a dependent load that
// misses L1d (but hits L2), so the per-instruction L1d miss rate is high and
// load-to-use stalls keep IPC low. A sparsely advanced cold chain supplies the
// occasional LLC miss.
struct ChainNode {
  ChainNode* next;
  char pad[48];
};

constexpr size_t kHotLen = 16384; // 1 MB working set, exceeds L1d, fits L2.
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

  // Eight independent walkers spread across the hot chain multiply the number
  // of dependent L1d-missing loads per iteration without growing the working
  // set, raising L1d MPKI and lowering IPC via load-to-use stalls.
  ChainNode* w0 = &g_hotNodes[(kHotLen / 8) * 0];
  ChainNode* w1 = &g_hotNodes[(kHotLen / 8) * 1];
  ChainNode* w2 = &g_hotNodes[(kHotLen / 8) * 2];
  ChainNode* w3 = &g_hotNodes[(kHotLen / 8) * 3];
  ChainNode* w4 = &g_hotNodes[(kHotLen / 8) * 4];
  ChainNode* w5 = &g_hotNodes[(kHotLen / 8) * 5];
  ChainNode* w6 = &g_hotNodes[(kHotLen / 8) * 6];
  ChainNode* w7 = &g_hotNodes[(kHotLen / 8) * 7];
  ChainNode* coldPos = &g_coldNodes[0];
  uint64_t acc = 0;

  braces.dismiss();

  for (unsigned j = 0; j < n; j++) {
    // Eight dependent pointer-chase loads, each an independent L1d miss.
    w0 = w0->next;
    w1 = w1->next;
    w2 = w2->next;
    w3 = w3->next;
    w4 = w4->next;
    w5 = w5->next;
    w6 = w6->next;
    w7 = w7->next;
    acc ^= reinterpret_cast<uintptr_t>(w0) ^ reinterpret_cast<uintptr_t>(w1) ^
        reinterpret_cast<uintptr_t>(w2) ^ reinterpret_cast<uintptr_t>(w3) ^
        reinterpret_cast<uintptr_t>(w4) ^ reinterpret_cast<uintptr_t>(w5) ^
        reinterpret_cast<uintptr_t>(w6) ^ reinterpret_cast<uintptr_t>(w7) ^
        static_cast<uint64_t>(rng());
    // Branchless cold-chain advance every 8th step for occasional LLC misses.
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
