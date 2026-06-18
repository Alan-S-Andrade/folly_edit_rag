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
// A large permuted chain creates dependent loads that stall the backend.
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

  // Primary lever this attempt: convert the previously independent walkers
  // into a single *serialized* dependent walk over the 1 MB hot chain. The
  // independent walkers overlapped their misses (high memory-level parallelism)
  // which kept IPC high and diluted the per-instruction miss density. By
  // chasing the same pointer through several dependent steps each iteration,
  // every L1d miss must complete before the next address is known, so the
  // misses can no longer overlap. The chain far exceeds L1d (48 KB) but fits in
  // L2, so each step reliably misses L1d while hitting L2. Non-overlapping
  // dependent misses both raise L1d MPKI and depress IPC toward the target,
  // which is the dominant remaining counter miss.
  constexpr size_t kSteps = 7;
  ChainNode* hotPos = &g_hotNodes[0];
  ChainNode* coldPos = &g_coldNodes[0];
  uint64_t acc = 0;

  braces.dismiss();

  for (unsigned j = 0; j < n; j++) {
    // A single serialized dependent walk: each step's address depends on the
    // load issued by the previous step, so these L1d misses cannot overlap.
    for (size_t s = 0; s < kSteps; ++s) {
      hotPos = hotPos->next;
      acc ^= static_cast<uint64_t>(reinterpret_cast<uintptr_t>(hotPos));
    }
    acc += static_cast<uint64_t>(rng());

    // Every third iteration take one dependent step over the 128 MB cold
    // chain. That load misses all the way to DRAM, lifting the average load
    // latency (further depressing IPC) without dominating the miss profile.
    if ((j % 3) == 0) {
      coldPos = coldPos->next;
      acc ^= static_cast<uint64_t>(reinterpret_cast<uintptr_t>(coldPos));
    }
  }
  doNotOptimizeAway(acc);
  doNotOptimizeAway(hotPos);
  doNotOptimizeAway(coldPos);
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
