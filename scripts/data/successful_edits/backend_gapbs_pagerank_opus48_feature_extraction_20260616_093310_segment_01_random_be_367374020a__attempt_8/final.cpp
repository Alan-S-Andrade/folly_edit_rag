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
struct ChainNode {
  ChainNode* next;
  char pad[48];
};

std::vector<ChainNode>& ranluxChasewalkChain() {
  // ~1 MB hot chain: 16384 nodes x 64 bytes. Exceeds L1d (48 KB) but fits
  // in L2 (2 MB). Fisher-Yates shuffle defeats the prefetcher so each step
  // is a true dependent load, serializing the loop to drive IPC down.
  static std::vector<ChainNode> chain = [] {
    const size_t kNodes = 16384;
    std::vector<ChainNode> nodes(kNodes);
    std::vector<size_t> order(kNodes);
    for (size_t i = 0; i < kNodes; i++) {
      order[i] = i;
    }
    std::mt19937 g(0xC0FFEEu);
    for (size_t i = kNodes - 1; i > 0; i--) {
      std::uniform_int_distribution<size_t> d(0, i);
      std::swap(order[i], order[d(g)]);
    }
    for (size_t i = 0; i + 1 < kNodes; i++) {
      nodes[order[i]].next = &nodes[order[i + 1]];
    }
    nodes[order[kNodes - 1]].next = &nodes[order[0]];
    return nodes;
  }();
  return chain;
}
} // namespace

BENCHMARK(ranlux24_base_chasewalk, n) {
  BenchmarkSuspender braces;
  std::random_device rd;
  std::ranlux24_base rng(rd());

  std::vector<ChainNode>& chain = ranluxChasewalkChain();
  ChainNode* p = &chain[0];

  braces.dismiss();

  for (unsigned i = 0; i < n; i++) {
    // Serial dependent pointer-chase: several latency-bound hops per RNG
    // step so the L2-latency dependency chain dominates execution and pulls
    // IPC toward the memory-bound target instead of letting RNG work fill
    // the stall slots.
    for (int hop = 0; hop < 7; hop++) {
      p = p->next;
    }
    doNotOptimizeAway(p);
    doNotOptimizeAway(rng());
  }
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
