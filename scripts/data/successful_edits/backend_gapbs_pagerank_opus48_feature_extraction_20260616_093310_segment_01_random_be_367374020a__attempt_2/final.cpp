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
// 64-byte cache-line-sized node for pointer-chase locality experiments.
struct ChainNode {
  ChainNode* next;
  char pad[56];
};

// Build a single ~1 MiB hot chain (16384 * 64B). The Fisher-Yates shuffle
// randomizes traversal order so each step is a dependent load that misses L1d
// but stays resident in L2, serializing the pipeline and lowering IPC.
ChainNode* makeHotChain() {
  const size_t kNodes = 16384; // 16384 * 64B = 1 MiB
  ChainNode* nodes = new ChainNode[kNodes];
  size_t* order = new size_t[kNodes];
  for (size_t i = 0; i < kNodes; ++i) {
    order[i] = i;
  }
  std::mt19937 g(0x9e3779b9u);
  for (size_t i = kNodes - 1; i > 0; --i) {
    std::uniform_int_distribution<size_t> d(0, i);
    size_t j = d(g);
    size_t tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }
  for (size_t i = 0; i + 1 < kNodes; ++i) {
    nodes[order[i]].next = &nodes[order[i + 1]];
  }
  nodes[order[kNodes - 1]].next = &nodes[order[0]];
  ChainNode* head = &nodes[order[0]];
  delete[] order;
  return head;
}
} // namespace

BENCHMARK(ranlux24_base_chasewalk, n) {
  BenchmarkSuspender braces;
  std::random_device rd;
  std::ranlux24_base rng(rd());
  static ChainNode* const head = makeHotChain();
  ChainNode* p = head;

  braces.dismiss();

  for (unsigned i = 0; i < n; i++) {
    // Multiple strictly dependent chase steps per iteration so the L2-latency
    // pointer-chase dominates and drives IPC down toward the target.
    p = p->next;
    p = p->next;
    p = p->next;
    p = p->next;
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
