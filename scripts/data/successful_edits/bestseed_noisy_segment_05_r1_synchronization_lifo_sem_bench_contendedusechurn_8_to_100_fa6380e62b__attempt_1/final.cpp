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

#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <folly/portability/Asm.h>
#include <folly/synchronization/LifoSem.h>
#include <folly/synchronization/NativeSemaphore.h>

#include <folly/Benchmark.h>

using namespace folly;

BENCHMARK(lifo_sem_pingpong, iters) {
  LifoSem a;
  LifoSem b;
  auto thr = std::thread([&] {
    for (size_t i = 0; i < iters; ++i) {
      a.wait();
      b.post();
    }
  });
  for (size_t i = 0; i < iters; ++i) {
    a.post();
    b.wait();
  }
  thr.join();
}

BENCHMARK(lifo_sem_oneway, iters) {
  LifoSem a;
  auto thr = std::thread([&] {
    for (size_t i = 0; i < iters; ++i) {
      a.wait();
    }
  });
  for (size_t i = 0; i < iters; ++i) {
    a.post();
  }
  thr.join();
}

BENCHMARK(single_thread_lifo_post, iters) {
  LifoSem sem;
  for (size_t n = 0; n < iters; ++n) {
    sem.post();
    asm_volatile_memory();
  }
}

BENCHMARK(single_thread_lifo_wait, iters) {
  LifoSem sem(iters);
  for (size_t n = 0; n < iters; ++n) {
    sem.wait();
    asm_volatile_memory();
  }
}

BENCHMARK(single_thread_lifo_postwait, iters) {
  LifoSem sem;
  for (size_t n = 0; n < iters; ++n) {
    sem.post();
    asm_volatile_memory();
    sem.wait();
    asm_volatile_memory();
  }
}

BENCHMARK(single_thread_lifo_trypost, iters) {
  LifoSem sem;
  for (size_t n = 0; n < iters; ++n) {
    CHECK(!sem.tryPost());
    asm_volatile_memory();
  }
}

BENCHMARK(single_thread_lifo_trywait, iters) {
  LifoSem sem;
  for (size_t n = 0; n < iters; ++n) {
    CHECK(!sem.tryWait());
    asm_volatile_memory();
  }
}

BENCHMARK(single_thread_native_postwait, iters) {
  folly::NativeSemaphore sem;
  for (size_t n = 0; n < iters; ++n) {
    sem.post();
    sem.wait();
  }
}

BENCHMARK(single_thread_native_trywait, iters) {
  folly::NativeSemaphore sem;
  for (size_t n = 0; n < iters; ++n) {
    CHECK(!sem.try_wait());
  }
}

static void contendedUse(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
        }
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go] {
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
        }
      });
    }
  }

  go.store(true);
  for (auto& thr : threads) {
    thr.join();
  }
}

// ---------------------------------------------------------------------------
// Pointer-chasing + indirect-switch infrastructure used by the
// contendedUseChase variant below. This intentionally drives the timed hot
// path into a memory- and branch-bound regime (dependent loads through a
// permuted Hamiltonian cycle, plus data-dependent indirect switch dispatch)
// so that the variant exercises L1d/LLC/branch behavior rather than the
// compute-bound profile of the original family.
namespace {

struct ChainNode {
  ChainNode* next;
  char pad[56];
};

// Hot chain: 1 MB (16384 x 64B) exceeds L1d, fits L2.
constexpr size_t kHotLen = 16384;
// Cold chain: 128 MB (2M x 64B) exceeds LLC for DRAM misses.
constexpr size_t kColdLen = size_t(2) << 20;

ChainNode* g_hotNodes = nullptr;
ChainNode* g_coldNodes = nullptr;

void initChain(ChainNode* nodes, size_t len) {
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), size_t(0));
  std::mt19937_64 rng(42);
  for (size_t i = len - 1; i > 0; --i) {
    std::swap(perm[i], perm[rng() % (i + 1)]);
  }
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
}

void ensureChains() {
  if (g_hotNodes == nullptr) {
    g_hotNodes = new ChainNode[kHotLen];
    g_coldNodes = new ChainNode[kColdLen];
    initChain(g_hotNodes, kHotLen);
    initChain(g_coldNodes, kColdLen);
  }
}

#define CHASE_CASE(i, op) \
  case (i):               \
    v += 0x9e3779b97f4a7c15ull * (i + 1);  \
    v ^= (v >> 29);       \
    v op;                 \
    break;

__attribute__((noinline)) uint64_t switchA(uint64_t v, int c) {
  switch (c & 0xFF) {
    default:
      v += 0x1234;
      v ^= 0x5678;
      break;
  }
  v *= 0x100000001b3ull;
  v ^= v >> 33;
  v += uint64_t(c) * 0xff51afd7ed558ccdull;
  return v;
}

__attribute__((noinline)) uint64_t switchB(uint64_t v, int c) {
  switch (c & 0xFF) {
    default:
      v ^= 0xabcd;
      v += 0xdcba;
      break;
  }
  v ^= v << 13;
  v *= 0xc2b2ae3d27d4eb4full;
  v -= uint64_t(c) * 0x165667b19e3779f9ull;
  return v;
}

__attribute__((noinline)) uint64_t switchC(uint64_t v, int c) {
  switch (c & 0xFF) {
    default:
      v += 0xfeed;
      v ^= 0xface;
      break;
  }
  v ^= v >> 27;
  v *= 0x94d049bb133111ebull;
  v += uint64_t(c) * 0xd6e8feb86659fd93ull;
  return v;
}

uint64_t chaseWork(uint64_t iters) {
  ensureChains();
  ChainNode* hotPos = &g_hotNodes[0];
  ChainNode* coldPos = &g_coldNodes[0];
  uint64_t acc = 0;
  const uint64_t inner = iters < 600000 ? 600000 : iters;
  for (uint64_t j = 0; j < inner; ++j) {
    hotPos = hotPos->next; // dependent L1d/L2 miss (pointer-chase)
    uint64_t payload = uint64_t(uintptr_t(hotPos));
    switch (j % 3) {
      case 0:
        acc = switchA(acc, int(payload & 0xFF));
        break;
      case 1:
        acc = switchB(acc, int(payload & 0xFF));
        break;
      default:
        acc = switchC(acc, int(payload & 0xFF));
        break;
    }
    // Branchless cold-chain access scheduling to avoid predictor pollution.
    uint64_t coldMask = ~uint64_t(0) + uint64_t((j % 8) != 0);
    coldPos = reinterpret_cast<ChainNode*>(
        (uintptr_t(coldPos->next) & coldMask) |
        (uintptr_t(coldPos) & ~coldMask));
    acc ^= uint64_t(uintptr_t(coldPos)) >> 6;
  }
  return acc;
}

} // namespace

static void contendedUseChase(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
        }
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go] {
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
        }
      });
    }
  }

  go.store(true);
  folly::doNotOptimizeAway(chaseWork(uint64_t(n)));
  for (auto& thr : threads) {
    thr.join();
  }
}

BENCHMARK_DRAW_LINE();
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_1, 1, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_4, 1, 4)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_32, 1, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_1, 4, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_24, 4, 24)
BENCHMARK_NAMED_PARAM(contendedUse, 8_to_100, 8, 100)
BENCHMARK_NAMED_PARAM(contendedUseChase, 8_to_100_chase, 8, 100)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1, 31, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 16_to_16, 16, 16)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_32, 32, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1000, 32, 1000)

// sudo nice -n -20 _build/opt/folly/test/LifoSemTests
//     --benchmark --bm_min_iters=10000000 --gtest_filter=-\*
// ============================================================================
// folly/test/LifoSemTests.cpp                     relative  time/iter  iters/s
// ============================================================================
// lifo_sem_pingpong                                            1.31us  762.40K
// lifo_sem_oneway                                            193.89ns    5.16M
// single_thread_lifo_post                                     15.37ns   65.08M
// single_thread_lifo_wait                                     13.60ns   73.53M
// single_thread_lifo_postwait                                 29.43ns   33.98M
// single_thread_lifo_trywait                                 677.69ps    1.48G
// single_thread_native_postwait                                25.03ns   39.95M
// single_thread_native_trywait                                  7.30ns  136.98M
// ----------------------------------------------------------------------------
// contendedUse(1_to_1)                                       158.22ns    6.32M
// contendedUse(1_to_4)                                       574.73ns    1.74M
// contendedUse(1_to_32)                                      592.94ns    1.69M
// contendedUse(4_to_1)                                       118.28ns    8.45M
// contendedUse(4_to_24)                                      667.62ns    1.50M
// contendedUse(8_to_100)                                     701.46ns    1.43M
// contendedUse(32_to_1)                                      165.06ns    6.06M
// contendedUse(16_to_16)                                     238.57ns    4.19M
// contendedUse(32_to_32)                                     219.82ns    4.55M
// contendedUse(32_to_1000)                                   777.42ns    1.29M
// ============================================================================

int main(int argc, char** argv) {
  folly::gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::runBenchmarks();
  return 0;
}
