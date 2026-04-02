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

#include <folly/portability/Asm.h>
#include <folly/synchronization/LifoSem.h>
#include <folly/synchronization/NativeSemaphore.h>

#include <folly/Benchmark.h>

#include <memory>
#include <numeric>
#include <random>
#include <vector>

using namespace folly;

namespace {

struct ChainNode {
  ChainNode* next;
  char pad[56];
};

static constexpr size_t kHotLen = 16384; // 1MB
static std::unique_ptr<ChainNode[]> hotNodes;

static constexpr size_t kColdLen = 1u << 20; // 128MB
static std::unique_ptr<ChainNode[]> coldNodes;

static void initChain(ChainNode* nodes, size_t len) {
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), 0u);
  std::mt19937_64 rng(42);
  for (size_t i = len - 1; i > 0; --i) {
    std::swap(perm[i], perm[std::uniform_int_distribution<size_t>(0, i)(rng)]);
  }
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
}

#define SWITCH_CASE(i)                                       \
  case i:                                                    \
    v += 0x1234;                                             \
    v ^= 0x5678;                                             \
    v = (v << 3) | (v >> 61); \
    break;

#define SWITCH_CHUNK(base)   \
  SWITCH_CASE(base + 0)      \
  SWITCH_CASE(base + 1)      \
  SWITCH_CASE(base + 2)      \
  SWITCH_CASE(base + 3)      \
  SWITCH_CASE(base + 4)      \
  SWITCH_CASE(base + 5)      \
  SWITCH_CASE(base + 6)      \
  SWITCH_CASE(base + 7)      \
  SWITCH_CASE(base + 8)      \
  SWITCH_CASE(base + 9)      \
  SWITCH_CASE(base + 10)     \
  SWITCH_CASE(base + 11)     \
  SWITCH_CASE(base + 12)     \
  SWITCH_CASE(base + 13)     \
  SWITCH_CASE(base + 14)     \
  SWITCH_CASE(base + 15)

__attribute__((noinline)) uint64_t switchA(uint64_t v, int c) {
  switch (c) {
    SWITCH_CHUNK(0)
    SWITCH_CHUNK(16)
    SWITCH_CHUNK(32)
    SWITCH_CHUNK(48)
    SWITCH_CHUNK(64)
    SWITCH_CHUNK(80)
    SWITCH_CHUNK(96)
    SWITCH_CHUNK(112)
    SWITCH_CHUNK(128)
    SWITCH_CHUNK(144)
    SWITCH_CHUNK(160)
    SWITCH_CHUNK(176)
    SWITCH_CHUNK(192)
    SWITCH_CHUNK(208)
    SWITCH_CHUNK(224)
    SWITCH_CHUNK(240)
  }
  return v;
}
__attribute__((noinline)) uint64_t switchB(uint64_t v, int c) {
  switch (c) {
    SWITCH_CHUNK(0)
    SWITCH_CHUNK(16)
    SWITCH_CHUNK(32)
    SWITCH_CHUNK(48)
    SWITCH_CHUNK(64)
    SWITCH_CHUNK(80)
    SWITCH_CHUNK(96)
    SWITCH_CHUNK(112)
    SWITCH_CHUNK(128)
    SWITCH_CHUNK(144)
    SWITCH_CHUNK(160)
    SWITCH_CHUNK(176)
    SWITCH_CHUNK(192)
    SWITCH_CHUNK(208)
    SWITCH_CHUNK(224)
    SWITCH_CHUNK(240)
  }
  return v;
}
__attribute__((noinline)) uint64_t switchC(uint64_t v, int c) {
  switch (c) {
    SWITCH_CHUNK(0)
    SWITCH_CHUNK(16)
    SWITCH_CHUNK(32)
    SWITCH_CHUNK(48)
    SWITCH_CHUNK(64)
    SWITCH_CHUNK(80)
    SWITCH_CHUNK(96)
    SWITCH_CHUNK(112)
    SWITCH_CHUNK(128)
    SWITCH_CHUNK(144)
    SWITCH_CHUNK(160)
    SWITCH_CHUNK(176)
    SWITCH_CHUNK(192)
    SWITCH_CHUNK(208)
    SWITCH_CHUNK(224)
    SWITCH_CHUNK(240)
  }
  return v;
}

struct ChaseBenchUtil {
  ChaseBenchUtil() {
    if (!hotNodes) {
      hotNodes = std::make_unique<ChainNode[]>(kHotLen);
      initChain(hotNodes.get(), kHotLen);
    }
    if (!coldNodes) {
      coldNodes = std::make_unique<ChainNode[]>(kColdLen);
      initChain(coldNodes.get(), kColdLen);
    }
  }
};
} // namespace

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

static void contendedUseWithChase(uint32_t n, int posters, int waiters) {
  static ChaseBenchUtil chaseBenchUtil;
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<int> waitersReady(0);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &go, &waitersReady] {
        ChainNode* hotPos = &hotNodes[t % kHotLen];
        ChainNode* coldPos = &coldNodes[t % kColdLen];
        uint64_t acc = 0;
        constexpr size_t INNER_ITERS = 16;

        waitersReady.fetch_add(1);
        while (!go.load()) {
          std::this_thread::yield();
        }

        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (size_t j = 0; j < INNER_ITERS; ++j) {
            hotPos = hotPos->next;
            uint64_t payload = (uint64_t)(uintptr_t)hotPos;
            switch (j % 3) {
              case 0:
                acc = switchA(acc, (int)(payload & 0xFF));
                break;
              case 1:
                acc = switchB(acc, (int)(payload & 0xFF));
                break;
              case 2:
                acc = switchC(acc, (int)(payload & 0xFF));
                break;
            }
            auto coldMask = -uint64_t(j % 8 == 0);
            auto nextCold = coldPos->next;
            coldPos = (ChainNode*)(
                ((uintptr_t)nextCold & coldMask) |
                ((uintptr_t)coldPos & ~coldMask));
          }
        }
        folly::doNotOptimizeAway(acc);
        folly::doNotOptimizeAway(hotPos);
        folly::doNotOptimizeAway(coldPos);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &waitersReady] {
        ChainNode* hotPos = &hotNodes[(t + waiters) % kHotLen];
        ChainNode* coldPos = &coldNodes[(t + waiters) % kColdLen];
        uint64_t acc = 0;
        constexpr size_t INNER_ITERS = 16;
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          for (size_t j = 0; j < INNER_ITERS; ++j) {
            hotPos = hotPos->next;
            uint64_t payload = (uint64_t)(uintptr_t)hotPos;
            switch (j % 3) {
              case 0:
                acc = switchA(acc, (int)(payload & 0xFF));
                break;
              case 1:
                acc = switchB(acc, (int)(payload & 0xFF));
                break;
              case 2:
                acc = switchC(acc, (int)(payload & 0xFF));
                break;
            }
            auto coldMask = -uint64_t(j % 8 == 0);
            auto nextCold = coldPos->next;
            coldPos = (ChainNode*)(
                ((uintptr_t)nextCold & coldMask) |
                ((uintptr_t)coldPos & ~coldMask));
          }
          sem.post();
        }
        folly::doNotOptimizeAway(acc);
        folly::doNotOptimizeAway(hotPos);
        folly::doNotOptimizeAway(coldPos);
      });
    }

    // Wait until all waiter threads are ready before releasing go
    while (waitersReady.load() < waiters) {
      std::this_thread::yield();
    }
  }

  go.store(true);
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
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1, 31, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 16_to_16, 16, 16)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_32, 32, 32)
BENCHMARK_NAMED_PARAM(
    contendedUseWithChase,
    contendedUse_v1_32_to_32,
    32,
    32)
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
