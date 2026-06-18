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
// Derived benchmark: contendedUseWork(8_to_100)
//
// Same contention structure as contendedUse, but each waiter weaves a hot
// instruction-footprint workload through three large noinline 256-case
// switches selected by j%3. The switch index comes from a per-thread
// pointer-chase load, and every case uses unique literal constants to defeat
// compiler case/function deduplication. This expands the executed instruction
// working set to raise L1-icache-load-misses (and thereby lower IPC) toward
// the Tier T0 target.
// ---------------------------------------------------------------------------
#define LSEM_SW_OPS(acc, K, SALT)                           \
  do {                                                      \
    (acc) += (uint64_t)(K) * 1000003u + (uint64_t)(SALT);   \
    (acc) ^= (uint64_t)(K) * 2654435761u + 17u;             \
    (acc) *= ((uint64_t)(K) | 1u);                          \
    (acc) -= ((uint64_t)(K) ^ (uint64_t)(SALT));            \
    (acc) ^= ((acc) >> 7);                                  \
    (acc) += (uint64_t)(K) * 374761393u + 5u;               \
  } while (0)

#define LSEM_CASE(k, S) \
  case (k):             \
    LSEM_SW_OPS(acc, (k), (S));  \
    break;
#define LSEM_CASE4(b, S)                                       \
  LSEM_CASE((b) + 0, S) LSEM_CASE((b) + 1, S)                  \
  LSEM_CASE((b) + 2, S) LSEM_CASE((b) + 3, S)
#define LSEM_CASE16(b, S)                                      \
  LSEM_CASE4((b) + 0, S) LSEM_CASE4((b) + 4, S)               \
  LSEM_CASE4((b) + 8, S) LSEM_CASE4((b) + 12, S)
#define LSEM_CASE256(S)                                              \
  LSEM_CASE16(0, S) LSEM_CASE16(16, S) LSEM_CASE16(32, S)            \
  LSEM_CASE16(48, S) LSEM_CASE16(64, S) LSEM_CASE16(80, S)          \
  LSEM_CASE16(96, S) LSEM_CASE16(112, S) LSEM_CASE16(128, S)        \
  LSEM_CASE16(144, S) LSEM_CASE16(160, S) LSEM_CASE16(176, S)       \
  LSEM_CASE16(192, S) LSEM_CASE16(208, S) LSEM_CASE16(224, S)       \
  LSEM_CASE16(240, S)

__attribute__((noinline)) static uint64_t lsemIcacheSwitchA(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    LSEM_CASE256(0x1357abcdu)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) static uint64_t lsemIcacheSwitchB(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    LSEM_CASE256(0x9e3779b1u)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) static uint64_t lsemIcacheSwitchC(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    LSEM_CASE256(0xc2b2ae35u)
    default:
      break;
  }
  return acc;
}

#undef LSEM_CASE256
#undef LSEM_CASE16
#undef LSEM_CASE4
#undef LSEM_CASE
#undef LSEM_SW_OPS

static void contendedUseWork(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        constexpr size_t kRing = 4096;
        std::vector<uint32_t> chase(kRing);
        for (size_t r = 0; r < kRing; ++r) {
          chase[r] = static_cast<uint32_t>(
              (r * 2654435761u + static_cast<uint32_t>(t) + 1u) % kRing);
        }
        uint64_t acc = static_cast<uint64_t>(t) + 1u;
        uint32_t cur = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          cur = chase[cur];
          uint8_t idx = static_cast<uint8_t>((acc ^ cur) & 0xFFu);
          switch (i % 3) {
            case 0:
              acc = lsemIcacheSwitchA(acc, idx);
              break;
            case 1:
              acc = lsemIcacheSwitchB(acc, idx);
              break;
            default:
              acc = lsemIcacheSwitchC(acc, idx);
              break;
          }
        }
        folly::doNotOptimizeAway(acc);
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
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1000, 32, 1000)
BENCHMARK_NAMED_PARAM(contendedUseWork, 8_to_100, 8, 100)

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
