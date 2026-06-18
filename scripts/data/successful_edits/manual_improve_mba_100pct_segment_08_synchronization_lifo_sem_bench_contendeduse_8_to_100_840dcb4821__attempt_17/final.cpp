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
// Frontend-pressure helpers for the icache-thrashing variant below.
// Three large noinline 256-case switches, each with 14 ALU ops per case and
// unique per-function salt constants so the compiler cannot deduplicate or
// fold them together.  The hot loop rotates among the three functions and
// indexes the switch with a pointer-chased payload byte.
// ---------------------------------------------------------------------------
#define LSB_ALU14(acc, k, s)                       \
  do {                                             \
    (acc) += (0x9E3779B1u ^ (s)) ^ (k);            \
    (acc) ^= (acc) << 5;                           \
    (acc) *= (0x85EBCA77u + (s)) + (k);            \
    (acc) ^= (acc) >> 7;                           \
    (acc) += (0xC2B2AE3Du ^ (s)) ^ ((k) << 1);     \
    (acc) *= (0x27D4EB2Fu + (s)) + ((k) << 2);     \
    (acc) ^= (acc) << 9;                           \
    (acc) += (0x165667B1u ^ (s)) ^ ((k) << 3);     \
    (acc) *= (0x9E3779B9u + (s)) + (k);            \
    (acc) ^= (acc) >> 11;                          \
    (acc) += (0xFF51AFD7u ^ (s)) ^ (k);            \
    (acc) *= (0xC4CEB9FEu + (s)) + ((k) << 1);     \
    (acc) ^= (acc) << 13;                          \
    (acc) += (0xD6E8FEB8u ^ (s)) ^ (k);            \
  } while (0)

#define LSB_SW_CASE(i, s) \
  case (i):               \
    LSB_ALU14(acc, static_cast<uint32_t>(i), (s)); \
    break;

#define LSB_SW_16(b, s)                                                 \
  LSB_SW_CASE((b) + 0, s) LSB_SW_CASE((b) + 1, s)                       \
  LSB_SW_CASE((b) + 2, s) LSB_SW_CASE((b) + 3, s)                       \
  LSB_SW_CASE((b) + 4, s) LSB_SW_CASE((b) + 5, s)                       \
  LSB_SW_CASE((b) + 6, s) LSB_SW_CASE((b) + 7, s)                       \
  LSB_SW_CASE((b) + 8, s) LSB_SW_CASE((b) + 9, s)                       \
  LSB_SW_CASE((b) + 10, s) LSB_SW_CASE((b) + 11, s)                     \
  LSB_SW_CASE((b) + 12, s) LSB_SW_CASE((b) + 13, s)                     \
  LSB_SW_CASE((b) + 14, s) LSB_SW_CASE((b) + 15, s)

#define LSB_SWITCH_BODY(s)                                              \
  switch (idx) {                                                        \
    LSB_SW_16(0, s) LSB_SW_16(16, s) LSB_SW_16(32, s) LSB_SW_16(48, s)  \
    LSB_SW_16(64, s) LSB_SW_16(80, s) LSB_SW_16(96, s) LSB_SW_16(112, s)\
    LSB_SW_16(128, s) LSB_SW_16(144, s) LSB_SW_16(160, s)              \
    LSB_SW_16(176, s) LSB_SW_16(192, s) LSB_SW_16(208, s)              \
    LSB_SW_16(224, s) LSB_SW_16(240, s)                                 \
  }

__attribute__((noinline)) static uint32_t lsb_switch_a(
    uint32_t acc, uint8_t idx) {
  LSB_SWITCH_BODY(0x11111111u);
  return acc;
}

__attribute__((noinline)) static uint32_t lsb_switch_b(
    uint32_t acc, uint8_t idx) {
  LSB_SWITCH_BODY(0x22222222u);
  return acc;
}

__attribute__((noinline)) static uint32_t lsb_switch_c(
    uint32_t acc, uint8_t idx) {
  LSB_SWITCH_BODY(0x33333333u);
  return acc;
}

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr size_t kChase = 8192;
  std::vector<uint32_t> chase(kChase);

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kChase; ++i) {
      chase[i] = static_cast<uint32_t>((i * 2654435761u + 1u) % kChase);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint32_t acc = static_cast<uint32_t>(t) + 1u;
        size_t p = static_cast<size_t>(t) % kChase;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p % kChase];
          uint8_t idx = static_cast<uint8_t>(p & 0xFFu);
          switch ((i / waiters) % 3) {
            case 0:
              acc = lsb_switch_a(acc, idx);
              break;
            case 1:
              acc = lsb_switch_b(acc, idx);
              break;
            default:
              acc = lsb_switch_c(acc, idx);
              break;
          }
        }
        folly::doNotOptimizeAway(acc);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &chase] {
        uint32_t acc = static_cast<uint32_t>(t) + 7u;
        size_t p = (static_cast<size_t>(t) + 3u) % kChase;
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          p = chase[p % kChase];
          uint8_t idx = static_cast<uint8_t>(p & 0xFFu);
          switch ((i / posters) % 3) {
            case 0:
              acc = lsb_switch_a(acc, idx);
              break;
            case 1:
              acc = lsb_switch_b(acc, idx);
              break;
            default:
              acc = lsb_switch_c(acc, idx);
              break;
          }
          sem.post();
        }
        folly::doNotOptimizeAway(acc);
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
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100_icache, 8, 100)

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
