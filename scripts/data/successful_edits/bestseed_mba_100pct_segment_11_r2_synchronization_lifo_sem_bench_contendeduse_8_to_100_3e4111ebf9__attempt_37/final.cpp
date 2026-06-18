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
// Derived benchmark: contendedSwitch
//
// Same contended LifoSem usage as contendedUse, but each waiter wakeup drives a
// pointer-chase load whose low byte selects a case in one of three large
// 256-case switch functions, rotated by iteration index. This widens the
// executed instruction working set to raise L1 icache pressure (and lower IPC)
// while keeping the synchronization structure of the reference benchmark.
// ---------------------------------------------------------------------------

#define HOT_CASE(n)                                                     \
  case (n): {                                                           \
    acc += 0x9E3779B97F4A7C15ull ^ ((uint64_t)(n) + 1u);               \
    acc ^= ((uint64_t)(n) * 2246822519u + 3u);                         \
    acc += (acc << 3) ^ ((uint64_t)(n) + 7u);                          \
    acc *= (((uint64_t)(n) | 1u) + 5u);                                \
    acc ^= ((uint64_t)(n) * 0x85EBCA6Bull ^ HOT_SALT);                 \
    acc += (acc >> 5) + (uint64_t)(n) + 11u;                           \
    acc ^= 0xC2B2AE3D27D4EB4Full + (uint64_t)(n);                      \
    acc *= (((uint64_t)(n) << 1) | 1u);                                \
    acc += (uint64_t)(n) * 374761393u + 13u;                           \
    acc ^= (acc << 7) + (uint64_t)(n);                                 \
    acc += 0x27D4EB2F165667C5ull ^ (uint64_t)(n);                      \
    acc *= (((uint64_t)(n) ^ 0x165667B1u) | 1u);                       \
    acc ^= (acc >> 9) + (uint64_t)(n) + 17u;                           \
    acc += (uint64_t)(n) * 2654435761u + 19u;                          \
    break;                                                              \
  }

#define HOT_CASES16(b)                                                  \
  HOT_CASE((b) + 0)                                                     \
  HOT_CASE((b) + 1)                                                     \
  HOT_CASE((b) + 2)                                                     \
  HOT_CASE((b) + 3)                                                     \
  HOT_CASE((b) + 4)                                                     \
  HOT_CASE((b) + 5)                                                     \
  HOT_CASE((b) + 6)                                                     \
  HOT_CASE((b) + 7)                                                     \
  HOT_CASE((b) + 8)                                                     \
  HOT_CASE((b) + 9)                                                     \
  HOT_CASE((b) + 10)                                                    \
  HOT_CASE((b) + 11)                                                    \
  HOT_CASE((b) + 12)                                                    \
  HOT_CASE((b) + 13)                                                    \
  HOT_CASE((b) + 14)                                                    \
  HOT_CASE((b) + 15)

#define HOT_SWITCH_BODY                                                 \
  switch (idx) {                                                        \
    HOT_CASES16(0)                                                      \
    HOT_CASES16(16)                                                     \
    HOT_CASES16(32)                                                     \
    HOT_CASES16(48)                                                     \
    HOT_CASES16(64)                                                     \
    HOT_CASES16(80)                                                     \
    HOT_CASES16(96)                                                     \
    HOT_CASES16(112)                                                    \
    HOT_CASES16(128)                                                    \
    HOT_CASES16(144)                                                    \
    HOT_CASES16(160)                                                    \
    HOT_CASES16(176)                                                    \
    HOT_CASES16(192)                                                    \
    HOT_CASES16(208)                                                    \
    HOT_CASES16(224)                                                    \
    HOT_CASES16(240)                                                    \
  }

#define HOT_SALT 0x1111111111111111ull
__attribute__((noinline)) static uint64_t hotSwitchA(
    uint64_t acc, uint8_t idx) {
  HOT_SWITCH_BODY
  return acc;
}
#undef HOT_SALT

#define HOT_SALT 0x2222222222222222ull
__attribute__((noinline)) static uint64_t hotSwitchB(
    uint64_t acc, uint8_t idx) {
  HOT_SWITCH_BODY
  return acc;
}
#undef HOT_SALT

#define HOT_SALT 0x3333333333333333ull
__attribute__((noinline)) static uint64_t hotSwitchC(
    uint64_t acc, uint8_t idx) {
  HOT_SWITCH_BODY
  return acc;
}
#undef HOT_SALT

static void contendedSwitch(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static const size_t kChase = 4096;
  std::vector<uint32_t> chase(kChase);

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kChase; ++i) {
      chase[i] = static_cast<uint32_t>((i * 2654435761u + 1u) % kChase);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1u;
        uint32_t cur = static_cast<uint32_t>(t) % kChase;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          cur = chase[cur % kChase];
          uint8_t idx = static_cast<uint8_t>((cur ^ acc) & 0xFFu);
          switch (i % 3u) {
            case 0:
              acc = hotSwitchA(acc, idx);
              break;
            case 1:
              acc = hotSwitchB(acc, idx);
              break;
            default:
              acc = hotSwitchC(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedSwitch, 8_to_100, 8, 100)

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
