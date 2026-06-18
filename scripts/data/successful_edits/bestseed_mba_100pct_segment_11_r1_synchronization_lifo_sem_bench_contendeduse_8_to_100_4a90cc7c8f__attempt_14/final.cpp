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

// ---------------------------------------------------------------------------
// Derived from contendedUse(8_to_100): same LifoSem contention skeleton, but
// each completed wait/post drives a pointer-chase load whose low byte selects
// a 256-way switch. Three noinline switch functions are rotated with j%3 to
// inflate the executed instruction working set and raise L1i pressure /
// lower IPC toward the frontend-bound target profile.
// ---------------------------------------------------------------------------

// 14 ALU ops per case; (n) and (s) keep the literal constants unique per case
// and per function so the compiler/linker cannot deduplicate the bodies.
#define LSB_OPS14(n, s)                       \
  v += (0x1000003ULL + (s)) * (n) + 1ULL;     \
  v ^= 0x2000005ULL + (n) + (s);              \
  v *= (2ULL * (n) + 3ULL + (s));             \
  v += 0x3000007ULL ^ ((n) + (s));            \
  v ^= (0x4000009ULL + (s)) * (n);            \
  v -= 0x500000BULL + (n) + (s);              \
  v *= (4ULL * (n) + 5ULL + (s));             \
  v ^= 0x600000DULL - (n) + (s);              \
  v += 0x700000FULL + (n) + (s);              \
  v ^= (0x8000011ULL + (s)) * (n);            \
  v *= (6ULL * (n) + 7ULL + (s));             \
  v -= 0x9000013ULL + (n) + (s);              \
  v ^= (0xA000015ULL ^ (n)) + (s);            \
  v += (0xB000017ULL + (s)) * (n);

#define LSB_C1(n, s) \
  case (n): {        \
    uint64_t v = acc; \
    LSB_OPS14(n, s)   \
    acc = v;          \
  } break;
#define LSB_C4(n, s) \
  LSB_C1((n) + 0, s) LSB_C1((n) + 1, s) LSB_C1((n) + 2, s) LSB_C1((n) + 3, s)
#define LSB_C16(n, s) \
  LSB_C4((n) + 0, s) LSB_C4((n) + 4, s) LSB_C4((n) + 8, s) LSB_C4((n) + 12, s)
#define LSB_C64(n, s)                                              \
  LSB_C16((n) + 0, s) LSB_C16((n) + 16, s) LSB_C16((n) + 32, s)    \
  LSB_C16((n) + 48, s)
#define LSB_C256(s) \
  LSB_C64(0, s) LSB_C64(64, s) LSB_C64(128, s) LSB_C64(192, s)

__attribute__((noinline)) static uint64_t lsb_switch0(
    uint64_t acc, uint8_t idx) {
  switch (idx) { LSB_C256(11) }
  return acc;
}
__attribute__((noinline)) static uint64_t lsb_switch1(
    uint64_t acc, uint8_t idx) {
  switch (idx) { LSB_C256(23) }
  return acc;
}
__attribute__((noinline)) static uint64_t lsb_switch2(
    uint64_t acc, uint8_t idx) {
  switch (idx) { LSB_C256(37) }
  return acc;
}

static void contendedUseSwitchHeavy(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr uint32_t kRing = 4096;
  std::vector<uint32_t> chase(kRing);

  BENCHMARK_SUSPEND {
    for (uint32_t i = 0; i < kRing; ++i) {
      chase[i] = (i * 2654435761u + 12345u) & (kRing - 1);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & (kRing - 1)];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = lsb_switch0(acc, idx);
              break;
            case 1:
              acc = lsb_switch1(acc, idx);
              break;
            default:
              acc = lsb_switch2(acc, idx);
              break;
          }
        }
        folly::doNotOptimizeAway(acc);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &chase] {
        while (!go.load()) {
          std::this_thread::yield();
        }
        uint64_t acc = static_cast<uint64_t>(t) + 7;
        uint32_t p = static_cast<uint32_t>(t) + 1;
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          p = chase[p & (kRing - 1)];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = lsb_switch1(acc, idx);
              break;
            case 1:
              acc = lsb_switch2(acc, idx);
              break;
            default:
              acc = lsb_switch0(acc, idx);
              break;
          }
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

BENCHMARK_NAMED_PARAM(contendedUseSwitchHeavy, 8_to_100, 8, 100)

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
