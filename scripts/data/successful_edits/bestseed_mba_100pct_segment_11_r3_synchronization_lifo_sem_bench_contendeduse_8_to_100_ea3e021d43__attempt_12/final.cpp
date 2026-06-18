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
// Frontend-bound code-footprint helpers for contendedUseFootprint.
//
// Three noinline 256-case switch functions, each with a unique per-function
// SALT constant and unique per-case literal constants. The hot loop rotates
// among the three with i%3 and indexes the switch with a pointer-chased
// payload byte. This inflates the executed instruction working set to raise
// L1i-cache pressure (and lower IPC) without changing the semaphore protocol.
// ---------------------------------------------------------------------------
#define LSB_ALU(K)                                       \
  acc += (uint64_t)(K) * 2654435761ull + 1469598103ull;  \
  acc ^= (uint64_t)(K) * 40503ull + 0x9E3779B9ull;       \
  acc *= ((uint64_t)(K) | 1ull);                         \
  acc -= (uint64_t)(K) * 2246822519ull + 7ull;           \
  acc ^= acc >> 7;                                       \
  acc += (uint64_t)(K) * 3266489917ull + 374761393ull;   \
  acc *= (0x2545F4914F6CDD1Dull ^ ((uint64_t)(K) << 1)); \
  acc += (uint64_t)(K) << 3;                             \
  acc ^= (uint64_t)(K) * 668265263ull + 11ull;           \
  acc -= acc << 5;                                       \
  acc += (uint64_t)(K) * 2147483647ull + 13ull;          \
  acc ^= (uint64_t)(K) * 374761393ull + 17ull;           \
  acc *= (((uint64_t)(K) * 5ull) | 1ull);                \
  acc ^= (uint64_t)(K) * SALT;

#define LSB_CASE(K) \
  case (K): {       \
    LSB_ALU(K)      \
    break;          \
  }
#define LSB_C16(b)                                                  \
  LSB_CASE(b + 0)                                                   \
  LSB_CASE(b + 1)                                                   \
  LSB_CASE(b + 2) LSB_CASE(b + 3) LSB_CASE(b + 4) LSB_CASE(b + 5)   \
      LSB_CASE(b + 6) LSB_CASE(b + 7) LSB_CASE(b + 8)               \
          LSB_CASE(b + 9) LSB_CASE(b + 10) LSB_CASE(b + 11)         \
              LSB_CASE(b + 12) LSB_CASE(b + 13) LSB_CASE(b + 14)    \
                  LSB_CASE(b + 15)
#define LSB_C256                                                       \
  LSB_C16(0) LSB_C16(16) LSB_C16(32) LSB_C16(48) LSB_C16(64)            \
      LSB_C16(80) LSB_C16(96) LSB_C16(112) LSB_C16(128) LSB_C16(144)    \
          LSB_C16(160) LSB_C16(176) LSB_C16(192) LSB_C16(208)          \
              LSB_C16(224) LSB_C16(240)

template <uint64_t SALT>
static __attribute__((noinline)) uint64_t lsbFootprintSwitch(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    LSB_C256
  }
  return acc + SALT;
}

static void contendedUseFootprint(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr size_t kChaseSize = 4096; // power of two
  std::vector<uint32_t> chase(kChaseSize);
  std::atomic<uint64_t> sink{0};

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kChaseSize; ++i) {
      chase[i] =
          (uint32_t)((i * 2654435761ull + 12345ull) & (kChaseSize - 1));
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase, &sink] {
        uint64_t acc = 0x1234567ull + (uint64_t)t;
        uint32_t p = (uint32_t)(t & (kChaseSize - 1));
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p];
          uint8_t idx = (uint8_t)(chase[p] & 0xFF);
          switch (i % 3) {
            case 0:
              acc = lsbFootprintSwitch<0xA5A5A5A5A5A5A5A5ull>(acc, idx);
              break;
            case 1:
              acc = lsbFootprintSwitch<0x5A5A5A5A5A5A5A5Aull>(acc, idx);
              break;
            default:
              acc = lsbFootprintSwitch<0x3C3C3C3C3C3C3C3Cull>(acc, idx);
              break;
          }
        }
        sink.fetch_add(acc, std::memory_order_relaxed);
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
  folly::doNotOptimizeAway(sink.load());
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
BENCHMARK_NAMED_PARAM(contendedUseFootprint, 8_to_200, 8, 200)

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
