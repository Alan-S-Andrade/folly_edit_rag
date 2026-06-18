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
// Frontend/icache-pressure derivative of contendedUse(8_to_100).
//
// Three noinline functions each carry a 256-case switch with 14 ALU ops per
// case (unique per-case literal constants to defeat compiler deduplication).
// The hot loop rotates among them with i%3 and indexes the switch with the low
// byte of a pointer-chase load.  This inflates the executed instruction working
// set to raise L1-icache-load-misses while keeping the LifoSem contention path
// intact.
// ---------------------------------------------------------------------------
#define LIFO_ICACHE_CASE(s, n)                                  \
  case (n): {                                                   \
    acc += 0x9E3779B9u + (uint32_t)(n) + (s);                   \
    acc ^= 0xC2B2AE35u ^ ((uint32_t)(n) << 3);                  \
    acc *= (0x85EBCA77u | ((uint32_t)(n) + 1));                 \
    acc += 0x27D4EB2Fu - (uint32_t)(n) + (s);                   \
    acc ^= 0x165667B1u + ((uint32_t)(n) << 1);                  \
    acc *= (0xD3A2646Du | ((uint32_t)(n) + 3));                 \
    acc += 0xFD7046C5u ^ (uint32_t)(n);                         \
    acc ^= 0xB55A4F09u - ((uint32_t)(n) << 2);                  \
    acc *= (0xCC9E2D51u | ((uint32_t)(n) + 5));                 \
    acc += 0x1B873593u + (uint32_t)(n) + (s);                   \
    acc ^= 0xE6546B64u ^ ((uint32_t)(n) << 4);                  \
    acc *= (0x9E3779B1u | ((uint32_t)(n) + 7));                 \
    acc += 0x85EBCA6Bu - (uint32_t)(n);                         \
    acc ^= 0xC2B2AE3Du + (uint32_t)(n) + (s);                   \
    break;                                                      \
  }

#define LIFO_ICACHE_REP16(s, b)                                 \
  LIFO_ICACHE_CASE(s, (b) + 0)                                  \
  LIFO_ICACHE_CASE(s, (b) + 1)                                  \
  LIFO_ICACHE_CASE(s, (b) + 2)                                  \
  LIFO_ICACHE_CASE(s, (b) + 3)                                  \
  LIFO_ICACHE_CASE(s, (b) + 4)                                  \
  LIFO_ICACHE_CASE(s, (b) + 5)                                  \
  LIFO_ICACHE_CASE(s, (b) + 6)                                  \
  LIFO_ICACHE_CASE(s, (b) + 7)                                  \
  LIFO_ICACHE_CASE(s, (b) + 8)                                  \
  LIFO_ICACHE_CASE(s, (b) + 9)                                  \
  LIFO_ICACHE_CASE(s, (b) + 10)                                 \
  LIFO_ICACHE_CASE(s, (b) + 11)                                 \
  LIFO_ICACHE_CASE(s, (b) + 12)                                 \
  LIFO_ICACHE_CASE(s, (b) + 13)                                 \
  LIFO_ICACHE_CASE(s, (b) + 14)                                 \
  LIFO_ICACHE_CASE(s, (b) + 15)

#define LIFO_ICACHE_REP256(s)                                   \
  LIFO_ICACHE_REP16(s, 0)                                       \
  LIFO_ICACHE_REP16(s, 16)                                      \
  LIFO_ICACHE_REP16(s, 32)                                      \
  LIFO_ICACHE_REP16(s, 48)                                      \
  LIFO_ICACHE_REP16(s, 64)                                      \
  LIFO_ICACHE_REP16(s, 80)                                      \
  LIFO_ICACHE_REP16(s, 96)                                      \
  LIFO_ICACHE_REP16(s, 112)                                     \
  LIFO_ICACHE_REP16(s, 128)                                     \
  LIFO_ICACHE_REP16(s, 144)                                     \
  LIFO_ICACHE_REP16(s, 160)                                     \
  LIFO_ICACHE_REP16(s, 176)                                     \
  LIFO_ICACHE_REP16(s, 192)                                     \
  LIFO_ICACHE_REP16(s, 208)                                     \
  LIFO_ICACHE_REP16(s, 224)                                     \
  LIFO_ICACHE_REP16(s, 240)

#define LIFO_ICACHE_FN(name, s)                                          \
  __attribute__((noinline)) static uint64_t name(                        \
      uint64_t acc, uint8_t idx) {                                       \
    switch (idx) { LIFO_ICACHE_REP256(s) }                               \
    return acc;                                                          \
  }

LIFO_ICACHE_FN(lifoIcacheMix0, 0x1111u)
LIFO_ICACHE_FN(lifoIcacheMix1, 0x2222u)
LIFO_ICACHE_FN(lifoIcacheMix2, 0x3333u)

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr size_t kChase = 4096;
  std::vector<uint32_t> chase(kChase);

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kChase; ++i) {
      chase[i] = (uint32_t)((i * 2654435761u + 12345u) % kChase);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = (uint64_t)t + 1;
        uint32_t p = (uint32_t)t;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p % kChase];
          uint8_t idx = (uint8_t)(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = lifoIcacheMix0(acc, idx);
              break;
            case 1:
              acc = lifoIcacheMix1(acc, idx);
              break;
            default:
              acc = lifoIcacheMix2(acc, idx);
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
