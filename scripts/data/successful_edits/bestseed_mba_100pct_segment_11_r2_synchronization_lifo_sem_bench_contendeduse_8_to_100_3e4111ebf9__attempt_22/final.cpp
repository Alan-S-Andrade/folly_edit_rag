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
// Derived benchmark: contendedUseIcacheMix
//
// Same LifoSem contention structure as contendedUse(8_to_100), but each
// waiter wakeup drives a large, branch-table-heavy code footprint to raise
// the executed instruction working set (L1i pressure). Three noinline
// functions each hold a full 256-case switch with 14 unique ALU ops per
// case; the hot loop rotates among them with i%3 and indexes the switch
// with a payload byte sourced from a pointer-chase load.
// ---------------------------------------------------------------------------

#define ICACHE_ALU(a, c, s)                                          \
  do {                                                               \
    (a) += (uint64_t)(c) * (0x9E3779B97F4A7C15ull ^ (uint64_t)(s));  \
    (a) ^= (a) >> 23;                                                \
    (a) += (uint64_t)(c) ^ (0xC2B2AE3D27D4EB4Full + (uint64_t)(s)); \
    (a) *= (0x165667B19E3779F9ull | 1u);                            \
    (a) ^= (a) << 17;                                                \
    (a) -= (uint64_t)(c) + (0x27D4EB2F165667C5ull ^ (uint64_t)(s)); \
    (a) ^= (a) >> 11;                                                \
    (a) += (a) << 5;                                                 \
    (a) *= (0xFF51AFD7ED558CCDull | 1u);                            \
    (a) ^= (uint64_t)(c) * 3u + (uint64_t)(s);                       \
    (a) += (0x9E3779B9ull ^ (uint64_t)(c));                          \
    (a) ^= (a) >> 7;                                                 \
    (a) *= (0xD6E8FEB86659FD93ull | 1u);                            \
    (a) += (uint64_t)(c) + (uint64_t)(s);                            \
  } while (0)

#define ICACHE_CASE(a, c, s) \
  case (c):                  \
    ICACHE_ALU(a, c, s);     \
    break;

#define ICACHE_CASES16(a, b, s)                                          \
  ICACHE_CASE(a, (b) + 0x0, s) ICACHE_CASE(a, (b) + 0x1, s)              \
      ICACHE_CASE(a, (b) + 0x2, s) ICACHE_CASE(a, (b) + 0x3, s)          \
          ICACHE_CASE(a, (b) + 0x4, s) ICACHE_CASE(a, (b) + 0x5, s)      \
              ICACHE_CASE(a, (b) + 0x6, s) ICACHE_CASE(a, (b) + 0x7, s)  \
                  ICACHE_CASE(a, (b) + 0x8, s) ICACHE_CASE(a, (b) + 0x9, \
                      s) ICACHE_CASE(a, (b) + 0xA, s)                    \
                      ICACHE_CASE(a, (b) + 0xB, s)                       \
                          ICACHE_CASE(a, (b) + 0xC, s)                   \
                              ICACHE_CASE(a, (b) + 0xD, s)               \
                                  ICACHE_CASE(a, (b) + 0xE, s)           \
                                      ICACHE_CASE(a, (b) + 0xF, s)

#define ICACHE_CASES256(a, s)                                             \
  ICACHE_CASES16(a, 0x00, s) ICACHE_CASES16(a, 0x10, s)                   \
      ICACHE_CASES16(a, 0x20, s) ICACHE_CASES16(a, 0x30, s)               \
          ICACHE_CASES16(a, 0x40, s) ICACHE_CASES16(a, 0x50, s)           \
              ICACHE_CASES16(a, 0x60, s) ICACHE_CASES16(a, 0x70, s)       \
                  ICACHE_CASES16(a, 0x80, s) ICACHE_CASES16(a, 0x90, s)   \
                      ICACHE_CASES16(a, 0xA0, s)                          \
                          ICACHE_CASES16(a, 0xB0, s)                      \
                              ICACHE_CASES16(a, 0xC0, s)                  \
                                  ICACHE_CASES16(a, 0xD0, s)              \
                                      ICACHE_CASES16(a, 0xE0, s)          \
                                          ICACHE_CASES16(a, 0xF0, s)

__attribute__((noinline)) static uint64_t icacheMix0(uint64_t a, uint8_t idx) {
  switch (idx) { ICACHE_CASES256(a, 0x1111111111111111ull) }
  return a;
}

__attribute__((noinline)) static uint64_t icacheMix1(uint64_t a, uint8_t idx) {
  switch (idx) { ICACHE_CASES256(a, 0x2222222222222222ull) }
  return a;
}

__attribute__((noinline)) static uint64_t icacheMix2(uint64_t a, uint8_t idx) {
  switch (idx) { ICACHE_CASES256(a, 0x3333333333333333ull) }
  return a;
}

static std::vector<uint32_t> makeChase(size_t sz) {
  std::vector<uint32_t> v(sz);
  for (size_t i = 0; i < sz; ++i) {
    v[i] = (uint32_t)((i * 2654435761ull + 1013904223ull) % sz);
  }
  return v;
}

static void contendedUseIcacheMix(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink{0};
  static const std::vector<uint32_t> chase = makeChase(4096);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t acc = 0x9E3779B97F4A7C15ull + (uint64_t)t;
        uint32_t p = (uint32_t)(t & 0xFFF);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p];
          uint8_t idx = (uint8_t)(chase[p] & 0xFF);
          switch (i % 3) {
            case 0:
              acc = icacheMix0(acc, idx);
              break;
            case 1:
              acc = icacheMix1(acc, idx);
              break;
            default:
              acc = icacheMix2(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUseIcacheMix, 8_to_100, 8, 100)
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
