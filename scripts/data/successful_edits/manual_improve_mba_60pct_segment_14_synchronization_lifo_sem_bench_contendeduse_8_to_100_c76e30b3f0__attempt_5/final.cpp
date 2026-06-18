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
// Derived benchmark: contendedUse_icache_8_to_100
//
// Same LifoSem contention shape as contendedUse(8_to_100), but each woken
// waiter executes a frontend-heavy, pointer-chase-driven switch workload.
// The three noinline 256-case switch functions create a large hot-code
// footprint that thrashes the L1 instruction cache (raising L1i MPKI and
// lowering IPC toward the contended-use target). Each case uses unique
// literal constants so the compiler cannot deduplicate case bodies, and a
// per-function salt prevents cross-function identical-code folding.
// ---------------------------------------------------------------------------

static std::atomic<uint64_t> g_icache_sink{0};

#define ALU_CASE(K)                                              \
  case (K): {                                                    \
    acc += (uint64_t)(K) * 0x9E3779B97F4A7C15ull + FSALT;        \
    acc ^= (acc >> 7);                                           \
    acc *= (2ull * (uint64_t)(K) + 1ull);                        \
    acc += (uint64_t)(K) * 7ull + 0x100ull;                      \
    acc ^= ((uint64_t)(K) << 3) ^ FSALT;                         \
    acc -= (uint64_t)(K) * 11ull + 0x55ull;                      \
    acc *= ((uint64_t)(K) | 1ull);                               \
    acc += (acc << 5);                                           \
    acc ^= (acc >> 11);                                          \
    acc += (uint64_t)(K) * 13ull + FSALT;                        \
    acc ^= (uint64_t)(K) * 17ull + 0xABCDull;                    \
    acc *= 0x100000001B3ull;                                     \
    acc += (uint64_t)(K) * 19ull;                                \
    acc ^= (acc >> 9);                                           \
    acc += (uint64_t)(K) ^ 0x5A5Aull;                            \
    acc *= (4ull * (uint64_t)(K) + 3ull);                        \
    break;                                                       \
  }

#define CASES_4(K) \
  ALU_CASE(K) ALU_CASE((K) + 1) ALU_CASE((K) + 2) ALU_CASE((K) + 3)
#define CASES_16(K) \
  CASES_4(K) CASES_4((K) + 4) CASES_4((K) + 8) CASES_4((K) + 12)
#define CASES_64(K) \
  CASES_16(K) CASES_16((K) + 16) CASES_16((K) + 32) CASES_16((K) + 48)
#define CASES_256 \
  CASES_64(0) CASES_64(64) CASES_64(128) CASES_64(192)

#define FSALT 0x1111111111111111ull
__attribute__((noinline)) static uint64_t icacheSwitchA(
    uint8_t idx, uint64_t acc) {
  switch (idx) {
    CASES_256
  }
  return acc;
}
#undef FSALT

#define FSALT 0x2222222222222222ull
__attribute__((noinline)) static uint64_t icacheSwitchB(
    uint8_t idx, uint64_t acc) {
  switch (idx) {
    CASES_256
  }
  return acc;
}
#undef FSALT

#define FSALT 0x3333333333333333ull
__attribute__((noinline)) static uint64_t icacheSwitchC(
    uint8_t idx, uint64_t acc) {
  switch (idx) {
    CASES_256
  }
  return acc;
}
#undef FSALT

#undef CASES_256
#undef CASES_64
#undef CASES_16
#undef CASES_4
#undef ALU_CASE

BENCHMARK(contendedUse_icache_8_to_100, iters) {
  const uint32_t n = iters;
  constexpr int posters = 8;
  constexpr int waiters = 100;
  constexpr int kInner = 8;
  constexpr size_t kRing = 4096;

  LifoSemImpl<std::atomic> sem;
  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::vector<uint32_t> ring(kRing);

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kRing; ++i) {
      ring[i] = (uint32_t)((i * 167 + 13) % kRing);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &ring] {
        uint64_t acc = 0x12345ull + (uint64_t)t;
        size_t p = ((size_t)t * 131u) % kRing;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int k = 0; k < kInner; ++k) {
            p = ring[p];
            uint8_t idx = (uint8_t)(ring[p] ^ acc);
            switch ((k + (int)(i / waiters)) % 3) {
              case 0:
                acc = icacheSwitchA(idx, acc);
                break;
              case 1:
                acc = icacheSwitchB(idx, acc);
                break;
              default:
                acc = icacheSwitchC(idx, acc);
                break;
            }
          }
        }
        g_icache_sink.fetch_add(acc, std::memory_order_relaxed);
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
