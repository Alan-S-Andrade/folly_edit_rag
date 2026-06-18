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
// Derived from contendedUse(8_to_100): same contention shape, but each waiter
// drives a large rotating switch-based ALU kernel indexed by a pointer-chase
// payload. This expands the executed instruction working set to raise
// frontend / L1 i-cache pressure relative to the bare reference benchmark.
// ---------------------------------------------------------------------------

#define ALU_CASE(i, S)                                          \
  case (i): {                                                   \
    acc += 0x9E3779B97F4A7C15ull + (uint64_t)(i) + (S);         \
    acc ^= 0xD1B54A32D192ED03ull * ((uint64_t)(i) + 1u);        \
    acc *= 0x2545F4914F6CDD1Dull | 1ull;                        \
    acc += ((uint64_t)(i) << 7) ^ (0xABCDEF01ull + (S));        \
    acc ^= ((uint64_t)(i) * 0x100000001B3ull);                  \
    acc += (uint64_t)(i) * 0xC2B2AE3D27D4EB4Full;               \
    acc ^= ~((uint64_t)(i) << 11) + (S);                        \
    acc *= 3ull + ((uint64_t)(i) << 1);                         \
    acc += 0xFF51AFD7ED558CCDull ^ ((uint64_t)(i) + (S));       \
    acc ^= 0xC4CEB9FE1A85EC53ull + (uint64_t)(i);               \
    acc += ((uint64_t)(i) | 0x55ull);                           \
    acc *= 1ull | ((uint64_t)(i) << 2);                         \
    acc ^= ((uint64_t)(i) * 7u + 13u + (S));                    \
    acc += ((uint64_t)(i) ^ 0xA5A5A5A5ull);                     \
    break;                                                      \
  }

#define CASES_4(b, S) \
  ALU_CASE(b + 0, S) ALU_CASE(b + 1, S) ALU_CASE(b + 2, S) ALU_CASE(b + 3, S)
#define CASES_16(b, S) \
  CASES_4(b, S) CASES_4(b + 4, S) CASES_4(b + 8, S) CASES_4(b + 12, S)
#define CASES_64(b, S) \
  CASES_16(b, S) CASES_16(b + 16, S) CASES_16(b + 32, S) CASES_16(b + 48, S)
#define CASES_256(S) \
  CASES_64(0, S) CASES_64(64, S) CASES_64(128, S) CASES_64(192, S)

__attribute__((noinline)) static uint64_t alu_churn_a(
    uint64_t acc, uint8_t idx) {
  switch (idx) { CASES_256(0x11ull) }
  return acc;
}

__attribute__((noinline)) static uint64_t alu_churn_b(
    uint64_t acc, uint8_t idx) {
  switch (idx) { CASES_256(0x22ull) }
  return acc;
}

__attribute__((noinline)) static uint64_t alu_churn_c(
    uint64_t acc, uint8_t idx) {
  switch (idx) { CASES_256(0x33ull) }
  return acc;
}

#undef CASES_256
#undef CASES_64
#undef CASES_16
#undef CASES_4
#undef ALU_CASE

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr size_t kBuf = 4096;
  static std::vector<uint32_t> chase = [] {
    std::vector<uint32_t> v(kBuf);
    for (size_t i = 0; i < kBuf; ++i) {
      v[i] = static_cast<uint32_t>((i * 2654435761u) % kBuf);
    }
    return v;
  }();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & (kBuf - 1)];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = alu_churn_a(acc, idx);
              break;
            case 1:
              acc = alu_churn_b(acc, idx);
              break;
            default:
              acc = alu_churn_c(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100, 8, 100)

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
