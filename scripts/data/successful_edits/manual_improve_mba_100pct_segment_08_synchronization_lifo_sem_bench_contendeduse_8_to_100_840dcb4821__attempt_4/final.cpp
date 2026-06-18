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
// Frontend-pressure derived benchmark (contendedUseFE).
//
// This deliberately inflates the executed instruction working set to pull IPC
// down toward the target by thrashing the L1 instruction cache.  The proven
// pattern is three __attribute__((noinline)) functions, each containing a
// 256-case switch with several ALU ops per case using unique literal constants
// (so the compiler cannot deduplicate cases or fold the three functions).
// The switch index comes from a data-dependent pointer-chase load, and all
// three large functions are invoked every iteration to maximize icache churn.
// ---------------------------------------------------------------------------

#define FE_OPS(acc, k)                          \
  do {                                          \
    (acc) += (uint64_t)((k)*2u + 1u);           \
    (acc) ^= (uint64_t)((k)*3u + 5u);           \
    (acc) *= (uint64_t)((k)*7u + 9u);           \
    (acc) += (uint64_t)((k)*11u + 13u);         \
    (acc) ^= (uint64_t)((k)*17u + 19u);         \
    (acc) *= (uint64_t)((k)*23u + 29u);         \
    (acc) += (uint64_t)((k)*31u + 37u);         \
    (acc) ^= (uint64_t)((k)*41u + 43u);         \
  } while (0)

#define FE_CASE(k) \
  case (k):        \
    FE_OPS(acc, (k)); \
    break;

#define FE_CASE16(b)                                              \
  FE_CASE(b + 0) FE_CASE(b + 1) FE_CASE(b + 2) FE_CASE(b + 3)     \
      FE_CASE(b + 4) FE_CASE(b + 5) FE_CASE(b + 6) FE_CASE(b + 7) \
          FE_CASE(b + 8) FE_CASE(b + 9) FE_CASE(b + 10)           \
              FE_CASE(b + 11) FE_CASE(b + 12) FE_CASE(b + 13)     \
                  FE_CASE(b + 14) FE_CASE(b + 15)

#define FE_CASES                                                          \
  FE_CASE16(0) FE_CASE16(16) FE_CASE16(32) FE_CASE16(48) FE_CASE16(64)    \
      FE_CASE16(80) FE_CASE16(96) FE_CASE16(112) FE_CASE16(128)           \
          FE_CASE16(144) FE_CASE16(160) FE_CASE16(176) FE_CASE16(192)     \
              FE_CASE16(208) FE_CASE16(224) FE_CASE16(240)

__attribute__((noinline)) static uint64_t feSwitch0(uint64_t acc, uint8_t idx) {
  switch (idx) { FE_CASES }
  return acc + 0x9e3779b97f4a7c15ull;
}

__attribute__((noinline)) static uint64_t feSwitch1(uint64_t acc, uint8_t idx) {
  switch (idx) { FE_CASES }
  return acc ^ 0x85ebca6b0c2b1d2full;
}

__attribute__((noinline)) static uint64_t feSwitch2(uint64_t acc, uint8_t idx) {
  switch (idx) { FE_CASES }
  return (acc * 0xc2b2ae3d27d4eb4full) + 1u;
}

static constexpr uint32_t kFeChaseSize = 1u << 13; // 8192 entries
static std::vector<uint32_t> feChase;
static std::atomic<uint64_t> feSink{0};

static void feInitChase() {
  if (!feChase.empty()) {
    return;
  }
  feChase.resize(kFeChaseSize);
  for (uint32_t i = 0; i < kFeChaseSize; ++i) {
    feChase[i] = i;
  }
  // Fisher-Yates shuffle with an LCG to build a hard-to-prefetch permutation.
  uint64_t s = 0x123456789abcdefull;
  for (uint32_t i = kFeChaseSize - 1; i > 0; --i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    uint32_t j = (uint32_t)((s >> 33) % (i + 1));
    uint32_t tmp = feChase[i];
    feChase[i] = feChase[j];
    feChase[j] = tmp;
  }
}

static void contendedUseFE(uint32_t n, int posters, int waiters) {
  feInitChase();

  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = (uint64_t)t + 1;
        uint32_t p = (uint32_t)(((uint32_t)t * 2654435761u) & (kFeChaseSize - 1));
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = feChase[p];
          uint8_t i0 = (uint8_t)(p & 0xFF);
          p = feChase[p];
          uint8_t i1 = (uint8_t)(p & 0xFF);
          p = feChase[p];
          uint8_t i2 = (uint8_t)(p & 0xFF);
          acc = feSwitch0(acc, i0);
          acc = feSwitch1(acc, i1);
          acc = feSwitch2(acc, i2);
        }
        feSink.fetch_add(acc, std::memory_order_relaxed);
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
BENCHMARK_NAMED_PARAM(contendedUseFE, 8_to_100_fe, 8, 100)

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
