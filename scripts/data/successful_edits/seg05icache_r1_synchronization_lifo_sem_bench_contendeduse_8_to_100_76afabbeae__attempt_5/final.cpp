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
// Same LifoSem contention pattern as contendedUse, but each completed wait
// drives a pointer-chase load whose low byte indexes one of four large
// 256-case switches. Rotating across the four noinline switch functions with
// (i % 4) inflates the executed instruction working set (hot code footprint),
// which raises L1-icache-load-misses_MPKI and pulls IPC back down toward the
// frontend-bound target. ALU ops-per-case is the primary L1i knob, and the
// fourth rotating switch widens the resident hot-code footprint further.
// ---------------------------------------------------------------------------

static uint8_t icacheChase[256];
static int icacheChaseInit = [] {
  for (int i = 0; i < 256; ++i) {
    icacheChase[i] = static_cast<uint8_t>((i * 167 + 13) & 0xFF);
  }
  return 0;
}();

// 6 ALU ops per case, each using unique literal constants (k and salt) to
// prevent compiler case/function deduplication.
#define ALU_BODY(acc, k, salt)                                  \
  acc += (uint64_t)((k) * 2654435761u + (salt) + 1u);           \
  acc ^= (uint64_t)((k) * 40503u + (salt) + 7u);                \
  acc *= (uint64_t)(((k) | 1u) + (salt));                       \
  acc += (uint64_t)((k) ^ (0x9E3779B9u + (salt)));              \
  acc ^= (uint64_t)((k) * 2246822519u + (salt));                \
  acc -= (uint64_t)((k) + 12345u + (salt));

#define ICACHE_CASE(k, salt) \
  case (k): {                \
    ALU_BODY(acc, (k), (salt)) \
  } break;

#define ICACHE_CASES_16(b, salt)                                          \
  ICACHE_CASE((b) + 0, salt) ICACHE_CASE((b) + 1, salt)                   \
  ICACHE_CASE((b) + 2, salt) ICACHE_CASE((b) + 3, salt)                   \
  ICACHE_CASE((b) + 4, salt) ICACHE_CASE((b) + 5, salt)                   \
  ICACHE_CASE((b) + 6, salt) ICACHE_CASE((b) + 7, salt)                   \
  ICACHE_CASE((b) + 8, salt) ICACHE_CASE((b) + 9, salt)                   \
  ICACHE_CASE((b) + 10, salt) ICACHE_CASE((b) + 11, salt)                 \
  ICACHE_CASE((b) + 12, salt) ICACHE_CASE((b) + 13, salt)                 \
  ICACHE_CASE((b) + 14, salt) ICACHE_CASE((b) + 15, salt)

#define ICACHE_CASES_256(salt)                                            \
  ICACHE_CASES_16(0, salt) ICACHE_CASES_16(16, salt)                      \
  ICACHE_CASES_16(32, salt) ICACHE_CASES_16(48, salt)                     \
  ICACHE_CASES_16(64, salt) ICACHE_CASES_16(80, salt)                     \
  ICACHE_CASES_16(96, salt) ICACHE_CASES_16(112, salt)                    \
  ICACHE_CASES_16(128, salt) ICACHE_CASES_16(144, salt)                   \
  ICACHE_CASES_16(160, salt) ICACHE_CASES_16(176, salt)                   \
  ICACHE_CASES_16(192, salt) ICACHE_CASES_16(208, salt)                   \
  ICACHE_CASES_16(224, salt) ICACHE_CASES_16(240, salt)

static __attribute__((noinline)) uint64_t icacheSwitch0(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_CASES_256(0x11u)
  }
  return acc;
}

static __attribute__((noinline)) uint64_t icacheSwitch1(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_CASES_256(0x22u)
  }
  return acc;
}

static __attribute__((noinline)) uint64_t icacheSwitch2(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_CASES_256(0x33u)
  }
  return acc;
}

static __attribute__((noinline)) uint64_t icacheSwitch3(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_CASES_256(0x44u)
  }
  return acc;
}

static void contendedUseIcacheMix(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint8_t idx = static_cast<uint8_t>(t * 97 + 1);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          uint8_t payload = icacheChase[idx];
          switch (i % 4) {
            case 0:
              acc = icacheSwitch0(acc, payload & 0xFF);
              break;
            case 1:
              acc = icacheSwitch1(acc, payload & 0xFF);
              break;
            case 2:
              acc = icacheSwitch2(acc, payload & 0xFF);
              break;
            default:
              acc = icacheSwitch3(acc, payload & 0xFF);
              break;
          }
          idx = static_cast<uint8_t>(payload ^ (acc & 0xFF));
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
BENCHMARK_NAMED_PARAM(contendedUseIcacheMix, 8_to_100_icache_x4, 8, 100)
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
