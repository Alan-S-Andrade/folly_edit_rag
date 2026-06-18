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
// Frontend-pressure derived variant of contendedUse(8_to_100).
//
// This adds a large, deduplication-resistant executed-code footprint to the
// hot loop so that the benchmark stops being purely backend/IPC bound and
// starts taking L1 instruction-cache misses (target tier T0 IPC, driven by
// L1-icache-load-misses_MPKI).  Three noinline functions each contain a full
// 256-case switch with several ALU ops per case using unique literal
// constants; the hot loop rotates among them with j%3 and indexes the switch
// with the low byte of a pointer-chase load (which also creates data-cache
// pressure).
// ---------------------------------------------------------------------------
#define FE_CASE(i, S)                                       \
  case (i): {                                               \
    acc ^= (uint64_t)((i) * 2654435761u + (S) + 1u);        \
    acc += (uint64_t)(i) * (0x9E3779B97F4A7C15ull + (S));   \
    acc *= (uint64_t)((i) | 1u);                            \
    acc ^= acc >> 7;                                        \
    acc += (uint64_t)((i) ^ (0xABCDu + (S)));               \
    acc *= 0x100000001B3ull;                                \
    acc ^= (uint64_t)(i) << 13;                             \
    acc += (uint64_t)((i) * 7u + 3u + (S));                 \
    acc ^= acc >> 11;                                       \
    acc *= (uint64_t)((i) * 3u + 5u);                       \
    acc += (uint64_t)((i) ^ (0x1234u + (S)));               \
    acc ^= (uint64_t)(i) << 5;                              \
    break;                                                  \
  }

#define FE_C4(b, S) \
  FE_CASE(b + 0, S) FE_CASE(b + 1, S) FE_CASE(b + 2, S) FE_CASE(b + 3, S)
#define FE_C16(b, S) \
  FE_C4(b + 0, S) FE_C4(b + 4, S) FE_C4(b + 8, S) FE_C4(b + 12, S)
#define FE_C64(b, S) \
  FE_C16(b + 0, S) FE_C16(b + 16, S) FE_C16(b + 32, S) FE_C16(b + 48, S)
#define FE_C256(S) \
  FE_C64(0, S) FE_C64(64, S) FE_C64(128, S) FE_C64(192, S)

static uint64_t feSink;

__attribute__((noinline)) static uint64_t feSwitch0(uint8_t idx, uint64_t acc) {
  switch (idx) {
    FE_C256(0u)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) static uint64_t feSwitch1(uint8_t idx, uint64_t acc) {
  switch (idx) {
    FE_C256(1u)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) static uint64_t feSwitch2(uint8_t idx, uint64_t acc) {
  switch (idx) {
    FE_C256(2u)
    default:
      break;
  }
  return acc;
}

static void contendedUseFE(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static const uint32_t kChain = 1u << 16;
  std::vector<uint32_t> chain(kChain);

  BENCHMARK_SUSPEND {
    for (uint32_t i = 0; i < kChain; ++i) {
      chain[i] = (i * 2654435761u + 1u) & (kChain - 1u);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chain] {
        uint64_t acc = (uint64_t)t + 1u;
        uint32_t p = (uint32_t)t & (kChain - 1u);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chain[p & (kChain - 1u)];
          uint8_t idx = (uint8_t)(p & 0xFFu);
          switch ((i / (uint32_t)waiters) % 3u) {
            case 0:
              acc = feSwitch0(idx, acc);
              break;
            case 1:
              acc = feSwitch1(idx, acc);
              break;
            default:
              acc = feSwitch2(idx, acc);
              break;
          }
        }
        feSink += acc;
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &chain] {
        while (!go.load()) {
          std::this_thread::yield();
        }
        uint64_t acc = (uint64_t)t + 7u;
        uint32_t p = ((uint32_t)t + 13u) & (kChain - 1u);
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          p = chain[p & (kChain - 1u)];
          uint8_t idx = (uint8_t)(p & 0xFFu);
          switch ((i / (uint32_t)posters) % 3u) {
            case 0:
              acc = feSwitch0(idx, acc);
              break;
            case 1:
              acc = feSwitch1(idx, acc);
              break;
            default:
              acc = feSwitch2(idx, acc);
              break;
          }
        }
        feSink += acc;
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
