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
// Frontend-footprint companion to contendedUse.
//
// To exercise the instruction cache, each waiter runs a large, rotating set of
// 256-case switches (three distinct noinline functions selected by j % 3). The
// switch index comes from a pointer-chase load so the branch target is
// data-dependent and unpredictable. Each case uses unique literal constants to
// prevent the compiler/linker from deduplicating the case bodies, and each of
// the three functions carries a distinct salt to prevent identical-code
// folding. The per-case ALU op count is the primary L1i-MPKI knob.
// ---------------------------------------------------------------------------
#define FP_CASE(K)                                       \
  case (K): {                                            \
    acc += 0x100u + (uint32_t)(K) * 2654435761u;         \
    acc ^= (acc >> 13) ^ (uint32_t)(K);                  \
    acc *= ((uint32_t)(K) | 1u);                         \
    acc += 0x9e3779b9u + (uint32_t)(K);                  \
    acc ^= (acc << 7);                                   \
    acc *= (0x85ebca6bu + (uint32_t)(K));                \
    acc += (uint32_t)(K) * 3u + 7u;                      \
    acc ^= (acc >> 11);                                  \
    acc *= ((uint32_t)(K) + 0xc2b2ae35u);                \
    acc += (uint32_t)(K) << 3;                           \
    acc ^= (acc >> 5);                                   \
    acc *= (0x27d4eb2fu ^ (uint32_t)(K));                \
    acc += (uint32_t)(K) * 5u + 13u;                     \
    acc ^= (acc << 9);                                   \
    break;                                               \
  }

#define FP_4(K) FP_CASE(K) FP_CASE((K) + 1) FP_CASE((K) + 2) FP_CASE((K) + 3)
#define FP_16(K) FP_4(K) FP_4((K) + 4) FP_4((K) + 8) FP_4((K) + 12)
#define FP_64(K) FP_16(K) FP_16((K) + 16) FP_16((K) + 32) FP_16((K) + 48)
#define FP_256(K) FP_64(K) FP_64((K) + 64) FP_64((K) + 128) FP_64((K) + 192)

#define MAKE_FP_FN(NAME, SALT)                                          \
  __attribute__((noinline)) static uint32_t NAME(                      \
      uint32_t acc, uint8_t idx) {                                     \
    acc ^= (SALT);                                                     \
    switch (idx) {                                                     \
      FP_256(0)                                                        \
      default:                                                         \
        acc += (SALT);                                                 \
        break;                                                         \
    }                                                                  \
    return acc;                                                        \
  }

MAKE_FP_FN(fpMixA, 0xA5A5A5A5u)
MAKE_FP_FN(fpMixB, 0x5A5A5A5Au)
MAKE_FP_FN(fpMixC, 0x3C3C3C3Cu)

static void contendedUseFootprint(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint32_t> sink(0);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        constexpr size_t kMask = 4096 - 1;
        std::vector<uint32_t> chase(kMask + 1);
        for (size_t k = 0; k <= kMask; ++k) {
          chase[k] = static_cast<uint32_t>((k * 2654435761u + 1u) & kMask);
        }
        uint32_t acc = static_cast<uint32_t>(t) + 1u;
        size_t p = static_cast<size_t>(t) & kMask;
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p];
          uint8_t idx = static_cast<uint8_t>(p & 0xFFu);
          switch (j % 3) {
            case 0:
              acc = fpMixA(acc, idx);
              break;
            case 1:
              acc = fpMixB(acc, idx);
              break;
            default:
              acc = fpMixC(acc, idx);
              break;
          }
          ++j;
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
BENCHMARK_DRAW_LINE();
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
