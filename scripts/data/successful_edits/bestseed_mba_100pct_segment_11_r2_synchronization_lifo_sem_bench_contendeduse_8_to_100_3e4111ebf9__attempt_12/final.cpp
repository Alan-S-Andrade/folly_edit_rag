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
// Derived from contendedUse(8_to_100): same poster/waiter LifoSem contention,
// but each waiter wakeup drives a large switch-based ALU workload spread across
// three noinline functions to enlarge the executed instruction working set
// (raises L1i footprint / lowers IPC toward the frontend-bound target).
// ---------------------------------------------------------------------------
namespace {

std::atomic<uint64_t> icacheSink{0};

#define ALU_CASE(K)                                          \
  case (K): {                                                \
    acc += (uint64_t)((K) * 2654435761u + 0x9e3779b9u);      \
    acc ^= (uint64_t)((K) * 40503u + 1013904223u);           \
    acc *= (uint64_t)(((K) | 1u));                           \
    acc += (uint64_t)((K) ^ 0x5bd1e995u);                    \
    acc ^= acc >> 7;                                         \
    acc *= (0xff51afd7ed558ccdULL ^ (uint64_t)(K));          \
    acc += (uint64_t)((K) * 7u + 3u);                        \
    acc ^= (uint64_t)((K) << 3);                             \
    acc *= (uint64_t)((K) + 0x1234u);                        \
    acc += (uint64_t)((K) ^ 0xabcdu);                        \
    acc ^= (uint64_t)((K) * 11u);                            \
    acc *= (uint64_t)(((K) | 0x100u));                       \
    acc += (uint64_t)((K) * 13u + 17u);                      \
    acc ^= (uint64_t)((K) * SALT + 0x77u);                   \
    break;                                                   \
  }

#define C16(B)                                                  \
  ALU_CASE((B) + 0) ALU_CASE((B) + 1) ALU_CASE((B) + 2)         \
  ALU_CASE((B) + 3) ALU_CASE((B) + 4) ALU_CASE((B) + 5)         \
  ALU_CASE((B) + 6) ALU_CASE((B) + 7) ALU_CASE((B) + 8)         \
  ALU_CASE((B) + 9) ALU_CASE((B) + 10) ALU_CASE((B) + 11)       \
  ALU_CASE((B) + 12) ALU_CASE((B) + 13) ALU_CASE((B) + 14)      \
  ALU_CASE((B) + 15)

#define C256                                                    \
  C16(0) C16(16) C16(32) C16(48) C16(64) C16(80) C16(96)        \
  C16(112) C16(128) C16(144) C16(160) C16(176) C16(192)         \
  C16(208) C16(224) C16(240)

#define SALT 0x85ebca6bu
__attribute__((noinline)) static uint64_t icacheMix0(
    uint8_t idx, uint64_t acc) {
  switch (idx) {
    C256
    default:
      break;
  }
  return acc;
}
#undef SALT

#define SALT 0xc2b2ae35u
__attribute__((noinline)) static uint64_t icacheMix1(
    uint8_t idx, uint64_t acc) {
  switch (idx) {
    C256
    default:
      break;
  }
  return acc;
}
#undef SALT

#define SALT 0x27d4eb2fu
__attribute__((noinline)) static uint64_t icacheMix2(
    uint8_t idx, uint64_t acc) {
  switch (idx) {
    C256
    default:
      break;
  }
  return acc;
}
#undef SALT

#undef C256
#undef C16
#undef ALU_CASE

} // namespace

static void contendedUseICache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr uint32_t kRingMask = 8192u - 1u;
  static std::vector<uint32_t> ring;

  BENCHMARK_SUSPEND {
    if (ring.empty()) {
      ring.resize(kRingMask + 1u);
      uint32_t x = 0x12345678u;
      for (uint32_t i = 0; i <= kRingMask; ++i) {
        x = x * 1664525u + 1013904223u;
        ring[i] = x;
      }
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = 0x9e3779b97f4a7c15ULL + (uint64_t)t;
        uint32_t p = ring[t & kRingMask];
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int j = 0; j < 16; ++j) {
            p = ring[p & kRingMask];
            uint8_t idx = (uint8_t)(p & 0xFFu);
            switch (j % 3) {
              case 0:
                acc = icacheMix0(idx, acc);
                break;
              case 1:
                acc = icacheMix1(idx, acc);
                break;
              default:
                acc = icacheMix2(idx, acc);
                break;
            }
          }
        }
        icacheSink.fetch_add(acc, std::memory_order_relaxed);
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
BENCHMARK_NAMED_PARAM(contendedUseICache, 8_to_100, 8, 100)

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
