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

// Derived from-scratch frontend-bound footprint variant.
// Three noinline 256-case switches (14 unique ALU ops/case) are rotated by
// j % 3 and indexed by a payload byte from a dependent pointer-chase load.
// This inflates the executed instruction working set to raise
// L1-icache-load-misses_MPKI and lower IPC toward the contendedUse target.
namespace {

#define LSB_ALU_OPS(K, S)                                           \
  acc += (uint32_t)(K) * (2654435761u + (S)) + (0x9e3779b9u ^ (S)); \
  acc ^= acc >> 13;                                                 \
  acc += ((uint32_t)(K) ^ (0xdeadbeefu + (S)));                     \
  acc *= ((uint32_t)(K) | 1u);                                      \
  acc -= (uint32_t)(K) * (40503u + (S));                            \
  acc ^= acc << 7;                                                  \
  acc += (uint32_t)(K) * (0x1000193u + (S));                        \
  acc ^= ((uint32_t)(K) + (0xcafef00du ^ (S)));                     \
  acc *= (0x85ebca6bu + (S));                                       \
  acc ^= acc >> 11;                                                 \
  acc += (uint32_t)(K) * (0x27d4eb2fu + (S));                       \
  acc -= ((uint32_t)(K) ^ (0x12345678u + (S)));                     \
  acc *= ((uint32_t)(K) | 3u);                                      \
  acc ^= acc >> 5;

#define LSB_CASE(K, S) \
  case (K): {          \
    LSB_ALU_OPS(K, S)  \
    break;             \
  }
#define LSB_C4(b, S)                                          \
  LSB_CASE((b) + 0, S) LSB_CASE((b) + 1, S) LSB_CASE((b) + 2, \
                                              S) LSB_CASE((b) + 3, S)
#define LSB_C16(b, S)                                            \
  LSB_C4((b) + 0, S) LSB_C4((b) + 4, S) LSB_C4((b) + 8, S) LSB_C4((b) + 12, S)
#define LSB_C64(b, S)                                              \
  LSB_C16((b) + 0, S) LSB_C16((b) + 16, S) LSB_C16((b) + 32, S)    \
  LSB_C16((b) + 48, S)
#define LSB_C256(S)                                                \
  LSB_C64(0, S) LSB_C64(64, S) LSB_C64(128, S) LSB_C64(192, S)

__attribute__((noinline)) uint32_t footprintFn0(uint32_t acc, uint8_t idx) {
  switch (idx) { LSB_C256(1u) }
  return acc;
}
__attribute__((noinline)) uint32_t footprintFn1(uint32_t acc, uint8_t idx) {
  switch (idx) { LSB_C256(2u) }
  return acc;
}
__attribute__((noinline)) uint32_t footprintFn2(uint32_t acc, uint8_t idx) {
  switch (idx) { LSB_C256(3u) }
  return acc;
}

#undef LSB_C256
#undef LSB_C64
#undef LSB_C16
#undef LSB_C4
#undef LSB_CASE
#undef LSB_ALU_OPS

} // namespace

static void contendedUseFootprint(uint32_t iters, int posters, int waiters) {
  const size_t kSize = 4096;
  std::vector<uint32_t> ring(kSize);

  BENCHMARK_SUSPEND {
    uint32_t seed = (uint32_t)(2 * waiters + posters) | 1u;
    for (size_t i = 0; i < kSize; ++i) {
      ring[i] = (uint32_t)((i * 2654435761u) ^ (i << 7) ^ (seed * (i + 1)));
    }
  }

  uint32_t acc = 5381;
  size_t pos = 0;
  for (uint32_t j = 0; j < iters; ++j) {
    uint32_t payload = ring[pos];
    uint8_t idx = (uint8_t)(payload & 0xFFu);
    switch (j % 3) {
      case 0:
        acc = footprintFn0(acc, idx);
        break;
      case 1:
        acc = footprintFn1(acc, idx);
        break;
      default:
        acc = footprintFn2(acc, idx);
        break;
    }
    pos = (payload ^ acc) & (kSize - 1);
  }
  folly::doNotOptimizeAway(acc);
}

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
