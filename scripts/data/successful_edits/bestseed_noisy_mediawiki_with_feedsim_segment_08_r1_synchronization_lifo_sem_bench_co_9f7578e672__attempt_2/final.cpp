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
// Frontend-bound (I-cache) derived variant of contendedUse(8_to_100).
//
// Three noinline functions each contain a 256-case switch with a fixed number
// of ALU ops per case. Each case uses unique literal constants (the case index
// is folded into every constant) to defeat compiler case deduplication, which
// gives a very large hot-code footprint. The hot loop rotates among the three
// functions with i%3 and indexes the switch with a pointer-chased payload byte,
// thrashing the L1 instruction cache.
// ---------------------------------------------------------------------------
#define FE_ALU(n, salt)                              \
  {                                                  \
    const uint64_t k = (uint64_t)((n) + (salt));     \
    acc += k * 0x9e3779b97f4a7c15ull;                \
    acc ^= acc >> 23;                                \
    acc += 0x100000001b3ull ^ k;                     \
    acc *= (0xff51afd7ed558ccdull + k);              \
    acc ^= (k << 7) | 0x5bd1e995ull;                 \
    acc -= (0xc6a4a7935bd1e995ull - k);              \
    acc += (k ^ 0xdeadbeefull);                      \
    acc ^= acc >> 17;                                \
    acc *= (0x27d4eb2f165667c5ull + (k << 1));       \
    acc += (0xcafebabeull + k);                      \
    acc ^= (0xfeedfaceull ^ (k * 3));                \
    acc += ((k << 5) - 0x12345ull);                  \
    acc *= (1469598103934665603ull + k);             \
    acc ^= (k + 0x7f4a7c15ull);                      \
    acc += (k * 0x65d200ce55b19ad8ull);              \
    acc ^= (acc >> 31);                              \
  }

#define FE_CASE(n, salt) \
  case (n):              \
    FE_ALU(n, salt)      \
    break;

#define FE_CASES_16(base, salt)                                       \
  FE_CASE((base) + 0, salt) FE_CASE((base) + 1, salt)                 \
  FE_CASE((base) + 2, salt) FE_CASE((base) + 3, salt)                 \
  FE_CASE((base) + 4, salt) FE_CASE((base) + 5, salt)                 \
  FE_CASE((base) + 6, salt) FE_CASE((base) + 7, salt)                 \
  FE_CASE((base) + 8, salt) FE_CASE((base) + 9, salt)                 \
  FE_CASE((base) + 10, salt) FE_CASE((base) + 11, salt)               \
  FE_CASE((base) + 12, salt) FE_CASE((base) + 13, salt)               \
  FE_CASE((base) + 14, salt) FE_CASE((base) + 15, salt)

#define FE_CASES_256(salt)                                            \
  FE_CASES_16(0, salt) FE_CASES_16(16, salt) FE_CASES_16(32, salt)    \
  FE_CASES_16(48, salt) FE_CASES_16(64, salt) FE_CASES_16(80, salt)   \
  FE_CASES_16(96, salt) FE_CASES_16(112, salt) FE_CASES_16(128, salt) \
  FE_CASES_16(144, salt) FE_CASES_16(160, salt) FE_CASES_16(176, salt) \
  FE_CASES_16(192, salt) FE_CASES_16(208, salt) FE_CASES_16(224, salt) \
  FE_CASES_16(240, salt)

__attribute__((noinline)) static uint64_t fe_switch_a(uint8_t idx, uint64_t acc) {
  switch (idx) { FE_CASES_256(0x1111) }
  return acc;
}

__attribute__((noinline)) static uint64_t fe_switch_b(uint8_t idx, uint64_t acc) {
  switch (idx) { FE_CASES_256(0x2222) }
  return acc;
}

__attribute__((noinline)) static uint64_t fe_switch_c(uint8_t idx, uint64_t acc) {
  switch (idx) { FE_CASES_256(0x3333) }
  return acc;
}

static void contendedUseFe(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink{0};

  static constexpr size_t kBufSize = 8192;
  static std::vector<uint32_t> chase = [] {
    std::vector<uint32_t> v(kBufSize);
    for (size_t i = 0; i < kBufSize; ++i) {
      v[i] = static_cast<uint32_t>((i * 2654435761u) ^ (i << 7));
    }
    return v;
  }();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t acc = 0x12345ull + static_cast<uint64_t>(t);
        uint32_t p = static_cast<uint32_t>(t) * 2654435761u;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & (kBufSize - 1)];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = fe_switch_a(idx, acc);
              break;
            case 1:
              acc = fe_switch_b(idx, acc);
              break;
            default:
              acc = fe_switch_c(idx, acc);
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
BENCHMARK_NAMED_PARAM(contendedUseFe, 8_to_100_icache, 8, 100)

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
