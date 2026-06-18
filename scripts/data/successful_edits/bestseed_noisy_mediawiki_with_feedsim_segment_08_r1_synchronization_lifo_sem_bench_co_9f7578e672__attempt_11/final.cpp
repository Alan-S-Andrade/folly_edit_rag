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
// Derived icache-footprint variant of contendedUse(8_to_100).
//
// The contended LifoSem traffic is preserved, but each post/wait iteration is
// interleaved with a heavy ALU switch dispatched from a pointer-chase load.
// Three noinline 256-case switch functions are rotated with j%3 so that the
// executed instruction working set is large enough to thrash the L1 icache,
// pulling IPC down toward the contendedUse target. Each case uses unique
// literal constants (derived from the case index) to defeat compiler
// deduplication, and the three function bodies differ in their epilogue so
// that identical-code-folding cannot merge them.
// ---------------------------------------------------------------------------
#define ALU_CASE(n)                                  \
  case (n): {                                        \
    acc += 0x9e3779b97f4a7c15ULL ^ (uint64_t)(n);    \
    acc ^= (uint64_t)(n) * 2654435761ULL;            \
    acc += ((uint64_t)(n) << 1) + 1ULL;              \
    acc *= ((uint64_t)(n) | 1ULL);                   \
    acc ^= (uint64_t)(n) + 0x12345ULL;               \
    acc -= (uint64_t)(n) * 7ULL;                     \
    acc += (uint64_t)(n) ^ 0xa5a5a5a5ULL;            \
    acc ^= acc >> 5;                                 \
    acc *= 0x100000001b3ULL;                         \
    acc += (uint64_t)(n) * 3ULL;                     \
    acc ^= (uint64_t)(n) << 2;                       \
    acc += 0xdeadbeefULL ^ (uint64_t)(n);            \
    acc *= ((uint64_t)(n) | 3ULL);                   \
    acc ^= acc >> 9;                                 \
    break;                                           \
  }

#define ALU_C16(b)                                                          \
  ALU_CASE((b) + 0x0)                                                       \
  ALU_CASE((b) + 0x1) ALU_CASE((b) + 0x2) ALU_CASE((b) + 0x3)               \
  ALU_CASE((b) + 0x4) ALU_CASE((b) + 0x5) ALU_CASE((b) + 0x6)               \
  ALU_CASE((b) + 0x7) ALU_CASE((b) + 0x8) ALU_CASE((b) + 0x9)               \
  ALU_CASE((b) + 0xA) ALU_CASE((b) + 0xB) ALU_CASE((b) + 0xC)               \
  ALU_CASE((b) + 0xD) ALU_CASE((b) + 0xE) ALU_CASE((b) + 0xF)

#define ALU_CASES                                                           \
  ALU_C16(0x00) ALU_C16(0x10) ALU_C16(0x20) ALU_C16(0x30)                   \
  ALU_C16(0x40) ALU_C16(0x50) ALU_C16(0x60) ALU_C16(0x70)                   \
  ALU_C16(0x80) ALU_C16(0x90) ALU_C16(0xA0) ALU_C16(0xB0)                   \
  ALU_C16(0xC0) ALU_C16(0xD0) ALU_C16(0xE0) ALU_C16(0xF0)

__attribute__((noinline)) static uint64_t aluMix0(uint64_t acc, uint32_t idx) {
  switch (idx & 0xFFu) {
    ALU_CASES
    default:
      break;
  }
  return acc + 0x1111ULL;
}

__attribute__((noinline)) static uint64_t aluMix1(uint64_t acc, uint32_t idx) {
  switch (idx & 0xFFu) {
    ALU_CASES
    default:
      break;
  }
  return acc + 0x2222ULL;
}

__attribute__((noinline)) static uint64_t aluMix2(uint64_t acc, uint32_t idx) {
  switch (idx & 0xFFu) {
    ALU_CASES
    default:
      break;
  }
  return acc + 0x3333ULL;
}

#undef ALU_CASES
#undef ALU_C16
#undef ALU_CASE

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr uint32_t kChase = 4096u;
  std::vector<uint32_t> chase(kChase);

  BENCHMARK_SUSPEND {
    for (uint32_t i = 0; i < kChase; ++i) {
      chase[i] = (i * 2654435761u + 1u) & (kChase - 1u);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = 0x100ULL + (uint64_t)t;
        uint32_t p = (uint32_t)t & (kChase - 1u);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p];
          uint32_t idx = chase[p] ^ (uint32_t)acc;
          switch (j % 3u) {
            case 0:
              acc = aluMix0(acc, idx);
              break;
            case 1:
              acc = aluMix1(acc, idx);
              break;
            default:
              acc = aluMix2(acc, idx);
              break;
          }
          ++j;
        }
        folly::doNotOptimizeAway(acc);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &chase] {
        while (!go.load()) {
          std::this_thread::yield();
        }
        uint64_t acc = 0x200ULL + (uint64_t)t;
        uint32_t p = ((uint32_t)t * 7u) & (kChase - 1u);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += posters) {
          p = chase[p];
          uint32_t idx = chase[p] ^ (uint32_t)acc;
          switch (j % 3u) {
            case 0:
              acc = aluMix0(acc, idx);
              break;
            case 1:
              acc = aluMix1(acc, idx);
              break;
            default:
              acc = aluMix2(acc, idx);
              break;
          }
          ++j;
          sem.post();
        }
        folly::doNotOptimizeAway(acc);
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
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100, 8, 100)
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
