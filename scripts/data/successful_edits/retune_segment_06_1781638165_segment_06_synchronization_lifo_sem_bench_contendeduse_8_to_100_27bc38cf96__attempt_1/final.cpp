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
// Frontend-pressure variant of contendedUse.
//
// The original contendedUse hot path is essentially a tight sem.wait() loop.
// This variant injects a large, branchy instruction footprint into the waiter
// hot path: three large noinline switch routines, rotated per-iteration, whose
// case index is driven by a pointer-chase load. This materially changes the
// timed hot path (it is not a rename-only clone) and expands the executed
// instruction working set to exercise the frontend / i-cache.
// ---------------------------------------------------------------------------

// 12 ALU ops per case. The literal case index `n` and the per-function salt
// `s` make every case body use unique constants, preventing compiler
// deduplication and identical-code folding across the three routines.
#define FE_CASE(n, s)                                  \
  case (n):                                            \
    acc += (uint64_t)((n) * 2654435761u + 1u + (s));   \
    acc ^= (uint64_t)((n) * 40503u + 7u + (s));        \
    acc *= (uint64_t)(((n) | 1u) + (s) + 2u);          \
    acc -= (uint64_t)((n) * 2246822519u + 11u + (s));  \
    acc ^= (uint64_t)((n) * 3266489917u + 13u + (s));  \
    acc += (uint64_t)((n) * 668265263u + 17u + (s));   \
    acc *= 0x9e3779b1u;                                \
    acc ^= acc >> 15;                                  \
    acc += (uint64_t)((n) * 374761393u + 19u + (s));   \
    acc ^= (uint64_t)((n) * 1274126177u + 23u + (s));  \
    acc -= (uint64_t)((n) * 2147483647u + 29u + (s));  \
    acc *= (uint64_t)(((n) << 1) | 1u);                \
    break;

#define FE_C1(b, s) FE_CASE(b, s)
#define FE_C4(b, s) FE_C1(b, s) FE_C1(b + 1, s) FE_C1(b + 2, s) FE_C1(b + 3, s)
#define FE_C16(b, s) \
  FE_C4(b, s) FE_C4(b + 4, s) FE_C4(b + 8, s) FE_C4(b + 12, s)
#define FE_C64(b, s) \
  FE_C16(b, s) FE_C16(b + 16, s) FE_C16(b + 32, s) FE_C16(b + 48, s)
#define FE_C256(b, s) \
  FE_C64(b, s) FE_C64(b + 64, s) FE_C64(b + 128, s) FE_C64(b + 192, s)

__attribute__((noinline)) static uint64_t feSwitchA(uint64_t acc, uint8_t idx) {
  switch (idx) { FE_C256(0, 0x11u) }
  return acc;
}

__attribute__((noinline)) static uint64_t feSwitchB(uint64_t acc, uint8_t idx) {
  switch (idx) { FE_C256(0, 0x22u) }
  return acc;
}

__attribute__((noinline)) static uint64_t feSwitchC(uint64_t acc, uint8_t idx) {
  switch (idx) { FE_C256(0, 0x33u) }
  return acc;
}

#undef FE_C256
#undef FE_C64
#undef FE_C16
#undef FE_C4
#undef FE_C1
#undef FE_CASE

static void contendedUseFE(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        constexpr size_t kChase = 256;
        std::vector<uint32_t> chase(kChase);
        for (size_t k = 0; k < kChase; ++k) {
          chase[k] = static_cast<uint32_t>((k * 1103515245u + 12345u) % kChase);
        }
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) & 0xFFu;
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p % kChase];
          uint8_t idx = static_cast<uint8_t>((acc ^ p) & 0xFFu);
          switch (j % 3u) {
            case 0:
              acc = feSwitchA(acc, idx);
              break;
            case 1:
              acc = feSwitchB(acc, idx);
              break;
            default:
              acc = feSwitchC(acc, idx);
              break;
          }
          ++j;
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
BENCHMARK_NAMED_PARAM(contendedUseFE, 8_to_100_icache, 8, 100)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_4, 1, 4)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_32, 1, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_1, 4, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_24, 4, 24)
BENCHMARK_NAMED_PARAM(contendedUse, 8_to_100, 8, 100)
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
