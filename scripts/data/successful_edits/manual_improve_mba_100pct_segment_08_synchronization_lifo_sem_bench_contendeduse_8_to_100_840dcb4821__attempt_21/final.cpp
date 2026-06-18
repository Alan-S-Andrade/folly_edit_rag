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
// Derived benchmark: contendedUseMix(8_to_100)
//
// Same producer/consumer LifoSem contention as contendedUse(8_to_100), but the
// poster threads additionally execute a large, branchy code footprint to push
// up the executed-instruction working set (front-end / L1i pressure). The hot
// loop pointer-chases a small ring to derive an unpredictable index payload,
// masks it to a byte, and dispatches it into one of three large 256-case switch
// functions, rotating among them with i % 3. Each case carries a chain of ALU
// ops on a local accumulator and per-case unique literals so the compiler
// cannot deduplicate the case bodies or the three functions.
// ---------------------------------------------------------------------------

#define MIXCASE(n, salt)                                  \
  case (n): {                                             \
    acc += (uint64_t)(n) * 2654435761u + (uint64_t)(salt);\
    acc ^= acc >> 13;                                     \
    acc *= ((uint64_t)(n) | 1u);                          \
    acc += 0x9e3779b97f4a7c15ull ^ (uint64_t)(n);         \
    acc ^= acc << 7;                                      \
    acc -= (uint64_t)(n) * 40503u + (uint64_t)(salt);     \
    acc *= 2246822519u;                                   \
    acc ^= acc >> 17;                                     \
    acc += (uint64_t)(n) * 668265263u;                    \
    acc ^= acc << 5;                                      \
    acc *= 374761393u;                                    \
    acc -= (uint64_t)(n) ^ (uint64_t)(salt);              \
    acc ^= acc >> 11;                                     \
    acc += (uint64_t)(n) << 3;                            \
  } break;

#define MIX_C16(M, base)                                            \
  M((base) + 0) M((base) + 1) M((base) + 2) M((base) + 3)           \
  M((base) + 4) M((base) + 5) M((base) + 6) M((base) + 7)           \
  M((base) + 8) M((base) + 9) M((base) + 10) M((base) + 11)         \
  M((base) + 12) M((base) + 13) M((base) + 14) M((base) + 15)

#define MIX_C256(M)                                                 \
  MIX_C16(M, 0) MIX_C16(M, 16) MIX_C16(M, 32) MIX_C16(M, 48)        \
  MIX_C16(M, 64) MIX_C16(M, 80) MIX_C16(M, 96) MIX_C16(M, 112)      \
  MIX_C16(M, 128) MIX_C16(M, 144) MIX_C16(M, 160) MIX_C16(M, 176)   \
  MIX_C16(M, 192) MIX_C16(M, 208) MIX_C16(M, 224) MIX_C16(M, 240)

#define MIXCASE_A(n) MIXCASE(n, 0x1111111111111111ull)
#define MIXCASE_B(n) MIXCASE(n, 0x2222222222222222ull)
#define MIXCASE_C(n) MIXCASE(n, 0x3333333333333333ull)

__attribute__((noinline)) static uint64_t mix_a(uint64_t acc, uint8_t idx) {
  switch (idx) {
    MIX_C256(MIXCASE_A)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t mix_b(uint64_t acc, uint8_t idx) {
  switch (idx) {
    MIX_C256(MIXCASE_B)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t mix_c(uint64_t acc, uint8_t idx) {
  switch (idx) {
    MIX_C256(MIXCASE_C)
  }
  return acc;
}

#undef MIXCASE_A
#undef MIXCASE_B
#undef MIXCASE_C
#undef MIX_C256
#undef MIX_C16
#undef MIXCASE

static void contendedUseMix(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr size_t kRing = 4096;
  std::vector<uint32_t> chase(kRing);

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kRing; ++i) {
      chase[i] = (uint32_t)((i * 2654435761u + 12345u) % kRing);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
        }
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &chase] {
        while (!go.load()) {
          std::this_thread::yield();
        }
        uint64_t acc = (uint64_t)t + 1;
        uint32_t p = (uint32_t)t;
        for (uint32_t i = t; i < n; i += posters) {
          p = chase[p & (kRing - 1)];
          uint8_t idx = (uint8_t)(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = mix_a(acc, idx);
              break;
            case 1:
              acc = mix_b(acc, idx);
              break;
            default:
              acc = mix_c(acc, idx);
              break;
          }
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
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1, 31, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 16_to_16, 16, 16)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_32, 32, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1000, 32, 1000)
BENCHMARK_NAMED_PARAM(contendedUseMix, 8_to_100, 8, 100)

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
