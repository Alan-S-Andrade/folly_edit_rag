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

// ---------------------------------------------------------------------------
// Frontend-pressure helpers for the contendedUse(8_to_100) family variant.
// Three noinline 256-case switches with unique per-case literal constants
// force a large hot-code working set and icache thrashing when rotated by
// j % 3 over an index drawn from a pointer-chase load.  This is intentionally
// off the deepest dependency chain of the semaphore wait/post path.
// ---------------------------------------------------------------------------
#define FE_ALU(K, S)                                                  \
  case (K): {                                                         \
    acc ^= (uint64_t)((K) * 2654435761u + (S) + 0x9e3779b9u);        \
    acc += (uint64_t)(K) * 0x100000001b3ull + (S);                   \
    acc ^= acc >> 7;                                                  \
    acc *= (0xff51afd7ed558ccdull ^ (uint64_t)((K) + (S)));          \
    acc += (uint64_t)((K) ^ 0xdeadbeefu) + (S);                      \
    acc ^= acc << 13;                                                 \
    acc += (uint64_t)(K) * 7u + 11u + (S);                           \
    acc *= 0x2545F4914F6CDD1Dull;                                     \
    acc ^= (uint64_t)((K) + 0xcafeu + (S));                          \
    acc += acc >> 3;                                                  \
    break;                                                            \
  }

#define FE_G4(K, S) FE_ALU(K + 0, S) FE_ALU(K + 1, S) FE_ALU(K + 2, S) FE_ALU(K + 3, S)
#define FE_G16(K, S) FE_G4(K + 0, S) FE_G4(K + 4, S) FE_G4(K + 8, S) FE_G4(K + 12, S)
#define FE_G64(K, S) FE_G16(K + 0, S) FE_G16(K + 16, S) FE_G16(K + 32, S) FE_G16(K + 48, S)
#define FE_G256(S) FE_G64(0, S) FE_G64(64, S) FE_G64(128, S) FE_G64(192, S)

__attribute__((noinline)) static uint64_t feMix0(uint64_t acc, uint8_t idx) {
  switch (idx) { FE_G256(0x11u) }
  return acc;
}

__attribute__((noinline)) static uint64_t feMix1(uint64_t acc, uint8_t idx) {
  switch (idx) { FE_G256(0x47u) }
  return acc;
}

__attribute__((noinline)) static uint64_t feMix2(uint64_t acc, uint8_t idx) {
  switch (idx) { FE_G256(0x8Du) }
  return acc;
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

// Nearby variant of contendedUse(8_to_100): identical contended semaphore
// structure, but each waiter additionally drives a pointer-chase-indexed
// rotation across three large noinline switch bodies, materially expanding
// the timed hot-path instruction footprint (frontend / icache pressure).
static void contendedUseFrontend(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        constexpr uint32_t kRing = 1024;
        std::vector<uint32_t> next(kRing);
        for (uint32_t r = 0; r < kRing; ++r) {
          next[r] = (r * 2654435761u + 1u) % kRing;
        }
        uint64_t acc = 0x12345678ull ^ static_cast<uint64_t>(t);
        uint32_t p = static_cast<uint32_t>(t) % kRing;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = next[p];
          uint8_t idx = static_cast<uint8_t>((acc ^ p) & 0xFF);
          switch ((i + static_cast<uint32_t>(t)) % 3) {
            case 0:
              acc = feMix0(acc, idx);
              break;
            case 1:
              acc = feMix1(acc, idx);
              break;
            default:
              acc = feMix2(acc, idx);
              break;
          }
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
BENCHMARK_NAMED_PARAM(contendedUseFrontend, 8_to_100_fe, 8, 100)
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
