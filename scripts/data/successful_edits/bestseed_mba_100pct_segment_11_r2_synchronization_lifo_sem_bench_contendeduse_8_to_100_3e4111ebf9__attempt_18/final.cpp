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

// ---------------------------------------------------------------------------
// Derived benchmark: contendedUseIcache(8_to_100)
//
// Same contended LifoSem post/wait shape as contendedUse(8_to_100), but each
// satisfied waiter performs a pointer-chase load and feeds the low byte of the
// chased payload into one of three large 256-case ALU switch dispatchers,
// rotated by j%3.  The three near-identical-but-salted dispatchers create a
// large hot instruction footprint so the executed I-stream working set no
// longer fits comfortably in the L1 instruction cache, lifting
// L1-icache-load-misses_MPKI and lowering IPC toward target.
// ---------------------------------------------------------------------------

// Repetition helpers to emit 256 distinct switch cases.
#define LIFO_R4(M, i) M(i) M((i) + 1) M((i) + 2) M((i) + 3)
#define LIFO_R16(M, i) \
  LIFO_R4(M, i) LIFO_R4(M, (i) + 4) LIFO_R4(M, (i) + 8) LIFO_R4(M, (i) + 12)
#define LIFO_R64(M, i)                                       \
  LIFO_R16(M, i) LIFO_R16(M, (i) + 16) LIFO_R16(M, (i) + 32) \
      LIFO_R16(M, (i) + 48)
#define LIFO_R256(M, i)                                       \
  LIFO_R64(M, i) LIFO_R64(M, (i) + 64) LIFO_R64(M, (i) + 128) \
      LIFO_R64(M, (i) + 192)

// ~14 ALU ops per case, each derived from the unique case index (and a
// per-function SALT) so the compiler/linker cannot deduplicate the bodies.
#define LIFO_ALUCASE(i)                                  \
  case (i): {                                            \
    uint64_t k = acc ^ (uint64_t)(SALT);                 \
    k += (uint64_t)(i) * 0x9E3779B97F4A7C15ULL + 1u;     \
    k ^= k >> 7;                                         \
    k *= ((uint64_t)(i) * 2654435761u) | 1u;             \
    k += (uint64_t)(i) ^ 0xABCDEF01u;                    \
    k ^= k << 17;                                        \
    k -= (uint64_t)(i) * 40503u + 3u;                    \
    k *= (0x100000001B3ULL + (uint64_t)(i)) | 1u;        \
    k ^= k >> 11;                                        \
    k += (uint64_t)(i) * 0x85EBCA6Bu;                    \
    k ^= (uint64_t)(i) << 3;                             \
    k *= (0xC2B2AE3D27D4EB4FULL ^ (uint64_t)(i)) | 1u;   \
    k -= (uint64_t)(i) + 0x27D4EB2Fu;                    \
    k ^= k >> 15;                                        \
    k += (uint64_t)(i) * 0xCC9E2D51u + 7u;               \
    acc = k;                                             \
  } break;

#define SALT 0x1111111111111111ULL
__attribute__((noinline)) static uint64_t lifoIcacheMixA(
    uint8_t idx, uint64_t acc) {
  switch (idx) { LIFO_R256(LIFO_ALUCASE, 0) }
  return acc;
}
#undef SALT

#define SALT 0x2222222222222222ULL
__attribute__((noinline)) static uint64_t lifoIcacheMixB(
    uint8_t idx, uint64_t acc) {
  switch (idx) { LIFO_R256(LIFO_ALUCASE, 0) }
  return acc;
}
#undef SALT

#define SALT 0x3333333333333333ULL
__attribute__((noinline)) static uint64_t lifoIcacheMixC(
    uint8_t idx, uint64_t acc) {
  switch (idx) { LIFO_R256(LIFO_ALUCASE, 0) }
  return acc;
}
#undef SALT

#undef LIFO_ALUCASE
#undef LIFO_R256
#undef LIFO_R64
#undef LIFO_R16
#undef LIFO_R4

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr size_t kRing = 4096;
  std::vector<uint32_t> ring(kRing);

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kRing; ++i) {
      ring[i] = static_cast<uint32_t>((i * 2654435761u + 1u) % kRing);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &ring] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) & (kRing - 1);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = ring[p & (kRing - 1)];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (j % 3) {
            case 0:
              acc = lifoIcacheMixA(idx, acc);
              break;
            case 1:
              acc = lifoIcacheMixB(idx, acc);
              break;
            default:
              acc = lifoIcacheMixC(idx, acc);
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

BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100, 8, 100)

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
