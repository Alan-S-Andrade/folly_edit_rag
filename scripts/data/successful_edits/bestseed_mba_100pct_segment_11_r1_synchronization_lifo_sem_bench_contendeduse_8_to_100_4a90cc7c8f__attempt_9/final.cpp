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
// Derived benchmark: icacheContendedUse
//
// Same LifoSem contention pattern as contendedUse, but each woken waiter runs
// a pointer-chase load followed by a large 256-case switch dispatched across
// three noinline mixers. This intentionally inflates the executed instruction
// working set (L1i footprint) to lower IPC toward the contendedUse target
// while keeping the synchronization behavior intact.
// ---------------------------------------------------------------------------

static constexpr size_t kChaseSize = 4096; // power of two
static uint32_t gChase[kChaseSize];
[[maybe_unused]] static const bool gChaseInit = [] {
  uint32_t x = 0x12345678u;
  for (size_t i = 0; i < kChaseSize; ++i) {
    x = x * 1664525u + 1013904223u;
    gChase[i] = x;
  }
  return true;
}();

// 14 ALU ops per case, each using unique literal constants derived from the
// case index and a per-function salt to defeat compiler/linker dedup.
#define ICACHE_CASE(S, i)                                  \
  case (i): {                                              \
    acc += 0x9E3779B9u + (uint32_t)(i) + (uint32_t)(S);    \
    acc ^= acc >> 3;                                       \
    acc *= 2654435761u + (uint32_t)(i) * 7u;               \
    acc += (uint32_t)(i) * 131u + 1u;                      \
    acc ^= acc << 5;                                       \
    acc -= (uint32_t)(i) * 17u + (uint32_t)(S);            \
    acc *= 40503u + (uint32_t)(i) * 3u;                    \
    acc ^= acc >> 7;                                       \
    acc += 0x85EBCA6Bu ^ (uint32_t)(i);                    \
    acc *= 0xC2B2AE35u + (uint32_t)(i) * 5u;               \
    acc ^= acc << 11;                                      \
    acc += (uint32_t)(i) * 2246822519u;                    \
    acc ^= acc >> 13;                                      \
    acc *= 3266489917u + (uint32_t)(i) * 9u;               \
    break;                                                 \
  }

#define ICACHE_C16(S, b)                                                   \
  ICACHE_CASE(S, (b) + 0) ICACHE_CASE(S, (b) + 1) ICACHE_CASE(S, (b) + 2)  \
  ICACHE_CASE(S, (b) + 3) ICACHE_CASE(S, (b) + 4) ICACHE_CASE(S, (b) + 5)  \
  ICACHE_CASE(S, (b) + 6) ICACHE_CASE(S, (b) + 7) ICACHE_CASE(S, (b) + 8)  \
  ICACHE_CASE(S, (b) + 9) ICACHE_CASE(S, (b) + 10) ICACHE_CASE(S, (b) + 11)\
  ICACHE_CASE(S, (b) + 12) ICACHE_CASE(S, (b) + 13)                        \
  ICACHE_CASE(S, (b) + 14) ICACHE_CASE(S, (b) + 15)

#define ICACHE_C256(S)                                                     \
  ICACHE_C16(S, 0) ICACHE_C16(S, 16) ICACHE_C16(S, 32) ICACHE_C16(S, 48)   \
  ICACHE_C16(S, 64) ICACHE_C16(S, 80) ICACHE_C16(S, 96) ICACHE_C16(S, 112) \
  ICACHE_C16(S, 128) ICACHE_C16(S, 144) ICACHE_C16(S, 160)                 \
  ICACHE_C16(S, 176) ICACHE_C16(S, 192) ICACHE_C16(S, 208)                 \
  ICACHE_C16(S, 224) ICACHE_C16(S, 240)

__attribute__((noinline)) static uint32_t icacheMix0(uint32_t payload) {
  uint32_t acc = payload;
  switch (payload & 0xFFu) {
    ICACHE_C256(0x11111111u)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) static uint32_t icacheMix1(uint32_t payload) {
  uint32_t acc = payload;
  switch (payload & 0xFFu) {
    ICACHE_C256(0x22222222u)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) static uint32_t icacheMix2(uint32_t payload) {
  uint32_t acc = payload;
  switch (payload & 0xFFu) {
    ICACHE_C256(0x33333333u)
    default:
      break;
  }
  return acc;
}

#undef ICACHE_C256
#undef ICACHE_C16
#undef ICACHE_CASE

static void icacheContendedUse(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint32_t idx = (static_cast<uint32_t>(t) * 131u) & (kChaseSize - 1);
        uint32_t acc = static_cast<uint32_t>(t) + 1u;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          uint32_t payload = gChase[idx] ^ acc;
          idx = payload & (kChaseSize - 1);
          switch (i % 3u) {
            case 0:
              acc += icacheMix0(payload);
              break;
            case 1:
              acc += icacheMix1(payload);
              break;
            default:
              acc += icacheMix2(payload);
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
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_4, 1, 4)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_32, 1, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_1, 4, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_24, 4, 24)
BENCHMARK_NAMED_PARAM(contendedUse, 8_to_100, 8, 100)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1, 31, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 16_to_16, 16, 16)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_32, 32, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1000, 32, 1000)
BENCHMARK_NAMED_PARAM(icacheContendedUse, 8_to_100, 8, 100)

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
