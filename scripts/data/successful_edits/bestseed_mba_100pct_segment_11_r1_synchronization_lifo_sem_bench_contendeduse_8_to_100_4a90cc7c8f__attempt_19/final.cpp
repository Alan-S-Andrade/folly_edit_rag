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
// Frontend-bound derivative of contendedUse(8_to_100).
//
// Three noinline 256-case switch functions form a large hot-code footprint.
// Each case performs 14 ALU ops on a local accumulator using unique literal
// constants (preventing compiler deduplication). The hot loop rotates among
// the three functions with i%3 and indexes the switch with the low byte of a
// pointer-chase load, forcing L1 instruction-cache thrashing to raise the
// executed instruction working set and lower IPC toward the target band.
// ---------------------------------------------------------------------------

#define ICACHE_CASE(i, s)                                          \
  case i: {                                                        \
    acc += 0x9E3779B9u + (uint32_t)((i) + (s));                    \
    acc ^= (acc << 5) ^ (uint32_t)((i) * 2654435761u + (s));      \
    acc *= (uint32_t)(0x85EBCA77u ^ ((uint32_t)(i) + (uint32_t)(s))); \
    acc += (uint32_t)((i) * 7u + (s) + 3u);                        \
    acc ^= (acc >> 3) + (uint32_t)((i) ^ (s));                     \
    acc += (uint32_t)((i) << 2) + (uint32_t)(s);                  \
    acc ^= 0xC2B2AE35u + (uint32_t)((i) + (s) * 3u);              \
    acc *= (0x27D4EB2Fu | 1u);                                    \
    acc += (uint32_t)((i) * 3u + (s));                            \
    acc ^= (acc << 7) ^ (uint32_t)((s) + (i));                    \
    acc -= (uint32_t)((i) + 17u + (s));                           \
    acc ^= (uint32_t)((i) * 11u + 0x5Au);                         \
    acc += (uint32_t)((i) ^ 0x55u);                               \
    acc *= 0x9E3779B1u;                                           \
    break;                                                        \
  }

#define ICACHE_R4(i, s) \
  ICACHE_CASE(i, s) ICACHE_CASE(i + 1, s) ICACHE_CASE(i + 2, s) ICACHE_CASE(i + 3, s)
#define ICACHE_R16(i, s) \
  ICACHE_R4(i, s) ICACHE_R4(i + 4, s) ICACHE_R4(i + 8, s) ICACHE_R4(i + 12, s)
#define ICACHE_R64(i, s) \
  ICACHE_R16(i, s) ICACHE_R16(i + 16, s) ICACHE_R16(i + 32, s) ICACHE_R16(i + 48, s)
#define ICACHE_R256(i, s) \
  ICACHE_R64(i, s) ICACHE_R64(i + 64, s) ICACHE_R64(i + 128, s) ICACHE_R64(i + 192, s)

__attribute__((noinline)) static uint32_t icacheSwitch0(uint32_t acc, uint8_t idx) {
  switch (idx) { ICACHE_R256(0, 0) }
  return acc;
}

__attribute__((noinline)) static uint32_t icacheSwitch1(uint32_t acc, uint8_t idx) {
  switch (idx) { ICACHE_R256(0, 101) }
  return acc;
}

__attribute__((noinline)) static uint32_t icacheSwitch2(uint32_t acc, uint8_t idx) {
  switch (idx) { ICACHE_R256(0, 202) }
  return acc;
}

BENCHMARK(contendedUse_icache_8_to_100, iters) {
  constexpr size_t kChase = 4096;
  std::vector<uint32_t> chase(kChase);
  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kChase; ++i) {
      chase[i] = (uint32_t)((i * 2654435761u + 12345u) % kChase);
    }
  }

  uint32_t acc = 1u;
  uint32_t pos = 0u;
  for (size_t i = 0; i < iters; ++i) {
    uint32_t payload = chase[pos];
    uint8_t sw = (uint8_t)(payload & 0xFFu);
    switch (i % 3) {
      case 0:
        acc = icacheSwitch0(acc, sw);
        break;
      case 1:
        acc = icacheSwitch1(acc, sw);
        break;
      default:
        acc = icacheSwitch2(acc, sw);
        break;
    }
    pos = payload;
  }
  folly::doNotOptimizeAway(acc);
}

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
