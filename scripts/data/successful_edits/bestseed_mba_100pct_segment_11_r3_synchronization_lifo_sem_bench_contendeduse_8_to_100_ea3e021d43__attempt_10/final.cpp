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

// ----------------------------------------------------------------------------
// From-scratch frontend-pressure experiment derived for contendedUse(8_to_100).
// Three large noinline 256-case switches with 14 unique ALU ops per case build
// a wide hot-code footprint; rotation by (i/waiters)%3 and a pointer-chase
// payload index force I-cache thrashing to lift L1i MPKI / lower IPC.
// ----------------------------------------------------------------------------
#define WDL_ALU14(K, S)                          \
  acc += ((uint64_t)(K) * 0x01u + 0x11u + (S));  \
  acc ^= ((uint64_t)(K) * 0x03u + 0x27u + (S));  \
  acc *= ((uint64_t)(K) | 0x05u);                \
  acc += ((uint64_t)(K) ^ (0x9Eu + (S)));        \
  acc -= ((uint64_t)(K) * 0x02u + 0x31u + (S));  \
  acc ^= ((uint64_t)(K) + 0x113u + (S));         \
  acc *= ((uint64_t)(K) + 0x101u);               \
  acc += ((uint64_t)(K) ^ (0x55u + (S)));        \
  acc -= ((uint64_t)(K) | 0x21u);                \
  acc ^= ((uint64_t)(K) * 0x05u + 0x43u + (S));  \
  acc += ((uint64_t)(K) + 0x77u + (S));          \
  acc *= ((uint64_t)(K) | 0x09u);                \
  acc -= ((uint64_t)(K) ^ (0x3Cu + (S)));        \
  acc += ((uint64_t)(K) * 0x07u + 0x1Bu + (S));

#define WDL_C1(n, S)  case (n): { WDL_ALU14(n, S); break; }
#define WDL_C4(n, S)  WDL_C1(n, S) WDL_C1(n + 1, S) WDL_C1(n + 2, S) WDL_C1(n + 3, S)
#define WDL_C16(n, S) WDL_C4(n, S) WDL_C4(n + 4, S) WDL_C4(n + 8, S) WDL_C4(n + 12, S)
#define WDL_C64(n, S) WDL_C16(n, S) WDL_C16(n + 16, S) WDL_C16(n + 32, S) WDL_C16(n + 48, S)
#define WDL_C256(S)   WDL_C64(0, S) WDL_C64(64, S) WDL_C64(128, S) WDL_C64(192, S)

static __attribute__((noinline)) uint64_t footprintSwitchA(
    uint64_t payload, uint64_t acc) {
  switch (payload & 0xFF) { WDL_C256(0x1000u) }
  return acc;
}
static __attribute__((noinline)) uint64_t footprintSwitchB(
    uint64_t payload, uint64_t acc) {
  switch (payload & 0xFF) { WDL_C256(0x2000u) }
  return acc;
}
static __attribute__((noinline)) uint64_t footprintSwitchC(
    uint64_t payload, uint64_t acc) {
  switch (payload & 0xFF) { WDL_C256(0x3000u) }
  return acc;
}

static void contendedUseFootprint(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr size_t kRing = 8192; // power of two
  std::vector<uint32_t> ring(kRing);

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kRing; ++i) {
      ring[i] = static_cast<uint32_t>((i * 2654435761u + 12345u) % kRing);
    }

    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &ring] {
        uint64_t acc = 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(t);
        uint32_t idx = static_cast<uint32_t>(t) & (kRing - 1);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          idx = ring[idx & (kRing - 1)];
          uint64_t payload = ring[idx] + acc;
          switch ((i / static_cast<uint32_t>(waiters)) % 3) {
            case 0:
              acc = footprintSwitchA(payload, acc);
              break;
            case 1:
              acc = footprintSwitchB(payload, acc);
              break;
            default:
              acc = footprintSwitchC(payload, acc);
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
