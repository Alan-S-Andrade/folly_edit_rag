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
// Derived frontend-bound variant of contendedUse(8_to_100).
//
// The reference contended benchmark has a very compact instruction working
// set, which keeps IPC high and L1i misses low. To match the target hardware
// profile (lower IPC, higher L1-icache MPKI) we inject a large, hard to cache
// executed-code footprint into the waiter hot path: three noinline functions,
// each holding a 256-case switch with 14 unique ALU ops per case. The switch
// index comes from a pointer-chase load and we rotate among the three
// functions with i % 3 so the icache is thrashed across a big code region.
// ---------------------------------------------------------------------------
namespace {

#define LSB_ALU(C)                                       \
  acc += (uint64_t)((C) * 2654435761u + 1u);             \
  acc ^= (uint64_t)((C) * 40503u + 13u);                 \
  acc *= ((uint64_t)((C) * 2246822519u + 1u) | 1u);      \
  acc += (uint64_t)((C) * 3266489917u + 7u);             \
  acc ^= (uint64_t)((C) * 668265263u + 17u);             \
  acc -= (uint64_t)((C) * 374761393u + 23u);             \
  acc *= ((uint64_t)((C) * 2147483647u + 3u) | 1u);      \
  acc ^= (uint64_t)((C) * 1274126177u + 29u);            \
  acc += (uint64_t)((C) * 9176u + 31u);                  \
  acc -= (uint64_t)((C) * 246822519u + 37u);             \
  acc *= ((uint64_t)((C) * 66489917u + 5u) | 1u);        \
  acc ^= (uint64_t)((C) * 374761u + 41u);                \
  acc += (uint64_t)((C) * 668265u + 43u);                \
  acc ^= (uint64_t)((C) * 2246822u + 47u);

#define LSB_CASE(C) \
  case (C): {       \
    LSB_ALU(C)      \
  } break;
#define LSB_CB(B)                                                  \
  LSB_CASE(B + 0) LSB_CASE(B + 1) LSB_CASE(B + 2) LSB_CASE(B + 3)   \
  LSB_CASE(B + 4) LSB_CASE(B + 5) LSB_CASE(B + 6) LSB_CASE(B + 7)   \
  LSB_CASE(B + 8) LSB_CASE(B + 9) LSB_CASE(B + 10) LSB_CASE(B + 11) \
  LSB_CASE(B + 12) LSB_CASE(B + 13) LSB_CASE(B + 14) LSB_CASE(B + 15)
#define LSB_SW256                                                    \
  LSB_CB(0) LSB_CB(16) LSB_CB(32) LSB_CB(48) LSB_CB(64) LSB_CB(80)    \
  LSB_CB(96) LSB_CB(112) LSB_CB(128) LSB_CB(144) LSB_CB(160)          \
  LSB_CB(176) LSB_CB(192) LSB_CB(208) LSB_CB(224) LSB_CB(240)

template <int K>
__attribute__((noinline)) uint64_t lsbSwitch(uint8_t idx, uint64_t acc) {
  acc += (uint64_t)K * 0x9e3779b97f4a7c15ull;
  switch (idx) {
    LSB_SW256
  }
  return acc;
}

} // namespace

static void contendedUseSwitch(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink{0};

  constexpr size_t kRing = 4096;
  static std::vector<uint32_t> chain;

  BENCHMARK_SUSPEND {
    chain.resize(kRing);
    for (size_t i = 0; i < kRing; ++i) {
      chain[i] = (uint32_t)((i * 2654435761u) % kRing);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t acc = (uint64_t)t + 1;
        uint32_t p = (uint32_t)((t * 7 + 1) % (int)kRing);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chain[p];
          uint8_t idx = (uint8_t)(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = lsbSwitch<0>(idx, acc);
              break;
            case 1:
              acc = lsbSwitch<1>(idx, acc);
              break;
            default:
              acc = lsbSwitch<2>(idx, acc);
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
  folly::doNotOptimizeAway(sink.load());
}

BENCHMARK_DRAW_LINE();
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_1, 1, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_4, 1, 4)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_32, 1, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_1, 4, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_24, 4, 24)
BENCHMARK_NAMED_PARAM(contendedUse, 8_to_100, 8, 100)
BENCHMARK_NAMED_PARAM(contendedUseSwitch, 8_to_100, 8, 100)
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
