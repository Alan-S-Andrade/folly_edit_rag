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
// This adds a compact but large instruction-footprint hot path on the waiter
// side: a pointer-chase load whose payload indexes three distinct 256-case
// switch functions, rotated by (i/waiters)%3. Each case performs a short
// sequence of ALU ops on a local accumulator using unique literal constants
// so the compiler cannot deduplicate cases. The result is a much larger
// executed instruction working set (more L1i pressure, lower IPC) without
// changing the semaphore post/wait operation volume or thread topology.
// ---------------------------------------------------------------------------
namespace {

#define LS_CASE_A(K)                            \
  case (K): {                                   \
    acc += (uint64_t)((K)*2654435761u + 1u);    \
    acc ^= (uint64_t)((K) + 0x9e37u) << 2;      \
    acc *= (uint64_t)(((K) | 1u) + 3u);         \
    acc += (uint64_t)((K) ^ 0x5bd1u);           \
    acc -= (uint64_t)((K)*7u + 11u);            \
    acc ^= (uint64_t)((K) << 1) + 0x1234u;      \
    acc *= 0x100000001b3ull;                    \
    acc += (uint64_t)(~(K) & 0xffu);            \
    acc ^= (uint64_t)((K)*13u + 17u);           \
    acc += (uint64_t)((K) + 0xa5a5u);           \
    acc -= (uint64_t)((K) << 3) ^ 0x77u;        \
    acc ^= (uint64_t)((K)*3u + 5u);             \
    break;                                      \
  }

#define LS_CASE_B(K)                            \
  case (K): {                                   \
    acc ^= (uint64_t)((K)*40503u + 7u);         \
    acc += (uint64_t)((K) + 0x1357u) << 1;      \
    acc *= (uint64_t)(((K) | 3u) + 5u);         \
    acc -= (uint64_t)((K) ^ 0x2468u);           \
    acc += (uint64_t)((K)*5u + 13u);            \
    acc ^= (uint64_t)((K) << 2) + 0x9999u;      \
    acc *= 0x9e3779b97f4a7c15ull;               \
    acc -= (uint64_t)(~(K) & 0x7fu);            \
    acc += (uint64_t)((K)*11u + 19u);           \
    acc ^= (uint64_t)((K) + 0x55aau);           \
    acc *= (uint64_t)(((K) << 1) | 1u);         \
    acc += (uint64_t)((K)*2u + 23u);            \
    break;                                      \
  }

#define LS_CASE_C(K)                            \
  case (K): {                                   \
    acc *= (uint64_t)(((K) | 1u) + 7u);         \
    acc += (uint64_t)((K)*65599u + 3u);         \
    acc ^= (uint64_t)((K) + 0x0f0fu) << 3;      \
    acc -= (uint64_t)((K)*9u + 29u);            \
    acc += (uint64_t)((K) ^ 0x3c3cu);           \
    acc ^= (uint64_t)((K) << 4) + 0xbeefu;      \
    acc *= 0xff51afd7ed558ccdull;               \
    acc += (uint64_t)(~(K) & 0x3fu);            \
    acc -= (uint64_t)((K)*15u + 31u);           \
    acc ^= (uint64_t)((K) + 0xc3c3u);           \
    acc += (uint64_t)(((K) << 2) | 5u);         \
    acc *= (uint64_t)((K)*3u + 37u) | 1u;       \
    break;                                      \
  }

#define LS_C4(M, K) M(K) M((K) + 1) M((K) + 2) M((K) + 3)
#define LS_C16(M, K) \
  LS_C4(M, K) LS_C4(M, (K) + 4) LS_C4(M, (K) + 8) LS_C4(M, (K) + 12)
#define LS_C64(M, K) \
  LS_C16(M, K) LS_C16(M, (K) + 16) LS_C16(M, (K) + 32) LS_C16(M, (K) + 48)
#define LS_C256(M, K)                                      \
  LS_C64(M, K) LS_C64(M, (K) + 64) LS_C64(M, (K) + 128)    \
  LS_C64(M, (K) + 192)

FOLLY_NOINLINE uint64_t lifoSwitchA(uint64_t acc, uint32_t idx) {
  switch (idx & 0xffu) {
    LS_C256(LS_CASE_A, 0)
  }
  return acc;
}

FOLLY_NOINLINE uint64_t lifoSwitchB(uint64_t acc, uint32_t idx) {
  switch (idx & 0xffu) {
    LS_C256(LS_CASE_B, 0)
  }
  return acc;
}

FOLLY_NOINLINE uint64_t lifoSwitchC(uint64_t acc, uint32_t idx) {
  switch (idx & 0xffu) {
    LS_C256(LS_CASE_C, 0)
  }
  return acc;
}

} // namespace

static void contendedUseFrontend(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        // Small per-thread pointer-chase table feeding the switch index.
        std::vector<uint32_t> chase(256);
        for (uint32_t k = 0; k < 256; ++k) {
          chase[k] = (k * 2654435761u + 1u) & 0xffu;
        }
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) & 0xffu;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p]; // dependent pointer-chase load
          uint32_t idx = static_cast<uint32_t>(acc ^ p) & 0xffu;
          switch ((i / static_cast<uint32_t>(waiters)) % 3u) {
            case 0:
              acc = lifoSwitchA(acc, idx);
              break;
            case 1:
              acc = lifoSwitchB(acc, idx);
              break;
            default:
              acc = lifoSwitchC(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUseFrontend, 8_to_100_frontend, 8, 100)
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
