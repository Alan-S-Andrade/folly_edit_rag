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
// The original contendedUse hot path executes an extremely compact loop body
// (a single sem.wait()/sem.post()), which gives a tiny executed-instruction
// working set and high IPC. To materially change the timed hot path of the
// derived benchmark we splice a large, branch-rich instruction footprint into
// the waiter loop: three noinline functions, each a 256-case switch with a
// long ALU chain per case using unique literal constants (so the compiler can
// neither deduplicate cases nor fold the functions together). The hot loop
// rotates among the three functions, thrashing the instruction cache and
// expanding the frontend footprint without altering the synchronization
// structure, thread counts, or operation volume of the family.
// ---------------------------------------------------------------------------

#define WDL_REP16(M, b)                                                       \
  M((b) + 0) M((b) + 1) M((b) + 2) M((b) + 3) M((b) + 4) M((b) + 5)          \
      M((b) + 6) M((b) + 7) M((b) + 8) M((b) + 9) M((b) + 10) M((b) + 11)    \
          M((b) + 12) M((b) + 13) M((b) + 14) M((b) + 15)

#define WDL_REP256(M)                                                        \
  WDL_REP16(M, 0) WDL_REP16(M, 16) WDL_REP16(M, 32) WDL_REP16(M, 48)         \
      WDL_REP16(M, 64) WDL_REP16(M, 80) WDL_REP16(M, 96) WDL_REP16(M, 112)   \
          WDL_REP16(M, 128) WDL_REP16(M, 144) WDL_REP16(M, 160)              \
              WDL_REP16(M, 176) WDL_REP16(M, 192) WDL_REP16(M, 208)          \
                  WDL_REP16(M, 224) WDL_REP16(M, 240)

#define WDL_CASE_A(k)                                                        \
  case (k): {                                                                \
    acc ^= (uint64_t)(((k)*2654435761u) + 0x9E3779B9u);                      \
    acc += (acc << 7) ^ (acc >> 3);                                          \
    acc *= 0x100000001b3ull;                                                 \
    acc += ((k) ^ 0x5bd1e995u);                                              \
    acc ^= (acc >> 11);                                                      \
    acc += ((k)*31u + 17u);                                                  \
    acc *= 0x27d4eb2full;                                                    \
    acc ^= (acc << 5);                                                       \
    acc += ((k) | 0x40u);                                                    \
    acc ^= ((k)&0xABu);                                                      \
    acc += (acc >> 9);                                                       \
    acc *= 0x85ebca6bull;                                                    \
    acc ^= ((k) + 0x1234u);                                                  \
    acc += (acc << 3);                                                       \
  } break;

#define WDL_CASE_B(k)                                                        \
  case (k): {                                                                \
    acc += (uint64_t)(((k)*40503u) ^ 0xCAFEBABEu);                           \
    acc ^= (acc << 13);                                                      \
    acc *= 0xD6E8FEB86659FD93ull;                                            \
    acc += ((k)*19u + 0x77u);                                                \
    acc ^= (acc >> 7);                                                       \
    acc += ((k) ^ 0xDEADBEEFu);                                              \
    acc *= 0xFF51AFD7ED558CCDull;                                            \
    acc ^= (acc << 17);                                                      \
    acc += ((k)&0x5Cu);                                                      \
    acc -= ((k) | 0x80u);                                                    \
    acc ^= (acc >> 23);                                                      \
    acc *= 0xC2B2AE3D27D4EB4Full;                                            \
    acc += ((k) + 0x9ABCu);                                                  \
    acc ^= (acc << 9);                                                       \
  } break;

#define WDL_CASE_C(k)                                                        \
  case (k): {                                                                \
    acc ^= (uint64_t)(((k)*2246822519u) + 0xB5297A4Du);                      \
    acc += (acc >> 15);                                                      \
    acc *= 0x165667B19E3779F9ull;                                            \
    acc ^= ((k)*23u + 0x3Bu);                                                \
    acc += (acc << 11);                                                      \
    acc ^= ((k) + 0x0F0Fu);                                                  \
    acc *= 0x9E3779B97F4A7C15ull;                                            \
    acc += (acc >> 6);                                                       \
    acc ^= ((k) | 0x12u);                                                    \
    acc += ((k)&0x3Du);                                                      \
    acc -= (acc << 4);                                                       \
    acc *= 0xBF58476D1CE4E5B9ull;                                            \
    acc ^= ((k) + 0x55AAu);                                                  \
    acc += (acc >> 8);                                                       \
  } break;

__attribute__((noinline)) static uint64_t wdlFeMixA(uint8_t idx, uint64_t acc) {
  switch (idx) { WDL_REP256(WDL_CASE_A) }
  return acc;
}

__attribute__((noinline)) static uint64_t wdlFeMixB(uint8_t idx, uint64_t acc) {
  switch (idx) { WDL_REP256(WDL_CASE_B) }
  return acc;
}

__attribute__((noinline)) static uint64_t wdlFeMixC(uint8_t idx, uint64_t acc) {
  switch (idx) { WDL_REP256(WDL_CASE_C) }
  return acc;
}

static void contendedUseFront(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          uint8_t idx = static_cast<uint8_t>(acc ^ (acc >> 8) ^ i);
          switch (i % 3) {
            case 0:
              acc = wdlFeMixA(idx, acc);
              break;
            case 1:
              acc = wdlFeMixB(idx, acc);
              break;
            default:
              acc = wdlFeMixC(idx, acc);
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
BENCHMARK_NAMED_PARAM(contendedUseFront, 8_to_100, 8, 100)
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
