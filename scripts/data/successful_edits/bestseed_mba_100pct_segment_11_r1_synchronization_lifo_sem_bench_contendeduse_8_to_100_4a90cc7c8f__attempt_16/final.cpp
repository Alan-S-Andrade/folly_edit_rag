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
// Derived from contendedUse(8_to_100): same LifoSem contention shape, but the
// poster/waiter hot loops also execute a large rotating switch dispatch to
// inflate the executed instruction working set (I-cache footprint) and lower
// IPC toward the frontend-bound target. Three noinline 256-case switches are
// rotated by (i % 3); each case performs a fixed batch of ALU ops on a local
// accumulator using unique literal constants (salt + case index) so the
// compiler cannot deduplicate/merge the bodies. The dispatch index comes from
// a small pointer-chase load so the branch target is data-dependent.
// ---------------------------------------------------------------------------
namespace {

// 14 ALU ops per case; constants are unique per (case index n, salt S).
#define DO_CASE(n, S)                                  \
  case (n): {                                          \
    a += (uint64_t)(n) * 0x9E3779B97F4A7C15ull + (S);  \
    a ^= a >> 7;                                        \
    a += ((uint64_t)(n) ^ (0xABCDull + (S)));          \
    a *= ((uint64_t)(n) | 1ull);                        \
    a -= ((uint64_t)(n) * 3ull + 7ull + (S));          \
    a ^= (a << 11);                                     \
    a += (uint64_t)(n) * 5ull;                          \
    a *= (0x100000001B3ull ^ (uint64_t)(n) ^ (S));     \
    a += (a >> 3);                                      \
    a ^= (uint64_t)(n) * 17ull;                         \
    a += (uint64_t)(n) * 0x1234ull + (S);              \
    a *= ((uint64_t)(n) + 3ull);                        \
    a ^= (a << 5);                                      \
    a += (uint64_t)(n) + (S);                           \
    break;                                              \
  }

#define CASES_16(B, S)                                              \
  DO_CASE((B) + 0, S) DO_CASE((B) + 1, S) DO_CASE((B) + 2, S)       \
      DO_CASE((B) + 3, S) DO_CASE((B) + 4, S) DO_CASE((B) + 5, S)   \
          DO_CASE((B) + 6, S) DO_CASE((B) + 7, S) DO_CASE((B) + 8, S) \
              DO_CASE((B) + 9, S) DO_CASE((B) + 10, S)              \
                  DO_CASE((B) + 11, S) DO_CASE((B) + 12, S)         \
                      DO_CASE((B) + 13, S) DO_CASE((B) + 14, S)     \
                          DO_CASE((B) + 15, S)

#define CASES_256(S)                                                  \
  CASES_16(0, S) CASES_16(16, S) CASES_16(32, S) CASES_16(48, S)      \
      CASES_16(64, S) CASES_16(80, S) CASES_16(96, S) CASES_16(112, S) \
          CASES_16(128, S) CASES_16(144, S) CASES_16(160, S)          \
              CASES_16(176, S) CASES_16(192, S) CASES_16(208, S)      \
                  CASES_16(224, S) CASES_16(240, S)

__attribute__((noinline)) static uint64_t hotSwitchA(uint64_t a, uint8_t idx) {
  switch (idx) { CASES_256(0x1111ull) }
  return a;
}

__attribute__((noinline)) static uint64_t hotSwitchB(uint64_t a, uint8_t idx) {
  switch (idx) { CASES_256(0x2222ull) }
  return a;
}

__attribute__((noinline)) static uint64_t hotSwitchC(uint64_t a, uint8_t idx) {
  switch (idx) { CASES_256(0x3333ull) }
  return a;
}

#undef CASES_256
#undef CASES_16
#undef DO_CASE

static std::vector<uint32_t>& chaseBuf() {
  static std::vector<uint32_t> buf = [] {
    std::vector<uint32_t> v(4096);
    for (size_t i = 0; i < v.size(); ++i) {
      v[i] = static_cast<uint32_t>((i * 2654435761ull + 1) & (v.size() - 1));
    }
    return v;
  }();
  return buf;
}

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  auto& chase = chaseBuf();
  const uint32_t mask = static_cast<uint32_t>(chase.size() - 1);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t a = 0x9E3779B9ull + static_cast<uint64_t>(t);
        uint32_t p = static_cast<uint32_t>(t);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & mask];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
            case 0:
              a = hotSwitchA(a, idx);
              break;
            case 1:
              a = hotSwitchB(a, idx);
              break;
            default:
              a = hotSwitchC(a, idx);
              break;
          }
        }
        folly::doNotOptimizeAway(a);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &chase] {
        while (!go.load()) {
          std::this_thread::yield();
        }
        uint64_t a = 0x1234567ull + static_cast<uint64_t>(t);
        uint32_t p = static_cast<uint32_t>(t) + 7u;
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          p = chase[p & mask];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
            case 0:
              a = hotSwitchA(a, idx);
              break;
            case 1:
              a = hotSwitchB(a, idx);
              break;
            default:
              a = hotSwitchC(a, idx);
              break;
          }
        }
        folly::doNotOptimizeAway(a);
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
