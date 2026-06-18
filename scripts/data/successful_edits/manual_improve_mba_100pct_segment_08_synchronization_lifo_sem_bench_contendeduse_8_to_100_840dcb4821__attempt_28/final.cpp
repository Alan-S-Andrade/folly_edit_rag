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
// Derived benchmark: contendedUseIcache
//
// Same contended LifoSem post/wait skeleton as contendedUse, but each waiter
// also runs a large, branchy switch-based payload between wakeups so that the
// executed instruction working set is much bigger.  This deliberately inflates
// the L1 instruction-cache footprint (three giant noinline 256-case switch
// functions rotated in the hot loop) to drive L1-icache-load-misses up and pull
// the frontend-bound IPC down toward target.
// ---------------------------------------------------------------------------
namespace {

// Exactly 14 unique ALU ops per case; constants derive from the case index and
// a per-function salt so the compiler cannot deduplicate cases or functions.
#define LIFO_OPS(k)                                                  \
  acc += (uint64_t)(k) * 2654435761ull + (LIFO_SALT) + 0x11u;        \
  acc ^= (uint64_t)(k) * 40503ull + (LIFO_SALT) + 0x22u;             \
  acc *= ((uint64_t)(k) | 0x33u) + (LIFO_SALT);                      \
  acc += (uint64_t)(k) << 3;                                         \
  acc ^= (uint64_t)(k) * 0x100000001b3ull + 0x44u;                   \
  acc -= (uint64_t)(k) * 7u + (LIFO_SALT) + 0x55u;                   \
  acc *= 0x2545F4914F6CDD1Dull | (uint64_t)(k);                      \
  acc += (uint64_t)(k) * 0xFF51AFD7ED558CCDull;                      \
  acc ^= (uint64_t)(k) + (LIFO_SALT) + 0x66u;                        \
  acc *= ((uint64_t)(k) * 3u) | 0x77u;                               \
  acc += (uint64_t)(k) * 0xC4CEB9FE1A85EC53ull;                      \
  acc ^= ((uint64_t)(k) << 5) + 0x88u;                               \
  acc -= (uint64_t)(k) * 11u + (LIFO_SALT);                          \
  acc *= ((uint64_t)(k) | 0x99u) + (LIFO_SALT) + 0x1u;

#define LIFO_CASE(k) \
  case (k): {        \
    LIFO_OPS(k)      \
    break;           \
  }
#define LIFO_C4(k) \
  LIFO_CASE(k) LIFO_CASE(k + 1) LIFO_CASE(k + 2) LIFO_CASE(k + 3)
#define LIFO_C16(k) \
  LIFO_C4(k) LIFO_C4(k + 4) LIFO_C4(k + 8) LIFO_C4(k + 12)
#define LIFO_C64(k) \
  LIFO_C16(k) LIFO_C16(k + 16) LIFO_C16(k + 32) LIFO_C16(k + 48)
#define LIFO_C256 LIFO_C64(0) LIFO_C64(64) LIFO_C64(128) LIFO_C64(192)

#define LIFO_SALT 0x9E37u
__attribute__((noinline)) uint64_t lifoIcacheA(uint8_t idx, uint64_t acc) {
  switch (idx) { LIFO_C256 }
  return acc;
}
#undef LIFO_SALT

#define LIFO_SALT 0x85EBu
__attribute__((noinline)) uint64_t lifoIcacheB(uint8_t idx, uint64_t acc) {
  switch (idx) { LIFO_C256 }
  return acc;
}
#undef LIFO_SALT

#define LIFO_SALT 0xC2B2u
__attribute__((noinline)) uint64_t lifoIcacheC(uint8_t idx, uint64_t acc) {
  switch (idx) { LIFO_C256 }
  return acc;
}
#undef LIFO_SALT

#undef LIFO_C256
#undef LIFO_C64
#undef LIFO_C16
#undef LIFO_C4
#undef LIFO_CASE
#undef LIFO_OPS

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  // Power-of-two pointer-chase ring shared by all waiters.
  static constexpr size_t kRing = 4096;
  static std::vector<uint64_t> ring = [] {
    std::vector<uint64_t> r(kRing);
    for (size_t i = 0; i < kRing; ++i) {
      r[i] = i * 2654435761ull + 12345ull;
    }
    return r;
  }();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        size_t p = (static_cast<size_t>(t) * 1099511628211ull) & (kRing - 1);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int j = 0; j < 32; ++j) {
            uint64_t payload = ring[p];
            uint8_t idx = static_cast<uint8_t>(payload & 0xFFu);
            switch (j % 3) {
              case 0:
                acc = lifoIcacheA(idx, acc);
                break;
              case 1:
                acc = lifoIcacheB(idx, acc);
                break;
              default:
                acc = lifoIcacheC(idx, acc);
                break;
            }
            p = static_cast<size_t>(payload ^ acc) & (kRing - 1);
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
