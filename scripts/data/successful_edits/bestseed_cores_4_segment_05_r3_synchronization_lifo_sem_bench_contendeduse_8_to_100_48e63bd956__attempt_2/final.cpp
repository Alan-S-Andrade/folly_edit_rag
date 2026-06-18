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

// Derived from contendedUse(8_to_100): identical contention structure, plus a
// large icache-resident switch dispatch driven by a pointer-chase index to
// raise the executed instruction working set (Tier T1 frontend / L1i MPKI).
namespace {

constexpr uint32_t kIcacheChaseSize = 4096;

std::vector<uint32_t>& icacheChase() {
  static std::vector<uint32_t> chase = [] {
    std::vector<uint32_t> v(kIcacheChaseSize);
    for (uint32_t i = 0; i < kIcacheChaseSize; ++i) {
      v[i] = (i * 2654435761u + 12345u) & (kIcacheChaseSize - 1);
    }
    return v;
  }();
  return chase;
}

std::atomic<uint64_t> g_icacheSink{0};

// 8 ALU ops per case; unique literal constants (seeded per function and per
// case) prevent the compiler from deduplicating case bodies.
#define ICACHE_ALU_OPS(n, S)                            \
  acc += (uint64_t)((n) * 2654435761ull + (S) + 0x1111ull); \
  acc ^= (uint64_t)((n) * 40503ull + (S) + 0x2222ull);      \
  acc *= (uint64_t)(((n) | 1u) + (S) + 0x3ull);             \
  acc += (uint64_t)(((n) ^ 0x33u) + (S) + 0x4444ull);       \
  acc ^= (uint64_t)((n) * 7ull + (S) + 0x5555ull);          \
  acc *= (uint64_t)((((n) << 1) | 1u) + (S) + 0x7ull);      \
  acc += (uint64_t)((n) * 13ull + (S) + 0x6666ull);         \
  acc ^= (uint64_t)((n) * 17ull + (S) + 0x7777ull);

#define ICACHE_CASE0(n) \
  case (n): {           \
    ICACHE_ALU_OPS(n, 0x10000ull) break;                  \
  }
#define ICACHE_CASE1(n) \
  case (n): {           \
    ICACHE_ALU_OPS(n, 0x20000ull) break;                  \
  }
#define ICACHE_CASE2(n) \
  case (n): {           \
    ICACHE_ALU_OPS(n, 0x30000ull) break;                  \
  }

#define ICACHE_C16(M, b)                                                  \
  M(b + 0) M(b + 1) M(b + 2) M(b + 3) M(b + 4) M(b + 5) M(b + 6) M(b + 7) \
  M(b + 8) M(b + 9) M(b + 10) M(b + 11) M(b + 12) M(b + 13) M(b + 14)     \
  M(b + 15)
#define ICACHE_C256(M)                                                      \
  ICACHE_C16(M, 0) ICACHE_C16(M, 16) ICACHE_C16(M, 32) ICACHE_C16(M, 48)    \
  ICACHE_C16(M, 64) ICACHE_C16(M, 80) ICACHE_C16(M, 96) ICACHE_C16(M, 112)  \
  ICACHE_C16(M, 128) ICACHE_C16(M, 144) ICACHE_C16(M, 160)                  \
  ICACHE_C16(M, 176) ICACHE_C16(M, 192) ICACHE_C16(M, 208)                  \
  ICACHE_C16(M, 224) ICACHE_C16(M, 240)

__attribute__((noinline)) uint64_t icacheSwitch0(uint64_t acc, uint8_t idx) {
  switch (idx) { ICACHE_C256(ICACHE_CASE0) }
  return acc;
}
__attribute__((noinline)) uint64_t icacheSwitch1(uint64_t acc, uint8_t idx) {
  switch (idx) { ICACHE_C256(ICACHE_CASE1) }
  return acc;
}
__attribute__((noinline)) uint64_t icacheSwitch2(uint64_t acc, uint8_t idx) {
  switch (idx) { ICACHE_C256(ICACHE_CASE2) }
  return acc;
}

#undef ICACHE_C256
#undef ICACHE_C16
#undef ICACHE_CASE0
#undef ICACHE_CASE1
#undef ICACHE_CASE2
#undef ICACHE_ALU_OPS

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  const std::vector<uint32_t>& chase = icacheChase();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int j = 0; j < 32; ++j) {
            p = chase[p & (kIcacheChaseSize - 1)];
            uint8_t idx = static_cast<uint8_t>(p & 0xFFu);
            switch (j % 3) {
              case 0:
                acc = icacheSwitch0(acc, idx);
                break;
              case 1:
                acc = icacheSwitch1(acc, idx);
                break;
              default:
                acc = icacheSwitch2(acc, idx);
                break;
            }
          }
        }
        g_icacheSink.fetch_add(acc, std::memory_order_relaxed);
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
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100, 8, 100)
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
