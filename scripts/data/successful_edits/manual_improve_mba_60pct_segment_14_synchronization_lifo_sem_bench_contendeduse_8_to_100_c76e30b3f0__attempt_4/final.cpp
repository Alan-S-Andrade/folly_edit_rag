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
// Frontend-bound derived workload (icache-pressure) used by contendedUseFE.
//
// Three noinline functions, each a fully-populated 256-case switch with many
// ALU ops per case (every case uses unique literal constants to prevent
// compiler de-duplication / identical-code-folding). Rotating across the three
// functions with i%3 forces a large executed instruction working set, raising
// L1i misses and lowering IPC toward the contended-use target.
// ---------------------------------------------------------------------------
namespace {

#define FE_BODY(acc, n, salt)                                                  \
  (acc) += UINT64_C(0x9E3779B97F4A7C15) + (uint64_t)(n) + (uint64_t)(salt);    \
  (acc) ^= (acc) << 13;                                                        \
  (acc) -= (uint64_t)(n) * 0x517CC1B7u + (uint64_t)(salt);                     \
  (acc) ^= (acc) >> 7;                                                         \
  (acc) *= UINT64_C(0x100000001B3) + (uint64_t)(n);                            \
  (acc) += ((acc) << 3) ^ (0xA5A5A5A5u + (uint64_t)(n) + (uint64_t)(salt));    \
  (acc) ^= (uint64_t)(n) * (uint64_t)(n) + 0x55u;                              \
  (acc) -= (acc) >> 11;                                                        \
  (acc) *= 0x2545F491u | 1u;                                                   \
  (acc) += (((uint64_t)(n) << 4) ^ (0xCAFEu + (uint64_t)(salt)));              \
  (acc) ^= (acc) << 5;                                                         \
  (acc) += (uint64_t)(n) * 7u + (uint64_t)(salt);                              \
  (acc) ^= (acc) >> 3;                                                         \
  (acc) *= 3u + ((uint64_t)(n) << 1);

#define FE_C1(b, salt) \
  case (b): {          \
    FE_BODY(acc, (b), salt)  \
  } break;
#define FE_C4(b, salt) \
  FE_C1((b) + 0, salt) FE_C1((b) + 1, salt) FE_C1((b) + 2, salt) \
  FE_C1((b) + 3, salt)
#define FE_C16(b, salt) \
  FE_C4((b) + 0, salt) FE_C4((b) + 4, salt) FE_C4((b) + 8, salt) \
  FE_C4((b) + 12, salt)
#define FE_C64(b, salt) \
  FE_C16((b) + 0, salt) FE_C16((b) + 16, salt) FE_C16((b) + 32, salt) \
  FE_C16((b) + 48, salt)
#define FE_C256(salt) \
  FE_C64(0, salt) FE_C64(64, salt) FE_C64(128, salt) FE_C64(192, salt)

__attribute__((noinline)) uint64_t feSwitchA(uint64_t acc, uint8_t idx) {
  switch (idx) { FE_C256(0x11) }
  return acc;
}
__attribute__((noinline)) uint64_t feSwitchB(uint64_t acc, uint8_t idx) {
  switch (idx) { FE_C256(0x22) }
  return acc;
}
__attribute__((noinline)) uint64_t feSwitchC(uint64_t acc, uint8_t idx) {
  switch (idx) { FE_C256(0x33) }
  return acc;
}

const std::vector<uint32_t>& feChase() {
  static const std::vector<uint32_t> c = [] {
    std::vector<uint32_t> v(1024);
    uint32_t x = 0x12345677u;
    for (uint32_t i = 0; i < 1024; ++i) {
      x = x * 1664525u + 1013904223u;
      v[i] = x & 1023u;
    }
    return v;
  }();
  return c;
}

} // namespace

static void contendedUseFE(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  const std::vector<uint32_t>& chase = feChase();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) & 1023u;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & 1023u];
          uint8_t idx = static_cast<uint8_t>(p ^ (p >> 8));
          switch (i % 3) {
            case 0:
              acc = feSwitchA(acc, idx);
              break;
            case 1:
              acc = feSwitchB(acc, idx);
              break;
            default:
              acc = feSwitchC(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUseFE, 8_to_100_fe, 8, 100)
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
