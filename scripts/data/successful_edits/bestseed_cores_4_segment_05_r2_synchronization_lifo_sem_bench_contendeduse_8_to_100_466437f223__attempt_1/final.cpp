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
// Frontend-perturbed variant of contendedUse used by the
// contendedUseFE(8_to_100) benchmark. Each waiter, after returning from a
// wait(), drives a pointer-chase index into one of three large 256-case
// switches. This expands the hot instruction footprint (icache pressure) and
// adds a small unpredictable control-flow site, without altering the
// throughput-carrying post/wait operation volume.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kLsemChaseSize = 256;
uint8_t g_lsemChase[kLsemChaseSize];

struct LsemChaseInit {
  LsemChaseInit() {
    for (size_t i = 0; i < kLsemChaseSize; ++i) {
      // Non-trivial permutation-like walk to defeat easy prefetch/predict.
      g_lsemChase[i] = uint8_t((i * 167u + 13u) & 0xFFu);
    }
  }
} g_lsemChaseInit;

#define LSEM_OP_CASE(n, salt)                              \
  case ((n)):                                              \
    acc += (uint64_t)((n) * 2654435761ull + (salt));       \
    acc ^= acc >> 7;                                       \
    acc *= (uint64_t)((n) * 40503ull + 3u + (salt));       \
    acc += (uint64_t)((n) ^ (0xA5A5u + (salt)));           \
    acc ^= acc << 3;                                       \
    acc -= (uint64_t)((n) * 17ull + (salt));               \
    acc *= 0x100000001b3ull;                               \
    acc ^= acc >> 11;                                      \
    acc += (uint64_t)((n) + 12345u + (salt));              \
    acc ^= (uint64_t)((n) * 31ull + (salt));               \
    break;

#define LSEM_CASES_16(b, salt)                                              \
  LSEM_OP_CASE((b) + 0, salt) LSEM_OP_CASE((b) + 1, salt)                   \
  LSEM_OP_CASE((b) + 2, salt) LSEM_OP_CASE((b) + 3, salt)                   \
  LSEM_OP_CASE((b) + 4, salt) LSEM_OP_CASE((b) + 5, salt)                   \
  LSEM_OP_CASE((b) + 6, salt) LSEM_OP_CASE((b) + 7, salt)                   \
  LSEM_OP_CASE((b) + 8, salt) LSEM_OP_CASE((b) + 9, salt)                   \
  LSEM_OP_CASE((b) + 10, salt) LSEM_OP_CASE((b) + 11, salt)                 \
  LSEM_OP_CASE((b) + 12, salt) LSEM_OP_CASE((b) + 13, salt)                 \
  LSEM_OP_CASE((b) + 14, salt) LSEM_OP_CASE((b) + 15, salt)

#define LSEM_SWITCH_BODY(salt)                                              \
  switch (idx) {                                                            \
    LSEM_CASES_16(0, salt) LSEM_CASES_16(16, salt)                         \
    LSEM_CASES_16(32, salt) LSEM_CASES_16(48, salt)                        \
    LSEM_CASES_16(64, salt) LSEM_CASES_16(80, salt)                        \
    LSEM_CASES_16(96, salt) LSEM_CASES_16(112, salt)                       \
    LSEM_CASES_16(128, salt) LSEM_CASES_16(144, salt)                      \
    LSEM_CASES_16(160, salt) LSEM_CASES_16(176, salt)                      \
    LSEM_CASES_16(192, salt) LSEM_CASES_16(208, salt)                      \
    LSEM_CASES_16(224, salt) LSEM_CASES_16(240, salt)                      \
  }

FOLLY_NOINLINE uint64_t lsemSwitchA(uint8_t idx, uint64_t acc) {
  LSEM_SWITCH_BODY(1u)
  return acc;
}

FOLLY_NOINLINE uint64_t lsemSwitchB(uint8_t idx, uint64_t acc) {
  LSEM_SWITCH_BODY(7u)
  return acc;
}

FOLLY_NOINLINE uint64_t lsemSwitchC(uint8_t idx, uint64_t acc) {
  LSEM_SWITCH_BODY(13u)
  return acc;
}

#undef LSEM_SWITCH_BODY
#undef LSEM_CASES_16
#undef LSEM_OP_CASE

} // namespace

static void contendedUseFE(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint8_t p = uint8_t(t * 53 + 1);
        uint64_t acc = uint64_t(t) * 0x9E3779B1u + 1;
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          // Frontend perturbation: pointer-chase index feeds a rotated set
          // of large switches. Kept off the post/wait dependency chain.
          p = g_lsemChase[p];
          uint8_t idx = p;
          switch (j % 3) {
            case 0:
              acc = lsemSwitchA(idx, acc);
              break;
            case 1:
              acc = lsemSwitchB(idx, acc);
              break;
            default:
              acc = lsemSwitchC(idx, acc);
              break;
          }
          ++j;
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
BENCHMARK_NAMED_PARAM(contendedUseFE, 8_to_100, 8, 100)
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
