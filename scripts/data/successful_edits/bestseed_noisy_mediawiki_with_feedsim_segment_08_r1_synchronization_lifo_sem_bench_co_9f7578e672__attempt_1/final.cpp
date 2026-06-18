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
// This keeps the exact same contended LifoSem post/wait steady state but
// injects a large, hard-to-cache instruction footprint into the waiter hot
// path. Three noinline functions each contain a 256-case switch driven by a
// pointer-chase load; the hot loop rotates among them with i%3. The goal is to
// expand the executed instruction working set (raising L1i pressure) and bring
// IPC down toward the contended-blocking regime without altering operation
// volume or the post/wait serialization structure.
// ---------------------------------------------------------------------------
namespace {

#define LIFO_FE_CASE(K, S)                                          \
  case (K): {                                                       \
    acc ^= (uint64_t)((K) * 2654435761u + (S));                    \
    acc += (acc << 7) ^ (acc >> 3);                                \
    acc *= (0x9E3779B97F4A7C15ull ^ (uint64_t)((K) + (S)));        \
    acc -= (uint64_t)((K) * 40503u + (S) * 3u);                    \
    acc ^= (acc >> 11) + (uint64_t)(K);                            \
    acc += (acc << 5) - (uint64_t)((K) * 7u + (S));                \
    acc *= 0x100000001B3ull;                                       \
    acc ^= (acc << 13);                                            \
    acc -= (uint64_t)((K) ^ (S));                                  \
    acc += (acc >> 17) ^ (uint64_t)((K) * 131u);                   \
    break;                                                          \
  }

#define LIFO_FE_4(B, S)                                             \
  LIFO_FE_CASE((B) + 0, S)                                          \
  LIFO_FE_CASE((B) + 1, S)                                          \
  LIFO_FE_CASE((B) + 2, S)                                          \
  LIFO_FE_CASE((B) + 3, S)

#define LIFO_FE_16(B, S)                                            \
  LIFO_FE_4((B) + 0, S)                                             \
  LIFO_FE_4((B) + 4, S)                                             \
  LIFO_FE_4((B) + 8, S)                                             \
  LIFO_FE_4((B) + 12, S)

#define LIFO_FE_64(B, S)                                            \
  LIFO_FE_16((B) + 0, S)                                            \
  LIFO_FE_16((B) + 16, S)                                           \
  LIFO_FE_16((B) + 32, S)                                           \
  LIFO_FE_16((B) + 48, S)

#define LIFO_FE_256(S)                                              \
  LIFO_FE_64(0, S)                                                  \
  LIFO_FE_64(64, S)                                                 \
  LIFO_FE_64(128, S)                                                \
  LIFO_FE_64(192, S)

__attribute__((noinline)) uint64_t lifoFeMix0(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFO_FE_256(0x9111)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) uint64_t lifoFeMix1(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFO_FE_256(0x5223)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) uint64_t lifoFeMix2(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFO_FE_256(0xA337)
    default:
      break;
  }
  return acc;
}

#undef LIFO_FE_256
#undef LIFO_FE_64
#undef LIFO_FE_16
#undef LIFO_FE_4
#undef LIFO_FE_CASE

} // namespace

static void contendedUseFe(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        // Per-thread pointer-chase table; the loaded payload (& 0xFF) selects
        // the switch case, defeating branch prediction on the case index.
        std::vector<uint32_t> chase(256);
        for (uint32_t k = 0; k < 256; ++k) {
          chase[k] = (k * 167u + 13u) & 0xFF;
        }
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) & 0xFF;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = lifoFeMix0(acc, idx);
              break;
            case 1:
              acc = lifoFeMix1(acc, idx);
              break;
            default:
              acc = lifoFeMix2(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUseFe, 8_to_100_icache, 8, 100)
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
