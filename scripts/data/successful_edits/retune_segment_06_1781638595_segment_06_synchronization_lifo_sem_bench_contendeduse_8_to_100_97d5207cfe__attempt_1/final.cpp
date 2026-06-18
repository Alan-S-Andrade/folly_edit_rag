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
// This is a deliberately distinct hot path: in addition to the contended
// LifoSem wait/post traffic, each waiter iteration performs a pointer-chase
// load and dispatches into one of three large noinline 256-case switch
// functions (rotated by j % 3). This materially expands the executed
// instruction working set (icache footprint) and inserts an unpredictable
// switch index derived from the chased payload, which lowers IPC relative to
// the lean original. The original contendedUse(...) benchmark is unchanged.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kChaseSize = 1024;
uint32_t gChase[kChaseSize];
std::atomic<uint64_t> gFeSink{0};

struct ChaseInit {
  ChaseInit() {
    for (size_t i = 0; i < kChaseSize; ++i) {
      gChase[i] =
          static_cast<uint32_t>((i * 2654435761ull + 12345ull) % kChaseSize);
    }
  }
} gChaseInit;

// ~8 ALU ops per case, each parameterized by the (unique) case literal K so
// the compiler cannot deduplicate cases.
#define ALU_A(K)                                          \
  {                                                       \
    acc += (uint64_t)((uint32_t)(K) * 2654435761u + 1u);  \
    acc ^= (uint64_t)((uint32_t)(K) * 40503u + 7u);       \
    acc *= ((uint64_t)(uint32_t)(K) | 1u);                \
    acc += acc >> 3;                                      \
    acc ^= (uint64_t)((uint32_t)(K) * 2246822519u);       \
    acc -= (uint64_t)((uint32_t)(K) * 3266489917u);       \
    acc *= 2654435761u;                                   \
    acc ^= acc << 5;                                      \
  }

#define ALU_B(K)                                          \
  {                                                       \
    acc ^= (uint64_t)((uint32_t)(K) * 2654435761u + 9u);  \
    acc += (uint64_t)((uint32_t)(K) * 974711u + 3u);      \
    acc *= ((uint64_t)(uint32_t)(K) | 3u);                \
    acc ^= acc >> 7;                                      \
    acc += (uint64_t)((uint32_t)(K) * 3266489917u);       \
    acc *= 40503u;                                        \
    acc -= (uint64_t)((uint32_t)(K) * 2246822519u);       \
    acc += acc << 11;                                     \
  }

#define ALU_C(K)                                          \
  {                                                       \
    acc += (uint64_t)((uint32_t)(K) * 19349663u + 5u);    \
    acc *= ((uint64_t)(uint32_t)(K) | 5u);                \
    acc ^= (uint64_t)((uint32_t)(K) * 83492791u + 11u);   \
    acc -= acc >> 9;                                      \
    acc += (uint64_t)((uint32_t)(K) * 2654435761u);       \
    acc ^= 2246822519u;                                   \
    acc *= 974711u;                                       \
    acc ^= acc << 13;                                     \
  }

#define A1(ALU, n) \
  case (n):        \
    ALU(n);        \
    break;
#define A4(ALU, n) A1(ALU, n) A1(ALU, (n) + 1) A1(ALU, (n) + 2) A1(ALU, (n) + 3)
#define A16(ALU, n) \
  A4(ALU, n) A4(ALU, (n) + 4) A4(ALU, (n) + 8) A4(ALU, (n) + 12)
#define A64(ALU, n) \
  A16(ALU, n) A16(ALU, (n) + 16) A16(ALU, (n) + 32) A16(ALU, (n) + 48)
#define A256(ALU, n) \
  A64(ALU, n) A64(ALU, (n) + 64) A64(ALU, (n) + 128) A64(ALU, (n) + 192)

FOLLY_NOINLINE uint64_t feSwitch0(uint32_t idx, uint64_t acc) {
  switch (idx & 0xFFu) {
    A256(ALU_A, 0)
    default:
      break;
  }
  return acc;
}

FOLLY_NOINLINE uint64_t feSwitch1(uint32_t idx, uint64_t acc) {
  switch (idx & 0xFFu) {
    A256(ALU_B, 0)
    default:
      break;
  }
  return acc;
}

FOLLY_NOINLINE uint64_t feSwitch2(uint32_t idx, uint64_t acc) {
  switch (idx & 0xFFu) {
    A256(ALU_C, 0)
    default:
      break;
  }
  return acc;
}

#undef A256
#undef A64
#undef A16
#undef A4
#undef A1
#undef ALU_C
#undef ALU_B
#undef ALU_A

} // namespace

static void contendedUseFE(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) & (kChaseSize - 1);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = gChase[p & (kChaseSize - 1)];
          uint32_t idx = p & 0xFFu;
          switch (j % 3) {
            case 0:
              acc = feSwitch0(idx, acc);
              break;
            case 1:
              acc = feSwitch1(idx, acc);
              break;
            default:
              acc = feSwitch2(idx, acc);
              break;
          }
          ++j;
        }
        gFeSink.fetch_add(acc, std::memory_order_relaxed);
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
BENCHMARK_NAMED_PARAM(contendedUseFE, 8_to_100_fe, 8, 100)
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
