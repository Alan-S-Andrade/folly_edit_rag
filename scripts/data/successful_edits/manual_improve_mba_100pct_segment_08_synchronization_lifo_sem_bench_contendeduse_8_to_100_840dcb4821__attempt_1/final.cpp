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

#include <array>
#include <cstdint>

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
// This injects a compact, hard-to-predict instruction-footprint expansion into
// the waiter hot path: three large noinline switch functions (256 unique cases
// each, with a chain of ALU ops per case) are rotated through with i % 3, and
// the switch index is sourced from a pointer-chase load. This materially
// enlarges the executed instruction working set (icache thrashing) and adds a
// short data-dependent chain, lowering IPC relative to the original variant
// without changing operation volume / loop trip counts.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kChaseLen = 1024; // power of two

struct ChaseNode {
  uint32_t next;
  uint32_t payload;
};

// One ALU "case" body: 12 dependent ALU ops, all literals salted with the
// case index K and a per-function SALT so the compiler cannot deduplicate.
#define FE_ALU(K)                                                      \
  case (K): {                                                          \
    acc += (0x9E3779B9u ^ (uint32_t)((K) + SALT));                     \
    acc ^= (acc << 5) + (uint32_t)((K) + 1u);                          \
    acc *= ((0x85EBCA6Bu + (uint32_t)(K)) | 1u);                       \
    acc += (acc >> 3) ^ (uint32_t)((K) * 3u + SALT);                   \
    acc ^= (0xDEADBEEFu - (uint32_t)((K) + SALT));                     \
    acc *= ((0xC2B2AE35u + (uint32_t)((K) << 1)) | 1u);                \
    acc += (acc << 7) ^ (uint32_t)((K) * 7u);                          \
    acc ^= (acc >> 11) + (uint32_t)((K) * 5u + SALT);                  \
    acc *= ((0x27D4EB2Fu + (uint32_t)(K)) | 1u);                       \
    acc += (uint32_t)((K) * 11u) ^ 0x165667B1u;                        \
    acc ^= (acc << 3) + (uint32_t)((K) + SALT);                        \
    acc *= ((0xFF51AFD7u + (uint32_t)((K) * 13u)) | 1u);               \
    break;                                                             \
  }

#define FE_C16(B)                                                      \
  FE_ALU(B + 0) FE_ALU(B + 1) FE_ALU(B + 2) FE_ALU(B + 3)              \
  FE_ALU(B + 4) FE_ALU(B + 5) FE_ALU(B + 6) FE_ALU(B + 7)              \
  FE_ALU(B + 8) FE_ALU(B + 9) FE_ALU(B + 10) FE_ALU(B + 11)            \
  FE_ALU(B + 12) FE_ALU(B + 13) FE_ALU(B + 14) FE_ALU(B + 15)

#define FE_C256                                                        \
  FE_C16(0) FE_C16(16) FE_C16(32) FE_C16(48)                           \
  FE_C16(64) FE_C16(80) FE_C16(96) FE_C16(112)                         \
  FE_C16(128) FE_C16(144) FE_C16(160) FE_C16(176)                      \
  FE_C16(192) FE_C16(208) FE_C16(224) FE_C16(240)

#define SALT 0x11u
__attribute__((noinline)) static uint32_t fe_mix0(uint32_t acc, uint8_t idx) {
  switch (idx) { FE_C256 }
  return acc;
}
#undef SALT

#define SALT 0x57u
__attribute__((noinline)) static uint32_t fe_mix1(uint32_t acc, uint8_t idx) {
  switch (idx) { FE_C256 }
  return acc;
}
#undef SALT

#define SALT 0xA3u
__attribute__((noinline)) static uint32_t fe_mix2(uint32_t acc, uint8_t idx) {
  switch (idx) { FE_C256 }
  return acc;
}
#undef SALT

#undef FE_C256
#undef FE_C16
#undef FE_ALU

} // namespace

static void contendedUseFE(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static std::array<ChaseNode, kChaseLen> chase;

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kChaseLen; ++i) {
      chase[i].next = (uint32_t)((i * 2654435761u + 1u) % kChaseLen);
      chase[i].payload = (uint32_t)(i * 2246822519u + 12345u);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint32_t acc = (uint32_t)t * 0x1000193u + 0xABCDu;
        uint32_t pos = (uint32_t)t & (kChaseLen - 1);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          pos = chase[pos].next;
          uint8_t idx = (uint8_t)(chase[pos].payload & 0xFFu);
          switch (i % 3) {
            case 0:
              acc = fe_mix0(acc, idx);
              break;
            case 1:
              acc = fe_mix1(acc, idx);
              break;
            default:
              acc = fe_mix2(acc, idx);
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
