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
// Derived benchmark: contendedUseSwitch
//
// This is a from-scratch derivative of contendedUse(8_to_100). In addition to
// the LifoSem contention pattern, each woken waiter executes a large, branchy
// switch-driven ALU kernel selected by a pointer-chase payload. Three distinct
// __attribute__((noinline)) 256-case dispatch functions are rotated with j%3
// in the hot loop to inflate the executed instruction working set (hot code
// footprint) so that L1-icache-load-misses_MPKI rises toward target while IPC
// drops out of the frontend-bound regime. Each case uses unique literal
// constants (via the case index and a per-function SALT) to defeat compiler
// deduplication. ALU ops-per-case is the primary L1i knob (~14 ops => ~15 MPKI).
// ---------------------------------------------------------------------------

#define ALU(i)                                                              \
  case (i): {                                                               \
    acc += (0x9E3779B9u ^ ((uint32_t)(i) * 2u + SALT));                     \
    acc ^= ((acc << 3) + ((uint32_t)(i) + SALT));                           \
    acc *= (0x85EBCA77u + (uint32_t)(i) + SALT);                            \
    acc += ((acc >> 5) ^ ((uint32_t)(i) * 7u + SALT));                      \
    acc ^= ((uint32_t)(i) * 131u + SALT);                                   \
    acc *= (0xC2B2AE3Du + (uint32_t)(i) + SALT);                            \
    acc += ((acc << 7) ^ ((uint32_t)(i) * 17u + SALT));                     \
    acc ^= ((acc >> 11) + ((uint32_t)(i) * 29u + SALT));                    \
    acc *= (0x27D4EB2Fu + (uint32_t)(i) + SALT);                            \
    acc += ((uint32_t)(i) * 5u + SALT);                                     \
    acc ^= ((acc << 13) + ((uint32_t)(i) * 3u + SALT));                     \
    acc *= (0x165667B1u + (uint32_t)(i) + SALT);                            \
    acc += ((acc >> 9) ^ ((uint32_t)(i) * 23u + SALT));                     \
    acc ^= ((uint32_t)(i) * 97u + SALT);                                    \
    break;                                                                  \
  }

#define ALU16(b)                                                            \
  ALU(b + 0) ALU(b + 1) ALU(b + 2) ALU(b + 3)                               \
  ALU(b + 4) ALU(b + 5) ALU(b + 6) ALU(b + 7)                               \
  ALU(b + 8) ALU(b + 9) ALU(b + 10) ALU(b + 11)                             \
  ALU(b + 12) ALU(b + 13) ALU(b + 14) ALU(b + 15)

#define ALU256                                                              \
  ALU16(0) ALU16(16) ALU16(32) ALU16(48)                                    \
  ALU16(64) ALU16(80) ALU16(96) ALU16(112)                                  \
  ALU16(128) ALU16(144) ALU16(160) ALU16(176)                              \
  ALU16(192) ALU16(208) ALU16(224) ALU16(240)

#define MK_SWITCH_FN(NAME)                                                  \
  __attribute__((noinline)) static uint32_t NAME(                          \
      uint32_t idx, uint32_t acc) {                                         \
    switch (idx & 0xFFu) {                                                  \
      ALU256                                                                \
    }                                                                       \
    return acc;                                                             \
  }

#define SALT 0x11u
MK_SWITCH_FN(switch_block_0)
#undef SALT
#define SALT 0x22u
MK_SWITCH_FN(switch_block_1)
#undef SALT
#define SALT 0x33u
MK_SWITCH_FN(switch_block_2)
#undef SALT

#undef MK_SWITCH_FN
#undef ALU256
#undef ALU16
#undef ALU

static std::vector<uint32_t> makeChaseBuffer(size_t len) {
  std::vector<uint32_t> v(len);
  for (size_t i = 0; i < len; ++i) {
    v[i] = (uint32_t)((i * 2654435761u) ^ (uint32_t)(i << 7) ^ 0xABCD1234u);
  }
  return v;
}

static void contendedUseSwitch(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  static const std::vector<uint32_t> chase = makeChaseBuffer(4096);
  const uint32_t mask = (uint32_t)(chase.size() - 1);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint32_t acc = (uint32_t)t + 1u;
        uint32_t p = (uint32_t)t * 2654435761u;
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & mask];
          uint32_t idx = p & 0xFFu;
          switch (j % 3u) {
            case 0:
              acc = switch_block_0(idx, acc);
              break;
            case 1:
              acc = switch_block_1(idx, acc);
              break;
            default:
              acc = switch_block_2(idx, acc);
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
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_4, 1, 4)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_32, 1, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_1, 4, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_24, 4, 24)
BENCHMARK_NAMED_PARAM(contendedUse, 8_to_100, 8, 100)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1, 31, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 16_to_16, 16, 16)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_32, 32, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1000, 32, 1000)
BENCHMARK_NAMED_PARAM(contendedUseSwitch, 8_to_100_switch, 8, 100)

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
