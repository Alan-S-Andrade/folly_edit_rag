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
// Frontend-pressure variant of the contendedUse(8_to_100) family.
//
// Each waiter, after returning from sem.wait(), performs a pointer-chase load
// into a shared chain and dispatches into one of three large noinline switch
// functions (rotated by i % 3). The switch index is the low byte of the
// chased payload. The combined ~768 unique switch cases create a large hot
// instruction footprint that increases L1i pressure without lengthening the
// dominant semaphore dependency chain, keeping IPC stable.
// ---------------------------------------------------------------------------
namespace {

const std::vector<uint32_t>& feChain() {
  static const std::vector<uint32_t> c = [] {
    std::vector<uint32_t> v(4096);
    uint32_t x = 2463534242u;
    for (size_t i = 0; i < v.size(); ++i) {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      v[i] = x & 4095u;
    }
    return v;
  }();
  return c;
}

#define FE_A(k)                                          \
  case (k):                                              \
    acc += 0x9e3779b97f4a7c15ULL ^ (unsigned long long)(k); \
    acc ^= acc >> 13;                                    \
    acc *= 0xff51afd7ed558ccdULL;                        \
    acc += (unsigned long long)(k) * 2654435761ULL;      \
    acc ^= acc << 7;                                     \
    break;
#define FE16_A(b)                                                       \
  FE_A((b) + 0) FE_A((b) + 1) FE_A((b) + 2) FE_A((b) + 3)               \
  FE_A((b) + 4) FE_A((b) + 5) FE_A((b) + 6) FE_A((b) + 7)               \
  FE_A((b) + 8) FE_A((b) + 9) FE_A((b) + 10) FE_A((b) + 11)             \
  FE_A((b) + 12) FE_A((b) + 13) FE_A((b) + 14) FE_A((b) + 15)

__attribute__((noinline)) uint64_t feSwitchA(uint64_t acc, uint8_t idx) {
  switch (idx) {
    FE16_A(0) FE16_A(16) FE16_A(32) FE16_A(48)
    FE16_A(64) FE16_A(80) FE16_A(96) FE16_A(112)
    FE16_A(128) FE16_A(144) FE16_A(160) FE16_A(176)
    FE16_A(192) FE16_A(208) FE16_A(224) FE16_A(240)
  }
  return acc;
}

#define FE_B(k)                                          \
  case (k):                                              \
    acc ^= 0xc2b2ae3d27d4eb4fULL + (unsigned long long)(k); \
    acc *= 0x165667b19e3779f9ULL;                        \
    acc -= (unsigned long long)(k) * 40503ULL;           \
    acc ^= acc >> 11;                                    \
    acc += acc << 3;                                     \
    break;
#define FE16_B(b)                                                       \
  FE_B((b) + 0) FE_B((b) + 1) FE_B((b) + 2) FE_B((b) + 3)               \
  FE_B((b) + 4) FE_B((b) + 5) FE_B((b) + 6) FE_B((b) + 7)               \
  FE_B((b) + 8) FE_B((b) + 9) FE_B((b) + 10) FE_B((b) + 11)             \
  FE_B((b) + 12) FE_B((b) + 13) FE_B((b) + 14) FE_B((b) + 15)

__attribute__((noinline)) uint64_t feSwitchB(uint64_t acc, uint8_t idx) {
  switch (idx) {
    FE16_B(0) FE16_B(16) FE16_B(32) FE16_B(48)
    FE16_B(64) FE16_B(80) FE16_B(96) FE16_B(112)
    FE16_B(128) FE16_B(144) FE16_B(160) FE16_B(176)
    FE16_B(192) FE16_B(208) FE16_B(224) FE16_B(240)
  }
  return acc;
}

#define FE_C(k)                                          \
  case (k):                                              \
    acc -= 0xa24baed4963ee407ULL ^ (unsigned long long)(k); \
    acc ^= acc << 5;                                     \
    acc *= 0x9fb21c651e98df25ULL + (unsigned long long)(k); \
    acc += (unsigned long long)(k) * 0x100000001b3ULL;   \
    acc ^= acc >> 9;                                     \
    break;
#define FE16_C(b)                                                       \
  FE_C((b) + 0) FE_C((b) + 1) FE_C((b) + 2) FE_C((b) + 3)               \
  FE_C((b) + 4) FE_C((b) + 5) FE_C((b) + 6) FE_C((b) + 7)               \
  FE_C((b) + 8) FE_C((b) + 9) FE_C((b) + 10) FE_C((b) + 11)             \
  FE_C((b) + 12) FE_C((b) + 13) FE_C((b) + 14) FE_C((b) + 15)

__attribute__((noinline)) uint64_t feSwitchC(uint64_t acc, uint8_t idx) {
  switch (idx) {
    FE16_C(0) FE16_C(16) FE16_C(32) FE16_C(48)
    FE16_C(64) FE16_C(80) FE16_C(96) FE16_C(112)
    FE16_C(128) FE16_C(144) FE16_C(160) FE16_C(176)
    FE16_C(192) FE16_C(208) FE16_C(224) FE16_C(240)
  }
  return acc;
}

#undef FE_A
#undef FE16_A
#undef FE_B
#undef FE16_B
#undef FE_C
#undef FE16_C

} // namespace

static void contendedUseSwitch(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  const std::vector<uint32_t>& chain = feChain();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chain] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = (static_cast<uint32_t>(t) * 2654435761u) & 4095u;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          uint32_t payload = chain[p];
          p = payload;
          uint8_t idx = static_cast<uint8_t>(payload & 0xFFu);
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
BENCHMARK_NAMED_PARAM(contendedUseSwitch, 8_to_100_fe, 8, 100)
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
