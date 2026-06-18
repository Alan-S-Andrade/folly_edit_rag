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
// The contended LifoSem hot path is extremely compact, so the executed
// instruction working set stays tiny and IPC runs high. To deliberately widen
// the hot-code footprint (and exercise the instruction frontend / i-cache) the
// waiter loop rotates among three large noinline 256-case switches, indexing
// each by a pointer-chased payload byte. Each case body performs a short chain
// of ALU ops on a thread-local accumulator with unique literal constants so the
// compiler cannot deduplicate the cases. This materially changes the timed hot
// path without altering the synchronization structure or the operation volume.
namespace {

// 256-case dispatch helpers. Each macro row generates 16 consecutive cases.
#define LIFO_FE_C16(M, base)                                              \
  M((base) + 0) M((base) + 1) M((base) + 2) M((base) + 3) M((base) + 4)   \
      M((base) + 5) M((base) + 6) M((base) + 7) M((base) + 8)             \
          M((base) + 9) M((base) + 10) M((base) + 11) M((base) + 12)      \
              M((base) + 13) M((base) + 14) M((base) + 15)

#define LIFO_FE_C256(M)                                                   \
  LIFO_FE_C16(M, 0) LIFO_FE_C16(M, 16) LIFO_FE_C16(M, 32)                 \
      LIFO_FE_C16(M, 48) LIFO_FE_C16(M, 64) LIFO_FE_C16(M, 80)            \
          LIFO_FE_C16(M, 96) LIFO_FE_C16(M, 112) LIFO_FE_C16(M, 128)      \
              LIFO_FE_C16(M, 144) LIFO_FE_C16(M, 160) LIFO_FE_C16(M, 176) \
                  LIFO_FE_C16(M, 192) LIFO_FE_C16(M, 208)                 \
                      LIFO_FE_C16(M, 224) LIFO_FE_C16(M, 240)

#define LIFO_FE_CASE_A(i)                              \
  case (i): {                                          \
    acc += 0x9e3779b97f4a7c15ULL + (uint64_t)(i);      \
    acc ^= acc >> 13;                                  \
    acc *= 0xff51afd7ed558ccdULL;                      \
    acc += (uint64_t)(i) * 2654435761ULL;              \
    acc ^= acc << 7;                                   \
    acc *= 0xc4ceb9fe1a85ec53ULL;                      \
    acc -= (uint64_t)(i) ^ 0xa5a5a5a5ULL;              \
    acc ^= acc >> 11;                                  \
    acc += (uint64_t)(i) * 40503ULL;                   \
    acc *= 0x2545f4914f6cdd1dULL;                      \
    acc ^= acc >> 17;                                  \
    acc += (uint64_t)(i) * 0x100000001b3ULL;           \
    break;                                             \
  }

#define LIFO_FE_CASE_B(i)                              \
  case (i): {                                          \
    acc ^= 0xd6e8feb86659fd93ULL + (uint64_t)(i);      \
    acc *= 0x9ddfea08eb382d69ULL;                      \
    acc += (uint64_t)(i) * 0x27d4eb2f165667c5ULL;      \
    acc ^= acc >> 15;                                  \
    acc -= (uint64_t)(i) * 374761393ULL;               \
    acc ^= acc << 9;                                   \
    acc *= 0x165667b19e3779f9ULL;                      \
    acc += (uint64_t)(i) ^ 0x5bd1e995ULL;              \
    acc ^= acc >> 19;                                  \
    acc *= 0x85ebca6bULL;                              \
    acc += (uint64_t)(i) * 0xcc9e2d51ULL;              \
    acc ^= acc >> 23;                                  \
    break;                                             \
  }

#define LIFO_FE_CASE_C(i)                              \
  case (i): {                                          \
    acc += 0xbf58476d1ce4e5b9ULL ^ (uint64_t)(i);      \
    acc ^= acc << 11;                                  \
    acc *= 0x94d049bb133111ebULL;                      \
    acc += (uint64_t)(i) * 0x1b873593ULL;              \
    acc ^= acc >> 14;                                  \
    acc -= (uint64_t)(i) * 2246822519ULL;              \
    acc *= 0xff51afd7ed558ccdULL;                      \
    acc ^= (uint64_t)(i) ^ 0xdeadbeefULL;              \
    acc += acc >> 12;                                  \
    acc *= 0x2545f4914f6cdd1dULL;                      \
    acc -= (uint64_t)(i) * 0x9e3779b1ULL;              \
    acc ^= acc >> 21;                                  \
    break;                                             \
  }

__attribute__((noinline)) uint64_t lifoFeSwitchA(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFO_FE_C256(LIFO_FE_CASE_A)
  }
  return acc;
}

__attribute__((noinline)) uint64_t lifoFeSwitchB(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFO_FE_C256(LIFO_FE_CASE_B)
  }
  return acc;
}

__attribute__((noinline)) uint64_t lifoFeSwitchC(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFO_FE_C256(LIFO_FE_CASE_C)
  }
  return acc;
}

// A small pointer-chase table feeds an unpredictable switch index byte.
const std::vector<uint8_t>& lifoFeChase() {
  static const std::vector<uint8_t> table = [] {
    std::vector<uint8_t> v(257);
    uint64_t x = 0x123456789abcdef0ULL;
    for (size_t i = 0; i < v.size(); ++i) {
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      v[i] = static_cast<uint8_t>(x);
    }
    return v;
  }();
  return table;
}

} // namespace

static void contendedUseFrontend(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  const std::vector<uint8_t>& chase = lifoFeChase();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) & 0xFF;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          uint8_t idx = static_cast<uint8_t>(chase[p] ^ (acc & 0xFF));
          switch (i % 3) {
            case 0:
              acc = lifoFeSwitchA(acc, idx);
              break;
            case 1:
              acc = lifoFeSwitchB(acc, idx);
              break;
            default:
              acc = lifoFeSwitchC(acc, idx);
              break;
          }
          p = idx;
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
BENCHMARK_NAMED_PARAM(contendedUseFrontend, 8_to_100_fe, 8, 100)
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
