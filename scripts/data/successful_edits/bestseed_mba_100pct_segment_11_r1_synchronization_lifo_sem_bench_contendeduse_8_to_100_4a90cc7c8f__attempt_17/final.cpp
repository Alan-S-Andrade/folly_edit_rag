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
// Frontend / i-cache pressure variant of contendedUse.
//
// Each waiter thread, in addition to draining the semaphore, walks a small
// pointer-chase table to obtain an 8-bit index and dispatches it into one of
// three large 256-case switch functions (rotated by j%3).  The switches inflate
// the executed instruction working set well past the L1 instruction cache so
// that the otherwise tiny semaphore loop becomes frontend-bound.  Each case uses
// unique literal constants (folded through a per-case k) to defeat compiler
// case/function deduplication.
// ---------------------------------------------------------------------------
namespace {

uint8_t gChase[256];

#define REP16(M, b)                                                          \
  M(b + 0) M(b + 1) M(b + 2) M(b + 3) M(b + 4) M(b + 5) M(b + 6) M(b + 7)    \
      M(b + 8) M(b + 9) M(b + 10) M(b + 11) M(b + 12) M(b + 13) M(b + 14)    \
          M(b + 15)
#define REP256(M)                                                           \
  REP16(M, 0) REP16(M, 16) REP16(M, 32) REP16(M, 48) REP16(M, 64)           \
      REP16(M, 80) REP16(M, 96) REP16(M, 112) REP16(M, 128) REP16(M, 144)   \
          REP16(M, 160) REP16(M, 176) REP16(M, 192) REP16(M, 208)           \
              REP16(M, 224) REP16(M, 240)

#define CASE_A(i)                                                           \
  case (i): {                                                               \
    uint64_t k = (uint64_t)(i);                                             \
    acc += 0x9e3779b97f4a7c15ull * (k + 1u);                                \
    acc ^= 0xff51afd7ed558ccdull ^ (k << 1);                                \
    acc *= (0xc4ceb9fe1a85ec53ull | (k + 3u));                              \
    acc += (k * 2654435761ull);                                             \
    acc -= (k * 40503ull + 7u);                                             \
    acc ^= (k << 5);                                                        \
    acc += (k | 0xA5ull);                                                   \
    acc *= ((k << 1) | 1u);                                                 \
    acc ^= (k * 0x100000001b3ull);                                          \
    acc += (k << 3);                                                        \
    acc -= (k * 374761393ull);                                              \
    acc ^= (k + 0x5bd1e995ull);                                             \
    acc += (k ^ 0x9e3779b1ull);                                             \
    acc *= ((k + 11u) | 1u);                                                \
  } break;

#define CASE_B(i)                                                           \
  case (i): {                                                               \
    uint64_t k = (uint64_t)(i) + 0x55ull;                                   \
    acc ^= 0xd6e8feb86659fd93ull * (k + 2u);                                \
    acc += 0xa0761d6478bd642full ^ (k << 2);                                \
    acc *= (0xe7037ed1a0b428dbull | (k + 5u));                              \
    acc -= (k * 2246822519ull);                                             \
    acc ^= (k * 3266489917ull + 9u);                                        \
    acc += (k << 4);                                                        \
    acc *= ((k << 2) | 1u);                                                 \
    acc ^= (k | 0x3Cull);                                                   \
    acc += (k * 0x9ddfea08eb382d69ull);                                     \
    acc -= (k << 6);                                                        \
    acc ^= (k * 668265263ull);                                              \
    acc += (k + 0x27d4eb2full);                                             \
    acc *= ((k + 13u) | 1u);                                                \
    acc ^= (k ^ 0x165667b1ull);                                             \
  } break;

#define CASE_C(i)                                                           \
  case (i): {                                                               \
    uint64_t k = (uint64_t)(i) + 0xAAull;                                   \
    acc += 0xbf58476d1ce4e5b9ull * (k + 4u);                                \
    acc *= (0x94d049bb133111ebull | (k + 7u));                              \
    acc ^= 0x2545f4914f6cdd1dull ^ (k << 3);                                \
    acc += (k * 2654435761ull + 1u);                                        \
    acc -= (k * 0x85ebca6bull);                                             \
    acc ^= (k << 7);                                                        \
    acc += (k | 0x5Aull);                                                   \
    acc *= ((k << 3) | 1u);                                                 \
    acc ^= (k * 0xff51afd7ull);                                             \
    acc -= (k << 2);                                                        \
    acc += (k * 0xc2b2ae35ull);                                             \
    acc ^= (k + 0x1b873593ull);                                             \
    acc *= ((k + 17u) | 1u);                                                \
    acc += (k ^ 0xcc9e2d51ull);                                             \
  } break;

__attribute__((noinline)) uint64_t icacheSwitchA(uint8_t idx, uint64_t acc) {
  switch (idx) { REP256(CASE_A) }
  return acc;
}

__attribute__((noinline)) uint64_t icacheSwitchB(uint8_t idx, uint64_t acc) {
  switch (idx) { REP256(CASE_B) }
  return acc;
}

__attribute__((noinline)) uint64_t icacheSwitchC(uint8_t idx, uint64_t acc) {
  switch (idx) { REP256(CASE_C) }
  return acc;
}

#undef CASE_A
#undef CASE_B
#undef CASE_C
#undef REP256
#undef REP16

} // namespace

static void contendedUseFrontend(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    // A coprime stride yields a full 256-entry permutation, i.e. a long
    // dependent pointer-chase cycle feeding the switch indices.
    for (int x = 0; x < 256; ++x) {
      gChase[x] = static_cast<uint8_t>((x * 167 + 13) & 0xFF);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = 0x12345678ull + static_cast<uint64_t>(t);
        uint8_t idx = static_cast<uint8_t>(t * 7 + 1);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          idx = gChase[idx];
          switch (j % 3) {
            case 0:
              acc = icacheSwitchA(idx, acc);
              break;
            case 1:
              acc = icacheSwitchB(idx, acc);
              break;
            default:
              acc = icacheSwitchC(idx, acc);
              break;
          }
          ++j;
          sem.wait();
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
BENCHMARK_NAMED_PARAM(contendedUseFrontend, 8_to_100, 8, 100)

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
