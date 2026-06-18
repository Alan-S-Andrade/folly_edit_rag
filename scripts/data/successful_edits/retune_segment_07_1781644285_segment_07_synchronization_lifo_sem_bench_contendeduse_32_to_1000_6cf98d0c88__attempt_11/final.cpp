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

// Pointer-chase table used to derive an unpredictable switch index for the
// frontend-stressing mixers below. Initialized once at process start; values
// are kept in [0, 256) so the chase stays self-contained and produces a
// genuine data-dependent (load-to-use) chain.
static uint32_t gChase[256];
static const bool gChaseInit = [] {
  for (uint32_t i = 0; i < 256; ++i) {
    gChase[i] = (i * 167u + 13u) & 0xFFu;
  }
  return true;
}();

// Tier T1 frontend (I-cache) structural lever.
//
// Each switch case carries a fixed dose of straight-line ALU work whose
// literal constants are unique per case (and per function via the salt S),
// which defeats compiler case/function deduplication and forces a large hot
// code footprint. The 256-case body is replicated across three noinline
// functions so that rotating among them (j % 3) in the hot loop thrashes the
// L1 instruction cache. ALU-ops-per-case is the primary knob: L1i MPKI scales
// roughly linearly with it (~12 ops -> ~12 MPKI, ~14 ops -> ~15 MPKI), so the
// per-case body is sized near 13 ops to land in the target MPKI window.
#define ALU_CASE(K, S)                                            \
  case (K): {                                                     \
    a += (uint32_t)((K) * 2654435761u + (S));                     \
    a ^= (uint32_t)(((a) << 7) ^ ((K) ^ 0x9e3779b9u));            \
    a *= (uint32_t)(((K) | 1u));                                  \
    a += (uint32_t)(((a) >> 3) + ((K) * 40503u + (S)));           \
    a ^= (uint32_t)((K) * 0x1000193u);                            \
    a -= (uint32_t)((K) ^ 0xdeadbeefu ^ (S));                     \
    a *= 2246822519u;                                             \
    a ^= (a) >> 13;                                               \
    a += (uint32_t)(((K) << 3) | 1u);                             \
    a *= (uint32_t)((((K) + 0x85ebca6bu) | 1u));                  \
    a ^= (uint32_t)(((a) >> 11) + (K) + (S));                     \
    a += (uint32_t)(0xc2b2ae35u ^ (K));                           \
    a *= (uint32_t)((((K) * 3u) | 1u));                           \
    break;                                                        \
  }

#define ALU_CASE16(B, S)                                          \
  ALU_CASE((B) + 0, S)                                            \
  ALU_CASE((B) + 1, S)                                            \
  ALU_CASE((B) + 2, S)                                            \
  ALU_CASE((B) + 3, S)                                            \
  ALU_CASE((B) + 4, S)                                            \
  ALU_CASE((B) + 5, S)                                            \
  ALU_CASE((B) + 6, S)                                            \
  ALU_CASE((B) + 7, S)                                            \
  ALU_CASE((B) + 8, S)                                            \
  ALU_CASE((B) + 9, S)                                            \
  ALU_CASE((B) + 10, S)                                           \
  ALU_CASE((B) + 11, S)                                           \
  ALU_CASE((B) + 12, S)                                           \
  ALU_CASE((B) + 13, S)                                           \
  ALU_CASE((B) + 14, S)                                           \
  ALU_CASE((B) + 15, S)

#define ALU_CASES(S)                                              \
  ALU_CASE16(0, S)                                                \
  ALU_CASE16(16, S)                                               \
  ALU_CASE16(32, S)                                               \
  ALU_CASE16(48, S)                                               \
  ALU_CASE16(64, S)                                               \
  ALU_CASE16(80, S)                                               \
  ALU_CASE16(96, S)                                               \
  ALU_CASE16(112, S)                                              \
  ALU_CASE16(128, S)                                              \
  ALU_CASE16(144, S)                                              \
  ALU_CASE16(160, S)                                              \
  ALU_CASE16(176, S)                                              \
  ALU_CASE16(192, S)                                              \
  ALU_CASE16(208, S)                                              \
  ALU_CASE16(224, S)                                              \
  ALU_CASE16(240, S)

__attribute__((noinline)) static uint32_t mixSwitchA(uint32_t a, uint8_t idx) {
  switch (idx) {
    ALU_CASES(0x12345678u)
  }
  return a;
}

__attribute__((noinline)) static uint32_t mixSwitchB(uint32_t a, uint8_t idx) {
  switch (idx) {
    ALU_CASES(0x9e3779b9u)
  }
  return a;
}

__attribute__((noinline)) static uint32_t mixSwitchC(uint32_t a, uint8_t idx) {
  switch (idx) {
    ALU_CASES(0x7f4a7c15u)
  }
  return a;
}

// Nearby variant of contendedUse: same contended wait/post structure and the
// same operation volume, but each hot-loop iteration first drives a large,
// instruction-cache-thrashing mixer selected from three noinline 256-case
// switch functions. The switch index comes from a data-dependent pointer
// chase, and the function is rotated with j % 3 so the executed instruction
// working set stays large instead of compact. The added work is off the
// cross-thread semaphore dependency chain and does not alter the benchmark's
// throughput-carrying structure.
//
// Attempt 11 corrective patch (single lever: Tier T1 frontend / L1i MPKI):
// replace the small straight-line LCG dose with the 3-function rotating
// switch footprint described above to raise L1-icache-load-misses_MPKI toward
// the target window while leaving the contended post/wait skeleton intact.
static void contendedUseBiasedSelect(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint32_t acc = 0x9e3779b9u ^ static_cast<uint32_t>(t);
        uint32_t chase = static_cast<uint32_t>(t) & 0xFFu;
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          chase = gChase[chase & 0xFFu];
          uint8_t idx = static_cast<uint8_t>(chase & 0xFFu);
          switch (j % 3) {
            case 0:
              acc = mixSwitchA(acc ^ chase, idx);
              break;
            case 1:
              acc = mixSwitchB(acc ^ chase, idx);
              break;
            default:
              acc = mixSwitchC(acc ^ chase, idx);
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
        uint32_t acc = 0x85ebca6bu ^ static_cast<uint32_t>(t);
        uint32_t chase = (static_cast<uint32_t>(t) * 7u + 3u) & 0xFFu;
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += posters) {
          chase = gChase[chase & 0xFFu];
          uint8_t idx = static_cast<uint8_t>(chase & 0xFFu);
          switch (j % 3) {
            case 0:
              acc = mixSwitchB(acc ^ chase, idx);
              break;
            case 1:
              acc = mixSwitchC(acc ^ chase, idx);
              break;
            default:
              acc = mixSwitchA(acc ^ chase, idx);
              break;
          }
          ++j;
          sem.post();
        }
        folly::doNotOptimizeAway(acc);
      });
    }
  }

  go.store(true);
  for (auto& thr : threads) {
    thr.join();
  }
}

#undef ALU_CASES
#undef ALU_CASE16
#undef ALU_CASE

BENCHMARK_DRAW_LINE();
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_1, 1, 1)
BENCHMARK_NAMED_PARAM(contendedUseBiasedSelect, 32_to_1000, 32, 1000)
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
