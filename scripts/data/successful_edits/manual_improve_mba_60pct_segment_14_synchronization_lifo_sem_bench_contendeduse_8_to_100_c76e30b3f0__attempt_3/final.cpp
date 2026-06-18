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
// Frontend-pressure derived variant of contendedUse.
//
// The hot loop performs a pointer-chase load and uses the low byte of the
// chased payload to index a 256-case switch. Three distinct noinline switch
// functions are rotated with j % 3 to inflate the executed instruction
// working set so the benchmark becomes frontend (L1i) bound rather than
// retiring at a high IPC.
// ---------------------------------------------------------------------------
namespace {

// 20 ALU ops per case, each with unique literal constants to defeat
// compiler case/function deduplication. This is the primary L1i knob.
#define FE_OPS(K)                          \
  acc += (0x100u + (K));                   \
  acc ^= (0x1000u + (K) * 7u);             \
  acc *= (2654435761u + (K));              \
  acc += (0x200u ^ (K));                   \
  acc ^= (0x9e3779b9u + (K) * 3u);         \
  acc *= (0x85ebca6bu + (K));              \
  acc += ((K) << 3);                       \
  acc ^= ((K) * 131u + 17u);               \
  acc += (0xc2b2ae35u - (K));              \
  acc *= (0x27d4eb2fu + (K) * 5u);         \
  acc ^= (0x165667b1u + (K));              \
  acc += ((K) ^ 0xdeadbeefu);              \
  acc *= (0x2545f491u + (K) * 9u);         \
  acc ^= (0xfd7046c5u + (K));              \
  acc += (0x1b873593u + (K) * 11u);        \
  acc ^= (0xcc9e2d51u - (K));              \
  acc *= (0x9ddfea08u + (K));              \
  acc += ((K) * 2246822519u);              \
  acc ^= ((K) * 3266489917u + 5u);         \
  acc *= (0x000001b3u + (K));

#define FE_CASE(K) \
  case (K):        \
    FE_OPS(K)      \
    break;

#define FE_CASE16(B)                                                  \
  FE_CASE((B) + 0) FE_CASE((B) + 1) FE_CASE((B) + 2) FE_CASE((B) + 3) \
  FE_CASE((B) + 4) FE_CASE((B) + 5) FE_CASE((B) + 6) FE_CASE((B) + 7) \
  FE_CASE((B) + 8) FE_CASE((B) + 9) FE_CASE((B) + 10)                 \
  FE_CASE((B) + 11) FE_CASE((B) + 12) FE_CASE((B) + 13)               \
  FE_CASE((B) + 14) FE_CASE((B) + 15)

#define FE_CASES                                                      \
  FE_CASE16(0) FE_CASE16(16) FE_CASE16(32) FE_CASE16(48)              \
  FE_CASE16(64) FE_CASE16(80) FE_CASE16(96) FE_CASE16(112)            \
  FE_CASE16(128) FE_CASE16(144) FE_CASE16(160) FE_CASE16(176)         \
  FE_CASE16(192) FE_CASE16(208) FE_CASE16(224) FE_CASE16(240)

#define FE_FUNC(NAME, SALT)                                          \
  __attribute__((noinline)) uint64_t NAME(uint64_t acc, uint8_t idx) { \
    acc ^= (SALT);                                                   \
    switch (idx) {                                                   \
      FE_CASES                                                       \
    }                                                                \
    acc += (SALT);                                                   \
    return acc;                                                      \
  }

FE_FUNC(feSwitch0, 0x1111u)
FE_FUNC(feSwitch1, 0x2222u)
FE_FUNC(feSwitch2, 0x3333u)

} // namespace

static void contendedUseFE(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        const size_t kRing = 1024;
        std::vector<uint32_t> next(kRing);
        for (size_t i = 0; i < kRing; ++i) {
          next[i] =
              static_cast<uint32_t>((i * 2654435761u + t * 40503u + 1) % kRing);
        }
        uint64_t acc = 0x12345678u + static_cast<uint64_t>(t);
        uint32_t p = static_cast<uint32_t>(t) % kRing;
        uint64_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = next[p];
          uint8_t idx = static_cast<uint8_t>((next[p] ^ acc) & 0xFF);
          switch (j % 3) {
            case 0:
              acc = feSwitch0(acc, idx);
              break;
            case 1:
              acc = feSwitch1(acc, idx);
              break;
            default:
              acc = feSwitch2(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUseFE, 8_to_100_fe, 8, 100)

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
