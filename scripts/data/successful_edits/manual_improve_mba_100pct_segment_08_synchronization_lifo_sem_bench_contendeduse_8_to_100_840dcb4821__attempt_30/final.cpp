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
// Derived frontend-bound variant of contendedUse(8_to_100).
//
// The contended LifoSem traffic by itself keeps the executed instruction
// working set tiny, so the i-cache footprint is far below the reference
// target. To raise L1i pressure we route each completed wait() through one of
// three large, noinline 256-case switch functions selected by i % 3. The
// switch index is derived from a pointer-chase load, so the branch target is
// data dependent and every case body (each with a fat run of ALU ops using
// unique literal constants) eventually executes, thrashing the i-cache.
// ---------------------------------------------------------------------------
namespace {

struct PChaseNode {
  const PChaseNode* next;
  uint64_t payload;
};

constexpr size_t kChainSize = 1024;

#define REP16(F, b)                                        \
  F((b) + 0) F((b) + 1) F((b) + 2) F((b) + 3)              \
  F((b) + 4) F((b) + 5) F((b) + 6) F((b) + 7)              \
  F((b) + 8) F((b) + 9) F((b) + 10) F((b) + 11)            \
  F((b) + 12) F((b) + 13) F((b) + 14) F((b) + 15)

#define REP256(F)                                          \
  REP16(F, 0) REP16(F, 16) REP16(F, 32) REP16(F, 48)       \
  REP16(F, 64) REP16(F, 80) REP16(F, 96) REP16(F, 112)     \
  REP16(F, 128) REP16(F, 144) REP16(F, 160) REP16(F, 176)  \
  REP16(F, 192) REP16(F, 208) REP16(F, 224) REP16(F, 240)

#define ALU_CASE(n)                                                  \
  case (n): {                                                        \
    a += 0x9E3779B97F4A7C15ull + (uint64_t)(n);                      \
    a ^= (a << 13) | ((uint64_t)(n) + 0x10001u);                     \
    a *= 0xD1B54A32D192ED03ull + (uint64_t)(n)*3u;                   \
    a += (a >> 7) ^ ((uint64_t)(n) * 7u + 1u);                       \
    a -= 0xC2B2AE3D27D4EB4Full - (uint64_t)(n);                      \
    a ^= (a << 17) + ((uint64_t)(n) ^ 0xA5A5u);                      \
    a *= 0x165667B19E3779F9ull + (uint64_t)(n)*5u;                   \
    a += (a >> 11) | ((uint64_t)(n) * 11u + 3u);                     \
    a ^= 0x27D4EB2F165667C5ull + (uint64_t)(n);                      \
    a += (a << 5) ^ ((uint64_t)(n) * 13u);                           \
    a -= 0x94D049BB133111EBull - (uint64_t)(n);                      \
    a *= 0xBF58476D1CE4E5B9ull + (uint64_t)(n)*7u;                   \
    a ^= (a >> 19) + ((uint64_t)(n) * 17u + 5u);                     \
    a += 0x2545F4914F6CDD1Dull + (uint64_t)(n);                      \
    a ^= (a << 23) | ((uint64_t)(n) ^ 0x5A5Au);                      \
    a *= 0x9FB21C651E98DF25ull + (uint64_t)(n)*9u;                   \
    a += (a >> 29) ^ ((uint64_t)(n) * 19u + 7u);                     \
    a -= 0xEB44ACCAB455D165ull - (uint64_t)(n);                      \
    a ^= 0xA24BAED4963EE407ull + (uint64_t)(n);                      \
    a += (a << 7) | ((uint64_t)(n) * 23u + 9u);                      \
    break;                                                           \
  }

#define MAKE_BURN(NAME, SALT)                                        \
  __attribute__((noinline)) uint64_t NAME(uint8_t idx, uint64_t a) { \
    a += (SALT);                                                     \
    switch (idx) {                                                   \
      REP256(ALU_CASE)                                               \
    }                                                                \
    return a ^ (SALT);                                               \
  }

MAKE_BURN(lifoBurnA, 0x1111111111111111ull)
MAKE_BURN(lifoBurnB, 0x2222222222222222ull)
MAKE_BURN(lifoBurnC, 0x3333333333333333ull)

#undef MAKE_BURN
#undef ALU_CASE
#undef REP256
#undef REP16

static void contendedUseBurn(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::vector<PChaseNode> chain(kChainSize);

  BENCHMARK_SUSPEND {
    for (size_t k = 0; k < kChainSize; ++k) {
      chain[k].payload = k * 0x9E3779B97F4A7C15ull + 0x1234567u;
      chain[k].next = &chain[(k * 131 + 17) % kChainSize];
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chain] {
        uint64_t a = static_cast<uint64_t>(t) + 1;
        const PChaseNode* p = &chain[static_cast<size_t>(t) & (kChainSize - 1)];
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = p->next;
          uint8_t idx = static_cast<uint8_t>(p->payload & 0xFF);
          switch (i % 3) {
            case 0:
              a = lifoBurnA(idx, a);
              break;
            case 1:
              a = lifoBurnB(idx, a);
              break;
            default:
              a = lifoBurnC(idx, a);
              break;
          }
        }
        folly::doNotOptimizeAway(a);
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

} // namespace

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
BENCHMARK_NAMED_PARAM(contendedUseBurn, 8_to_100, 8, 100)

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
