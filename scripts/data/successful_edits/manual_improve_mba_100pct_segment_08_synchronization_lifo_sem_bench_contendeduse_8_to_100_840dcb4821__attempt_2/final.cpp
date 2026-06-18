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
// The IPC of the plain contendedUse family runs far above target because the
// executed instruction working set is tiny and reused. To grow the hot code
// footprint (and thus L1-icache pressure) we rotate among three large,
// non-inlinable 256-case switch functions. Each case performs a chain of ALU
// ops on a local accumulator using case-unique literal constants, and each of
// the three functions is salted with a distinct constant so the linker cannot
// fold them together. The switch index comes from a pointer-chase load so it
// is hard to predict.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kFeChaseLen = 4096; // power of two
constexpr size_t kFeChaseMask = kFeChaseLen - 1;

std::vector<uint64_t> feChase = [] {
  std::vector<uint64_t> v(kFeChaseLen);
  for (size_t i = 0; i < kFeChaseLen; ++i) {
    v[i] = (i * 2654435761ull + 1469598103934665603ull) & kFeChaseMask;
  }
  return v;
}();

#define FE_OPS(n)                                                  \
  acc += 0x9e3779b97f4a7c15ull * (uint64_t)((n) + 1u);             \
  acc ^= (0xff51afd7ed558ccdull ^ ((uint64_t)(n) << 7));           \
  acc *= (2ull * (uint64_t)(n) + 1ull);                            \
  acc -= (0xc4ceb9fe1a85ec53ull + (uint64_t)(n));                  \
  acc ^= acc >> 29;                                                \
  acc += 0x165667b19e3779f9ull * (uint64_t)(n);                    \
  acc *= (0x2545f4914f6cdd1dull | (uint64_t)((n) + 3u));           \
  acc ^= acc << 17;                                                \
  acc += (kSalt ^ (uint64_t)(n));                                  \
  acc *= (0x589965cc75374cc3ull | (uint64_t)((n) + 5u));           \
  acc ^= acc >> 23;                                                \
  acc += kSalt * (uint64_t)((n) + 7u);

#define FE_C(n) \
  case (uint8_t)(n):    \
    FE_OPS(n)           \
    break;

#define FE_R16(b)                                                 \
  FE_C(b + 0) FE_C(b + 1) FE_C(b + 2) FE_C(b + 3)                 \
  FE_C(b + 4) FE_C(b + 5) FE_C(b + 6) FE_C(b + 7)                 \
  FE_C(b + 8) FE_C(b + 9) FE_C(b + 10) FE_C(b + 11)               \
  FE_C(b + 12) FE_C(b + 13) FE_C(b + 14) FE_C(b + 15)

#define FE_R256                                                   \
  FE_R16(0) FE_R16(16) FE_R16(32) FE_R16(48)                      \
  FE_R16(64) FE_R16(80) FE_R16(96) FE_R16(112)                    \
  FE_R16(128) FE_R16(144) FE_R16(160) FE_R16(176)                 \
  FE_R16(192) FE_R16(208) FE_R16(224) FE_R16(240)

__attribute__((noinline)) static uint64_t feSwitch0(uint8_t idx, uint64_t acc) {
  const uint64_t kSalt = 0x1111111111111111ull;
  switch (idx) {
    FE_R256
  }
  return acc;
}

__attribute__((noinline)) static uint64_t feSwitch1(uint8_t idx, uint64_t acc) {
  const uint64_t kSalt = 0x2222222222222222ull;
  switch (idx) {
    FE_R256
  }
  return acc;
}

__attribute__((noinline)) static uint64_t feSwitch2(uint8_t idx, uint64_t acc) {
  const uint64_t kSalt = 0x3333333333333333ull;
  switch (idx) {
    FE_R256
  }
  return acc;
}

#undef FE_R256
#undef FE_R16
#undef FE_C
#undef FE_OPS

} // namespace

static void contendedUseFE(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        size_t idx = (static_cast<size_t>(t) * 2654435761ull) & kFeChaseMask;
        size_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          idx = feChase[idx];
          uint8_t payload = static_cast<uint8_t>(idx & 0xFF);
          switch (j % 3) {
            case 0:
              acc = feSwitch0(payload, acc);
              break;
            case 1:
              acc = feSwitch1(payload, acc);
              break;
            default:
              acc = feSwitch2(payload, acc);
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
BENCHMARK_NAMED_PARAM(contendedUseFE, 8_to_100_fe, 8, 100)
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
