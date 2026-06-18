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

// --- Frontend-pressure variant support (contendedUseFE) -------------------
// A compact 3-function / 256-case switch rotation that expands the executed
// instruction working set in the waiter hot path. Each case uses unique
// literal constants (derived from the case index and a per-function salt) to
// prevent compiler deduplication / ICF folding so that the icache footprint
// stays large. The switch index is taken from a pointer-chase load so the
// rotation stays off the deepest dependency chain while still perturbing the
// frontend.

#define FE_OPS(i)                                            \
  acc ^= (uint64_t)((i)*2654435761u + SALT);                 \
  acc += (uint64_t)((i) ^ 0x5bd1e995u);                      \
  acc *= (uint64_t)((i) | 0x10000001u);                      \
  acc ^= (acc >> 7) + (uint64_t)((i)*3u + 1u);               \
  acc += (uint64_t)((i) << 3) ^ 0xdeadbeefu;                 \
  acc *= 0x100000001b3ull;                                   \
  acc ^= (acc << 5) + (uint64_t)((i)*7u + SALT);             \
  acc += (uint64_t)(i)*0xff51afd7ull;

#define FE_CASE(i) \
  case (i): {      \
    FE_OPS(i)      \
  } break;
#define FE_C4(i) FE_CASE(i) FE_CASE(i + 1) FE_CASE(i + 2) FE_CASE(i + 3)
#define FE_C16(i) FE_C4(i) FE_C4(i + 4) FE_C4(i + 8) FE_C4(i + 12)
#define FE_C64(i) FE_C16(i) FE_C16(i + 16) FE_C16(i + 32) FE_C16(i + 48)
#define FE_C256(i) FE_C64(i) FE_C64(i + 64) FE_C64(i + 128) FE_C64(i + 192)

#define SALT 0x9e3779b9u
__attribute__((noinline)) static uint64_t feMix0(uint64_t acc, uint8_t idx) {
  switch (idx) {
    FE_C256(0)
  }
  return acc;
}
#undef SALT

#define SALT 0x85ebca6bu
__attribute__((noinline)) static uint64_t feMix1(uint64_t acc, uint8_t idx) {
  switch (idx) {
    FE_C256(0)
  }
  return acc;
}
#undef SALT

#define SALT 0xc2b2ae35u
__attribute__((noinline)) static uint64_t feMix2(uint64_t acc, uint8_t idx) {
  switch (idx) {
    FE_C256(0)
  }
  return acc;
}
#undef SALT

static uint32_t feChase[256];
static bool feChaseInit = [] {
  for (uint32_t i = 0; i < 256; ++i) {
    feChase[i] = (i * 167u + 13u) & 0xFFu;
  }
  return true;
}();

static void contendedUseFE(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t acc = 0x12345678u ^ static_cast<uint64_t>(t);
        uint32_t cur = static_cast<uint32_t>(t) & 0xFFu;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          uint32_t next = feChase[cur];
          uint8_t idx = static_cast<uint8_t>(next & 0xFFu);
          switch (i % 3u) {
            case 0:
              acc = feMix0(acc, idx);
              break;
            case 1:
              acc = feMix1(acc, idx);
              break;
            default:
              acc = feMix2(acc, idx);
              break;
          }
          cur = next;
        }
        sink.fetch_add(acc, std::memory_order_relaxed);
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

#undef FE_C256
#undef FE_C64
#undef FE_C16
#undef FE_C4
#undef FE_CASE
#undef FE_OPS

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
