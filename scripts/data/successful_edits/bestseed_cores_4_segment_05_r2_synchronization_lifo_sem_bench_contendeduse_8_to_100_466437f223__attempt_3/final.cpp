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
// Derived icache-pressure variant of contendedUse(8_to_100).
//
// The IPC of the baseline contendedUse(8_to_100) runs hotter than target
// because its executed instruction working set is tiny and reused. To raise
// L1-icache-load-misses_MPKI (and thereby pull IPC down toward target), each
// waiter, after returning from sem.wait(), performs a pointer-chase load and
// dispatches through one of three large 256-case switch functions selected by
// j%3. Each case uses unique literal constants so the compiler cannot
// deduplicate the code, producing a large hot code footprint that thrashes the
// instruction cache.
//
// Primary tuning lever for L1i MPKI is the number of ALU ops per switch case.
namespace {

// Per-function salt drives unique constants across the three rotation targets,
// preventing identical-code folding of the three switch functions.
#define ICACHE_OPS(n)                                                   \
  acc += UINT64_C(0x9E3779B97F4A7C15) * ((uint64_t)(n) + 1u) + SALT;    \
  acc ^= UINT64_C(0xD1B54A32D192ED03) + ((uint64_t)(n) * 7u) + SALT;    \
  acc *= ((uint64_t)(n) | 1u) + SALT;                                   \
  acc += (uint64_t)(n) * UINT64_C(2654435761) + SALT;                   \
  acc ^= ((uint64_t)(n) << 1) + SALT;                                   \
  acc -= (uint64_t)(n) * 40503u + SALT;                                 \
  acc *= (UINT64_C(0xFF51AFD7ED558CCD) | (uint64_t)(n)) + SALT;         \
  acc ^= (uint64_t)(n) * 0x100u + 13u + SALT;

#define ICASE(n)  \
  case (n): {     \
    ICACHE_OPS(n) \
  } break;
#define ICASE4(b) ICASE(b) ICASE(b + 1) ICASE(b + 2) ICASE(b + 3)
#define ICASE16(b) ICASE4(b) ICASE4(b + 4) ICASE4(b + 8) ICASE4(b + 12)
#define ICASE64(b) ICASE16(b) ICASE16(b + 16) ICASE16(b + 32) ICASE16(b + 48)
#define ICASE256 ICASE64(0) ICASE64(64) ICASE64(128) ICASE64(192)

#define SALT UINT64_C(0x1111111111111111)
__attribute__((noinline)) static uint64_t icacheMix0(
    uint8_t idx, uint64_t acc) {
  switch (idx) { ICASE256 }
  return acc;
}
#undef SALT

#define SALT UINT64_C(0x2222222222222222)
__attribute__((noinline)) static uint64_t icacheMix1(
    uint8_t idx, uint64_t acc) {
  switch (idx) { ICASE256 }
  return acc;
}
#undef SALT

#define SALT UINT64_C(0x3333333333333333)
__attribute__((noinline)) static uint64_t icacheMix2(
    uint8_t idx, uint64_t acc) {
  switch (idx) { ICASE256 }
  return acc;
}
#undef SALT

uint8_t icacheChase[256];

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int i = 0; i < 256; ++i) {
      icacheChase[i] = (uint8_t)((i * 167 + 13) & 0xFF);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = (uint64_t)t + 1u;
        uint8_t p = (uint8_t)t;
        for (uint32_t j = t; j < n; j += waiters) {
          sem.wait();
          p = icacheChase[p];
          uint8_t idx = p & 0xFF;
          switch (j % 3) {
            case 0:
              acc = icacheMix0(idx, acc);
              break;
            case 1:
              acc = icacheMix1(idx, acc);
              break;
            default:
              acc = icacheMix2(idx, acc);
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
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_4, 1, 4)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_32, 1, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_1, 4, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_24, 4, 24)
BENCHMARK_NAMED_PARAM(contendedUse, 8_to_100, 8, 100)
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100, 8, 100)
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
