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

// I-cache footprint generator: three noinline functions, each a 256-case
// switch with 14 unique ALU ops per case. Rotating among them with j%3 in the
// hot loop forces a large, low-reuse executed instruction working set, which
// raises L1i misses (and lowers IPC) without changing the semaphore protocol.
namespace {

#define LIFO_ALU_CASE(n)                                          \
  case (n): {                                                     \
    acc += (uint64_t)(n) * 0x9E3779B97F4A7C15ull + 0x1u;          \
    acc ^= (uint64_t)(n) + 0xABCD0u;                              \
    acc *= ((uint64_t)(n) | 0x3u);                                \
    acc += (uint64_t)(n) * 7u + 0x5u;                             \
    acc ^= ((uint64_t)(n) << 1) | 0x1u;                           \
    acc -= (uint64_t)(n) + 0x2Cu;                                 \
    acc *= ((uint64_t)(n) | 0x9u);                                \
    acc += (uint64_t)(n) ^ 0xA5u;                                 \
    acc ^= (uint64_t)(n) + 0x1Fu;                                 \
    acc += (uint64_t)(n) * 3u + 0x7u;                             \
    acc *= ((uint64_t)(n) | 0xDu);                                \
    acc -= (uint64_t)(n) ^ 0x3Cu;                                 \
    acc ^= (uint64_t)(n) + 0x77u;                                 \
    acc += ((uint64_t)(n) | 0x41u);                               \
    break;                                                        \
  }

#define LIFO_ALU_16(b)                                                       \
  LIFO_ALU_CASE(b + 0) LIFO_ALU_CASE(b + 1) LIFO_ALU_CASE(b + 2)             \
  LIFO_ALU_CASE(b + 3) LIFO_ALU_CASE(b + 4) LIFO_ALU_CASE(b + 5)             \
  LIFO_ALU_CASE(b + 6) LIFO_ALU_CASE(b + 7) LIFO_ALU_CASE(b + 8)             \
  LIFO_ALU_CASE(b + 9) LIFO_ALU_CASE(b + 10) LIFO_ALU_CASE(b + 11)           \
  LIFO_ALU_CASE(b + 12) LIFO_ALU_CASE(b + 13) LIFO_ALU_CASE(b + 14)          \
  LIFO_ALU_CASE(b + 15)

#define LIFO_ALU_256                                                         \
  LIFO_ALU_16(0) LIFO_ALU_16(16) LIFO_ALU_16(32) LIFO_ALU_16(48)             \
  LIFO_ALU_16(64) LIFO_ALU_16(80) LIFO_ALU_16(96) LIFO_ALU_16(112)           \
  LIFO_ALU_16(128) LIFO_ALU_16(144) LIFO_ALU_16(160) LIFO_ALU_16(176)        \
  LIFO_ALU_16(192) LIFO_ALU_16(208) LIFO_ALU_16(224) LIFO_ALU_16(240)

__attribute__((noinline)) uint64_t icacheFn0(uint64_t acc, uint8_t idx) {
  acc += 0x1111u;
  switch (idx) {
    LIFO_ALU_256
  }
  return acc;
}

__attribute__((noinline)) uint64_t icacheFn1(uint64_t acc, uint8_t idx) {
  acc += 0x2222u;
  switch (idx) {
    LIFO_ALU_256
  }
  return acc;
}

__attribute__((noinline)) uint64_t icacheFn2(uint64_t acc, uint8_t idx) {
  acc += 0x3333u;
  switch (idx) {
    LIFO_ALU_256
  }
  return acc;
}

#undef LIFO_ALU_256
#undef LIFO_ALU_16
#undef LIFO_ALU_CASE

const std::vector<uint32_t>& icacheChase() {
  static const std::vector<uint32_t> chase = [] {
    std::vector<uint32_t> v(4096);
    for (size_t i = 0; i < v.size(); ++i) {
      v[i] = static_cast<uint32_t>((i * 2654435761u + 12345u) % v.size());
    }
    return v;
  }();
  return chase;
}

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  const std::vector<uint32_t>& chase = icacheChase();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t * 97 + 1) % chase.size();
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = icacheFn0(acc, idx);
              break;
            case 1:
              acc = icacheFn1(acc, idx);
              break;
            default:
              acc = icacheFn2(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1, 31, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 16_to_16, 16, 16)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_32, 32, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1000, 32, 1000)
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100, 8, 100)

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
