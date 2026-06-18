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

// Derived from contendedUse(8_to_100): a frontend-bound variant that exercises
// a large hot instruction footprint to raise L1 i-cache pressure. Three
// noinline 256-case switch functions are rotated (j % 3) using an index drawn
// from a pointer-chase load, forcing i-cache thrashing across a wide code span.
namespace {

struct ChaseNode {
  ChaseNode* next;
  uint8_t payload;
};

#define ICACHE_C(n) \
  case (n): { \
    a += ((uint32_t)(n) * 2654435761u + SALT); \
    a ^= a >> 13; \
    a *= (((uint32_t)(n) | 1u) * 40503u + 1u); \
    a += ((uint32_t)(n) ^ (0xABCDu ^ SALT)); \
    a ^= a << 7; \
    a *= (0x85ebca6bu + (uint32_t)(n)); \
    a -= ((uint32_t)(n) * 7u + 3u + SALT); \
    a ^= a >> 11; \
    a += ((uint32_t)(n) * 13u + 5u); \
    a *= (0xc2b2ae35u ^ SALT); \
    a ^= ((uint32_t)(n) << 3); \
    a += (0x27d4eb2fu ^ (uint32_t)(n)); \
    break; \
  }

#define ICACHE_C16(b) \
  ICACHE_C((b) + 0) ICACHE_C((b) + 1) ICACHE_C((b) + 2) ICACHE_C((b) + 3) \
  ICACHE_C((b) + 4) ICACHE_C((b) + 5) ICACHE_C((b) + 6) ICACHE_C((b) + 7) \
  ICACHE_C((b) + 8) ICACHE_C((b) + 9) ICACHE_C((b) + 10) ICACHE_C((b) + 11) \
  ICACHE_C((b) + 12) ICACHE_C((b) + 13) ICACHE_C((b) + 14) ICACHE_C((b) + 15)

#define ICACHE_SWITCH256 \
  switch (idx) { \
    ICACHE_C16(0) ICACHE_C16(16) ICACHE_C16(32) ICACHE_C16(48) \
    ICACHE_C16(64) ICACHE_C16(80) ICACHE_C16(96) ICACHE_C16(112) \
    ICACHE_C16(128) ICACHE_C16(144) ICACHE_C16(160) ICACHE_C16(176) \
    ICACHE_C16(192) ICACHE_C16(208) ICACHE_C16(224) ICACHE_C16(240) \
  }

#define SALT 0x9e3779b9u
__attribute__((noinline)) static uint32_t icacheMixA(uint32_t a, uint8_t idx) {
  ICACHE_SWITCH256
  return a;
}
#undef SALT

#define SALT 0x85ebca6bu
__attribute__((noinline)) static uint32_t icacheMixB(uint32_t a, uint8_t idx) {
  ICACHE_SWITCH256
  return a;
}
#undef SALT

#define SALT 0xc2b2ae35u
__attribute__((noinline)) static uint32_t icacheMixC(uint32_t a, uint8_t idx) {
  ICACHE_SWITCH256
  return a;
}
#undef SALT

#undef ICACHE_SWITCH256
#undef ICACHE_C16
#undef ICACHE_C

} // namespace

BENCHMARK(contendedUse_8_to_100_icache, iters) {
  constexpr size_t kNodes = 4096;
  std::vector<ChaseNode> nodes(kNodes);
  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kNodes; ++i) {
      nodes[i].next = &nodes[(i * 2654435761u + 1u) % kNodes];
      nodes[i].payload = (uint8_t)((i * 1103515245u + 12345u) >> 16);
    }
  }

  ChaseNode* p = &nodes[0];
  uint32_t acc = (uint32_t)iters;
  for (size_t j = 0; j < iters; ++j) {
    uint8_t idx = p->payload;
    switch (j % 3) {
      case 0:
        acc = icacheMixA(acc, idx);
        break;
      case 1:
        acc = icacheMixB(acc, idx);
        break;
      default:
        acc = icacheMixC(acc, idx);
        break;
    }
    p = p->next;
  }
  folly::doNotOptimizeAway(acc);
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
