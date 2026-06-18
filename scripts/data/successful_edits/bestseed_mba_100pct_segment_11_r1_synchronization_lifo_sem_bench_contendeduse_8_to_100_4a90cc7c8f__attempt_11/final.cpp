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
// icacheContendedUse: a from-scratch frontend-bound variant of the contended
// LifoSem workload. The waiter threads drive a large executed-instruction
// working set (three noinline 256-case switches rotated by j % 3, indexed by a
// pointer-chase payload byte) to raise L1-icache pressure and lower IPC into
// the contended-use target band. The semaphore traffic itself is unchanged in
// spirit from contendedUse.
// ---------------------------------------------------------------------------

namespace {

struct ChaseNode {
  uint32_t next;
  uint32_t payload;
};

std::vector<ChaseNode> makeChaseRing(size_t len) {
  std::vector<ChaseNode> ring(len);
  for (size_t i = 0; i < len; ++i) {
    ring[i].next = static_cast<uint32_t>((i * 2654435761u + 1u) % len);
    ring[i].payload = static_cast<uint32_t>(i * 0x9E3779B9u);
  }
  return ring;
}

} // namespace

// 14 unique ALU ops per case; the salt S and key K keep every literal site
// distinct so the compiler cannot deduplicate cases or functions.
#define ICACHE_ALU(S, K)                                    \
  do {                                                      \
    acc += 0x9E3779B9u + (uint32_t)(K) + (uint32_t)(S);     \
    acc ^= (acc << 13) ^ (0x100u + (uint32_t)(K));          \
    acc *= 0x85EBCA77u ^ (uint32_t)(K);                     \
    acc += (uint32_t)(K) * 2654435761u + (uint32_t)(S);     \
    acc ^= acc >> 7;                                        \
    acc *= 0xC2B2AE3Du + (uint32_t)(K);                     \
    acc += 0x27D4EB2Fu ^ (uint32_t)(K) ^ (uint32_t)(S);     \
    acc ^= (acc << 5) + (uint32_t)(K);                      \
    acc *= 0x165667B1u + (uint32_t)(K);                     \
    acc += (uint32_t)(K) ^ 0xDEADBEEFu;                     \
    acc ^= acc >> 11;                                       \
    acc *= 0x9E3779B1u + (uint32_t)(K) + (uint32_t)(S);     \
    acc += 0xFEEDFACEu ^ (uint32_t)(K);                     \
    acc ^= (uint32_t)(K) * 0x7FEDu + (uint32_t)(S);         \
  } while (0)

#define ICACHE_CASE(S, K) \
  case (K):               \
    ICACHE_ALU((S), (K)); \
    break;

#define ICACHE_CASES16(S, B)                            \
  ICACHE_CASE(S, (B) + 0) ICACHE_CASE(S, (B) + 1)       \
  ICACHE_CASE(S, (B) + 2) ICACHE_CASE(S, (B) + 3)       \
  ICACHE_CASE(S, (B) + 4) ICACHE_CASE(S, (B) + 5)       \
  ICACHE_CASE(S, (B) + 6) ICACHE_CASE(S, (B) + 7)       \
  ICACHE_CASE(S, (B) + 8) ICACHE_CASE(S, (B) + 9)       \
  ICACHE_CASE(S, (B) + 10) ICACHE_CASE(S, (B) + 11)     \
  ICACHE_CASE(S, (B) + 12) ICACHE_CASE(S, (B) + 13)     \
  ICACHE_CASE(S, (B) + 14) ICACHE_CASE(S, (B) + 15)

#define ICACHE_CASES256(S)                              \
  ICACHE_CASES16(S, 0) ICACHE_CASES16(S, 16)            \
  ICACHE_CASES16(S, 32) ICACHE_CASES16(S, 48)           \
  ICACHE_CASES16(S, 64) ICACHE_CASES16(S, 80)           \
  ICACHE_CASES16(S, 96) ICACHE_CASES16(S, 112)          \
  ICACHE_CASES16(S, 128) ICACHE_CASES16(S, 144)         \
  ICACHE_CASES16(S, 160) ICACHE_CASES16(S, 176)         \
  ICACHE_CASES16(S, 192) ICACHE_CASES16(S, 208)         \
  ICACHE_CASES16(S, 224) ICACHE_CASES16(S, 240)

__attribute__((noinline)) static uint32_t icacheSwitch0(
    uint32_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_CASES256(0x11u)
  }
  return acc;
}

__attribute__((noinline)) static uint32_t icacheSwitch1(
    uint32_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_CASES256(0x2323u)
  }
  return acc;
}

__attribute__((noinline)) static uint32_t icacheSwitch2(
    uint32_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_CASES256(0x4545u)
  }
  return acc;
}

#undef ICACHE_CASES256
#undef ICACHE_CASES16
#undef ICACHE_CASE
#undef ICACHE_ALU

static void icacheContendedUse(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  static const std::vector<ChaseNode> ring = makeChaseRing(8192);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint32_t pos = static_cast<uint32_t>(t) % ring.size();
        uint32_t acc = 0x12345678u ^ static_cast<uint32_t>(t);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          pos = ring[pos].next;
          uint8_t idx = static_cast<uint8_t>(ring[pos].payload & 0xFFu);
          switch (i % 3u) {
            case 0:
              acc = icacheSwitch0(acc, idx);
              break;
            case 1:
              acc = icacheSwitch1(acc, idx);
              break;
            default:
              acc = icacheSwitch2(acc, idx);
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
BENCHMARK_NAMED_PARAM(icacheContendedUse, 8_to_100, 8, 100)

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
