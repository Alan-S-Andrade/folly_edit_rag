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
// Derived frontend-bound (I-cache) variant of contendedUse(8_to_100).
//
// The contention skeleton is preserved, but each loop iteration also runs a
// rotating dispatch through three very large __attribute__((noinline)) switch
// functions.  Each switch has 256 cases, each case performs a block of ALU ops
// on a local accumulator using case- and function-unique literal constants to
// defeat compiler deduplication.  The case index is derived from a pointer
// chase load (payload & 0xFF) so dispatch is data dependent.  Rotating among
// the three functions with j % 3 produces a large hot instruction footprint
// that thrashes the L1 instruction cache and lowers IPC.
// ---------------------------------------------------------------------------
namespace {

#define IC_OP_BLOCK(i)                                                       \
  acc += 0x9e3779b97f4a7c15ull + (uint64_t)(i) * 2654435761ull + IC_S;       \
  acc ^= (uint64_t)(i) * 40503ull + 0x1234567ull;                           \
  acc *= (((uint64_t)(i) << 1) | 1ull);                                      \
  acc += ((uint64_t)(i) ^ 0xdeadbeefull ^ IC_S);                            \
  acc ^= (uint64_t)(i) * 7919ull + 0xabcdefull;                             \
  acc *= (((uint64_t)(i) * 3ull) | 1ull);                                    \
  acc += (uint64_t)(i) * 0x100000001b3ull;                                   \
  acc ^= ~((uint64_t)(i) * 11ull + IC_S);                                    \
  acc += (uint64_t)(i) * 1000003ull + 17ull;                                 \
  acc ^= ((uint64_t)(i) << 3) ^ 0x55555555ull;                              \
  acc *= (((uint64_t)(i) + 5ull) | 1ull);                                    \
  acc += (uint64_t)(i) * 0x9ddfea08eb382d69ull;                             \
  acc ^= (uint64_t)(i) * 65599ull + IC_S;                                    \
  acc += (uint64_t)(i) * 31ull + 0x7eull;                                    \
  acc *= (((uint64_t)(i) ^ 0xaaull) | 1ull);                                 \
  acc ^= (uint64_t)(i) * 2246822519ull;                                      \
  acc += (uint64_t)(i) * 3266489917ull + IC_S;                              \
  acc ^= (uint64_t)(i) * 668265263ull;

#define IC_CASE(i) \
  case (i): {      \
    IC_OP_BLOCK(i) \
    break;         \
  }

#define IC_CASE16(b)                                                  \
  IC_CASE((b) + 0) IC_CASE((b) + 1) IC_CASE((b) + 2) IC_CASE((b) + 3) \
  IC_CASE((b) + 4) IC_CASE((b) + 5) IC_CASE((b) + 6) IC_CASE((b) + 7) \
  IC_CASE((b) + 8) IC_CASE((b) + 9) IC_CASE((b) + 10)                 \
  IC_CASE((b) + 11) IC_CASE((b) + 12) IC_CASE((b) + 13)               \
  IC_CASE((b) + 14) IC_CASE((b) + 15)

#define IC_SWITCH_BODY                                              \
  IC_CASE16(0) IC_CASE16(16) IC_CASE16(32) IC_CASE16(48)           \
  IC_CASE16(64) IC_CASE16(80) IC_CASE16(96) IC_CASE16(112)         \
  IC_CASE16(128) IC_CASE16(144) IC_CASE16(160) IC_CASE16(176)      \
  IC_CASE16(192) IC_CASE16(208) IC_CASE16(224) IC_CASE16(240)

#define IC_S 0x1111111111111111ull
__attribute__((__noinline__)) static uint64_t icacheSwitchA(
    uint8_t idx, uint64_t acc) {
  switch (idx) {
    IC_SWITCH_BODY
  }
  return acc;
}
#undef IC_S

#define IC_S 0x2222222222222222ull
__attribute__((__noinline__)) static uint64_t icacheSwitchB(
    uint8_t idx, uint64_t acc) {
  switch (idx) {
    IC_SWITCH_BODY
  }
  return acc;
}
#undef IC_S

#define IC_S 0x3333333333333333ull
__attribute__((__noinline__)) static uint64_t icacheSwitchC(
    uint8_t idx, uint64_t acc) {
  switch (idx) {
    IC_SWITCH_BODY
  }
  return acc;
}
#undef IC_S

#undef IC_SWITCH_BODY
#undef IC_CASE16
#undef IC_CASE
#undef IC_OP_BLOCK

std::vector<uint8_t> makeChase(size_t len) {
  std::vector<uint8_t> v(len);
  uint32_t s = 2463534242u;
  for (size_t i = 0; i < len; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    v[i] = static_cast<uint8_t>(s >> 16);
  }
  return v;
}

__attribute__((__noinline__)) static uint64_t icacheWork(
    const uint8_t* chase, size_t len, uint64_t seed) {
  uint64_t acc = seed;
  size_t p = seed % len;
  for (size_t j = 0; j < len; ++j) {
    uint8_t payload = chase[p];
    uint8_t idx = payload & 0xFF;
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
    p = (p + payload + 1u) % len;
  }
  return acc;
}

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  static const std::vector<uint8_t> chase = makeChase(256);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        for (uint32_t i = t; i < n; i += waiters) {
          acc = icacheWork(chase.data(), chase.size(), acc + i);
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
        uint64_t acc = static_cast<uint64_t>(t) + 7;
        for (uint32_t i = t; i < n; i += posters) {
          acc = icacheWork(chase.data(), chase.size(), acc + i);
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
