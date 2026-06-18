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

// ----------------------------------------------------------------------------
// Derived from-scratch benchmark for contendedUse(8_to_100).
//
// The contendedUse(8_to_100) reference is frontend/icache starved relative to
// its target counter profile: its executed instruction working set is tiny and
// gets too much temporal reuse, which drives IPC up and L1i MPKI down. To match
// the target hardware-counter profile (notably a higher L1-icache-load-misses
// MPKI and a lower IPC) this companion benchmark deliberately inflates the hot
// code footprint using the proven 3-function 256-case switch rotation pattern.
//
// Each of the three noinline functions holds a 256-entry switch; every case
// runs 14 ALU ops (add/xor/mul) on a local accumulator using per-case-unique
// constants (derived from the case index and a per-function salt) so the
// compiler cannot deduplicate the bodies. The hot loop rotates among the three
// functions with j%3 and indexes each switch with a pointer-chase derived
// payload byte (payload & 0xFF), forcing icache thrashing across a large
// executed-instruction footprint while keeping IPC modest.
namespace {

struct ChaseNode {
  uint64_t payload;
  ChaseNode* next;
};

// 14 ALU ops per case; (s) is a per-function salt to keep the three bodies
// distinct, (n) is the case index to keep each case body unique.
#define ICACHE_OPS14(acc, n, s)                                  \
  do {                                                           \
    (acc) += 0x9E3779B97F4A7C15ull * (uint64_t)(n) + (s) + 1ull; \
    (acc) ^= 0xC2B2AE3D27D4EB4Full * (uint64_t)(n) + 3ull;       \
    (acc) *= (0x165667B19E3779F9ull * (uint64_t)(n)) | 1ull;     \
    (acc) += 0xD6E8FEB86659FD93ull ^ (uint64_t)(n);              \
    (acc) ^= 0xFF51AFD7ED558CCDull + (uint64_t)(n) + (s);        \
    (acc) *= (0xC4CEB9FE1A85EC53ull * (uint64_t)(n)) | 1ull;     \
    (acc) += (uint64_t)(n) * 11400714785074694791ull;            \
    (acc) ^= (uint64_t)(n) * 14029467366897019727ull;            \
    (acc) *= ((uint64_t)(n) * 1609587929392839161ull) | 1ull;    \
    (acc) += (uint64_t)(n) * 9650029242287828579ull + (s);       \
    (acc) ^= (uint64_t)(n) * 0x27D4EB2F165667C5ull;              \
    (acc) *= ((uint64_t)(n) * 0x85EBCA77C2B2AE63ull) | 1ull;     \
    (acc) += (uint64_t)(n) + 0xDEADBEEFull;                      \
    (acc) ^= (uint64_t)(n) + 0xCAFEBABEull + (s);                \
  } while (0)

#define ICASE1(n, s) \
  case (n): {        \
    ICACHE_OPS14(acc, (n), (s)); \
  } break;
#define ICASE16(b, s)                                                       \
  ICASE1((b) + 0, s) ICASE1((b) + 1, s) ICASE1((b) + 2, s)                  \
  ICASE1((b) + 3, s) ICASE1((b) + 4, s) ICASE1((b) + 5, s)                  \
  ICASE1((b) + 6, s) ICASE1((b) + 7, s) ICASE1((b) + 8, s)                  \
  ICASE1((b) + 9, s) ICASE1((b) + 10, s) ICASE1((b) + 11, s)                \
  ICASE1((b) + 12, s) ICASE1((b) + 13, s) ICASE1((b) + 14, s)               \
  ICASE1((b) + 15, s)
#define ICASE256(s)                                                         \
  ICASE16(0, s) ICASE16(16, s) ICASE16(32, s) ICASE16(48, s)                \
  ICASE16(64, s) ICASE16(80, s) ICASE16(96, s) ICASE16(112, s)              \
  ICASE16(128, s) ICASE16(144, s) ICASE16(160, s) ICASE16(176, s)           \
  ICASE16(192, s) ICASE16(208, s) ICASE16(224, s) ICASE16(240, s)

__attribute__((noinline)) static uint64_t icacheSwitch0(
    uint64_t acc, uint32_t idx) {
  switch (idx & 0xFF) {
    ICASE256(0x1111111111111111ull)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheSwitch1(
    uint64_t acc, uint32_t idx) {
  switch (idx & 0xFF) {
    ICASE256(0x2222222222222222ull)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheSwitch2(
    uint64_t acc, uint32_t idx) {
  switch (idx & 0xFF) {
    ICASE256(0x3333333333333333ull)
  }
  return acc;
}

#undef ICASE256
#undef ICASE16
#undef ICASE1
#undef ICACHE_OPS14

} // namespace

BENCHMARK(icacheSwitch_8_to_100, iters) {
  static std::vector<ChaseNode> ring;
  BENCHMARK_SUSPEND {
    if (ring.empty()) {
      const size_t N = 4096;
      ring.resize(N);
      for (size_t i = 0; i < N; ++i) {
        ring[i].payload = i * 0x9E3779B1ull + 0x12345u;
        ring[i].next = &ring[(i + 1) % N];
      }
    }
  }

  ChaseNode* p = ring.data();
  uint64_t acc = 0;
  for (size_t j = 0; j < iters; ++j) {
    uint32_t idx = static_cast<uint32_t>(p->payload) & 0xFFu;
    switch (j % 3) {
      case 0:
        acc += icacheSwitch0(acc, idx);
        break;
      case 1:
        acc += icacheSwitch1(acc, idx);
        break;
      default:
        acc += icacheSwitch2(acc, idx);
        break;
    }
    p = p->next;
  }
  folly::doNotOptimizeAway(acc);
}

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
