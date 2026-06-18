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

#include <mutex>

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
// The reference benchmark executes a very compact instruction working set,
// which yields an artificially high IPC and a tiny L1 i-cache footprint.
// To match the hardware-counter profile we inject a large, branchy code
// footprint: three noinline functions each containing a 256-way switch whose
// index is derived from a pointer-chase load. Rotating across the three
// functions with j%3 forces L1 i-cache thrashing across a big code region.
// Each case uses unique literal constants to defeat compiler de-duplication.
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t kFeRingSize = 4096;

struct FeNode {
  uint32_t next;
  uint32_t payload;
};

std::vector<FeNode> feRing;
std::once_flag feInitFlag;

void feInitRing() {
  feRing.resize(kFeRingSize);
  for (uint32_t i = 0; i < kFeRingSize; ++i) {
    feRing[i].next = (i * 2654435761u + 1u) % kFeRingSize;
    feRing[i].payload = i * 40503u + 12345u;
  }
}

// 16 ALU ops per case, all keyed off the literal case index K and a
// per-function salt S so that no two case bodies are identical.
#define FE_OPS(K, S)                                       \
  acc += (uint64_t)((K) * 2654435761u + (S) + 1u);         \
  acc ^= (uint64_t)((K) ^ ((S) + 0x9e3779b9u));            \
  acc *= (uint64_t)(((K) | 1u) + (S));                     \
  acc += (uint64_t)((K) * 7u + (S) + 3u);                  \
  acc ^= (uint64_t)(((K) << 3) + (S));                     \
  acc *= (uint64_t)(((K) * 5u) | 1u);                      \
  acc += (uint64_t)((K) + 0xABCDu + (S));                  \
  acc ^= (uint64_t)((~(uint32_t)(K) & 0xFFu) + (S));       \
  acc += (uint64_t)((K) * 3u + (S) + 7u);                  \
  acc ^= (uint64_t)(((K) >> 1) + (S) + 0x55u);             \
  acc *= (uint64_t)(((K) ^ 0x33u) | 1u);                   \
  acc += (uint64_t)((K) * 11u + (S));                      \
  acc ^= (uint64_t)(((K) << 1) + (S) + 0xAAu);             \
  acc *= (uint64_t)(((K) + 3u) | 1u);                      \
  acc += (uint64_t)((K) * 13u + (S) + 17u);                \
  acc ^= (uint64_t)(((K) | 0x80u) + (S));

#define FE_CASE(K, S) \
  case (K): {         \
    FE_OPS(K, S)      \
    break;            \
  }

#define FE_C4(K, S) \
  FE_CASE(K + 0, S) FE_CASE(K + 1, S) FE_CASE(K + 2, S) FE_CASE(K + 3, S)
#define FE_C16(K, S) \
  FE_C4(K + 0, S) FE_C4(K + 4, S) FE_C4(K + 8, S) FE_C4(K + 12, S)
#define FE_C64(K, S) \
  FE_C16(K + 0, S) FE_C16(K + 16, S) FE_C16(K + 32, S) FE_C16(K + 48, S)
#define FE_C256(S) \
  FE_C64(0, S) FE_C64(64, S) FE_C64(128, S) FE_C64(192, S)

__attribute__((noinline)) uint64_t feSwitchA(uint8_t s, uint64_t acc) {
  switch (s) { FE_C256(0x11u) }
  return acc;
}

__attribute__((noinline)) uint64_t feSwitchB(uint8_t s, uint64_t acc) {
  switch (s) { FE_C256(0x22u) }
  return acc;
}

__attribute__((noinline)) uint64_t feSwitchC(uint8_t s, uint64_t acc) {
  switch (s) { FE_C256(0x33u) }
  return acc;
}

#undef FE_C256
#undef FE_C64
#undef FE_C16
#undef FE_C4
#undef FE_CASE
#undef FE_OPS

} // namespace

static void contendedUseFe(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;
  std::call_once(feInitFlag, feInitRing);

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t acc = 0x1234567u + (uint64_t)t;
        uint32_t idx = (uint32_t)(t * 97u) % kFeRingSize;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int j = 0; j < 8; ++j) {
            idx = feRing[idx].next;
            uint8_t s = (uint8_t)(feRing[idx].payload & 0xFFu);
            switch (j % 3) {
              case 0:
                acc = feSwitchA(s, acc);
                break;
              case 1:
                acc = feSwitchB(s, acc);
                break;
              default:
                acc = feSwitchC(s, acc);
                break;
            }
          }
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
  folly::doNotOptimizeAway(sink.load());
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
BENCHMARK_NAMED_PARAM(contendedUseFe, 8_to_100_icache, 8, 100)

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
