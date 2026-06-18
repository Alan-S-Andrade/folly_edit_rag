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

// Derived from contendedUse(8_to_100): same contended LifoSem traffic, but each
// satisfied waiter executes a rotating, noinline 256-case ALU switch indexed by
// a pointer-chase payload. This inflates the executed instruction working set to
// raise L1 i-cache pressure (and thus lower IPC toward the frontend-bound
// target) without disturbing the underlying semaphore contention pattern.
namespace {

#define ALU_CASE(N)                                              \
  case (N):                                                      \
    acc += (uint32_t)((N) * 0x9e3779b1u + SEED + 0x01u);         \
    acc ^= (uint32_t)((N) * 0x85ebca77u + SEED + 0x02u);         \
    acc *= (uint32_t)((((N) << 1) | 1u) + SEED + 0x03u);         \
    acc += (uint32_t)(((N) ^ 0xdeadbeefu) + SEED + 0x04u);       \
    acc ^= (uint32_t)((N) * 0xc2b2ae35u + SEED + 0x05u);         \
    acc -= (uint32_t)((N) * 0x27d4eb2fu + SEED + 0x06u);         \
    acc += (uint32_t)((N) * 0x165667b1u + SEED + 0x07u);         \
    acc ^= (uint32_t)((N) * 0xff51afd7u + SEED + 0x08u);         \
    acc *= (uint32_t)((((N) << 2) | 1u) + SEED + 0x09u);         \
    acc += (uint32_t)((N) * 0xd6e8feb3u + SEED + 0x0au);         \
    acc ^= (uint32_t)((N) * 0xcc9e2d51u + SEED + 0x0bu);         \
    acc -= (uint32_t)((N) * 0x1b873593u + SEED + 0x0cu);         \
    acc += (uint32_t)((N) * 0xe6546b64u + SEED + 0x0du);         \
    acc ^= (uint32_t)((N) * 0x3c6ef372u + SEED + 0x0eu);         \
    break;

#define ALU_C4(B) \
  ALU_CASE(B) ALU_CASE((B) + 1) ALU_CASE((B) + 2) ALU_CASE((B) + 3)
#define ALU_C16(B) \
  ALU_C4(B) ALU_C4((B) + 4) ALU_C4((B) + 8) ALU_C4((B) + 12)
#define ALU_C64(B) \
  ALU_C16(B) ALU_C16((B) + 16) ALU_C16((B) + 32) ALU_C16((B) + 48)
#define ALU_C256 ALU_C64(0) ALU_C64(64) ALU_C64(128) ALU_C64(192)

#define SEED 0x10000001u
__attribute__((noinline)) uint32_t aluSwitch0(uint32_t acc, uint8_t idx) {
  switch (idx) {
    ALU_C256
  }
  return acc;
}
#undef SEED

#define SEED 0x20000002u
__attribute__((noinline)) uint32_t aluSwitch1(uint32_t acc, uint8_t idx) {
  switch (idx) {
    ALU_C256
  }
  return acc;
}
#undef SEED

#define SEED 0x30000003u
__attribute__((noinline)) uint32_t aluSwitch2(uint32_t acc, uint8_t idx) {
  switch (idx) {
    ALU_C256
  }
  return acc;
}
#undef SEED

#undef ALU_C256
#undef ALU_C64
#undef ALU_C16
#undef ALU_C4
#undef ALU_CASE

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink{0};

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint32_t chain[256];
        for (uint32_t k = 0; k < 256; ++k) {
          chain[k] = (k * 167u + 13u) & 0xFFu;
        }
        uint32_t acc = 0x12345678u ^ static_cast<uint32_t>(t);
        uint8_t p = static_cast<uint8_t>(t);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = static_cast<uint8_t>(chain[p]);
          switch (j % 3) {
            case 0:
              acc = aluSwitch0(acc, p);
              break;
            case 1:
              acc = aluSwitch1(acc, p);
              break;
            default:
              acc = aluSwitch2(acc, p);
              break;
          }
          ++j;
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
  folly::doNotOptimizeAway(sink.load(std::memory_order_relaxed));
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
BENCHMARK_DRAW_LINE();
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
