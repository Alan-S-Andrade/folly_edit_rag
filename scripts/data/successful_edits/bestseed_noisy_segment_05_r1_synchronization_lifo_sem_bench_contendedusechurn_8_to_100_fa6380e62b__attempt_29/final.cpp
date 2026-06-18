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
// Derived benchmark: contendedUseChase
//
// This variant augments the contended LifoSem traffic with a large
// pointer-chase load feeding a rotation of three big switch-dispatch
// functions. The pointer chase introduces backend memory stalls while the
// 3-function 256-case switch rotation inflates the executed instruction
// footprint, together driving IPC down toward the reference target.
// ---------------------------------------------------------------------------

// One ALU-heavy switch case body. Every case uses unique literal constants
// derived from its index so the compiler cannot deduplicate the cases.
#define CHASE_CASE(i)                                                        \
  case (i):                                                                  \
    acc += (uint64_t)(SALT) * (uint64_t)((i) * 2654435761u + 0x9e3779b9u);  \
    acc ^= (acc << 13);                                                      \
    acc *= (uint64_t)((i) * 0x100000001b3ull + 0x1234567u + (SALT));        \
    acc += (uint64_t)((i) ^ 0x5bd1e995u) ^ (p);                             \
    acc ^= (acc >> 7) ^ (uint64_t)((i) * 40503u);                           \
    acc *= (uint64_t)(2246822519u + (i) + (SALT));                          \
    acc += (uint64_t)((i) ^ 0xcc9e2d51u);                                   \
    acc ^= (acc << 5) ^ (uint64_t)((i) * 0x1b873593u);                      \
    break;

#define CHASE_C4(b) CHASE_CASE(b) CHASE_CASE(b + 1) CHASE_CASE(b + 2) CHASE_CASE(b + 3)
#define CHASE_C16(b) CHASE_C4(b) CHASE_C4(b + 4) CHASE_C4(b + 8) CHASE_C4(b + 12)
#define CHASE_C64(b) CHASE_C16(b) CHASE_C16(b + 16) CHASE_C16(b + 32) CHASE_C16(b + 48)
#define CHASE_C256 CHASE_C64(0) CHASE_C64(64) CHASE_C64(128) CHASE_C64(192)

__attribute__((noinline)) static uint64_t chaseSwitchA(
    uint64_t acc, uint32_t idx, uint64_t p) {
#define SALT 0x1111u
  switch (idx & 0xFFu) { CHASE_C256 }
#undef SALT
  return acc;
}

__attribute__((noinline)) static uint64_t chaseSwitchB(
    uint64_t acc, uint32_t idx, uint64_t p) {
#define SALT 0x2222u
  switch (idx & 0xFFu) { CHASE_C256 }
#undef SALT
  return acc;
}

__attribute__((noinline)) static uint64_t chaseSwitchC(
    uint64_t acc, uint32_t idx, uint64_t p) {
#define SALT 0x3333u
  switch (idx & 0xFFu) { CHASE_C256 }
#undef SALT
  return acc;
}

// Large pointer-chase ring (power-of-two sized, ~32MB) shared read-only by
// all threads. Built once on first use.
static const std::vector<uint64_t>& chaseBuffer() {
  static const std::vector<uint64_t> buf = [] {
    const size_t n = size_t(1) << 22; // 4M entries * 8B = 32MB
    std::vector<uint64_t> order(n);
    for (size_t i = 0; i < n; ++i) {
      order[i] = i;
    }
    uint64_t s = 0x9e3779b97f4a7c15ull;
    for (size_t i = n; i > 1; --i) {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      size_t j = (size_t)(s % i);
      std::swap(order[i - 1], order[j]);
    }
    std::vector<uint64_t> chase(n);
    for (size_t i = 0; i < n; ++i) {
      chase[order[i]] = order[(i + 1) % n];
    }
    return chase;
  }();
  return buf;
}

static void contendedUseChase(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;
  const auto& chase = chaseBuffer();
  const size_t mask = chase.size() - 1;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = (uint64_t)t + 1;
        size_t cur = ((size_t)t * 2654435761u) & mask;
        uint64_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          cur = (size_t)(chase[cur] & mask);
          uint64_t payload = (uint64_t)cur ^ (acc << 1);
          // Primary corrective lever (Tier T0 IPC, frontend-bound):
          // PROVEN PATTERN. Instead of executing every case of all three
          // dispatch functions on each wait (which kept the executed working
          // set compact via temporal reuse and held L1i MPKI far too low),
          // dispatch into exactly one of the three big switch functions per
          // wait, rotating with j%3, with a switch index taken from the low
          // byte of the pointer-chase payload. Each wait now lands on an
          // essentially unpredictable case in a different oversized function,
          // scattering the executed instruction footprint across a >L1i code
          // region with little reuse. This thrashes the L1 instruction cache,
          // stalls the frontend, and pulls IPC down toward the target.
          uint32_t sw = (uint32_t)(payload & 0xFFu);
          switch (j % 3) {
            case 0:
              acc = chaseSwitchA(acc, sw, payload);
              break;
            case 1:
              acc = chaseSwitchB(acc, sw, payload);
              break;
            default:
              acc = chaseSwitchC(acc, sw, payload);
              break;
          }
          ++j;
        }
        folly::doNotOptimizeAway(acc);
        folly::doNotOptimizeAway(j);
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
BENCHMARK_NAMED_PARAM(contendedUseChase, 8_to_100_chase, 8, 100)

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
