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

// ---------------------------------------------------------------------------
// Derived frontend-pressure variant of contendedUse(8_to_100).
//
// A large, deduplication-resistant instruction working set is generated via
// three noinline 256-case switch routines (rotated by j%3) that each apply a
// run of unique-constant ALU ops to a local accumulator. The switch index is
// driven by a pointer-chase load (payload & 0xFF). This intentionally inflates
// L1 instruction-cache pressure to bring the measured IPC down toward target.
// ---------------------------------------------------------------------------

static uint32_t* g_chase = nullptr;
static void initChase() {
  static std::vector<uint32_t> buf;
  if (buf.empty()) {
    buf.resize(4096);
    for (size_t i = 0; i < buf.size(); ++i) {
      buf[i] = static_cast<uint32_t>((i * 2654435761u) ^ (i << 7) ^ 0x9e3779b9u);
    }
    g_chase = buf.data();
  }
}

#define ALU_BODY(n, salt)                          \
  acc += ((n) * 2654435761u + (salt));             \
  acc ^= (acc >> 15);                              \
  acc *= (2246822519u + (salt));                   \
  acc += ((n) * 374761393u + 13u);                 \
  acc = (acc << 3) | (acc >> 29);                  \
  acc -= ((n) * 668265263u + (salt));              \
  acc ^= ((n) << 5) ^ (0x5bd1e995u + (salt));      \
  acc *= 0x85ebca6bu;                              \
  acc += ((n) ^ (0xc2b2ae35u + (salt)));           \
  acc ^= (acc >> 13);                              \
  acc += ((n) * 9176u + 3u);                       \
  acc *= 3u;                                        \
  acc ^= ((n) + (0x27d4eb2fu + (salt)));           \
  acc += ((n) * 40503u + 7u);

#define CASE_A(n) \
  case (n): {     \
    ALU_BODY(n, 0x11u) \
  } break;
#define CASE_B(n) \
  case (n): {     \
    ALU_BODY(n, 0x37u) \
  } break;
#define CASE_C(n) \
  case (n): {     \
    ALU_BODY(n, 0x5Du) \
  } break;

#define R4(M, b) M(b + 0) M(b + 1) M(b + 2) M(b + 3)
#define R16(M, b) R4(M, b + 0) R4(M, b + 4) R4(M, b + 8) R4(M, b + 12)
#define R64(M, b) R16(M, b + 0) R16(M, b + 16) R16(M, b + 32) R16(M, b + 48)
#define R256(M) R64(M, 0) R64(M, 64) R64(M, 128) R64(M, 192)

__attribute__((noinline)) static uint32_t alu_switch_a(
    uint32_t idx, uint32_t acc) {
  switch (idx & 0xFFu) { R256(CASE_A) }
  return acc;
}

__attribute__((noinline)) static uint32_t alu_switch_b(
    uint32_t idx, uint32_t acc) {
  switch (idx & 0xFFu) { R256(CASE_B) }
  return acc;
}

__attribute__((noinline)) static uint32_t alu_switch_c(
    uint32_t idx, uint32_t acc) {
  switch (idx & 0xFFu) { R256(CASE_C) }
  return acc;
}

__attribute__((noinline)) static uint32_t frontendStress(uint32_t seed) {
  uint32_t acc = seed;
  uint32_t p = g_chase[seed & 4095u];
  for (int j = 0; j < 3; ++j) {
    uint32_t idx = (p ^ acc) & 0xFFu;
    switch (j % 3) {
      case 0:
        acc = alu_switch_a(idx, acc);
        break;
      case 1:
        acc = alu_switch_b(idx, acc);
        break;
      default:
        acc = alu_switch_c(idx, acc);
        break;
    }
    p = g_chase[acc & 4095u];
  }
  return acc;
}

static void contendedUseFront(uint32_t n, int posters, int waiters) {
  initChase();
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint32_t> sink(0);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint32_t acc = static_cast<uint32_t>(t) + 1u;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          acc = frontendStress(acc + i);
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

BENCHMARK_NAMED_PARAM(contendedUseFront, 8_to_100, 8, 100)

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
