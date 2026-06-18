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

#include <array>
#include <cstdint>

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

// Compact frontend-pressure helper used by the contendedUseFnPad variant.
// Three noinline 256-case switches (8 ALU ops/case, unique literals per case
// and per function via the salt) deliberately inflate the executed instruction
// working set without adding a long serial dependency chain, so the
// throughput-carrying semaphore structure of the original benchmark is kept.
namespace {

std::array<uint32_t, 256> makeLifoChase() {
  std::array<uint32_t, 256> a{};
  uint32_t x = 0x243F6A88u;
  for (uint32_t i = 0; i < 256; ++i) {
    x = x * 1103515245u + 12345u;
    a[i] = (x >> 13) & 0xFFu;
  }
  return a;
}

std::array<uint32_t, 256> lifoChase = makeLifoChase();

#define LIFO_CASE(n, salt)                                         \
  case (n):                                                        \
    acc += (uint64_t)((uint32_t)(n) * 2654435761u + (salt) + 1u); \
    acc ^= (uint64_t)((uint32_t)(n) * 40503u + (salt) + 7u);      \
    acc *= (uint64_t)(((uint32_t)(n) | 1u) + (salt));             \
    acc += (uint64_t)((uint32_t)(n) ^ (0x9E3779B9u + (salt)));    \
    acc ^= (uint64_t)((uint32_t)(n) + 0x7F4A7C15u + (salt));      \
    acc *= (uint64_t)((((uint32_t)(n) << 1) | 1u) + (salt));      \
    acc += (uint64_t)((uint32_t)(n) * 2246822519u + (salt));      \
    acc ^= (uint64_t)((uint32_t)(n) * 3266489917u + (salt));      \
    break;

#define LIFO_CASES_16(b, salt)                                            \
  LIFO_CASE(b + 0, salt) LIFO_CASE(b + 1, salt) LIFO_CASE(b + 2, salt)    \
      LIFO_CASE(b + 3, salt) LIFO_CASE(b + 4, salt) LIFO_CASE(b + 5, salt) \
          LIFO_CASE(b + 6, salt) LIFO_CASE(b + 7, salt)                   \
              LIFO_CASE(b + 8, salt) LIFO_CASE(b + 9, salt)               \
                  LIFO_CASE(b + 10, salt) LIFO_CASE(b + 11, salt)         \
                      LIFO_CASE(b + 12, salt) LIFO_CASE(b + 13, salt)     \
                          LIFO_CASE(b + 14, salt) LIFO_CASE(b + 15, salt)

#define LIFO_CASES_256(salt)                                            \
  LIFO_CASES_16(0, salt) LIFO_CASES_16(16, salt) LIFO_CASES_16(32, salt) \
      LIFO_CASES_16(48, salt) LIFO_CASES_16(64, salt)                   \
          LIFO_CASES_16(80, salt) LIFO_CASES_16(96, salt)               \
              LIFO_CASES_16(112, salt) LIFO_CASES_16(128, salt)         \
                  LIFO_CASES_16(144, salt) LIFO_CASES_16(160, salt)     \
                      LIFO_CASES_16(176, salt) LIFO_CASES_16(192, salt) \
                          LIFO_CASES_16(208, salt) LIFO_CASES_16(224, salt) \
                              LIFO_CASES_16(240, salt)

__attribute__((noinline)) uint64_t lifoSwitchA(uint32_t idx, uint64_t acc) {
  switch (idx & 0xFFu) {
    LIFO_CASES_256(0x11u)
    default:
      acc += 1u;
      break;
  }
  return acc;
}

__attribute__((noinline)) uint64_t lifoSwitchB(uint32_t idx, uint64_t acc) {
  switch (idx & 0xFFu) {
    LIFO_CASES_256(0x29u)
    default:
      acc += 2u;
      break;
  }
  return acc;
}

__attribute__((noinline)) uint64_t lifoSwitchC(uint32_t idx, uint64_t acc) {
  switch (idx & 0xFFu) {
    LIFO_CASES_256(0x3Du)
    default:
      acc += 3u;
      break;
  }
  return acc;
}

#undef LIFO_CASES_256
#undef LIFO_CASES_16
#undef LIFO_CASE

std::atomic<uint64_t> lifoSwitchSink{0};

} // namespace

static void contendedUseFnPad(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1u;
        uint32_t idx = (static_cast<uint32_t>(t) * 7u) & 0xFFu;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          idx = lifoChase[idx & 0xFFu];
          switch (i % 3u) {
            case 0:
              acc = lifoSwitchA(idx, acc);
              break;
            case 1:
              acc = lifoSwitchB(idx, acc);
              break;
            default:
              acc = lifoSwitchC(idx, acc);
              break;
          }
        }
        lifoSwitchSink.fetch_add(acc, std::memory_order_relaxed);
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
BENCHMARK_NAMED_PARAM(contendedUseFnPad, 8_to_100_fnpad, 8, 100)
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
