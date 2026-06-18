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

namespace {

// Three large, near-identical switch bodies used to inflate the instruction
// footprint of the contended waiter hot path. Rotating among them while the
// switch index comes from a pointer-chase load makes the indirect branch
// targets unpredictable, exercising the frontend (icache + indirect branch).
#define LIFO_OP(x)                                  \
  case (x):                                         \
    acc = acc * 6364136223846793005ULL + (x) + k;  \
    acc ^= acc >> 27;                               \
    break;

#define LIFO_OP16(b)                                                  \
  LIFO_OP(b + 0) LIFO_OP(b + 1) LIFO_OP(b + 2) LIFO_OP(b + 3)         \
  LIFO_OP(b + 4) LIFO_OP(b + 5) LIFO_OP(b + 6) LIFO_OP(b + 7)         \
  LIFO_OP(b + 8) LIFO_OP(b + 9) LIFO_OP(b + 10) LIFO_OP(b + 11)       \
  LIFO_OP(b + 12) LIFO_OP(b + 13) LIFO_OP(b + 14) LIFO_OP(b + 15)

#define LIFO_SWITCH                                                  \
  switch (idx) {                                                     \
    LIFO_OP16(0) LIFO_OP16(16) LIFO_OP16(32) LIFO_OP16(48)           \
    LIFO_OP16(64) LIFO_OP16(80) LIFO_OP16(96) LIFO_OP16(112)         \
    LIFO_OP16(128) LIFO_OP16(144) LIFO_OP16(160) LIFO_OP16(176)      \
    LIFO_OP16(192) LIFO_OP16(208) LIFO_OP16(224) LIFO_OP16(240)      \
    default:                                                         \
      break;                                                         \
  }

FOLLY_NOINLINE uint64_t lifoMixA(uint64_t acc, uint8_t idx) {
  const uint64_t k = 0x1111111111111111ULL;
  LIFO_SWITCH
  return acc;
}

FOLLY_NOINLINE uint64_t lifoMixB(uint64_t acc, uint8_t idx) {
  const uint64_t k = 0x2222222222222222ULL;
  LIFO_SWITCH
  return acc;
}

FOLLY_NOINLINE uint64_t lifoMixC(uint64_t acc, uint8_t idx) {
  const uint64_t k = 0x3333333333333333ULL;
  LIFO_SWITCH
  return acc;
}

#undef LIFO_SWITCH
#undef LIFO_OP16
#undef LIFO_OP

} // namespace

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

// Nearby variant of contendedUse(32_to_32): same contended post/wait
// throughput structure, but each completed wait drives a pointer-chase load
// whose payload selects one of three large switch bodies. This materially
// changes the timed hot path by adding frontend (icache/indirect-branch)
// pressure without altering the operation volume or serialization.
static void contendedUseChase(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  static constexpr uint32_t kRing = 4096;
  std::vector<uint32_t> ring(kRing);
  for (uint32_t i = 0; i < kRing; ++i) {
    ring[i] = (i * 2654435761u + 1) & (kRing - 1);
  }
  std::atomic<uint64_t> sink{0};

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &ring, &sink] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) & (kRing - 1);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = ring[p];
          uint8_t idx = static_cast<uint8_t>(acc ^ p);
          switch ((i + static_cast<uint32_t>(t)) % 3) {
            case 0:
              acc = lifoMixA(acc, idx);
              break;
            case 1:
              acc = lifoMixB(acc, idx);
              break;
            default:
              acc = lifoMixC(acc, idx);
              break;
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
BENCHMARK_NAMED_PARAM(contendedUseChase, 32_to_32, 32, 32)
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
