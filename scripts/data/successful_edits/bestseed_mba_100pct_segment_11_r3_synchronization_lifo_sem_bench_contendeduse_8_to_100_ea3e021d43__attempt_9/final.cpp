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
// Frontend-bound (I-cache footprint) variant of contendedUse.
//
// Each waiter, after acquiring the semaphore, performs a short bout of work
// that rotates among three very large noinline 256-case switch functions.
// The switch index is derived from a pointer-chase load so the executed
// instruction working set is large and poorly reused, which deliberately
// raises L1i pressure and lowers IPC toward the contended target profile.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kChaseSize = 4096; // power of two

struct ChaseBuf {
  std::vector<uint32_t> next;
  ChaseBuf() : next(kChaseSize) {
    for (size_t i = 0; i < kChaseSize; ++i) {
      next[i] =
          static_cast<uint32_t>((i * 2654435761u + 1013904223u) % kChaseSize);
    }
  }
};

// 14 ALU ops per case, all using literals derived from the case index and a
// per-function salt so the compiler cannot deduplicate cases or functions.
#define ALU_CASE_S(i, S)                                          \
  case (i): {                                                     \
    acc += 0x9E3779B9u ^ (static_cast<uint32_t>(i) * 2654435761u) ^ (S); \
    acc ^= (static_cast<uint32_t>(i) + 0x1234u) * 0x85EBCA6Bu + (S);     \
    acc *= ((static_cast<uint32_t>(i)) | 1u) + 0xC2B2AE35u;              \
    acc += (acc >> 13) ^ (static_cast<uint32_t>(i) * 0x27D4EB2Fu);       \
    acc ^= static_cast<uint32_t>(i) * 0x165667B1u + 0xDEADBEEFu + (S);   \
    acc *= 0x9E3779B1u + static_cast<uint32_t>(i);                       \
    acc += ((static_cast<uint32_t>(i)) << 3) ^ 0xCAFEBABEu;             \
    acc ^= (acc << 7) + static_cast<uint32_t>(i) * 0x7FEB352Du;          \
    acc *= ((static_cast<uint32_t>(i)) ^ 0xABCDu) | 1u;                  \
    acc += 0x632BE5ABu - static_cast<uint32_t>(i) + (S);                 \
    acc ^= static_cast<uint32_t>(i) * 0x9E3779B9u + 0xFEEDu;             \
    acc *= ((static_cast<uint32_t>(i)) + 3u) | 1u;                       \
    acc += (acc >> 11) ^ static_cast<uint32_t>(i);                       \
    acc ^= 0x5BD1E995u + static_cast<uint32_t>(i) * 7u + (S);            \
    break;                                                              \
  }

#define REP16(M, b)                                                       \
  M(b + 0) M(b + 1) M(b + 2) M(b + 3) M(b + 4) M(b + 5) M(b + 6) M(b + 7) \
      M(b + 8) M(b + 9) M(b + 10) M(b + 11) M(b + 12) M(b + 13) M(b + 14) \
          M(b + 15)
#define REP256(M)                                                      \
  REP16(M, 0) REP16(M, 16) REP16(M, 32) REP16(M, 48) REP16(M, 64)       \
      REP16(M, 80) REP16(M, 96) REP16(M, 112) REP16(M, 128)            \
          REP16(M, 144) REP16(M, 160) REP16(M, 176) REP16(M, 192)      \
              REP16(M, 208) REP16(M, 224) REP16(M, 240)

#define ALU_CASE_A(i) ALU_CASE_S(i, 0x11111111u)
#define ALU_CASE_B(i) ALU_CASE_S(i, 0x22222222u)
#define ALU_CASE_C(i) ALU_CASE_S(i, 0x33333333u)

__attribute__((noinline)) uint32_t footprintFn0(uint32_t acc, uint8_t idx) {
  switch (idx) { REP256(ALU_CASE_A) }
  return acc;
}
__attribute__((noinline)) uint32_t footprintFn1(uint32_t acc, uint8_t idx) {
  switch (idx) { REP256(ALU_CASE_B) }
  return acc;
}
__attribute__((noinline)) uint32_t footprintFn2(uint32_t acc, uint8_t idx) {
  switch (idx) { REP256(ALU_CASE_C) }
  return acc;
}

#undef ALU_CASE_A
#undef ALU_CASE_B
#undef ALU_CASE_C
#undef REP256
#undef REP16
#undef ALU_CASE_S

} // namespace

static void contendedUseFootprint(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint32_t> sink(0);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        ChaseBuf chase;
        uint32_t pos = static_cast<uint32_t>(t);
        uint32_t acc = 0x12345678u + static_cast<uint32_t>(t);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int j = 0; j < 24; ++j) {
            pos = chase.next[pos & (kChaseSize - 1)];
            uint8_t idx = static_cast<uint8_t>(pos & 0xFFu);
            switch (j % 3) {
              case 0:
                acc = footprintFn0(acc, idx);
                break;
              case 1:
                acc = footprintFn1(acc, idx);
                break;
              default:
                acc = footprintFn2(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUseFootprint, 8_to_200, 8, 200)

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
