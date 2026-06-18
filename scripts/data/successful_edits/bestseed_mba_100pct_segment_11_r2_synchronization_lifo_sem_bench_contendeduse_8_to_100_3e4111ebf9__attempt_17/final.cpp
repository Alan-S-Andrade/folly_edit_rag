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
// Derived benchmark: contendedUseIcache
//
// Same LifoSem contention structure as contendedUse(8_to_100), but each waiter
// wakeup also drives a large, branchy instruction working set to add frontend
// (L1 instruction-cache) pressure and reduce the otherwise-high IPC. The hot
// path pointer-chases a small buffer to obtain a switch index (payload & 0xFF),
// then rotates among three noinline 256-case switches via i%3. Each case body
// carries 14 ALU ops with case- and function-unique literal constants to defeat
// compiler deduplication and force a wide executed code footprint.
// ---------------------------------------------------------------------------
namespace {

#define ALU_OPS(N, S)                                  \
  acc += ((uint64_t)(N) * 7u + (S) + 1u);              \
  acc ^= ((uint64_t)(N) * 13u + (S) + 3u);             \
  acc *= (((uint64_t)(N) + (S)) | 1u);                 \
  acc += ((uint64_t)(N) * 5u + (S) + 7u);              \
  acc ^= ((uint64_t)(N) * 11u + (S) + 2u);             \
  acc *= (((uint64_t)(N) + (S) + 3u) | 1u);            \
  acc += ((uint64_t)(N) * 17u + (S) + 9u);             \
  acc ^= ((uint64_t)(N) * 19u + (S) + 4u);             \
  acc *= (((uint64_t)(N) + (S) + 5u) | 1u);            \
  acc += ((uint64_t)(N) * 23u + (S) + 6u);             \
  acc ^= ((uint64_t)(N) * 29u + (S) + 8u);             \
  acc *= (((uint64_t)(N) + (S) + 7u) | 1u);            \
  acc += ((uint64_t)(N) * 31u + (S) + 10u);            \
  acc ^= ((uint64_t)(N) * 37u + (S) + 11u);

#define CASE_A(N) \
  case (N): {     \
    ALU_OPS(N, 0x1111u) break; \
  }
#define CASE_B(N) \
  case (N): {     \
    ALU_OPS(N, 0x2222u) break; \
  }
#define CASE_C(N) \
  case (N): {     \
    ALU_OPS(N, 0x3333u) break; \
  }

#define R16(M, b)                                                     \
  M(b + 0) M(b + 1) M(b + 2) M(b + 3) M(b + 4) M(b + 5) M(b + 6)      \
      M(b + 7) M(b + 8) M(b + 9) M(b + 10) M(b + 11) M(b + 12)        \
          M(b + 13) M(b + 14) M(b + 15)

#define R256(M)                                                       \
  R16(M, 0) R16(M, 16) R16(M, 32) R16(M, 48) R16(M, 64) R16(M, 80)    \
      R16(M, 96) R16(M, 112) R16(M, 128) R16(M, 144) R16(M, 160)      \
          R16(M, 176) R16(M, 192) R16(M, 208) R16(M, 224) R16(M, 240)

__attribute__((noinline)) uint64_t icacheA(uint64_t acc, uint8_t idx) {
  switch (idx) { R256(CASE_A) }
  return acc;
}

__attribute__((noinline)) uint64_t icacheB(uint64_t acc, uint8_t idx) {
  switch (idx) { R256(CASE_B) }
  return acc;
}

__attribute__((noinline)) uint64_t icacheC(uint64_t acc, uint8_t idx) {
  switch (idx) { R256(CASE_C) }
  return acc;
}

#undef R256
#undef R16
#undef CASE_A
#undef CASE_B
#undef CASE_C
#undef ALU_OPS

constexpr size_t kChaseSize = 4096;

const std::vector<uint32_t>& chaseBuffer() {
  static const std::vector<uint32_t> chase = [] {
    std::vector<uint32_t> c(kChaseSize);
    for (size_t i = 0; i < kChaseSize; ++i) {
      c[i] = static_cast<uint32_t>((i * 2654435761u + 12345u) % kChaseSize);
    }
    return c;
  }();
  return chase;
}

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  const std::vector<uint32_t>& chase = chaseBuffer();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) % kChaseSize;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p % kChaseSize];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = icacheA(acc, idx);
              break;
            case 1:
              acc = icacheB(acc, idx);
              break;
            default:
              acc = icacheC(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100, 8, 100)
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
