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
// This variant mirrors the contended LifoSem usage but each woken waiter runs
// through a large, statically-generated instruction working set in order to
// stress the L1 instruction cache (frontend-bound). The hot path performs a
// pointer-chase load, extracts a byte to index a 256-case switch, and rotates
// across three noinline switch functions so the executed code footprint is
// large enough that it cannot stay resident in L1i.
// ---------------------------------------------------------------------------
namespace {

struct ChaseNode {
  uint64_t payload;
  uint32_t next;
};

constexpr size_t kChaseSize = 8192; // power of two

static std::vector<ChaseNode>& chaseBuffer() {
  static std::vector<ChaseNode> buf = [] {
    std::vector<ChaseNode> v(kChaseSize);
    for (size_t i = 0; i < kChaseSize; ++i) {
      v[i].payload = i * 2654435761ull + (i << 13);
      v[i].next =
          (uint32_t)((i * 1103515245ull + 12345ull) & (kChaseSize - 1));
    }
    return v;
  }();
  return buf;
}

// 28 ALU ops per case; all constants fold uniquely per (case, salt) so the
// compiler cannot deduplicate either cases or the three functions.
#define CASE_OPS(n, S)                                  \
  case (n): {                                           \
    uint64_t k = (uint64_t)((unsigned)(n) + (unsigned)(S)); \
    acc += k * 2654435761ull + 0x9e3779b97f4a7c15ull;   \
    acc ^= (k << 3) ^ 0x0123456789abcdefull;            \
    acc *= (k | 0x3ull);                                \
    acc -= k * 0x100000001b3ull;                        \
    acc += (k >> 1) + 0xfeedface0ull;                   \
    acc ^= (k << 5) + 0xcafebabeull;                    \
    acc *= ((k & 0x3f) | 0x5ull);                       \
    acc += k * 0x9bull + 0x11ull;                       \
    acc -= (k << 2) ^ 0x22ull;                          \
    acc ^= k * 0x1full + 0x33ull;                       \
    acc += (k >> 2) + 0x44ull;                          \
    acc *= ((k & 0x1f) | 0x7ull);                       \
    acc -= k * 0x2dull + 0x55ull;                       \
    acc ^= (k << 7) + 0x66ull;                          \
    acc += k * 0x3bull + 0x77ull;                       \
    acc -= (k >> 3) ^ 0x88ull;                          \
    acc ^= k * 0x4dull + 0x99ull;                       \
    acc *= ((k & 0x0f) | 0x9ull);                       \
    acc += (k << 1) + 0xaaull;                          \
    acc -= k * 0x5full + 0xbbull;                        \
    acc ^= (k >> 4) + 0xccull;                          \
    acc += k * 0x6bull + 0xddull;                       \
    acc *= ((k & 0x07) | 0xbull);                       \
    acc -= (k << 4) ^ 0xeeull;                          \
    acc ^= k * 0x7dull + 0xffull;                       \
    acc += (k >> 5) + 0x1a2bull;                        \
    acc -= k * 0x8full + 0x3c4dull;                     \
    acc ^= (k << 6) + 0x5e6full;                        \
    break;                                              \
  }

#define REP4(M, n, S) M(n, S) M((n) + 1, S) M((n) + 2, S) M((n) + 3, S)
#define REP16(M, n, S)                                              \
  REP4(M, n, S) REP4(M, (n) + 4, S) REP4(M, (n) + 8, S)            \
      REP4(M, (n) + 12, S)
#define REP64(M, n, S)                                              \
  REP16(M, n, S) REP16(M, (n) + 16, S) REP16(M, (n) + 32, S)        \
      REP16(M, (n) + 48, S)
#define REP256(M, S)                                                \
  REP64(M, 0, S) REP64(M, 64, S) REP64(M, 128, S) REP64(M, 192, S)

__attribute__((noinline)) static uint64_t icacheMix0(uint64_t acc, uint8_t idx) {
  switch (idx) { REP256(CASE_OPS, 0x0000) }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheMix1(uint64_t acc, uint8_t idx) {
  switch (idx) { REP256(CASE_OPS, 0x1000) }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheMix2(uint64_t acc, uint8_t idx) {
  switch (idx) { REP256(CASE_OPS, 0x2000) }
  return acc;
}

#undef REP256
#undef REP64
#undef REP16
#undef REP4
#undef CASE_OPS

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  auto& chase = chaseBuffer();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink, &chase] {
        uint64_t acc = (uint64_t)t + 1;
        uint32_t p = (uint32_t)((t * 2654435761u) & (kChaseSize - 1));
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int j = 0; j < 24; ++j) {
            const ChaseNode& node = chase[p];
            uint8_t idx = (uint8_t)(node.payload & 0xFFu);
            switch (j % 3) {
              case 0:
                acc = icacheMix0(acc, idx);
                break;
              case 1:
                acc = icacheMix1(acc, idx);
                break;
              default:
                acc = icacheMix2(acc, idx);
                break;
            }
            p = node.next;
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
