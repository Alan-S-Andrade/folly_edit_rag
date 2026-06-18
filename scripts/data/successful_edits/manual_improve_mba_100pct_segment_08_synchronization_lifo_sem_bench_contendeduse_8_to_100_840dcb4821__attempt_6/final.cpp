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

#include <cstdint>
#include <vector>

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
// Derived frontend-bound benchmark for contendedUse(8_to_100).
//
// Goal: drive the executed instruction working set wide enough to raise
// L1-icache-load-misses_MPKI (and thereby lower the overly-high IPC) without
// touching the reference benchmark family above. The hot loop pointer-chases a
// buffer, derives a switch index from the loaded payload, and rotates among
// three noinline 256-case switches whose cases each perform a chain of ALU ops
// with unique literal constants (defeating compiler deduplication).
// ---------------------------------------------------------------------------
namespace {

struct ICacheNode {
  uint32_t next;
  uint32_t payload;
};

// 14 ALU ops per case; constants depend on both case index (i) and a
// per-function seed (S) so no two cases collapse to identical code.
#define MIX_BODY(i, S)                                  \
  acc += (uint32_t)(i) + (uint32_t)(S);                 \
  acc ^= (acc << 7) ^ ((uint32_t)(i) * 2654435761u + (uint32_t)(S)); \
  acc *= (3u + ((uint32_t)(i) << 1));                   \
  acc += (uint32_t)(i) ^ ((uint32_t)(S) * 40503u);      \
  acc -= (acc >> 5) + (uint32_t)(i);                    \
  acc ^= (uint32_t)(i) * 0x9E3779B9u;                   \
  acc += (acc << 11) ^ (uint32_t)(S);                   \
  acc *= (1u | (uint32_t)(i));                          \
  acc ^= acc >> 9;                                      \
  acc += (uint32_t)(i) * 5u + (uint32_t)(S);            \
  acc -= (uint32_t)(i) ^ 0x5BD1E995u;                   \
  acc ^= (acc << 3) + (uint32_t)(i);                    \
  acc += (uint32_t)(i) * 17u ^ (uint32_t)(S);           \
  acc ^= (acc >> 6) + (uint32_t)(S);

#define CASE_OPS0(i) \
  case (i): {        \
    MIX_BODY(i, 0x11u) break; \
  }
#define CASE_OPS1(i) \
  case (i): {        \
    MIX_BODY(i, 0x22u) break; \
  }
#define CASE_OPS2(i) \
  case (i): {        \
    MIX_BODY(i, 0x33u) break; \
  }

#define REP16(M, b)                                                          \
  M(b + 0) M(b + 1) M(b + 2) M(b + 3) M(b + 4) M(b + 5) M(b + 6) M(b + 7)    \
      M(b + 8) M(b + 9) M(b + 10) M(b + 11) M(b + 12) M(b + 13) M(b + 14)    \
          M(b + 15)
#define REP256(M)                                                            \
  REP16(M, 0) REP16(M, 16) REP16(M, 32) REP16(M, 48) REP16(M, 64)            \
      REP16(M, 80) REP16(M, 96) REP16(M, 112) REP16(M, 128) REP16(M, 144)    \
          REP16(M, 160) REP16(M, 176) REP16(M, 192) REP16(M, 208)            \
              REP16(M, 224) REP16(M, 240)

__attribute__((noinline)) uint32_t icache_mix0(uint32_t acc, uint32_t idx) {
  switch (idx & 0xFFu) {
    REP256(CASE_OPS0)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) uint32_t icache_mix1(uint32_t acc, uint32_t idx) {
  switch (idx & 0xFFu) {
    REP256(CASE_OPS1)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) uint32_t icache_mix2(uint32_t acc, uint32_t idx) {
  switch (idx & 0xFFu) {
    REP256(CASE_OPS2)
    default:
      break;
  }
  return acc;
}

#undef REP256
#undef REP16
#undef CASE_OPS0
#undef CASE_OPS1
#undef CASE_OPS2
#undef MIX_BODY

} // namespace

BENCHMARK(contendedUse_8_to_100_icache, iters) {
  constexpr size_t kBuf = 4096;
  static const std::vector<ICacheNode> buf = [] {
    std::vector<ICacheNode> v(kBuf);
    for (size_t i = 0; i < kBuf; ++i) {
      v[i].next = static_cast<uint32_t>((i * 2654435761u + 1u) % kBuf);
      v[i].payload = static_cast<uint32_t>(i * 40503u + 12345u);
    }
    return v;
  }();

  uint32_t acc = 0x12345678u;
  uint32_t idx = 0;
  for (size_t j = 0; j < iters; ++j) {
    const ICacheNode& node = buf[idx];
    const uint32_t payload = node.payload;
    switch (j % 3) {
      case 0:
        acc = icache_mix0(acc, payload & 0xFFu);
        break;
      case 1:
        acc = icache_mix1(acc, payload & 0xFFu);
        break;
      default:
        acc = icache_mix2(acc, payload & 0xFFu);
        break;
    }
    idx = node.next;
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
