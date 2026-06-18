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
// Derived benchmark: contendedUseIcache(8_to_100_icache)
//
// Same contended LifoSem post/wait skeleton as contendedUse(8_to_100), but each
// waiter, after being woken, walks a pointer-chase array and dispatches the
// loaded payload byte through one of three large 256-case switch functions,
// rotating among them with j%3. The three functions form a deliberately large
// executed-instruction working set to raise the L1 i-cache footprint (and thus
// lower the over-high IPC of the plain contended workload).
// ---------------------------------------------------------------------------

namespace {

struct ChaseNode {
  uint32_t next;
  uint8_t payload;
};

std::vector<ChaseNode>& chaseArray() {
  static std::vector<ChaseNode> v = [] {
    constexpr size_t kN = 4096;
    std::vector<ChaseNode> nodes(kN);
    for (size_t i = 0; i < kN; ++i) {
      nodes[i].next = static_cast<uint32_t>((i * 2654435761u + 12345u) % kN);
      nodes[i].payload = static_cast<uint8_t>(i * 31u + 7u);
    }
    return nodes;
  }();
  return v;
}

// 14 ALU ops per case, each using the case index and a per-function salt so the
// constants are unique and the compiler cannot deduplicate cases or functions.
#define CASE_OPS(c)                                                        \
  case (c): {                                                              \
    acc += 0x9E3779B97F4A7C15ull + (uint64_t)(c) + FN_SALT;               \
    acc ^= (acc << 13) ^ ((uint64_t)(c) * 2654435761ull + FN_SALT);       \
    acc *= 0x100000001B3ull + (uint64_t)(c);                              \
    acc += (uint64_t)(c) * 0x51ull + 7ull + FN_SALT;                      \
    acc ^= (acc >> 7) + (uint64_t)(c);                                    \
    acc *= 0xFF51AFD7ED558CCDull ^ ((uint64_t)(c) + FN_SALT);             \
    acc += ((uint64_t)(c) ^ 0xABCDull) + FN_SALT;                         \
    acc ^= (acc << 11) + (uint64_t)(c);                                   \
    acc *= 0xC4CEB9FE1A85EC53ull + (uint64_t)(c);                         \
    acc += (uint64_t)(c) * 13ull + 101ull;                               \
    acc ^= (acc >> 5) ^ ((uint64_t)(c) + FN_SALT);                        \
    acc *= 0x2545F4914F6CDD1Dull + (uint64_t)(c);                         \
    acc += ((uint64_t)(c) ^ 0x1234ull);                                   \
    acc ^= (acc << 3) + (uint64_t)(c) + FN_SALT;                          \
    break;                                                                 \
  }

#define C16(b)                                                             \
  CASE_OPS((b) + 0) CASE_OPS((b) + 1) CASE_OPS((b) + 2) CASE_OPS((b) + 3)  \
  CASE_OPS((b) + 4) CASE_OPS((b) + 5) CASE_OPS((b) + 6) CASE_OPS((b) + 7)  \
  CASE_OPS((b) + 8) CASE_OPS((b) + 9) CASE_OPS((b) + 10)                   \
  CASE_OPS((b) + 11) CASE_OPS((b) + 12) CASE_OPS((b) + 13)                 \
  CASE_OPS((b) + 14) CASE_OPS((b) + 15)

#define SWITCH256                                                          \
  switch (idx) {                                                           \
    C16(0) C16(16) C16(32) C16(48) C16(64) C16(80) C16(96) C16(112)        \
    C16(128) C16(144) C16(160) C16(176) C16(192) C16(208) C16(224)         \
    C16(240) default: break;                                               \
  }

#define FN_SALT 0x11ull
__attribute__((noinline)) uint64_t icache_switch_a(uint8_t idx, uint64_t acc) {
  SWITCH256
  return acc;
}
#undef FN_SALT

#define FN_SALT 0x9173ull
__attribute__((noinline)) uint64_t icache_switch_b(uint8_t idx, uint64_t acc) {
  SWITCH256
  return acc;
}
#undef FN_SALT

#define FN_SALT 0xC0FFEEull
__attribute__((noinline)) uint64_t icache_switch_c(uint8_t idx, uint64_t acc) {
  SWITCH256
  return acc;
}
#undef FN_SALT

#undef SWITCH256
#undef C16
#undef CASE_OPS

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::vector<ChaseNode>& chase = chaseArray();
  const uint32_t mask = static_cast<uint32_t>(chase.size() - 1);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t cur = static_cast<uint32_t>(t) & mask;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int j = 0; j < 64; ++j) {
            const ChaseNode& node = chase[cur & mask];
            uint8_t p = node.payload;
            cur = node.next;
            switch (j % 3) {
              case 0:
                acc = icache_switch_a(p, acc);
                break;
              case 1:
                acc = icache_switch_b(p, acc);
                break;
              default:
                acc = icache_switch_c(p, acc);
                break;
            }
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
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100_icache, 8, 100)
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
