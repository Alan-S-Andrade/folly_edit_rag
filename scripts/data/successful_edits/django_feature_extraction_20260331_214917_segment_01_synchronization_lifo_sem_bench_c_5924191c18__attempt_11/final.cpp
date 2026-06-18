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

#include <numeric>
#include <random>
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

namespace {
struct ChainNode {
  ChainNode* next;
  char pad[56];
};

constexpr size_t kHotLen = 16384;
constexpr size_t kColdLen = 2u << 20;

alignas(64) static ChainNode hotNodes[kHotLen];
static ChainNode* coldNodes;

static void initChain(ChainNode* nodes, size_t len) {
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), 0u);
  std::mt19937_64 rng(42);
  for (size_t i = len - 1; i > 0; --i) {
    std::swap(perm[i], perm[rng() % (i + 1)]);
  }
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
}

constexpr uint32_t kNumAluOpsPerCase = 38;
// A list of 3 * kNumAluOpsPerCase primes to ensure unique constants
// across the three switch functions, and unique constants within each
// case for kNumAluOpsPerCase operations.
static constexpr uint32_t g_primes[] = { // Fix: Changed uint33_t to uint32_t
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, // Group 0 for switchA
    173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, // Group 1 for switchB
    397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 641  // Group 2 for switchC
};

// Macro to generate kNumAluOpsPerCase ALU operations within a single switch case.
// Each operation uses a unique multiplier from g_primes and a unique additive constant.
#define ALU_OPS_MANY_TIMES(v, c_val, case_label_val, multiplier_idx_offset, add_idx_offset) \
  v = (v * g_primes[multiplier_idx_offset + 0]) + (uint64_t)c_val + case_label_val + add_idx_offset + 0; \
  v = (v * g_primes[multiplier_idx_offset + 1]) + (uint64_t)c_val + case_label_val + add_idx_offset + 1; \
  v = (v * g_primes[multiplier_idx_offset + 2]) + (uint64_t)c_val + case_label_val + add_idx_offset + 2; \
  v = (v * g_primes[multiplier_idx_offset + 3]) + (uint64_t)c_val + case_label_val + add_idx_offset + 3; \
  v = (v * g_primes[multiplier_idx_offset + 4]) + (uint64_t)c_val + case_label_val + add_idx_offset + 4; \
  v = (v * g_primes[multiplier_idx_offset + 5]) + (uint64_t)c_val + case_label_val + add_idx_offset + 5; \
  v = (v * g_primes[multiplier_idx_offset + 6]) + (uint64_t)c_val + case_label_val + add_idx_offset + 6; \
  v = (v * g_primes[multiplier_idx_offset + 7]) + (uint64_t)c_val + case_label_val + add_idx_offset + 7; \
  v = (v * g_primes[multiplier_idx_offset + 8]) + (uint64_t)c_val + case_label_val + add_idx_offset + 8; \
  v = (v * g_primes[multiplier_idx_offset + 9]) + (uint64_t)c_val + case_label_val + add_idx_offset + 9; \
  v = (v * g_primes[multiplier_idx_offset + 10]) + (uint64_t)c_val + case_label_val + add_idx_offset + 10; \
  v = (v * g_primes[multiplier_idx_offset + 11]) + (uint64_t)c_val + case_label_val + add_idx_offset + 11; \
  v = (v * g_primes[multiplier_idx_offset + 12]) + (uint64_t)c_val + case_label_val + add_idx_offset + 12; \
  v = (v * g_primes[multiplier_idx_offset + 13]) + (uint64_t)c_val + case_label_val + add_idx_offset + 13; \
  v = (v * g_primes[multiplier_idx_offset + 14]) + (uint64_t)c_val + case_label_val + add_idx_offset + 14; \
  v = (v * g_primes[multiplier_idx_offset + 15]) + (uint64_t)c_val + case_label_val + add_idx_offset + 15; \
  v = (v * g_primes[multiplier_idx_offset + 16]) + (uint64_t)c_val + case_label_val + add_idx_offset + 16; \
  v = (v * g_primes[multiplier_idx_offset + 17]) + (uint64_t)c_val + case_label_val + add_idx_offset + 17; \
  v = (v * g_primes[multiplier_idx_offset + 18]) + (uint64_t)c_val + case_label_val + add_idx_offset + 18; \
  v = (v * g_primes[multiplier_idx_offset + 19]) + (uint64_t)c_val + case_label_val + add_idx_offset + 19; \
  v = (v * g_primes[multiplier_idx_offset + 20]) + (uint64_t)c_val + case_label_val + add_idx_offset + 20; \
  v = (v * g_primes[multiplier_idx_offset + 21]) + (uint64_t)c_val + case_label_val + add_idx_offset + 21; \
  v = (v * g_primes[multiplier_idx_offset + 22]) + (uint64_t)c_val + case_label_val + add_idx_offset + 22; \
  v = (v * g_primes[multiplier_idx_offset + 23]) + (uint64_t)c_val + case_label_val + add_idx_offset + 23; \
  v = (v * g_primes[multiplier_idx_offset + 24]) + (uint64_t)c_val + case_label_val + add_idx_offset + 24; \
  v = (v * g_primes[multiplier_idx_offset + 25]) + (uint64_t)c_val + case_label_val + add_idx_offset + 25; \
  v = (v * g_primes[multiplier_idx_offset + 26]) + (uint64_t)c_val + case_label_val + add_idx_offset + 26; \
  v = (v * g_primes[multiplier_idx_offset + 27]) + (uint64_t)c_val + case_label_val + add_idx_offset + 27; \
  v = (v * g_primes[multiplier_idx_offset + 28]) + (uint64_t)c_val + case_label_val + add_idx_offset + 28; \
  v = (v * g_primes[multiplier_idx_offset + 29]) + (uint64_t)c_val + case_label_val + add_idx_offset + 29; \
  v = (v * g_primes[multiplier_idx_offset + 30]) + (uint64_t)c_val + case_label_val + add_idx_offset + 30; \
  v = (v * g_primes[multiplier_idx_offset + 31]) + (uint64_t)c_val + case_label_val + add_idx_offset + 31; \
  v = (v * g_primes[multiplier_idx_offset + 32]) + (uint64_t)c_val + case_label_val + add_idx_offset + 32; \
  v = (v * g_primes[multiplier_idx_offset + 33]) + (uint64_t)c_val + case_label_val + add_idx_offset + 33; \
  v = (v * g_primes[multiplier_idx_offset + 34]) + (uint64_t)c_val + case_label_val + add_idx_offset + 34; \
  v = (v * g_primes[multiplier_idx_offset + 35]) + (uint64_t)c_val + case_label_val + add_idx_offset + 35; \
  v = (v * g_primes[multiplier_idx_offset + 36]) + (uint64_t)c_val + case_label_val + add_idx_offset + 36; \
  v = (v * g_primes[multiplier_idx_offset + 37]) + (uint64_t)c_val + case_label_val + add_idx_offset + 37;

// Redefine CASES_8 to embed ALU_OPS_MANY_TIMES for kNumAluOpsPerCase operations per real case.
#undef CASES_8
#define CASES_8(v, c, base, multiplier_offset, global_add_offset)              \
  case base + 0: ALU_OPS_MANY_TIMES(v, c, base + 0, multiplier_offset, global_add_offset + (0 * kNumAluOpsPerCase)); break; \
  case base + 1: ALU_OPS_MANY_TIMES(v, c, base + 1, multiplier_offset, global_add_offset + (1 * kNumAluOpsPerCase)); break; \
  case base + 2: ALU_OPS_MANY_TIMES(v, c, base + 2, multiplier_offset, global_add_offset + (2 * kNumAluOpsPerCase)); break; \
  case base + 3: ALU_OPS_MANY_TIMES(v, c, base + 3, multiplier_offset, global_add_offset + (3 * kNumAluOpsPerCase)); break; \
  case base + 4: ALU_OPS_MANY_TIMES(v, c, base + 4, multiplier_offset, global_add_offset + (4 * kNumAluOpsPerCase)); break; \
  case base + 5: ALU_OPS_MANY_TIMES(v, c, base + 5, multiplier_offset, global_add_offset + (5 * kNumAluOpsPerCase)); break; \
  case base + 6: ALU_OPS_MANY_TIMES(v, c, base + 6, multiplier_offset, global_add_offset + (6 * kNumAluOpsPerCase)); break; \
  case base + 7: ALU_OPS_MANY_TIMES(v, c, base + 7, multiplier_offset, global_add_offset + (7 * kNumAluOpsPerCase)); break;

// Redefine CASES_256 to pass multiplier and addition offsets.
#undef CASES_256
#define CASES_256(v, c, multiplier_offset, global_add_offset_base)             \
  CASES_8(v, c, 0, multiplier_offset, global_add_offset_base + (0 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 8, multiplier_offset, global_add_offset_base + (1 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 16, multiplier_offset, global_add_offset_base + (2 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 24, multiplier_offset, global_add_offset_base + (3 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 32, multiplier_offset, global_add_offset_base + (4 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 40, multiplier_offset, global_add_offset_base + (5 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 48, multiplier_offset, global_add_offset_base + (6 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 56, multiplier_offset, global_add_offset_base + (7 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 64, multiplier_offset, global_add_offset_base + (8 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 72, multiplier_offset, global_add_offset_base + (9 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 80, multiplier_offset, global_add_offset_base + (10 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 88, multiplier_offset, global_add_offset_base + (11 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 96, multiplier_offset, global_add_offset_base + (12 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 104, multiplier_offset, global_add_offset_base + (13 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 112, multiplier_offset, global_add_offset_base + (14 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 120, multiplier_offset, global_add_offset_base + (15 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 128, multiplier_offset, global_add_offset_base + (16 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 136, multiplier_offset, global_add_offset_base + (17 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 144, multiplier_offset, global_add_offset_base + (18 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 152, multiplier_offset, global_add_offset_base + (19 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 160, multiplier_offset, global_add_offset_base + (20 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 168, multiplier_offset, global_add_offset_base + (21 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 176, multiplier_offset, global_add_offset_base + (22 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 184, multiplier_offset, global_add_offset_base + (23 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 192, multiplier_offset, global_add_offset_base + (24 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 200, multiplier_offset, global_add_offset_base + (25 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 208, multiplier_offset, global_add_offset_base + (26 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 216, multiplier_offset, global_add_offset_base + (27 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 224, multiplier_offset, global_add_offset_base + (28 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 232, multiplier_offset, global_add_offset_base + (29 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 240, multiplier_offset, global_add_offset_base + (30 * 8 * kNumAluOpsPerCase)) \
  CASES_8(v, c, 248, multiplier_offset, global_add_offset_base + (31 * 8 * kNumAluOpsPerCase))

#undef switchA
__attribute__((noinline)) uint64_t switchA(uint64_t v, int c) {
  switch (c) {
    CASES_256(v, c, 0, 0); // multiplier_offset: 0, global_add_offset_base: 0
    default:;
  }
  return v;
}
#undef switchB
__attribute__((noinline)) uint64_t switchB(uint64_t v, int c) {
  switch (c) {
    // multiplier_offset: start of second prime group, global_add_offset_base: after all 256*kNumAluOpsPerCase from switchA
    CASES_256(v, c * 3, kNumAluOpsPerCase, 256 * kNumAluOpsPerCase);
    default:;
  }
  return v;
}
#undef switchC
__attribute__((noinline)) uint64_t switchC(uint64_t v, int c) {
  switch (c) {
    // multiplier_offset: start of third prime group, global_add_offset_base: after all 256*kNumAluOpsPerCase*2 from switchA/B
    CASES_256(v, c * 5, kNumAluOpsPerCase * 2, 256 * kNumAluOpsPerCase * 2);
    default:;
  }
  return v;
}

struct DepWorkInitializer {
  DepWorkInitializer() {
    coldNodes = new ChainNode[kColdLen];
    initChain(hotNodes, kHotLen);
    initChain(coldNodes, kColdLen);
  }
  ~DepWorkInitializer() { delete[] coldNodes; }
};
static DepWorkInitializer initializer;

static void contendedUse_v2(uint32_t n, int posters, int waiters) {
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
        ChainNode* hotPos = &hotNodes[t % kHotLen];
        ChainNode* coldPos = &coldNodes[t % kColdLen];
        uint64_t acc = t;

        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          hotPos = hotPos->next;
          uint64_t payload = (uint64_t)(uintptr_t)hotPos;
          switch (i % 3) {
            case 0:
              acc = switchA(acc, payload & 0xFF);
              break;
            case 1:
              acc = switchB(acc, payload & 0xFF);
              break;
            case 2:
              acc = switchC(acc, payload & 0xFF);
              break;
          }

          auto coldMask = -uint64_t(i % 8 == 0);
          coldPos = (ChainNode*)(((uintptr_t)coldPos->next & coldMask) |
                                 ((uintptr_t)coldPos & ~coldMask));
          sem.post();
        }
        folly::doNotOptimizeAway(acc);
        folly::doNotOptimizeAway(hotPos);
        folly::doNotOptimizeAway(coldPos);
      });
    }
  }

  go.store(true);
  for (auto& thr : threads) {
    thr.join();
  }
}
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

BENCHMARK_DRAW_LINE();
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_1, 1, 1)
BENCHMARK_NAMED_PARAM(contendedUse_v2, 1_to_1_branchy, 1, 1)
BENCHMARK_NAMED_PARAM(contendedUse_v2, 32_to_1000_branchy, 32, 1000) // NEW
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_4, 1, 4)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_32, 1, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_1, 4, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_24, 4, 24)
BENCHMARK_NAMED_PARAM(contendedUse, 8_to_100, 8, 100)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1, 31, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 16_to_16, 16, 16)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_32, 32, 32)
BENCHMARK_NAMED_PARAM(contendedUse_v2, 32_to_32_v2, 32, 32)
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
