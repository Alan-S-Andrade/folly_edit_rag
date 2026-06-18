/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may in a copy of the License at
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

#define CASES_8(v, c, base)                                                    \
  case base + 0:                                                               \
    v = (v * 3) + (uint64_t)c + base + 0;                                      \
    v = (v * 31) + (uint64_t)c + base + 0;                                     \
    v = (v ^ 101) + (uint64_t)c + base + 0;                                    \
    v = (v * 103) - (uint64_t)c + base + 0;                                    \
    v = (v ^ 107) + (uint64_t)c + base + 0;                                    \
    v = (v * 109) - (uint64_t)c + base + 0;                                    \
    v = (v ^ 113) + (uint64_t)c + base + 0;                                    \
    v = (v * 127) - (uint64_t)c + base + 0;                                    \
    break;                                                                     \
  case base + 1:                                                               \
    v = (v * 5) + (uint64_t)c + base + 1;                                      \
    v = (v * 41) + (uint64_t)c + base + 1;                                     \
    v = (v ^ 131) + (uint64_t)c + base + 1;                                    \
    v = (v * 137) - (uint64_t)c + base + 1;                                    \
    v = (v ^ 139) + (uint64_t)c + base + 1;                                    \
    v = (v * 149) - (uint64_t)c + base + 1;                                    \
    v = (v ^ 151) + (uint64_t)c + base + 1;                                    \
    v = (v * 157) - (uint64_t)c + base + 1;                                    \
    break;                                                                     \
  case base + 2:                                                               \
    v = (v * 7) + (uint64_t)c + base + 2;                                      \
    v = (v * 47) + (uint64_t)c + base + 2;                                     \
    v = (v ^ 163) + (uint64_t)c + base + 2;                                    \
    v = (v * 167) - (uint64_t)c + base + 2;                                    \
    v = (v ^ 173) + (uint64_t)c + base + 2;                                    \
    v = (v * 179) - (uint64_t)c + base + 2;                                    \
    v = (v ^ 181) + (uint64_t)c + base + 2;                                    \
    v = (v * 191) - (uint64_t)c + base + 2;                                    \
    break;                                                                     \
  case base + 3:                                                               \
    v = (v * 11) + (uint64_t)c + base + 3;                                     \
    v = (v * 59) + (uint64_t)c + base + 3;                                     \
    v = (v ^ 193) + (uint64_t)c + base + 3;                                    \
    v = (v * 197) - (uint64_t)c + base + 3;                                    \
    v = (v ^ 199) + (uint64_t)c + base + 3;                                    \
    v = (v * 211) - (uint64_t)c + base + 3;                                    \
    v = (v ^ 223) + (uint64_t)c + base + 3;                                    \
    v = (v * 227) - (uint64_t)c + base + 3;                                    \
    break;                                                                     \
  case base + 4:                                                               \
    v = (v * 13) + (uint64_t)c + base + 4;                                     \
    v = (v * 67) + (uint64_t)c + base + 4;                                     \
    v = (v ^ 229) + (uint64_t)c + base + 4;                                    \
    v = (v * 233) - (uint64_t)c + base + 4;                                    \
    v = (v ^ 239) + (uint64_t)c + base + 4;                                    \
    v = (v * 241) - (uint64_t)c + base + 4;                                    \
    v = (v ^ 251) + (uint64_t)c + base + 4;                                    \
    v = (v * 257) - (uint64_t)c + base + 4;                                    \
    break;                                                                     \
  case base + 5:                                                               \
    v = (v * 17) + (uint64_t)c + base + 5;                                     \
    v = (v * 73) + (uint64_t)c + base + 5;                                     \
    v = (v ^ 263) + (uint64_t)c + base + 5;                                    \
    v = (v * 269) - (uint64_t)c + base + 5;                                    \
    v = (v ^ 271) + (uint64_t)c + base + 5;                                    \
    v = (v * 277) - (uint64_t)c + base + 5;                                    \
    v = (v ^ 281) + (uint64_t)c + base + 5;                                    \
    v = (v * 283) - (uint64_t)c + base + 5;                                    \
    break;                                                                     \
  case base + 6:                                                               \
    v = (v * 19) + (uint64_t)c + base + 6;                                     \
    v = (v * 83) + (uint64_t)c + base + 6;                                     \
    v = (v ^ 293) + (uint64_t)c + base + 6;                                    \
    v = (v * 307) - (uint64_t)c + base + 6;                                    \
    v = (v ^ 311) + (uint64_t)c + base + 6;                                    \
    v = (v * 313) - (uint64_t)c + base + 6;                                    \
    v = (v ^ 317) + (uint64_t)c + base + 6;                                    \
    v = (v * 331) - (uint64_t)c + base + 6;                                    \
    break;                                                                     \
  case base + 7:                                                               \
    v = (v * 23) + (uint64_t)c + base + 7;                                     \
    v = (v * 97) + (uint64_t)c + base + 7;                                     \
    v = (v ^ 337) + (uint64_t)c + base + 7;                                    \
    v = (v * 347) - (uint64_t)c + base + 7;                                    \
    v = (v ^ 349) + (uint64_t)c + base + 7;                                    \
    v = (v * 353) - (uint64_t)c + base + 7;                                    \
    v = (v ^ 359) + (uint64_t)c + base + 7;                                    \
    v = (v * 367) - (uint64_t)c + base + 7;                                    \
    break;

#define CASES_256(v, c)                                                        \
  CASES_8(v, c, 0)                                                             \
  CASES_8(v, c, 8) CASES_8(v, c, 16) CASES_8(v, c, 24) CASES_8(v, c, 32)       \
      CASES_8(v, c, 40) CASES_8(v, c, 48) CASES_8(v, c, 56)                     \
          CASES_8(v, c, 64) CASES_8(v, c, 72) CASES_8(v, c, 80)                 \
              CASES_8(v, c, 88) CASES_8(v, c, 96) CASES_8(v, c, 104)            \
                  CASES_8(v, c, 112) CASES_8(v, c, 120) CASES_8(v, c, 128)      \
                      CASES_8(v, c, 136) CASES_8(v, c, 144)                     \
                          CASES_8(v, c, 152) CASES_8(v, c, 160)                 \
                              CASES_8(v, c, 168) CASES_8(v, c, 176)             \
                                  CASES_8(v, c, 184) CASES_8(v, c, 192)         \
                                      CASES_8(v, c, 200) CASES_8(v, c, 208)     \
                                          CASES_8(v, c, 216)                   \
                                              CASES_8(v, c, 224)               \
                                                  CASES_8(v, c, 232)           \
                                                      CASES_8(v, c, 240)       \
                                                          CASES_8(v, c, 248)

__attribute__((noinline)) uint64_t switchA(uint64_t v, int c) {
  switch (c) {
    CASES_256(v, c);
    default:;
  }
  return v;
}
__attribute__((noinline)) uint64_t switchB(uint64_t v, int c) {
  switch (c) {
    CASES_256(v, c * 3);
    default:;
  }
  return v;
}
__attribute__((noinline)) uint64_t switchC(uint64_t v, int c) {
  switch (c) {
    CASES_256(v, c * 5);
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

static void contendedUse_v3(uint32_t n, int posters, int waiters) {
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

          sem.post();
        }
        folly::doNotOptimizeAway(acc);
        folly::doNotOptimizeAway(hotPos);
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
BENCHMARK_NAMED_PARAM(contendedUse_v3, contendedUse_ipc_v1, 32, 1000)
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
