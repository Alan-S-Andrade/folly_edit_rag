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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>

namespace {

struct ChainNode {
  ChainNode* next;
  char pad[56];
};

constexpr size_t kHotLen = 16384; // 1MB
static ChainNode hotNodes[kHotLen];

constexpr size_t kColdLen = 2 * 1024 * 1024; // 128MB
static ChainNode* coldNodes = nullptr;

static void initChain(ChainNode* nodes, size_t len) {
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), 0u);
  std::mt19937 rng(42);
  std::shuffle(perm.begin(), perm.end(), rng);
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
}

static std::once_flag chains_initialized;

static void initialize_chains() {
  std::call_once(chains_initialized, [] {
    if (!coldNodes) {
      coldNodes = new ChainNode[kColdLen];
    }
    initChain(hotNodes, kHotLen);
    initChain(coldNodes, kColdLen);
  });
}

#define R16(X, OP) \
  OP(X + 0) OP(X + 1) OP(X + 2) OP(X + 3) OP(X + 4) OP(X + 5) OP(X + 6) \
      OP(X + 7) OP(X + 8) OP(X + 9) OP(X + 10) OP(X + 11) OP(X + 12) \
      OP(X + 13) OP(X + 14) OP(X + 15)
#define R256(OP) \
  R16(0, OP) R16(16, OP) R16(32, OP) R16(48, OP) R16(64, OP) R16(80, OP) \
      R16(96, OP) R16(112, OP) R16(128, OP) R16(144, OP) R16(160, OP) \
      R16(176, OP) R16(192, OP) R16(208, OP) R16(224, OP) R16(240, OP)

__attribute__((noinline)) uint64_t switchA(uint64_t v, int c) {
#define A_OP(X) \
  case X: \
    v = (v * 3) + X; \
    break;
  switch (c & 255) {
    R256(A_OP)
  }
#undef A_OP
  return v;
}

__attribute__((noinline)) uint64_t switchB(uint64_t v, int c) {
#define B_OP(X) \
  case X: \
    v = (v ^ 0xdeadbeef) + X; \
    break;
  switch (c & 255) {
    R256(B_OP)
  }
#undef B_OP
  return v;
}

__attribute__((noinline)) uint64_t switchC(uint64_t v, int c) {
#define C_OP(X) \
  case X: \
    v = v - (v >> 3) * X; \
    break;
  switch (c & 255) {
    R256(C_OP)
  }
#undef C_OP
  return v;
}
#undef R16
#undef R256

} // namespace

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

static void contendedUseWithWork_v1(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    initialize_chains();
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        ChainNode* hotPos = &hotNodes[(t * 13) % kHotLen];
        ChainNode* coldPos = &coldNodes[(t * 29) % kColdLen];
        uint64_t acc = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          constexpr size_t WORK_PER_WAIT = 4;
          for (size_t k = 0; k < WORK_PER_WAIT; ++k) {
            size_t j = (i / waiters) * WORK_PER_WAIT + k;
            hotPos = hotPos->next;
            uint64_t payload = (uint64_t)(uintptr_t)hotPos;
            switch (j % 3) {
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

            auto coldMask = -uint64_t((j % 8) == 0);
            coldPos = (ChainNode*)((uintptr_t)coldPos->next & coldMask |
                                   (uintptr_t)coldPos & ~coldMask);
          }
        }
        folly::doNotOptimizeAway(acc);
        folly::doNotOptimizeAway(hotPos);
        folly::doNotOptimizeAway(coldPos);
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
BENCHMARK_NAMED_PARAM(
    contendedUseWithWork_v1, contendedUseWithWork_v1_32_to_32, 32, 32)
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
