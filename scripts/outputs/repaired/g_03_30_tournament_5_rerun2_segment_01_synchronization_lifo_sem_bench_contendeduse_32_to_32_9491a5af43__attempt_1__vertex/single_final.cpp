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
#include <folly/lang/Align.h>

#include <atomic>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
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
  char pad[64 - sizeof(ChainNode*)];
};
static_assert(sizeof(ChainNode) == 64, "ChainNode should be 64 bytes");

// Hot chain: 1 MB (16384 nodes * 64 B)
static constexpr size_t kHotLen = 16384;
static cacheline_aligned<ChainNode> hotNodes[kHotLen];

// Cold chain: 128 MB (2M nodes * 64 B)
static constexpr size_t kColdLen = 2 * 1024 * 1024;
static ChainNode* coldNodes;

#define GEN_CASE(i, salt) \
  case i:                 \
    v ^= (v >> 13);       \
    v += (i * salt);      \
    break;

#define GEN_SWITCH_CHUNK(start, salt) \
  GEN_CASE(start + 0, salt) GEN_CASE(start + 1, salt) GEN_CASE(start + 2, salt) GEN_CASE(start + 3, salt) \
  GEN_CASE(start + 4, salt) GEN_CASE(start + 5, salt) GEN_CASE(start + 6, salt) GEN_CASE(start + 7, salt)

#define GEN_SWITCH_BODY(prefix, salt) \
  GEN_SWITCH_CHUNK(prefix + 0, salt) GEN_SWITCH_CHUNK(prefix + 8, salt) \
  GEN_SWITCH_CHUNK(prefix + 16, salt) GEN_SWITCH_CHUNK(prefix + 24, salt) \
  GEN_SWITCH_CHUNK(prefix + 32, salt) GEN_SWITCH_CHUNK(prefix + 40, salt) \
  GEN_SWITCH_CHUNK(prefix + 48, salt) GEN_SWITCH_CHUNK(prefix + 56, salt)

__attribute__((noinline)) uint64_t switchA(uint64_t v, int c) {
  switch (c) {
    GEN_SWITCH_BODY(0, 0x123456789abcdef1ULL)
    GEN_SWITCH_BODY(64, 0x123456789abcdef1ULL)
    GEN_SWITCH_BODY(128, 0x123456789abcdef1ULL)
    GEN_SWITCH_BODY(192, 0x123456789abcdef1ULL)
  }
  return v;
}

__attribute__((noinline)) uint64_t switchB(uint64_t v, int c) {
  switch (c) {
    GEN_SWITCH_BODY(0, 0x9abcdef123456781ULL)
    GEN_SWITCH_BODY(64, 0x9abcdef123456781ULL)
    GEN_SWITCH_BODY(128, 0x9abcdef123456781ULL)
    GEN_SWITCH_BODY(192, 0x9abcdef123456781ULL)
  }
  return v;
}

__attribute__((noinline)) uint64_t switchC(uint64_t v, int c) {
  switch (c) {
    GEN_SWITCH_BODY(0, 0xfedcba9876543211ULL)
    GEN_SWITCH_BODY(64, 0xfedcba9876543211ULL)
    GEN_SWITCH_BODY(128, 0xfedcba9876543211ULL)
    GEN_SWITCH_BODY(192, 0xfedcba9876543211ULL)
  }
  return v;
}

#undef GEN_CASE
#undef GEN_SWITCH_CHUNK
#undef GEN_SWITCH_BODY

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

static std::once_flag initFlag;
static void initialize() {
  std::call_once(initFlag, [] {
    initChain(reinterpret_cast<ChainNode*>(hotNodes), kHotLen);
    coldNodes = new ChainNode[kColdLen];
    initChain(coldNodes, kColdLen);
  });
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

static void contendedUse_v1(uint32_t n, int posters, int waiters) {
  initialize();
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  constexpr size_t kWorkIters = 512;

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

        ChainNode* hotPos =
            reinterpret_cast<ChainNode*>(&hotNodes[(t * 37) % kHotLen]);
        ChainNode* coldPos = &coldNodes[(t * 37) % kColdLen];
        uint64_t acc = 0;

        for (uint32_t i = t; i < n; i += posters) {
          for (size_t j = 0; j < kWorkIters; ++j) {
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
            auto coldMask = -uint64_t((j & 7) == 0);
            coldPos = (ChainNode*)(((uintptr_t)coldPos->next & coldMask) |
                                    ((uintptr_t)coldPos & ~coldMask));
          }
          sem.post();
        }
        folly::doNotOptimizeAway(acc);
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
BENCHMARK_NAMED_PARAM(contendedUse_v1, 32_to_32_v1, 32, 32)
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
