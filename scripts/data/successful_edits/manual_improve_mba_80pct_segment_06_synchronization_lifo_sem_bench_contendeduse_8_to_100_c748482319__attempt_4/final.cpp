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
// Pointer-chase + switch machinery used by contendedUseChase below.
//
// This intentionally adds a large permuted pointer-chase footprint and a
// rotation of three big switch functions into the timed hot path so that the
// variant is memory/branch/frontend bound instead of compute bound. The chase
// chains exceed L1d and the switch functions inflate the executed instruction
// working set well beyond L1i.
// ---------------------------------------------------------------------------
namespace {

struct ChainNode {
  ChainNode* next;
  char pad[56];
};

constexpr size_t kHotLen = 16384; // 16384 * 64B = 1 MiB
constexpr size_t kColdLen = size_t(2) << 20; // 2,097,152 nodes ~= 128 MiB

ChainNode gHotNodes[kHotLen];
ChainNode* gColdNodes = nullptr;

void initChain(ChainNode* nodes, size_t len) {
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), size_t(0));
  std::mt19937_64 rng(42);
  for (size_t i = len - 1; i > 0; --i) {
    std::swap(perm[i], perm[rng() % (i + 1)]);
  }
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
}

void initChase() {
  static std::once_flag once;
  std::call_once(once, [] {
    gColdNodes = new ChainNode[kColdLen];
    initChain(gHotNodes, kHotLen);
    initChain(gColdNodes, kColdLen);
  });
}

// Three large switch bodies generated via macro expansion. Each case uses
// unique literal constants (derived from the case index) so the compiler
// cannot deduplicate them, producing a big hot code footprint.
#define CHASE_X16(M, b)                                                     \
  M((b) + 0) M((b) + 1) M((b) + 2) M((b) + 3) M((b) + 4) M((b) + 5)         \
  M((b) + 6) M((b) + 7) M((b) + 8) M((b) + 9) M((b) + 10) M((b) + 11)       \
  M((b) + 12) M((b) + 13) M((b) + 14) M((b) + 15)

#define CHASE_X256(M)                                                       \
  CHASE_X16(M, 0) CHASE_X16(M, 16) CHASE_X16(M, 32) CHASE_X16(M, 48)        \
  CHASE_X16(M, 64) CHASE_X16(M, 80) CHASE_X16(M, 96) CHASE_X16(M, 112)      \
  CHASE_X16(M, 128) CHASE_X16(M, 144) CHASE_X16(M, 160) CHASE_X16(M, 176)   \
  CHASE_X16(M, 192) CHASE_X16(M, 208) CHASE_X16(M, 224) CHASE_X16(M, 240)

#define CHASE_CASE_A(i)                                                     \
  case (i):                                                                 \
    v += UINT64_C(0x9E3779B97F4A7C15) * ((i) + 1u);                         \
    v ^= v >> 7;                                                            \
    v *= UINT64_C(0xFF51AFD7ED558CCD) + (i);                                \
    v += UINT64_C(0x100000001B3) * ((i) + 2u);                             \
    v ^= v << 13;                                                           \
    v -= UINT64_C(0xC2B2AE3D27D4EB4F) * ((i) + 3u);                         \
    v *= UINT64_C(0x165667B19E3779F9);                                      \
    v ^= (v >> 11) + (i);                                                   \
    v += UINT64_C(0xD6E8FEB86659FD93) ^ (i);                                \
    v ^= v << 5;                                                            \
    break;

#define CHASE_CASE_B(i)                                                     \
  case (i):                                                                 \
    v ^= UINT64_C(0xA0761D6478BD642F) + (i);                                \
    v += UINT64_C(0xE7037ED1A0B428DB) * ((i) + 5u);                         \
    v ^= v >> 9;                                                            \
    v *= UINT64_C(0x8EBC6AF09C88C6E3) ^ (i);                                \
    v -= UINT64_C(0x589965CC75374CC3) * ((i) + 7u);                         \
    v ^= v << 17;                                                           \
    v += UINT64_C(0x1D8E4E27C47D124F) + (i);                                \
    v *= UINT64_C(0x2545F4914F6CDD1D);                                      \
    v ^= (v >> 6) + ((i) << 1);                                            \
    v += UINT64_C(0x94D049BB133111EB) ^ (i);                                \
    break;

#define CHASE_CASE_C(i)                                                     \
  case (i):                                                                 \
    v += UINT64_C(0xBF58476D1CE4E5B9) ^ (i);                                \
    v ^= v >> 12;                                                           \
    v *= UINT64_C(0x9FB21C651E98DF25) + (i);                                \
    v -= UINT64_C(0xEB44ACCAB455D165) * ((i) + 11u);                        \
    v ^= v << 7;                                                            \
    v += UINT64_C(0x27D4EB2F165667C5) * ((i) + 13u);                        \
    v ^= (v >> 14) + (i);                                                   \
    v *= UINT64_C(0x3C79AC492BA7B653);                                      \
    v += UINT64_C(0x1C69B3F74AC4AE35) ^ ((i) + 17u);                        \
    v ^= v << 3;                                                            \
    break;

__attribute__((noinline)) uint64_t chaseSwitchA(uint64_t v, int c) {
  switch (c) {
    CHASE_X256(CHASE_CASE_A)
    default:
      v ^= UINT64_C(0xABCDEF0123456789);
      break;
  }
  return v;
}

__attribute__((noinline)) uint64_t chaseSwitchB(uint64_t v, int c) {
  switch (c) {
    CHASE_X256(CHASE_CASE_B)
    default:
      v ^= UINT64_C(0x0123456789ABCDEF);
      break;
  }
  return v;
}

__attribute__((noinline)) uint64_t chaseSwitchC(uint64_t v, int c) {
  switch (c) {
    CHASE_X256(CHASE_CASE_C)
    default:
      v ^= UINT64_C(0xFEDCBA9876543210);
      break;
  }
  return v;
}

#undef CHASE_CASE_A
#undef CHASE_CASE_B
#undef CHASE_CASE_C
#undef CHASE_X256
#undef CHASE_X16

} // namespace

// Variant of contendedUse(8_to_100): same contended post/wait traffic, but the
// timed main thread additionally drives a large permuted pointer-chase loop
// with data-dependent rotation across three big switch functions. This makes
// the hot path memory/branch/frontend bound (lower IPC, higher L1i/branch/LLC
// MPKI) while preserving the contended-semaphore character.
static void contendedUseChase(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    initChase();
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

  // Timed pointer-chase hot path; iteration count scales with the benchmark
  // iters so it dominates the instruction mix.
  ChainNode* hotPos = &gHotNodes[0];
  ChainNode* coldPos = &gColdNodes[0];
  uint64_t acc = 0;
  for (uint32_t j = 0; j < n; ++j) {
    hotPos = hotPos->next; // dependent L1d-missing load
    uint64_t payload = (uint64_t)(uintptr_t)hotPos;
    switch (j % 3) {
      case 0:
        acc = chaseSwitchA(acc, static_cast<int>(payload & 0xFF));
        break;
      case 1:
        acc = chaseSwitchB(acc, static_cast<int>(payload & 0xFF));
        break;
      default:
        acc = chaseSwitchC(acc, static_cast<int>(payload & 0xFF));
        break;
    }
    // IPC lever: advance the 128 MiB cold chain far more frequently (every
    // other iteration instead of every 8th) so that distinct LLC-missing
    // dependent loads dominate the cycle budget and pull IPC back down toward
    // the contended-semaphore target. The advance is still branchless.
    uint64_t coldMask = -uint64_t((j & 1) == 0);
    coldPos = (ChainNode*)(((uintptr_t)coldPos->next & coldMask) |
                           ((uintptr_t)coldPos & ~coldMask));
    acc ^= (uint64_t)(uintptr_t)coldPos;
  }
  folly::doNotOptimizeAway(acc);

  for (auto& thr : threads) {
    thr.join();
  }
}

BENCHMARK_DRAW_LINE();
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_1, 1, 1)
BENCHMARK_NAMED_PARAM(contendedUseChase, 8_to_100, 8, 100)
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
