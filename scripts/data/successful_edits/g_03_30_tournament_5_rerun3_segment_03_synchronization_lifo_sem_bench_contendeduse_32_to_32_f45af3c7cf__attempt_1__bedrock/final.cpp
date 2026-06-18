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
// Pointer-chase chains + 3 noinline 256-case switch functions
// for contendedUse_chase variant
// ---------------------------------------------------------------------------

namespace {

struct ChaseNode {
  ChaseNode* next;
  char pad[56]; // 64 bytes total
};

static constexpr size_t kHotLen = 16384; // 1 MB
static ChaseNode hotNodes[kHotLen];

static constexpr size_t kColdLen = 1u << 17; // 128K nodes x 64B = 8 MB
static ChaseNode coldNodes[kColdLen];

static bool chainsInited = false;

static void initChain(ChaseNode* nodes, size_t len, uint64_t seed) {
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), 0u);
  std::mt19937_64 rng(seed);
  for (size_t i = len - 1; i > 0; --i) {
    std::swap(perm[i], perm[rng() % (i + 1)]);
  }
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
}

static void ensureChainsInited() {
  if (!chainsInited) {
    initChain(hotNodes, kHotLen, 42u);
    initChain(coldNodes, kColdLen, 137u);
    chainsInited = true;
  }
}

__attribute__((noinline)) uint64_t chaseSwitch0(uint64_t v, int c) {
  switch (c & 0xFF) {
    case 0: v += 0xA1B2C3D4; v ^= 0x1234ABCD; break;
    case 1: v *= 0x9E3779B9; v ^= 0xDEADBEEF; break;
    case 2: v += 0x6C62272E; v ^= 0x07BB0142; break;
    case 3: v *= 0xBF58476D; v ^= 0x94D049BB; break;
    case 4: v += 0xC4CEB9FE; v ^= 0x1A85EC53; break;
    case 5: v *= 0xFF51AFD7; v ^= 0xC4CEB9FE; break;
    case 6: v += 0x3CD0EB9D; v ^= 0xBEEFCAFE; break;
    case 7: v *= 0x2545F491; v ^= 0x4915FEA2; break;
    case 8: v += 0xA3B1BAAD; v ^= 0x13198A2E; break;
    case 9: v *= 0xD2A98B26; v ^= 0x625B7065; break;
    case 10: v += 0x0123456789ABCDEF; v ^= 0xFEDCBA9876543210; break;
    case 11: v *= 0x1111111111111111; v ^= 0x2222222222222222; break;
    case 12: v += 0x3333333333333333; v ^= 0x4444444444444444; break;
    case 13: v *= 0x5555555555555555; v ^= 0x6666666666666666; break;
    case 14: v += 0x7777777777777777; v ^= 0x8888888888888888; break;
    case 15: v *= 0x9999999999999999; v ^= 0xAAAAAAAAAAAAAAAA; break;
    case 16: v += 0xBBBBBBBBBBBBBBBB; v ^= 0xCCCCCCCCCCCCCCCC; break;
    case 17: v *= 0xDDDDDDDDDDDDDDDD; v ^= 0xEEEEEEEEEEEEEEEE; break;
    case 18: v += 0xFFFFFFFFFFFFFFFF; v ^= 0x0F0F0F0F0F0F0F0F; break;
    case 19: v *= 0x1F1F1F1F1F1F1F1F; v ^= 0x2F2F2F2F2F2F2F2F; break;
    case 20: v += 0x3F3F3F3F3F3F3F3F; v ^= 0x4F4F4F4F4F4F4F4F; break;
    case 21: v *= 0x5F5F5F5F5F5F5F5F; v ^= 0x6F6F6F6F6F6F6F6F; break;
    case 22: v += 0x7F7F7F7F7F7F7F7F; v ^= 0x8F8F8F8F8F8F8F8F; break;
    case 23: v *= 0x9F9F9F9F9F9F9F9F; v ^= 0xAFAFAFAFAFAFAFAF; break;
    case 24: v += 0xBFBFBFBFBFBFBFBF; v ^= 0xCFCFCFCFCFCFCFCF; break;
    case 25: v *= 0xDFDFDFDFDFDFDFDF; v ^= 0xEFEFEFEFEFEFEFEF; break;
    case 26: v += 0xA0B0C0D0E0F01020; v ^= 0x3040506070809000; break;
    case 27: v *= 0x1020304050607080; v ^= 0x90A0B0C0D0E0F001; break;
    case 28: v += 0x0201040803020108; v ^= 0x0804020110804020; break;
    case 29: v *= 0x2010080402010804; v ^= 0x0201008040201008; break;
    case 30: v += 0x0100010001000100; v ^= 0x0001000100010001; break;
    case 31: v *= 0x0100000001000000; v ^= 0x0000000100000001; break;
    default: v ^= (uint64_t)c * 0x9E3779B97F4A7C15ULL; break;
  }
  return v;
}

__attribute__((noinline)) uint64_t chaseSwitch1(uint64_t v, int c) {
  switch (c & 0xFF) {
    case 0: v ^= 0xFEDCBA9876543210; v += 0x0123456789ABCDEF; break;
    case 1: v ^= 0x2222222222222222; v *= 0x1111111111111111; break;
    case 2: v ^= 0x4444444444444444; v += 0x3333333333333333; break;
    case 3: v ^= 0x6666666666666666; v *= 0x5555555555555555; break;
    case 4: v ^= 0x8888888888888888; v += 0x7777777777777777; break;
    case 5: v ^= 0xAAAAAAAAAAAAAAAA; v *= 0x9999999999999999; break;
    case 6: v ^= 0xCCCCCCCCCCCCCCCC; v += 0xBBBBBBBBBBBBBBBB; break;
    case 7: v ^= 0xEEEEEEEEEEEEEEEE; v *= 0xDDDDDDDDDDDDDDDD; break;
    case 8: v ^= 0x0F0F0F0F0F0F0F0F; v += 0xFFFFFFFFFFFFFFFF; break;
    case 9: v ^= 0x2F2F2F2F2F2F2F2F; v *= 0x1F1F1F1F1F1F1F1F; break;
    case 10: v ^= 0x4F4F4F4F4F4F4F4F; v += 0x3F3F3F3F3F3F3F3F; break;
    case 11: v ^= 0x6F6F6F6F6F6F6F6F; v *= 0x5F5F5F5F5F5F5F5F; break;
    case 12: v ^= 0x8F8F8F8F8F8F8F8F; v += 0x7F7F7F7F7F7F7F7F; break;
    case 13: v ^= 0xAFAFAFAFAFAFAFAF; v *= 0x9F9F9F9F9F9F9F9F; break;
    case 14: v ^= 0xCFCFCFCFCFCFCFCF; v += 0xBFBFBFBFBFBFBFBF; break;
    case 15: v ^= 0xEFEFEFEFEFEFEFEF; v *= 0xDFDFDFDFDFDFDFDF; break;
    case 16: v ^= 0xA1B2C3D4E5F60718; v += 0x192A3B4C5D6E7F80; break;
    case 17: v ^= 0x91A2B3C4D5E6F708; v *= 0x192A3B4C5D6E7F80; break;
    case 18: v ^= 0x81929384A5B6C7D8; v += 0xE9FA0B1C2D3E4F50; break;
    case 19: v ^= 0x71828394A5B6C7D8; v *= 0xE9FA0B1C2D3E4F50; break;
    case 20: v ^= 0x61728384A5B6C7D8; v += 0xA9BA0B1C2D3E4F50; break;
    case 21: v ^= 0x51628374A5B6C7D8; v *= 0xA9BA0B1C2D3E4F50; break;
    case 22: v ^= 0x41526374A5B6C7D8; v += 0x69BA0B1C2D3E4F50; break;
    case 23: v ^= 0x31425364A5B6C7D8; v *= 0x69BA0B1C2D3E4F50; break;
    case 24: v ^= 0x21324354A5B6C7D8; v += 0x29BA0B1C2D3E4F50; break;
    case 25: v ^= 0x11224344A5B6C7D8; v *= 0x29BA0B1C2D3E4F50; break;
    case 26: v ^= 0x01122334A5B6C7D8; v += 0x09BA0B1C2D3E4F50; break;
    case 27: v ^= 0xF1020314A5B6C7D8; v *= 0x09BA0B1C2D3E4F50; break;
    case 28: v ^= 0xE1F20304A5B6C7D8; v += 0xE9FA0B0C2D3E4F50; break;
    case 29: v ^= 0xD1E2F304A5B6C7D8; v *= 0xD9EA0B0C2D3E4F50; break;
    case 30: v ^= 0xC1D2E3F4A5B6C7D8; v += 0xC9DA0B0C2D3E4F50; break;
    case 31: v ^= 0xB1C2D3E4F5A6B7C8; v *= 0xB9CA0B0C2D3E4F50; break;
    default: v ^= (uint64_t)c * 0x6C62272E07BB0142ULL; break;
  }
  return v;
}

__attribute__((noinline)) uint64_t chaseSwitch2(uint64_t v, int c) {
  switch (c & 0xFF) {
    case 0: v = (v << 13) | (v >> 51); v += 0xABCDEF0123456789; break;
    case 1: v = (v << 17) | (v >> 47); v ^= 0xBCDEF01234567890; break;
    case 2: v = (v << 19) | (v >> 45); v += 0xCDEF012345678901; break;
    case 3: v = (v << 23) | (v >> 41); v ^= 0xDEF0123456789012; break;
    case 4: v = (v << 29) | (v >> 35); v += 0xEF01234567890123; break;
    case 5: v = (v << 31) | (v >> 33); v ^= 0xF012345678901234; break;
    case 6: v = (v << 37) | (v >> 27); v += 0x0123456789012345; break;
    case 7: v = (v << 41) | (v >> 23); v ^= 0x1234567890123456; break;
    case 8: v = (v << 43) | (v >> 21); v += 0x2345678901234567; break;
    case 9: v = (v << 47) | (v >> 17); v ^= 0x3456789012345678; break;
    case 10: v = (v << 53) | (v >> 11); v += 0x4567890123456789; break;
    case 11: v = (v << 59) | (v >> 5); v ^= 0x567890123456789A; break;
    case 12: v = (v << 7) | (v >> 57); v += 0x67890123456789AB; break;
    case 13: v = (v << 11) | (v >> 53); v ^= 0x7890123456789ABC; break;
    case 14: v = (v << 13) | (v >> 51); v *= 0x890123456789ABCD; break;
    case 15: v = (v << 17) | (v >> 47); v *= 0x90123456789ABCDE; break;
    case 16: v = (v << 19) | (v >> 45); v *= 0xA0123456789ABCDF; break;
    case 17: v = (v << 23) | (v >> 41); v *= 0xB0123456789ABCE0; break;
    case 18: v = (v << 29) | (v >> 35); v *= 0xC0123456789ABCF1; break;
    case 19: v = (v << 31) | (v >> 33); v *= 0xD0123456789ABD02; break;
    case 20: v = (v << 37) | (v >> 27); v *= 0xE0123456789ABE13; break;
    case 21: v = (v << 41) | (v >> 23); v *= 0xF0123456789ABF24; break;
    case 22: v = (v << 43) | (v >> 21); v *= 0x00123456789AC035; break;
    case 23: v = (v << 47) | (v >> 17); v *= 0x10123456789AC146; break;
    case 24: v = (v << 53) | (v >> 11); v *= 0x20123456789AC257; break;
    case 25: v = (v << 59) | (v >> 5); v *= 0x30123456789AC368; break;
    case 26: v = (v << 7) | (v >> 57); v *= 0x40123456789AC479; break;
    case 27: v = (v << 11) | (v >> 53); v *= 0x50123456789AC58A; break;
    case 28: v = (v << 3) | (v >> 61); v *= 0x60123456789AC69B; break;
    case 29: v = (v << 5) | (v >> 59); v *= 0x70123456789AC7AC; break;
    case 30: v = (v << 61) | (v >> 3); v *= 0x80123456789AC8BD; break;
    case 31: v = (v << 63) | (v >> 1); v *= 0x90123456789AC9CE; break;
    default: v = ((v << 7) | (v >> 57)) ^ ((uint64_t)c * 0xBF58476D1CE4E5B9ULL); break;
  }
  return v;
}

static constexpr size_t kInnerIters = 500000;

static void contendedUseChase(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    ensureChainsInited();

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

  // Timed hot path: pointer-chasing + 3-way switch dispatch
  ChaseNode* hotPos = &hotNodes[0];
  ChaseNode* coldPos = &coldNodes[0];
  uint64_t acc = 0;
  for (size_t j = 0; j < kInnerIters; ++j) {
    hotPos = hotPos->next;
    uint64_t payload = (uint64_t)(uintptr_t)hotPos;
    int sel = (int)(j % 3);
    if (sel == 0) {
      acc = chaseSwitch0(acc ^ payload, (int)(payload & 0xFF));
    } else if (sel == 1) {
      acc = chaseSwitch1(acc ^ payload, (int)(payload & 0xFF));
    } else {
      acc = chaseSwitch2(acc ^ payload, (int)(payload & 0xFF));
    }
    // branchless cold-chain step every 8 iterations
    uint64_t doStep = ((j & 7) == 0) ? ~uint64_t(0) : uint64_t(0);
    uintptr_t nextCold = (uintptr_t)coldPos->next;
    uintptr_t keepCold = (uintptr_t)coldPos;
    coldPos = (ChaseNode*)((nextCold & doStep) | (keepCold & ~doStep));
  }
  folly::doNotOptimizeAway(acc);
  folly::doNotOptimizeAway(coldPos);

  go.store(true);
  for (auto& thr : threads) {
    thr.join();
  }
}

} // namespace

BENCHMARK_DRAW_LINE();
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_1, 1, 1)
BENCHMARK_NAMED_PARAM(contendedUseChase, 32_to_32_chase, 32, 32)
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
