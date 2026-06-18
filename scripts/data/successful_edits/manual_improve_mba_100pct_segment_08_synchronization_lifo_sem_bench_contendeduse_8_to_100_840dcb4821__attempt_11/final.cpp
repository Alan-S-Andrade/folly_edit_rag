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

// --------------------------------------------------------------------------
// Derived I-cache-footprint variant of contendedUse(8_to_100).
//
// The hot path of the reference benchmark is far too compact, keeping the
// executed instruction working set tiny. This variant interleaves a large,
// non-deduplicated code footprint (three noinline 256-case switches with a
// rotating dispatch driven by a pointer-chase load) into each semaphore
// post/wait iteration to widen the executed instruction working set.
// --------------------------------------------------------------------------
namespace {

constexpr int kChainLen = 4096;

struct ChaseNode {
  uint32_t next;
  uint8_t payload;
};

std::vector<ChaseNode> makeChain() {
  std::vector<ChaseNode> v(kChainLen);
  for (int i = 0; i < kChainLen; ++i) {
    v[i].next = static_cast<uint32_t>((i * 2654435761u + 1) % kChainLen);
    v[i].payload = static_cast<uint8_t>(i * 31 + 7);
  }
  return v;
}

std::vector<ChaseNode> gChain = makeChain();

// 14 ALU ops per case. Each case uses unique literal constants derived from the
// case index plus a per-function salt (S) so that the three switch bodies are
// not folded together by the compiler.
#define ICASE(i, S)                                                          \
  case (i):                                                                  \
    acc += 0x9e3779b97f4a7c15ull + (uint64_t)(i) * 2654435761ull + (S);      \
    acc ^= ((uint64_t)(i) << 13) ^ (S);                                      \
    acc *= (2246822519ull + (uint64_t)(i) * 3);                              \
    acc += ((uint64_t)(i) ^ 0xdeadbeefull) + (S);                           \
    acc ^= (acc >> 7);                                                       \
    acc *= (0x100000001b3ull + (uint64_t)(i) + (S));                        \
    acc += ((uint64_t)(i) * 40503ull + 11);                                  \
    acc ^= (acc << 11);                                                      \
    acc *= (0xff51afd7ed558ccdull ^ (uint64_t)(i));                         \
    acc += (0xc4ceb9fe1a85ec53ull + (uint64_t)(i) + (S));                   \
    acc ^= (acc >> 17);                                                      \
    acc += ((uint64_t)(i) * 2654435761ull);                                  \
    acc *= (1469598103934665603ull + (uint64_t)(i));                        \
    acc ^= ((uint64_t)(i) + 0x12345ull) ^ (S);                              \
    break;

#define C2(n, S) ICASE(n, S) ICASE((n) + 1, S)
#define C8(n, S) C2(n, S) C2((n) + 2, S) C2((n) + 4, S) C2((n) + 6, S)
#define C32(n, S) C8(n, S) C8((n) + 8, S) C8((n) + 16, S) C8((n) + 24, S)
#define C128(n, S) \
  C32(n, S) C32((n) + 32, S) C32((n) + 64, S) C32((n) + 96, S)
#define C256(S) C128(0, S) C128(128, S)

FOLLY_NOINLINE uint64_t icacheSwitchA(uint64_t acc, uint8_t idx) {
  switch (idx) {
    C256(0x1111111111111111ull)
  }
  return acc;
}

FOLLY_NOINLINE uint64_t icacheSwitchB(uint64_t acc, uint8_t idx) {
  switch (idx) {
    C256(0x2222222222222222ull)
  }
  return acc;
}

FOLLY_NOINLINE uint64_t icacheSwitchC(uint64_t acc, uint8_t idx) {
  switch (idx) {
    C256(0x3333333333333333ull)
  }
  return acc;
}

#undef C256
#undef C128
#undef C32
#undef C8
#undef C2
#undef ICASE

FOLLY_NOINLINE uint64_t icacheChurn(uint64_t acc, uint32_t& pos) {
  const auto& chain = gChain;
  for (int j = 0; j < 24; ++j) {
    pos = chain[pos].next;
    uint8_t idx = static_cast<uint8_t>(chain[pos].payload ^ (uint8_t)acc);
    switch (j % 3) {
      case 0:
        acc = icacheSwitchA(acc, idx);
        break;
      case 1:
        acc = icacheSwitchB(acc, idx);
        break;
      default:
        acc = icacheSwitchC(acc, idx);
        break;
    }
  }
  return acc;
}

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t acc = 0x12345ull + t;
        uint32_t pos = static_cast<uint32_t>((t * 97) % kChainLen);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          acc = icacheChurn(acc, pos);
        }
        sink.fetch_add(acc, std::memory_order_relaxed);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &sink] {
        uint64_t acc = 0x6789aull + t;
        uint32_t pos = static_cast<uint32_t>((t * 131) % kChainLen);
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          acc = icacheChurn(acc, pos);
        }
        sink.fetch_add(acc, std::memory_order_relaxed);
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
BENCHMARK_DRAW_LINE();
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
