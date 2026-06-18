#include <folly/portability/Asm.h>
#include <folly/synchronization/LifoSem.h>
#include <folly/synchronization/NativeSemaphore.h>

#include <folly/Benchmark.h>

#include <cstdint>

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
// Frontend-bound i-cache footprint helpers used by contendedUseIcache below.
// Three noinline functions each holding a 256-case switch with 14 ALU ops per
// case (unique literal constants per case/function to defeat deduplication).
// Rotating among them with (i % 3) and indexing by a pointer-chased payload
// byte produces a large hot instruction working set.
// ---------------------------------------------------------------------------
#define ICACHE_ALU_OPS(k, S)                                       \
  acc ^= (uint64_t)(k) * 0x9E3779B97F4A7C15ull + (uint64_t)(S);    \
  acc += (uint64_t)(k) + 0x1111u + (uint64_t)(S);                  \
  acc *= ((uint64_t)(k) | 1u);                                     \
  acc ^= acc >> 7;                                                 \
  acc += (uint64_t)(k) * 3u + 0x2222u + (uint64_t)(S);             \
  acc *= (uint64_t)(((k) & 0xFFu) + 3u);                           \
  acc ^= (uint64_t)(k) << 3;                                       \
  acc += (uint64_t)((k) ^ (0x3333u + (S)));                        \
  acc *= (uint64_t)((k) + 5u);                                     \
  acc ^= (uint64_t)(k) * 7u + 0x4444u;                             \
  acc += acc >> 3;                                                 \
  acc *= (uint64_t)((k) | 3u);                                     \
  acc ^= (uint64_t)(k) + 0x5555u + (uint64_t)(S);                  \
  acc += (uint64_t)(k) * 11u;

#define ICACHE_CASE(k, S) \
  case (k): {             \
    ICACHE_ALU_OPS(k, S)  \
  } break;

#define ICACHE_CASES16(b, S)                                        \
  ICACHE_CASE(b + 0, S)                                             \
  ICACHE_CASE(b + 1, S)                                             \
  ICACHE_CASE(b + 2, S)                                             \
  ICACHE_CASE(b + 3, S)                                             \
  ICACHE_CASE(b + 4, S)                                             \
  ICACHE_CASE(b + 5, S)                                             \
  ICACHE_CASE(b + 6, S)                                             \
  ICACHE_CASE(b + 7, S)                                             \
  ICACHE_CASE(b + 8, S)                                             \
  ICACHE_CASE(b + 9, S)                                             \
  ICACHE_CASE(b + 10, S)                                            \
  ICACHE_CASE(b + 11, S)                                            \
  ICACHE_CASE(b + 12, S)                                            \
  ICACHE_CASE(b + 13, S)                                            \
  ICACHE_CASE(b + 14, S)                                            \
  ICACHE_CASE(b + 15, S)

#define ICACHE_ALL_CASES(S)                                         \
  ICACHE_CASES16(0, S)                                              \
  ICACHE_CASES16(16, S)                                             \
  ICACHE_CASES16(32, S)                                             \
  ICACHE_CASES16(48, S)                                             \
  ICACHE_CASES16(64, S)                                             \
  ICACHE_CASES16(80, S)                                             \
  ICACHE_CASES16(96, S)                                             \
  ICACHE_CASES16(112, S)                                            \
  ICACHE_CASES16(128, S)                                            \
  ICACHE_CASES16(144, S)                                            \
  ICACHE_CASES16(160, S)                                            \
  ICACHE_CASES16(176, S)                                            \
  ICACHE_CASES16(192, S)                                            \
  ICACHE_CASES16(208, S)                                            \
  ICACHE_CASES16(224, S)                                            \
  ICACHE_CASES16(240, S)

__attribute__((noinline)) static uint64_t icache_fn0(uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_ALL_CASES(0xA1u)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t icache_fn1(uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_ALL_CASES(0xB2u)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t icache_fn2(uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_ALL_CASES(0xC3u)
  }
  return acc;
}

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static std::vector<uint32_t> ring;

  BENCHMARK_SUSPEND {
    if (ring.empty()) {
      ring.resize(4096);
      for (size_t i = 0; i < ring.size(); ++i) {
        ring[i] = (uint32_t)((i * 1103515245u + 12345u) % ring.size());
      }
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = (uint64_t)t + 1;
        uint32_t cur = (uint32_t)t & 4095u;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          cur = ring[cur & 4095u];
          uint8_t idx = (uint8_t)(cur & 0xFFu);
          switch (i % 3) {
            case 0:
              acc = icache_fn0(acc, idx);
              break;
            case 1:
              acc = icache_fn1(acc, idx);
              break;
            default:
              acc = icache_fn2(acc, idx);
              break;
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
