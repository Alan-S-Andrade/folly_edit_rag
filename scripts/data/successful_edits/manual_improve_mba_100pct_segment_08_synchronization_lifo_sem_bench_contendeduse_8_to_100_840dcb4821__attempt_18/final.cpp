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
// Derived frontend-bound variant of contendedUse(8_to_100).
//
// The contended LifoSem hot path has a very compact instruction working set,
// which keeps L1i misses far below the target profile and pushes IPC too high.
// To match the reference hardware-counter profile we splice a large, hard to
// cache code footprint into the waiter hot path: three noinline functions each
// holding a 256-case switch with a fixed number of ALU ops per case, selected
// by the low byte of a pointer-chase load and rotated with j%3. The unique
// per-case / per-function literal constants prevent compiler deduplication so
// the three switches stay distinct in the instruction cache.
// ---------------------------------------------------------------------------

#define ALU_CASE(n, s)                                                     \
  case (n): {                                                              \
    acc += (uint64_t)((n) * 2654435761u + 17u + (s));                      \
    acc ^= ((acc << 13) | (uint64_t)((n) + 0x100 + (s)));                  \
    acc *= (uint64_t)(((n) | 1) + 0x9E3779B97F4A7C15ull + (s));            \
    acc -= (uint64_t)((n) * 40503u + 7u + (s));                            \
    acc ^= (acc >> 11) ^ (uint64_t)((n) * 3u + (s));                       \
    acc += (uint64_t)(((n) ^ 0xABCD) * 31u + (s));                         \
    acc ^= (uint64_t)(((n) << 3) + 0xDEAD + (s));                          \
    acc *= (0x100000001B3ull ^ (uint64_t)(n)) + (s);                       \
    acc += (uint64_t)((n) * 7u + 3u + (s));                                \
    acc ^= (acc << 5) + (uint64_t)((n) + 0x55 + (s));                      \
    acc *= (uint64_t)(((n) * 5u) | 3u) + (s);                              \
    acc -= (uint64_t)((n) ^ 0x1234) + (s);                                 \
    break;                                                                 \
  }

#define ICACHE_C1(n, s) ALU_CASE(n, s)
#define ICACHE_C4(n, s) \
  ICACHE_C1(n, s) ICACHE_C1(n + 1, s) ICACHE_C1(n + 2, s) ICACHE_C1(n + 3, s)
#define ICACHE_C16(n, s) \
  ICACHE_C4(n, s) ICACHE_C4(n + 4, s) ICACHE_C4(n + 8, s) ICACHE_C4(n + 12, s)
#define ICACHE_C64(n, s)                                          \
  ICACHE_C16(n, s) ICACHE_C16(n + 16, s) ICACHE_C16(n + 32, s)    \
      ICACHE_C16(n + 48, s)
#define ICACHE_C256(s)                                                 \
  ICACHE_C64(0, s) ICACHE_C64(64, s) ICACHE_C64(128, s) ICACHE_C64(192, s)

__attribute__((noinline)) static uint64_t icacheSwitchA(
    uint64_t acc, uint8_t idx) {
  switch (idx) { ICACHE_C256(1u) }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheSwitchB(
    uint64_t acc, uint8_t idx) {
  switch (idx) { ICACHE_C256(2u) }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheSwitchC(
    uint64_t acc, uint8_t idx) {
  switch (idx) { ICACHE_C256(3u) }
  return acc;
}

#undef ICACHE_C256
#undef ICACHE_C64
#undef ICACHE_C16
#undef ICACHE_C4
#undef ICACHE_C1
#undef ALU_CASE

static uint64_t runIcacheWork(
    const uint32_t* chase, uint32_t start, uint64_t acc) {
  uint32_t p = start;
  for (int j = 0; j < 24; ++j) {
    p = chase[p];
    uint8_t idx = static_cast<uint8_t>(p & 0xFF);
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

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  static std::vector<uint32_t> chase;

  BENCHMARK_SUSPEND {
    if (chase.empty()) {
      chase.resize(4096);
      for (uint32_t i = 0; i < chase.size(); ++i) {
        chase[i] = (i * 2654435761u + 1u) % chase.size();
      }
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          acc = runIcacheWork(chase.data(), (i + t) & 4095u, acc);
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
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100, 8, 100)
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
