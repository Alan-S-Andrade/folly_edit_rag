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
// The original contendedUse hot path is small and reuses the same handful of
// instructions, so its executed instruction working set is far too compact
// (low L1i pressure, high IPC). To grow the hot code footprint we interleave
// the semaphore contention with a rotation across three large noinline
// 256-case switch functions whose case index comes from a pointer-chase
// load. Each case carries a block of ALU ops with unique literal constants so
// the compiler cannot deduplicate cases or functions, forcing icache
// thrashing across a large code region.
// ---------------------------------------------------------------------------

#define ICACHE_OP_CASE(n, s)                           \
  case (n): {                                          \
    acc += 0x9E3779B1ull * (uint64_t)((n) + 1u) + (s); \
    acc ^= (uint64_t)((n) * 2246822519u + 1u);         \
    acc *= (uint64_t)(((n) << 1) | 1u);                \
    acc += (uint64_t)((n) ^ (0xA5u + (s)));            \
    acc ^= (uint64_t)((n) * 3266489917u + 7u);         \
    acc -= (uint64_t)((n) * 668265263u + 11u);         \
    acc *= (uint64_t)(((n) >> 1) | 1u);                \
    acc += (uint64_t)((n) * 374761393u + 13u);         \
    acc ^= (uint64_t)((n) * 2654435761u + 17u);        \
    acc -= (uint64_t)((n) * 40503u + 19u);             \
    acc *= (uint64_t)(((n) ^ (s)) | 1u);               \
    acc += (uint64_t)((n) * 257u + 23u);               \
    acc ^= (uint64_t)((n) * 65537u + 29u);             \
    acc -= (uint64_t)((n) * 99u + 31u);                \
    break;                                             \
  }

#define ICACHE_CASES_4(b, s)                  \
  ICACHE_OP_CASE((b) + 0, s)                  \
  ICACHE_OP_CASE((b) + 1, s)                  \
  ICACHE_OP_CASE((b) + 2, s)                  \
  ICACHE_OP_CASE((b) + 3, s)
#define ICACHE_CASES_16(b, s)                 \
  ICACHE_CASES_4((b) + 0, s)                  \
  ICACHE_CASES_4((b) + 4, s)                  \
  ICACHE_CASES_4((b) + 8, s)                  \
  ICACHE_CASES_4((b) + 12, s)
#define ICACHE_CASES_64(b, s)                 \
  ICACHE_CASES_16((b) + 0, s)                 \
  ICACHE_CASES_16((b) + 16, s)                \
  ICACHE_CASES_16((b) + 32, s)                \
  ICACHE_CASES_16((b) + 48, s)
#define ICACHE_CASES_256(s)                   \
  ICACHE_CASES_64(0, s)                       \
  ICACHE_CASES_64(64, s)                      \
  ICACHE_CASES_64(128, s)                     \
  ICACHE_CASES_64(192, s)

__attribute__((noinline)) static uint64_t icacheSwitchA(
    uint8_t idx, uint64_t acc) {
  switch (idx) { ICACHE_CASES_256(0x11u) }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheSwitchB(
    uint8_t idx, uint64_t acc) {
  switch (idx) { ICACHE_CASES_256(0x22u) }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheSwitchC(
    uint8_t idx, uint64_t acc) {
  switch (idx) { ICACHE_CASES_256(0x33u) }
  return acc;
}

static constexpr size_t kIcacheChaseLen = 4096;

static const std::vector<uint8_t>& icacheChase() {
  static const std::vector<uint8_t> chase = [] {
    std::vector<uint8_t> v(kIcacheChaseLen);
    uint32_t x = 0x12345678u;
    for (size_t i = 0; i < kIcacheChaseLen; ++i) {
      x = x * 1664525u + 1013904223u;
      v[i] = static_cast<uint8_t>(x >> 16);
    }
    return v;
  }();
  return chase;
}

static void contendedUseICache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  const std::vector<uint8_t>& chase = icacheChase();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        size_t p = static_cast<size_t>(t) * 131u + 7u;
        for (uint32_t i = t; i < n; i += waiters) {
          for (int j = 0; j < 3; ++j) {
            uint8_t idx = chase[p & (kIcacheChaseLen - 1)];
            p = idx + ((p * 6364136223846793005ull) >> 32);
            switch (j % 3) {
              case 0:
                acc = icacheSwitchA(idx, acc);
                break;
              case 1:
                acc = icacheSwitchB(idx, acc);
                break;
              default:
                acc = icacheSwitchC(idx, acc);
                break;
            }
          }
          sem.wait();
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
BENCHMARK_NAMED_PARAM(contendedUseICache, 8_to_100, 8, 100)
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
