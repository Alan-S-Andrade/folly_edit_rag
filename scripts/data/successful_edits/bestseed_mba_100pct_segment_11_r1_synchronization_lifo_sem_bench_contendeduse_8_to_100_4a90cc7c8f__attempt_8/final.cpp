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

// --- Derived frontend-bound variant: icacheContendedUse ---------------------
// This variant keeps the LifoSem contention pattern but interleaves a large,
// hard-to-cache instruction working set into the hot loop so that the executed
// instruction footprint (and L1i pressure) grows without changing the
// semaphore semantics. Three noinline functions each hold a 256-case switch
// with many ALU ops per case (unique literal constants to defeat compiler
// deduplication); the hot loop rotates among them via j % 3, indexing each
// switch by a payload byte from a pointer-chase load.

static constexpr size_t kChaseLen = 4096;
static uint32_t icacheChaseNext[kChaseLen];
static uint8_t icacheChasePayload[kChaseLen];
static volatile uint64_t icacheSink;

static const bool icacheChaseInit = [] {
  for (size_t i = 0; i < kChaseLen; ++i) {
    icacheChaseNext[i] =
        static_cast<uint32_t>((i * 2654435761u + 12345u) % kChaseLen);
    icacheChasePayload[i] = static_cast<uint8_t>((i * 97u + 13u) & 0xFFu);
  }
  return true;
}();

#define ICACHE_CASE(c, k)                                       \
  case (c): {                                                   \
    a ^= (uint64_t)((c) * 2654435761u + (k) + 1u);              \
    a += (uint64_t)((c) * 40503u + (k) + 7u);                   \
    a *= (uint64_t)(((c) | 1u) + (k));                          \
    a ^= (uint64_t)((c) * 2246822519u + (k) + 13u);             \
    a += (uint64_t)((c) * 3266489917u + (k) + 17u);             \
    a *= (uint64_t)((((c) << 1) | 1u) + (k));                   \
    a ^= (uint64_t)((c) * 668265263u + (k) + 19u);              \
    a += (uint64_t)((c) * 374761393u + (k) + 23u);              \
    a *= (uint64_t)((((c) << 2) | 1u) + (k));                   \
    a ^= (uint64_t)((c) * 2654435761u + (k) + 29u);             \
    a += (uint64_t)((c) * 246822519u + (k) + 31u);             \
    a *= (uint64_t)((((c) << 3) | 1u) + (k));                   \
    a ^= (uint64_t)((c) * 668265263u + (k) + 37u);             \
    a += (uint64_t)((c) * 374761393u + (k) + 41u);             \
    break;                                                      \
  }

#define ICACHE_C16(b, k)                                            \
  ICACHE_CASE((b) + 0, k)                                           \
  ICACHE_CASE((b) + 1, k) ICACHE_CASE((b) + 2, k)                   \
  ICACHE_CASE((b) + 3, k) ICACHE_CASE((b) + 4, k)                   \
  ICACHE_CASE((b) + 5, k) ICACHE_CASE((b) + 6, k)                   \
  ICACHE_CASE((b) + 7, k) ICACHE_CASE((b) + 8, k)                   \
  ICACHE_CASE((b) + 9, k) ICACHE_CASE((b) + 10, k)                  \
  ICACHE_CASE((b) + 11, k) ICACHE_CASE((b) + 12, k)                 \
  ICACHE_CASE((b) + 13, k) ICACHE_CASE((b) + 14, k)                 \
  ICACHE_CASE((b) + 15, k)

#define ICACHE_SWITCH_BODY(k)                                          \
  switch (idx & 0xFFu) {                                               \
    ICACHE_C16(0, k) ICACHE_C16(16, k) ICACHE_C16(32, k)              \
    ICACHE_C16(48, k) ICACHE_C16(64, k) ICACHE_C16(80, k)            \
    ICACHE_C16(96, k) ICACHE_C16(112, k) ICACHE_C16(128, k)         \
    ICACHE_C16(144, k) ICACHE_C16(160, k) ICACHE_C16(176, k)        \
    ICACHE_C16(192, k) ICACHE_C16(208, k) ICACHE_C16(224, k)        \
    ICACHE_C16(240, k)                                               \
  }

__attribute__((noinline)) static uint64_t icacheSwitchA(
    uint64_t a, uint32_t idx) {
  ICACHE_SWITCH_BODY(101u)
  return a;
}

__attribute__((noinline)) static uint64_t icacheSwitchB(
    uint64_t a, uint32_t idx) {
  ICACHE_SWITCH_BODY(211u)
  return a;
}

__attribute__((noinline)) static uint64_t icacheSwitchC(
    uint64_t a, uint32_t idx) {
  ICACHE_SWITCH_BODY(331u)
  return a;
}

static inline uint64_t icacheWork(uint64_t a, uint32_t& pos) {
  for (int j = 0; j < 3; ++j) {
    pos = icacheChaseNext[pos];
    uint32_t idx = icacheChasePayload[pos];
    switch (j % 3) {
      case 0:
        a = icacheSwitchA(a, idx);
        break;
      case 1:
        a = icacheSwitchB(a, idx);
        break;
      default:
        a = icacheSwitchC(a, idx);
        break;
    }
  }
  return a;
}

static void icacheContendedUse(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t a = 0x9e3779b97f4a7c15ull + static_cast<uint64_t>(t);
        uint32_t pos = static_cast<uint32_t>(t * 131u) % kChaseLen;
        for (uint32_t i = t; i < n; i += waiters) {
          a = icacheWork(a, pos);
          sem.wait();
        }
        icacheSink += a;
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go] {
        uint64_t a = 0xff51afd7ed558ccdull + static_cast<uint64_t>(t);
        uint32_t pos = static_cast<uint32_t>(t * 257u) % kChaseLen;
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          a = icacheWork(a, pos);
          sem.post();
        }
        icacheSink += a;
      });
    }
  }

  go.store(true);
  for (auto& thr : threads) {
    thr.join();
  }
}

BENCHMARK_NAMED_PARAM(icacheContendedUse, 8_to_100, 8, 100)

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
