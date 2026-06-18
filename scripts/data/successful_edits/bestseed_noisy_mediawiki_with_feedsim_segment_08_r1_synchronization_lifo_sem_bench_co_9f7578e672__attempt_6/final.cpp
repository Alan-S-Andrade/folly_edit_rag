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
// Frontend/i-cache pressure variant of contendedUse.
//
// Each contended waiter, after being woken, walks a pointer-chase buffer and
// dispatches into one of three large 256-case switch functions selected by
// j % 3.  The switch functions are noinline and each case carries unique
// literal constants so the compiler cannot deduplicate or merge them.  This
// creates a large hot instruction working set that thrashes the L1 i-cache and
// drives IPC down toward the contended-semaphore target.
// ---------------------------------------------------------------------------

static constexpr size_t kChaseLen = 4096;
static uint32_t gChase[kChaseLen];

static void initIcacheChase() {
  for (size_t i = 0; i < kChaseLen; ++i) {
    gChase[i] = (uint32_t)((i * 2654435761u + 1013904223u) % kChaseLen);
  }
}

#define ALU_CASE(n, s)                                              \
  case (n):                                                         \
    acc += (uint64_t)((n) * 2654435761u + (s) * 40503u + 1u);       \
    acc ^= (uint64_t)((n) * 374761393u + (s) * 668265263u + 3u);    \
    acc *= (uint64_t)((n) * 2u + 1u);                               \
    acc += (uint64_t)((n) * 17u + (s) * 19u + 5u);                  \
    acc ^= (uint64_t)((n) * 23u + (s) * 29u + 7u);                  \
    acc *= (uint64_t)((n) * 4u + 3u);                               \
    acc += (uint64_t)((n) * 31u + (s) * 37u + 11u);                 \
    acc ^= (uint64_t)((n) * 41u + (s) * 43u + 13u);                 \
    acc *= (uint64_t)((n) * 6u + 5u);                               \
    acc += (uint64_t)((n) * 47u + (s) * 53u + 17u);                 \
    acc ^= (uint64_t)((n) * 59u + (s) * 61u + 19u);                 \
    acc *= (uint64_t)((n) * 8u + 7u);                               \
    acc += (uint64_t)((n) * 67u + (s) * 71u + 23u);                 \
    acc ^= (uint64_t)((n) * 73u + (s) * 79u + 29u);                 \
    acc *= (uint64_t)((n) * 10u + 9u);                              \
    acc += (uint64_t)((n) * 83u + (s) * 89u + 31u);                 \
    acc ^= (uint64_t)((n) * 97u + (s) * 101u + 37u);                \
    acc *= (uint64_t)((n) * 12u + 11u);                             \
    break;

#define C16(b, s)                                                              \
  ALU_CASE((b) + 0, s) ALU_CASE((b) + 1, s) ALU_CASE((b) + 2, s)               \
  ALU_CASE((b) + 3, s) ALU_CASE((b) + 4, s) ALU_CASE((b) + 5, s)               \
  ALU_CASE((b) + 6, s) ALU_CASE((b) + 7, s) ALU_CASE((b) + 8, s)               \
  ALU_CASE((b) + 9, s) ALU_CASE((b) + 10, s) ALU_CASE((b) + 11, s)             \
  ALU_CASE((b) + 12, s) ALU_CASE((b) + 13, s) ALU_CASE((b) + 14, s)            \
  ALU_CASE((b) + 15, s)

#define ICACHE_SWITCH_BODY(s)                                                  \
  switch (idx) {                                                               \
    C16(0, s) C16(16, s) C16(32, s) C16(48, s)                                 \
    C16(64, s) C16(80, s) C16(96, s) C16(112, s)                               \
    C16(128, s) C16(144, s) C16(160, s) C16(176, s)                            \
    C16(192, s) C16(208, s) C16(224, s) C16(240, s)                            \
  }

__attribute__((noinline)) static uint64_t icacheBurnA(uint64_t acc,
                                                       uint8_t idx) {
  ICACHE_SWITCH_BODY(1)
  return acc;
}

__attribute__((noinline)) static uint64_t icacheBurnB(uint64_t acc,
                                                       uint8_t idx) {
  ICACHE_SWITCH_BODY(2)
  return acc;
}

__attribute__((noinline)) static uint64_t icacheBurnC(uint64_t acc,
                                                      uint8_t idx) {
  ICACHE_SWITCH_BODY(3)
  return acc;
}

#undef ICACHE_SWITCH_BODY
#undef C16
#undef ALU_CASE

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    initIcacheChase();
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = 0x12345u + (uint64_t)t;
        uint32_t p = (uint32_t)((t * 1099087573u + 7u) % kChaseLen);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int k = 0; k < 24; ++k) {
            p = gChase[p];
            uint8_t idx = (uint8_t)(p & 0xFFu);
            switch (k % 3) {
              case 0:
                acc = icacheBurnA(acc, idx);
                break;
              case 1:
                acc = icacheBurnB(acc, idx);
                break;
              default:
                acc = icacheBurnC(acc, idx);
                break;
            }
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
