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

// --- icache-stress helpers for contendedUseICache (from-scratch experiment) --
//
// Three large noinline 256-case switches with many unique-literal ALU ops per
// case.  Rotating among them with j%3 on a pointer-chased index forces a large
// hot executed-instruction working set (raises L1-icache-load-misses_MPKI and
// lowers IPC toward the contended-semaphore target).
#define LIFOSEM_ALU_OPS(i, K)                                  \
  acc += (uint64_t)(0x9E3779B97F4A7C15ULL + (i) * (K));        \
  acc ^= (uint64_t)(0xC2B2AE3D27D4EB4FULL ^ ((i) + (K)));      \
  acc *= (uint64_t)(0x100000001B3ULL | ((i) * 7u + 1u));       \
  acc += (acc >> 13);                                          \
  acc ^= (uint64_t)((i) * 2654435761u + (K)*40503u);           \
  acc -= (uint64_t)(0xFF51AFD7ED558CCDULL + (i) + (K));         \
  acc *= (uint64_t)(0x0D2511F3u | 1u);                         \
  acc += (acc << 7);                                           \
  acc ^= (acc >> 17);                                          \
  acc += (uint64_t)((i) ^ (0xABCD0000u + (K)*131u));            \
  acc *= 3ULL;                                                 \
  acc += (uint64_t)((i) + (K)*7919u);                          \
  acc ^= (acc << 11);                                          \
  acc -= (uint64_t)(0x55555555u + (i) * (K));                  \
  acc += (acc >> 5);                                           \
  acc *= (uint64_t)(0x9u | ((i) + 1u));

#define LIFOSEM_CASE(i, K) \
  case (i): {              \
    LIFOSEM_ALU_OPS(i, K)  \
    break;                 \
  }
#define LIFOSEM_CASES_16(b, K)                                  \
  LIFOSEM_CASE((b) + 0, K) LIFOSEM_CASE((b) + 1, K)             \
  LIFOSEM_CASE((b) + 2, K) LIFOSEM_CASE((b) + 3, K)             \
  LIFOSEM_CASE((b) + 4, K) LIFOSEM_CASE((b) + 5, K)             \
  LIFOSEM_CASE((b) + 6, K) LIFOSEM_CASE((b) + 7, K)             \
  LIFOSEM_CASE((b) + 8, K) LIFOSEM_CASE((b) + 9, K)             \
  LIFOSEM_CASE((b) + 10, K) LIFOSEM_CASE((b) + 11, K)           \
  LIFOSEM_CASE((b) + 12, K) LIFOSEM_CASE((b) + 13, K)           \
  LIFOSEM_CASE((b) + 14, K) LIFOSEM_CASE((b) + 15, K)
#define LIFOSEM_CASES_256(K)                                    \
  LIFOSEM_CASES_16(0, K) LIFOSEM_CASES_16(16, K)                \
  LIFOSEM_CASES_16(32, K) LIFOSEM_CASES_16(48, K)               \
  LIFOSEM_CASES_16(64, K) LIFOSEM_CASES_16(80, K)               \
  LIFOSEM_CASES_16(96, K) LIFOSEM_CASES_16(112, K)              \
  LIFOSEM_CASES_16(128, K) LIFOSEM_CASES_16(144, K)             \
  LIFOSEM_CASES_16(160, K) LIFOSEM_CASES_16(176, K)             \
  LIFOSEM_CASES_16(192, K) LIFOSEM_CASES_16(208, K)             \
  LIFOSEM_CASES_16(224, K) LIFOSEM_CASES_16(240, K)

__attribute__((__noinline__)) static uint64_t icacheBurnA(
    uint8_t idx, uint64_t acc) {
  switch (idx) {
    LIFOSEM_CASES_256(1)
  }
  return acc;
}

__attribute__((__noinline__)) static uint64_t icacheBurnB(
    uint8_t idx, uint64_t acc) {
  switch (idx) {
    LIFOSEM_CASES_256(2)
  }
  return acc;
}

__attribute__((__noinline__)) static uint64_t icacheBurnC(
    uint8_t idx, uint64_t acc) {
  switch (idx) {
    LIFOSEM_CASES_256(3)
  }
  return acc;
}

static void contendedUseICache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr size_t kRing = 4096;
  std::vector<uint32_t> chase(kRing);
  for (size_t i = 0; i < kRing; ++i) {
    chase[i] = (uint32_t)((i * 2654435761u + 12345u) & (kRing - 1));
  }

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = (uint64_t)t + 1;
        uint32_t p = (uint32_t)t & (kRing - 1);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & (kRing - 1)];
          uint8_t idx = (uint8_t)(p ^ (acc >> 3));
          switch (j % 3) {
            case 0:
              acc = icacheBurnA(idx, acc);
              break;
            case 1:
              acc = icacheBurnB(idx, acc);
              break;
            default:
              acc = icacheBurnC(idx, acc);
              break;
          }
          ++j;
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
