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
// Derived from contendedUse(8_to_100): adds a large, low-reuse executed-code
// footprint to raise frontend/L1-icache pressure (lower IPC). Three noinline
// 256-case switches with unique per-case immediates are rotated by j%3 in the
// hot path, indexed by a pointer-chase payload byte so the branch target /
// instruction working set stays wide.
// ---------------------------------------------------------------------------

#define ICACHE_OPS(n, S)                          \
  do {                                            \
    uint64_t k = (uint64_t)(n) ^ (uint64_t)(S);   \
    a += k * 1000003u + 17u;                      \
    a ^= k * 2654435761u + 3u;                    \
    a *= (k * 6u + 1u);                           \
    a += (k << 4) ^ 0xA5u;                        \
    a -= k * 31u + 5u;                            \
    a ^= (k * 65537u);                            \
    a += k * 7u + 11u;                            \
    a *= ((k & 0xFFu) | 1u);                       \
    a ^= k * 99991u + 13u;                        \
    a += (k << 2) + 9u;                           \
    a -= k * 19u;                                 \
    a ^= k * 40503u + 7u;                         \
  } while (0)

#define ICACHE_CASE(n, S) \
  case (n):               \
    ICACHE_OPS(n, S);     \
    break;

#define ICACHE_C16(b, S)                                            \
  ICACHE_CASE(b + 0, S) ICACHE_CASE(b + 1, S) ICACHE_CASE(b + 2, S) \
  ICACHE_CASE(b + 3, S) ICACHE_CASE(b + 4, S) ICACHE_CASE(b + 5, S) \
  ICACHE_CASE(b + 6, S) ICACHE_CASE(b + 7, S) ICACHE_CASE(b + 8, S) \
  ICACHE_CASE(b + 9, S) ICACHE_CASE(b + 10, S) ICACHE_CASE(b + 11, S) \
  ICACHE_CASE(b + 12, S) ICACHE_CASE(b + 13, S) ICACHE_CASE(b + 14, S) \
  ICACHE_CASE(b + 15, S)

#define ICACHE_C256(S)                                                   \
  ICACHE_C16(0, S) ICACHE_C16(16, S) ICACHE_C16(32, S) ICACHE_C16(48, S) \
  ICACHE_C16(64, S) ICACHE_C16(80, S) ICACHE_C16(96, S) ICACHE_C16(112, S) \
  ICACHE_C16(128, S) ICACHE_C16(144, S) ICACHE_C16(160, S) ICACHE_C16(176, S) \
  ICACHE_C16(192, S) ICACHE_C16(208, S) ICACHE_C16(224, S) ICACHE_C16(240, S)

__attribute__((noinline)) static uint64_t icacheSwitch0(uint8_t idx, uint64_t a) {
  switch (idx) {
    ICACHE_C256(0x9E3779B1u)
    default:
      break;
  }
  return a;
}

__attribute__((noinline)) static uint64_t icacheSwitch1(uint8_t idx, uint64_t a) {
  switch (idx) {
    ICACHE_C256(0x85EBCA77u)
    default:
      break;
  }
  return a;
}

__attribute__((noinline)) static uint64_t icacheSwitch2(uint8_t idx, uint64_t a) {
  switch (idx) {
    ICACHE_C256(0xC2B2AE3Du)
    default:
      break;
  }
  return a;
}

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        std::vector<uint64_t> chase(256);
        for (size_t z = 0; z < chase.size(); ++z) {
          chase[z] = (z * 2654435761u + 1013904223u + (uint64_t)t);
        }
        uint64_t a = (uint64_t)t + 1;
        uint64_t p = (uint64_t)t;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & 0xFF];
          uint8_t idx = (uint8_t)(p & 0xFF);
          switch (i % 3) {
            case 0:
              a = icacheSwitch0(idx, a);
              break;
            case 1:
              a = icacheSwitch1(idx, a);
              break;
            default:
              a = icacheSwitch2(idx, a);
              break;
          }
        }
        folly::doNotOptimizeAway(a);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go] {
        std::vector<uint64_t> chase(256);
        for (size_t z = 0; z < chase.size(); ++z) {
          chase[z] = (z * 40503u + 19349663u + (uint64_t)t);
        }
        uint64_t a = (uint64_t)t + 7;
        uint64_t p = (uint64_t)t + 3;
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          p = chase[p & 0xFF];
          uint8_t idx = (uint8_t)(p & 0xFF);
          switch (i % 3) {
            case 0:
              a = icacheSwitch1(idx, a);
              break;
            case 1:
              a = icacheSwitch2(idx, a);
              break;
            default:
              a = icacheSwitch0(idx, a);
              break;
          }
        }
        folly::doNotOptimizeAway(a);
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
