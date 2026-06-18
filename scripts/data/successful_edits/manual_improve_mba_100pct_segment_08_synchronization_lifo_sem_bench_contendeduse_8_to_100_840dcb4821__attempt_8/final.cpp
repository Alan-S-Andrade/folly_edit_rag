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
// Frontend-pressure variant of contendedUse(8_to_100).
//
// The contended LifoSem traffic is kept identical to contendedUse(8, 100),
// but each waiter/poster iteration also drives a large, mostly-cold switch
// dispatch.  Three noinline functions each hold a 256-case switch with 14
// unique ALU ops per case; rotating among them with j%3 and indexing each
// switch with a pointer-chase-derived payload spreads the executed
// instruction working set across a big code footprint, raising
// L1-icache-load-misses and lowering IPC toward the contended target.
// ---------------------------------------------------------------------------
#define ALU_CASE(i, s)                                       \
  case (i): {                                                \
    acc += 0x9E3779B97F4A7C15ull ^ (uint64_t)((i) + (s));    \
    acc ^= acc >> 7;                                         \
    acc *= 0xFF51AFD7ED558CCDull;                            \
    acc += (uint64_t)((i) * 2654435761u) ^ (uint64_t)(s);    \
    acc ^= acc << 13;                                        \
    acc -= (uint64_t)((i) ^ 0x5BD1E995u);                    \
    acc *= 0xC2B2AE3D27D4EB4Full;                            \
    acc ^= acc >> 11;                                        \
    acc += (uint64_t)(((i) << 3) | 1u) + (uint64_t)(s);      \
    acc ^= (uint64_t)((i) * 0x85EBCA6Bu);                    \
    acc *= 0x2545F4914F6CDD1Dull;                            \
    acc += (uint64_t)((i) + 0xDEADBEEFu);                    \
    acc ^= acc >> 15;                                        \
    acc -= (uint64_t)((i) * 7u + (s));                       \
    break;                                                   \
  }

#define ICACHE_C16(b, s)                                                  \
  ALU_CASE((b) + 0, s) ALU_CASE((b) + 1, s) ALU_CASE((b) + 2, s)          \
  ALU_CASE((b) + 3, s) ALU_CASE((b) + 4, s) ALU_CASE((b) + 5, s)          \
  ALU_CASE((b) + 6, s) ALU_CASE((b) + 7, s) ALU_CASE((b) + 8, s)          \
  ALU_CASE((b) + 9, s) ALU_CASE((b) + 10, s) ALU_CASE((b) + 11, s)        \
  ALU_CASE((b) + 12, s) ALU_CASE((b) + 13, s) ALU_CASE((b) + 14, s)       \
  ALU_CASE((b) + 15, s)

#define ICACHE_C256(s)                                                    \
  ICACHE_C16(0, s) ICACHE_C16(16, s) ICACHE_C16(32, s) ICACHE_C16(48, s)  \
  ICACHE_C16(64, s) ICACHE_C16(80, s) ICACHE_C16(96, s)                   \
  ICACHE_C16(112, s) ICACHE_C16(128, s) ICACHE_C16(144, s)                \
  ICACHE_C16(160, s) ICACHE_C16(176, s) ICACHE_C16(192, s)                \
  ICACHE_C16(208, s) ICACHE_C16(224, s) ICACHE_C16(240, s)

#define MAKE_ICACHE_FN(NAME, SALT)                                        \
  __attribute__((noinline)) static uint64_t NAME(                         \
      uint64_t acc, uint32_t idx) {                                       \
    switch (idx & 0xFFu) {                                                \
      ICACHE_C256(SALT)                                                   \
      default:                                                            \
        acc += 1;                                                         \
        break;                                                            \
    }                                                                     \
    return acc;                                                           \
  }

MAKE_ICACHE_FN(icacheWorkA, 0x1234u)
MAKE_ICACHE_FN(icacheWorkB, 0xABCDu)
MAKE_ICACHE_FN(icacheWorkC, 0x7F31u)

static uint8_t icacheChase[256];
static const bool icacheChaseInit = [] {
  for (int i = 0; i < 256; ++i) {
    icacheChase[i] = (uint8_t)((i * 167 + 13) & 0xFF);
  }
  return true;
}();

static void contendedUseICacheImpl(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = 0x100ull + t;
        uint32_t idx = (uint32_t)(t * 2654435761u);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          idx = icacheChase[idx & 0xFFu];
          switch (j % 3) {
            case 0:
              acc = icacheWorkA(acc, idx);
              break;
            case 1:
              acc = icacheWorkB(acc, idx);
              break;
            default:
              acc = icacheWorkC(acc, idx);
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
        uint64_t acc = 0x200ull + t;
        uint32_t idx = (uint32_t)(t * 40503u + 7u);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += posters) {
          idx = icacheChase[idx & 0xFFu];
          switch (j % 3) {
            case 0:
              acc = icacheWorkB(acc, idx);
              break;
            case 1:
              acc = icacheWorkC(acc, idx);
              break;
            default:
              acc = icacheWorkA(acc, idx);
              break;
          }
          ++j;
          sem.post();
        }
        folly::doNotOptimizeAway(acc);
      });
    }
  }

  go.store(true);
  for (auto& thr : threads) {
    thr.join();
  }
}

#undef MAKE_ICACHE_FN
#undef ICACHE_C256
#undef ICACHE_C16
#undef ALU_CASE

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

BENCHMARK(contendedUse_8_to_100_icache, iters) {
  contendedUseICacheImpl(iters, 8, 100);
}

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
