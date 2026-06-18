#include <folly/portability/Asm.h>
#include <folly/synchronization/LifoSem.h>
#include <folly/synchronization/NativeSemaphore.h>

#include <folly/Benchmark.h>

#include <cstdint>
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

// ----------------------------------------------------------------------------
// Derived icache-stress variant of contendedUse(8_to_100).
//
// The contended LifoSem traffic is identical to contendedUse, but each waiter
// drives a large, deduplication-resistant instruction working set: three
// noinline 256-case switch functions, each case performing 14 ALU ops on a
// local accumulator with case-unique folded constants. The switch index comes
// from an L1-resident pointer-chase load (payload & 0xFF), and the hot loop
// rotates among the three functions with i % 3 to force icache thrashing
// across a massive code footprint. This is the primary L1i-MPKI lever.
// ----------------------------------------------------------------------------

#define ALU_CASE(K, S)                                  \
  case (K): {                                           \
    uint64_t k = (uint64_t)(K);                         \
    acc += k * 2654435761ull + 1ull + (uint64_t)(S);    \
    acc ^= acc << 13;                                   \
    acc *= (k | 1ull);                                  \
    acc += (k ^ 0xABCDull);                             \
    acc ^= acc >> 7;                                    \
    acc *= (0x9E3779B97F4A7C15ull ^ k);                 \
    acc += k + 12345ull + (uint64_t)(S);                \
    acc ^= acc << 17;                                   \
    acc *= (k * 3ull + 7ull);                           \
    acc += k * 5ull + 11ull;                            \
    acc ^= acc >> 11;                                   \
    acc *= (k | 0x100ull);                              \
    acc += k * 7ull + 13ull + (uint64_t)(S);            \
    acc ^= acc << 5;                                    \
  } break;

#define CASE16(b, S)                                                    \
  ALU_CASE((b) + 0, S) ALU_CASE((b) + 1, S) ALU_CASE((b) + 2, S)        \
  ALU_CASE((b) + 3, S) ALU_CASE((b) + 4, S) ALU_CASE((b) + 5, S)        \
  ALU_CASE((b) + 6, S) ALU_CASE((b) + 7, S) ALU_CASE((b) + 8, S)        \
  ALU_CASE((b) + 9, S) ALU_CASE((b) + 10, S) ALU_CASE((b) + 11, S)      \
  ALU_CASE((b) + 12, S) ALU_CASE((b) + 13, S) ALU_CASE((b) + 14, S)     \
  ALU_CASE((b) + 15, S)

#define ALL_CASES(S)                                                    \
  CASE16(0, S) CASE16(16, S) CASE16(32, S) CASE16(48, S)                \
  CASE16(64, S) CASE16(80, S) CASE16(96, S) CASE16(112, S)              \
  CASE16(128, S) CASE16(144, S) CASE16(160, S) CASE16(176, S)           \
  CASE16(192, S) CASE16(208, S) CASE16(224, S) CASE16(240, S)

FOLLY_NOINLINE static uint64_t icacheSwitchA(uint8_t idx, uint64_t acc) {
  switch (idx) {
    ALL_CASES(0x1111u)
    default:
      acc += 0xA5A5u;
      break;
  }
  return acc;
}

FOLLY_NOINLINE static uint64_t icacheSwitchB(uint8_t idx, uint64_t acc) {
  switch (idx) {
    ALL_CASES(0x2222u)
    default:
      acc += 0x5A5Au;
      break;
  }
  return acc;
}

FOLLY_NOINLINE static uint64_t icacheSwitchC(uint8_t idx, uint64_t acc) {
  switch (idx) {
    ALL_CASES(0x3333u)
    default:
      acc += 0xC3C3u;
      break;
  }
  return acc;
}

#undef ALL_CASES
#undef CASE16
#undef ALU_CASE

static void contendedUseICache(uint32_t n, int posters, int waiters) {
  static constexpr uint32_t kBuf = 4096; // L1-resident pointer-chase buffer
  static const std::vector<uint32_t> chase = [] {
    std::vector<uint32_t> v(kBuf);
    for (uint32_t i = 0; i < kBuf; ++i) {
      v[i] = static_cast<uint32_t>((i * 2654435761ull + 1013904223ull) % kBuf);
    }
    return v;
  }();

  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) & (kBuf - 1);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & (kBuf - 1)];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
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
BENCHMARK_NAMED_PARAM(contendedUseICache, 8_to_100, 8, 100)

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
