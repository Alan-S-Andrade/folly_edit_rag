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
// Derived from-scratch benchmark: contendedUseChurn(8_to_100)
//
// This benchmark intentionally inflates the executed-instruction working set to
// push the frontend (L1 i-cache) harder than the tight reference loop, lowering
// IPC toward the target band.  Three large noinline 256-case switch functions
// are rotated with j%3 in the hot loop; the switch index comes from a
// pointer-chase load (payload & 0xFF).  ~14 unique ALU ops per case give the
// targeted L1i MPKI.  Each case uses unique literal constants derived from the
// case index to prevent compiler case/function deduplication.
// ---------------------------------------------------------------------------

#define ALU_CASE(K)                                                  \
  case (K): {                                                        \
    acc += 0x9e3779b1u + (uint64_t)((K) * 2654435761u);              \
    acc ^= (uint64_t)((K) * 40503u + 7u);                           \
    acc *= (uint64_t)((((K) << 1) | 1u));                           \
    acc += (uint64_t)((K) ^ 0x5bd1e995u);                           \
    acc ^= (uint64_t)((K) * 0x27d4eb2fu + 11u);                     \
    acc -= (uint64_t)((K) + 13u);                                   \
    acc *= 0x100000001b3ull;                                        \
    acc ^= acc >> 13;                                               \
    acc += (uint64_t)((K) * 3u + 17u);                              \
    acc ^= (uint64_t)((K) * 19u + 23u);                             \
    acc *= (uint64_t)((K) + 0x101u);                                \
    acc -= (uint64_t)((K) * 7u + 29u);                              \
    acc ^= (uint64_t)((K) * 0x85ebca6bu);                           \
    acc += (uint64_t)((K) * 5u + 31u);                              \
    break;                                                          \
  }

#define ALU_CASE16(b)                                                \
  ALU_CASE((b) + 0) ALU_CASE((b) + 1) ALU_CASE((b) + 2)              \
  ALU_CASE((b) + 3) ALU_CASE((b) + 4) ALU_CASE((b) + 5)              \
  ALU_CASE((b) + 6) ALU_CASE((b) + 7) ALU_CASE((b) + 8)              \
  ALU_CASE((b) + 9) ALU_CASE((b) + 10) ALU_CASE((b) + 11)            \
  ALU_CASE((b) + 12) ALU_CASE((b) + 13) ALU_CASE((b) + 14)           \
  ALU_CASE((b) + 15)

#define ALU_CASE256                                                  \
  ALU_CASE16(0) ALU_CASE16(16) ALU_CASE16(32) ALU_CASE16(48)         \
  ALU_CASE16(64) ALU_CASE16(80) ALU_CASE16(96) ALU_CASE16(112)       \
  ALU_CASE16(128) ALU_CASE16(144) ALU_CASE16(160) ALU_CASE16(176)    \
  ALU_CASE16(192) ALU_CASE16(208) ALU_CASE16(224) ALU_CASE16(240)

#define DEFINE_SWITCH_FN(NAME, UNIQ)                                 \
  static __attribute__((noinline)) uint64_t NAME(                    \
      uint8_t idx, uint64_t acc) {                                   \
    acc += (UNIQ);                                                   \
    switch (idx) { ALU_CASE256 }                                     \
    return acc;                                                      \
  }

DEFINE_SWITCH_FN(churnSwitchA, 0xABCDEF0123456789ull)
DEFINE_SWITCH_FN(churnSwitchB, 0x0123456789ABCDEFull)
DEFINE_SWITCH_FN(churnSwitchC, 0xFEDCBA9876543210ull)

static __attribute__((noinline)) uint64_t icacheChurn(
    const uint32_t* chain, size_t len, size_t steps, uint64_t acc) {
  uint32_t p = (uint32_t)(acc % len);
  for (size_t j = 0; j < steps; ++j) {
    p = chain[p];
    uint8_t idx = (uint8_t)(p & 0xFF);
    switch (j % 3) {
      case 0:
        acc = churnSwitchA(idx, acc);
        break;
      case 1:
        acc = churnSwitchB(idx, acc);
        break;
      default:
        acc = churnSwitchC(idx, acc);
        break;
    }
  }
  return acc;
}

static void contendedUseChurn(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static std::vector<uint32_t> chain;

  BENCHMARK_SUSPEND {
    if (chain.empty()) {
      chain.resize(4096);
      uint32_t x = 1u;
      for (size_t i = 0; i < chain.size(); ++i) {
        x = x * 1103515245u + 12345u;
        chain[i] = (x >> 8) % (uint32_t)chain.size();
      }
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = (uint64_t)t + 1;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          acc = icacheChurn(chain.data(), chain.size(), 48, acc);
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
BENCHMARK_NAMED_PARAM(contendedUseChurn, churn_8_to_100, 8, 100)

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
