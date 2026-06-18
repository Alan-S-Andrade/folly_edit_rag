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

// --- Derived from contendedUse(8_to_100): large hot-code-footprint variant ---
// Three noinline 256-case switch functions with many unique-constant ALU ops
// per case. Rotating among them (j%3) in the contended hot loop blows up the
// executed-instruction working set to raise L1 icache pressure / lower IPC.
#define LSB_ALU(n, s)                                  \
  do {                                                 \
    uint64_t k = (uint64_t)(n) ^ (uint64_t)(s);        \
    acc += k * 0x9E3779B1ull + 0x1ull;                 \
    acc ^= (acc >> 5) + k;                             \
    acc *= (k | 1ull);                                 \
    acc += k * 2654435761ull;                          \
    acc ^= k << 3;                                     \
    acc -= k * 40503ull;                               \
    acc += (acc << 7) ^ k;                             \
    acc ^= k * 2246822519ull;                          \
    acc += k ^ 0xABCDull;                              \
    acc *= 0x100000001B3ull ^ (k & 0xFFull);           \
    acc ^= (acc >> 11);                                \
    acc += k * 374761393ull;                           \
    acc ^= k + 0xDEADBEEFull;                          \
    acc -= (acc >> 2) ^ k;                             \
    acc += k * 668265263ull;                           \
    acc ^= (acc << 13) + k;                            \
  } while (0)

#define LSB_CASES_4(M, b) M(b) M((b) + 1) M((b) + 2) M((b) + 3)
#define LSB_CASES_16(M, b) \
  LSB_CASES_4(M, b)        \
  LSB_CASES_4(M, (b) + 4) LSB_CASES_4(M, (b) + 8) LSB_CASES_4(M, (b) + 12)
#define LSB_CASES_64(M, b) \
  LSB_CASES_16(M, b)       \
  LSB_CASES_16(M, (b) + 16) LSB_CASES_16(M, (b) + 32) LSB_CASES_16(M, (b) + 48)
#define LSB_CASES_256(M) \
  LSB_CASES_64(M, 0)     \
  LSB_CASES_64(M, 64) LSB_CASES_64(M, 128) LSB_CASES_64(M, 192)

#define LSB_CASE_A(n) \
  case (n):           \
    LSB_ALU(n, 0x11u);  \
    break;
#define LSB_CASE_B(n) \
  case (n):           \
    LSB_ALU(n, 0x53u);  \
    break;
#define LSB_CASE_C(n) \
  case (n):           \
    LSB_ALU(n, 0xA7u);  \
    break;

__attribute__((noinline)) static uint64_t lsbSwitchA(uint64_t acc, uint8_t idx) {
  switch (idx) { LSB_CASES_256(LSB_CASE_A) }
  return acc;
}
__attribute__((noinline)) static uint64_t lsbSwitchB(uint64_t acc, uint8_t idx) {
  switch (idx) { LSB_CASES_256(LSB_CASE_B) }
  return acc;
}
__attribute__((noinline)) static uint64_t lsbSwitchC(uint64_t acc, uint8_t idx) {
  switch (idx) { LSB_CASES_256(LSB_CASE_C) }
  return acc;
}

static void contendedSwitch(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  // Pointer-chase buffer: each waiter walks it to derive an unpredictable
  // switch index (payload & 0xFF) feeding the large code-footprint switches.
  static constexpr uint32_t kChaseSize = 4096; // power of two
  std::vector<uint32_t> chase(kChaseSize);

  BENCHMARK_SUSPEND {
    for (uint32_t i = 0; i < kChaseSize; ++i) {
      chase[i] = (i * 2654435761u + 12345u) & (kChaseSize - 1);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = (uint64_t)t + 1;
        uint32_t p = (uint32_t)t;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & (kChaseSize - 1)];
          uint8_t idx = (uint8_t)(p ^ acc);
          switch ((i / (uint32_t)waiters) % 3) {
            case 0:
              acc = lsbSwitchA(acc, idx);
              break;
            case 1:
              acc = lsbSwitchB(acc, idx);
              break;
            default:
              acc = lsbSwitchC(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedSwitch, 8_to_100, 8, 100)

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
