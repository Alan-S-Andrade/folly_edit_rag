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
// icacheContendedUse: contended LifoSem use augmented with a large, low-reuse
// executed-instruction footprint.  Three noinline functions each hold a
// 256-case switch (selected by a pointer-chase payload byte) with 14 ALU ops
// per case using unique literal constants.  Rotating across the three
// functions with j%3 thrashes the L1 instruction cache, lowering IPC into the
// frontend-bound regime without altering the synchronization semantics.
#define ICACHE_CASE(n)                                       \
  case (n):                                                  \
    acc += (uint64_t)(n) * 2654435761ull + (SALT);           \
    acc ^= (uint64_t)(n) * 40503ull + ((SALT) * 3ull);       \
    acc *= ((uint64_t)(n) | 1ull);                           \
    acc += (uint64_t)((n) ^ 0x5bd1e995u);                    \
    acc ^= ((uint64_t)(n) << 1) + (SALT);                    \
    acc -= (uint64_t)(n) * 7919ull;                          \
    acc += (uint64_t)((n) + 0x9e3779b9u);                    \
    acc ^= (uint64_t)(n) * 2246822519ull;                    \
    acc *= ((uint64_t)(n) | 3ull);                           \
    acc += (uint64_t)((n) ^ 0xcc9e2d51u);                    \
    acc ^= (uint64_t)(n) * 0x85ebca6bull + (SALT);           \
    acc -= (uint64_t)((n) << 2);                             \
    acc += (uint64_t)(n) * 0xc2b2ae35ull;                    \
    acc ^= (uint64_t)((n) + (SALT));                         \
    break;

#define ICACHE_C1(n) ICACHE_CASE(n)
#define ICACHE_C4(n) \
  ICACHE_C1(n) ICACHE_C1((n) + 1) ICACHE_C1((n) + 2) ICACHE_C1((n) + 3)
#define ICACHE_C16(n) \
  ICACHE_C4(n) ICACHE_C4((n) + 4) ICACHE_C4((n) + 8) ICACHE_C4((n) + 12)
#define ICACHE_C64(n) \
  ICACHE_C16(n) ICACHE_C16((n) + 16) ICACHE_C16((n) + 32) ICACHE_C16((n) + 48)
#define ICACHE_C256(n) \
  ICACHE_C64(n) ICACHE_C64((n) + 64) ICACHE_C64((n) + 128) ICACHE_C64((n) + 192)

#define SALT 0x1111111111111111ull
__attribute__((noinline)) static uint64_t icache_mix0(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_C256(0)
  }
  return acc;
}
#undef SALT

#define SALT 0x2222222222222222ull
__attribute__((noinline)) static uint64_t icache_mix1(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_C256(0)
  }
  return acc;
}
#undef SALT

#define SALT 0x3333333333333333ull
__attribute__((noinline)) static uint64_t icache_mix2(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_C256(0)
  }
  return acc;
}
#undef SALT

static const std::vector<uint32_t>& icacheChase() {
  static const std::vector<uint32_t> chase = [] {
    const size_t N = 4096;
    std::vector<uint32_t> v(N);
    for (size_t i = 0; i < N; ++i) {
      v[i] = (uint32_t)((i * 2654435761u + 12345u) % N);
    }
    return v;
  }();
  return chase;
}

static void icacheContendedUse(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;
  const std::vector<uint32_t>& chase = icacheChase();

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = (uint64_t)t + 1;
        uint32_t p = (uint32_t)t;
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & 0xFFFu];
          uint8_t idx = (uint8_t)(p & 0xFFu);
          switch (j % 3) {
            case 0:
              acc = icache_mix0(acc, idx);
              break;
            case 1:
              acc = icache_mix1(acc, idx);
              break;
            default:
              acc = icache_mix2(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1, 31, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 16_to_16, 16, 16)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_32, 32, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1000, 32, 1000)
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
