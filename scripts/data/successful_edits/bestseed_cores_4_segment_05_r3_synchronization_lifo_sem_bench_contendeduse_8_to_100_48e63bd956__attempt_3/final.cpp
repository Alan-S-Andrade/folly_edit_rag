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
// Derived frontend/i-cache pressure variant of contendedUse(8_to_100).
//
// Proven I-cache counter-matching pattern: three large noinline 256-case
// switches with several ALU ops per case (each using unique literal constants
// to defeat compiler deduplication). The poster threads rotate among the three
// functions (j % 3) and index the switch with a payload byte obtained from a
// data-dependent (pointer-chase) load. This inflates the executed instruction
// working set to raise L1-icache-load-misses_MPKI toward the target without
// disturbing the underlying contention behavior.
// ---------------------------------------------------------------------------

// 8 ALU ops per case; the (S) salt makes each function's cases distinct.
#define ICACHE_OPS(i, S)                                                  \
  acc += UINT64_C(0x9E3779B97F4A7C15) ^                                   \
      (uint64_t((i)) * 2654435761ull + (S) + 1ull);                       \
  acc ^= (uint64_t((i)) * 40503ull + (S) + 7ull);                         \
  acc *= ((uint64_t((i)) * 2246822519ull) | 1ull) + (S);                  \
  acc += (uint64_t((i)) * 3266489917ull + (S) + 13ull);                   \
  acc ^= (uint64_t((i)) << (((i) & 31u) + 1u)) ^ (S);                     \
  acc -= (uint64_t((i)) * 668265263ull + (S) + 17ull);                    \
  acc *= ((uint64_t((i)) * 374761393ull) | 3ull) + (S);                   \
  acc ^= (uint64_t((i)) * 2057ull + (S) + 23ull);

#define ICACHE_CASE_A(i) \
  case (i): {            \
    ICACHE_OPS(i, 0x11)  \
    break;               \
  }
#define ICACHE_CASE_B(i) \
  case (i): {            \
    ICACHE_OPS(i, 0x22)  \
    break;               \
  }
#define ICACHE_CASE_C(i) \
  case (i): {            \
    ICACHE_OPS(i, 0x33)  \
    break;               \
  }

#define ICACHE_REP16(M, b)                                              \
  M(b + 0) M(b + 1) M(b + 2) M(b + 3) M(b + 4) M(b + 5) M(b + 6)        \
      M(b + 7) M(b + 8) M(b + 9) M(b + 10) M(b + 11) M(b + 12)          \
          M(b + 13) M(b + 14) M(b + 15)

#define ICACHE_REP256(M)                                                  \
  ICACHE_REP16(M, 0) ICACHE_REP16(M, 16) ICACHE_REP16(M, 32)             \
      ICACHE_REP16(M, 48) ICACHE_REP16(M, 64) ICACHE_REP16(M, 80)        \
          ICACHE_REP16(M, 96) ICACHE_REP16(M, 112) ICACHE_REP16(M, 128)  \
              ICACHE_REP16(M, 144) ICACHE_REP16(M, 160)                  \
                  ICACHE_REP16(M, 176) ICACHE_REP16(M, 192)              \
                      ICACHE_REP16(M, 208) ICACHE_REP16(M, 224)          \
                          ICACHE_REP16(M, 240)

static __attribute__((noinline)) uint64_t icacheMixA(uint64_t acc, uint8_t idx) {
  switch (idx) { ICACHE_REP256(ICACHE_CASE_A) }
  return acc;
}

static __attribute__((noinline)) uint64_t icacheMixB(uint64_t acc, uint8_t idx) {
  switch (idx) { ICACHE_REP256(ICACHE_CASE_B) }
  return acc;
}

static __attribute__((noinline)) uint64_t icacheMixC(uint64_t acc, uint8_t idx) {
  switch (idx) { ICACHE_REP256(ICACHE_CASE_C) }
  return acc;
}

static std::vector<uint8_t> makeIcacheChase() {
  std::vector<uint8_t> v(1024);
  uint32_t x = 2463534242u;
  for (size_t i = 0; i < v.size(); ++i) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    v[i] = uint8_t(x);
  }
  return v;
}

static void contendedUseMix(uint32_t n, int posters, int waiters) {
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
        static thread_local std::vector<uint8_t> chase = makeIcacheChase();
        const uint32_t mask = uint32_t(chase.size() - 1);
        while (!go.load()) {
          std::this_thread::yield();
        }
        uint64_t acc = uint64_t(t) + 1;
        uint32_t p = uint32_t(t);
        for (uint32_t i = t; i < n; i += posters) {
          // data-dependent (pointer-chase) load supplies the switch index
          p = chase[p & mask];
          uint8_t idx = uint8_t(acc ^ p);
          switch ((i / posters) % 3) {
            case 0:
              acc = icacheMixA(acc, idx);
              break;
            case 1:
              acc = icacheMixB(acc, idx);
              break;
            default:
              acc = icacheMixC(acc, idx);
              break;
          }
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
BENCHMARK_NAMED_PARAM(contendedUseMix, 8_to_100, 8, 100)

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
