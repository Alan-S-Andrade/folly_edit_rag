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

// ----------------------------------------------------------------------------
// Derived benchmark: contendedUsePerturb
//
// This variant keeps the exact LifoSem contention pattern of contendedUse but
// folds in a frontend-pressure payload so that the executed instruction
// working set is large enough to match the reference hardware-counter profile
// (the unmodified family runs with an instruction footprint that is too small,
// driving IPC above target and L1i misses below target).
//
// The payload is three noinline functions, each a 256-case switch with 14
// distinct ALU ops per case (unique literal constants prevent the compiler
// from deduplicating cases or functions). The hot loop rotates among the three
// functions, thrashing the instruction cache across a massive code footprint.
// ALU-ops-per-case is the primary L1i MPKI knob (~14 ops -> ~15 MPKI).
// ----------------------------------------------------------------------------

#define LIFO_REP4(M, n) M(n) M((n) + 1) M((n) + 2) M((n) + 3)
#define LIFO_REP16(M, n)                                                    \
  LIFO_REP4(M, n) LIFO_REP4(M, (n) + 4) LIFO_REP4(M, (n) + 8)               \
      LIFO_REP4(M, (n) + 12)
#define LIFO_REP64(M, n)                                                    \
  LIFO_REP16(M, n) LIFO_REP16(M, (n) + 16) LIFO_REP16(M, (n) + 32)          \
      LIFO_REP16(M, (n) + 48)
#define LIFO_REP256(M)                                                      \
  LIFO_REP64(M, 0) LIFO_REP64(M, 64) LIFO_REP64(M, 128) LIFO_REP64(M, 192)

#define LIFO_CASE_A(n)                  \
  case (n): {                           \
    acc += (n) * 2654435761u + 1u;      \
    acc ^= (n) * 40503u + 2u;           \
    acc *= (n) * 2246822519u + 3u;      \
    acc -= (n) * 3266489917u + 4u;      \
    acc ^= (n) * 668265263u + 5u;       \
    acc += (n) * 374761393u + 6u;       \
    acc *= (n) * 2147483647u + 7u;      \
    acc ^= (n) * 1274126177u + 8u;      \
    acc += (n) * 1u + 9u;               \
    acc -= (n) * 13u + 10u;             \
    acc ^= (n) * 17u + 11u;             \
    acc *= (n) * 19u + 13u;             \
    acc += (n) * 23u + 15u;             \
    acc ^= (n) * 29u + 17u;             \
    break;                              \
  }

#define LIFO_CASE_B(n)                  \
  case (n): {                           \
    acc ^= (n) * 2246822519u + 19u;     \
    acc += (n) * 3266489917u + 23u;     \
    acc *= (n) * 668265263u + 29u;      \
    acc ^= (n) * 374761393u + 31u;      \
    acc -= (n) * 2654435761u + 37u;     \
    acc += (n) * 40503u + 41u;          \
    acc *= (n) * 2147483647u + 43u;     \
    acc ^= (n) * 1274126177u + 47u;     \
    acc -= (n) * 7u + 53u;              \
    acc += (n) * 11u + 59u;             \
    acc ^= (n) * 13u + 61u;             \
    acc *= (n) * 17u + 67u;             \
    acc -= (n) * 19u + 71u;             \
    acc += (n) * 23u + 73u;             \
    break;                              \
  }

#define LIFO_CASE_C(n)                  \
  case (n): {                           \
    acc *= (n) * 3266489917u + 79u;     \
    acc ^= (n) * 668265263u + 83u;      \
    acc += (n) * 2246822519u + 89u;     \
    acc -= (n) * 374761393u + 97u;      \
    acc ^= (n) * 2654435761u + 101u;    \
    acc *= (n) * 40503u + 103u;         \
    acc += (n) * 2147483647u + 107u;    \
    acc ^= (n) * 1274126177u + 109u;    \
    acc *= (n) * 29u + 113u;            \
    acc -= (n) * 31u + 127u;            \
    acc ^= (n) * 37u + 131u;            \
    acc += (n) * 41u + 137u;            \
    acc *= (n) * 43u + 139u;            \
    acc ^= (n) * 47u + 149u;            \
    break;                              \
  }

__attribute__((noinline)) static uint64_t lifoSwitchA(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFO_REP256(LIFO_CASE_A)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t lifoSwitchB(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFO_REP256(LIFO_CASE_B)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t lifoSwitchC(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFO_REP256(LIFO_CASE_C)
  }
  return acc;
}

static std::atomic<uint64_t> g_lifoPerturbSink{0};

static void contendedUsePerturb(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint64_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          // pointer-chase-like payload index from the running accumulator
          uint8_t idx = static_cast<uint8_t>(acc & 0xFF);
          switch (j % 3) {
            case 0:
              acc = lifoSwitchA(acc, idx);
              break;
            case 1:
              acc = lifoSwitchB(acc, idx);
              break;
            default:
              acc = lifoSwitchC(acc, idx);
              break;
          }
          ++j;
        }
        g_lifoPerturbSink.fetch_add(acc, std::memory_order_relaxed);
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
BENCHMARK_NAMED_PARAM(contendedUsePerturb, 8_to_100_perturb, 8, 100)

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
