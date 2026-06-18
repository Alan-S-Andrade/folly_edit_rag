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
// Frontend-bound derived variant of contendedUse.
//
// The original contendedUse(8_to_100) is heavily backend/contention bound with
// a very small executed instruction working set, which yields an IPC well above
// the desired target. To bring IPC down we deliberately inflate the hot code
// footprint so the workload becomes L1-icache bound: each waiter, after being
// released, performs a pointer-chase load and then dispatches into one of three
// large noinline 256-case switch functions (rotated by i % 3). Each case runs a
// chain of ALU ops on a local accumulator using unique literal constants so the
// compiler cannot deduplicate cases or functions, forcing icache thrashing.
//
// The number of ALU ops per case is the primary L1i knob.
constexpr size_t kFeChaseSize = 4096;
static uint32_t feChase[kFeChaseSize];

#define FE_CASE(SALT, n)                                          \
  case (n): {                                                     \
    acc += ((SALT) + 0x9E3779B9ULL) ^ ((uint64_t)(n) + 1);       \
    acc *= (2ull * (n) + 1 + (SALT));                            \
    acc ^= (0xFFULL * ((n) + 3) + (SALT));                       \
    acc += (((uint64_t)(n) << 7) + 0x1234 + (SALT));            \
    acc *= (3ull + (n) + (SALT));                                \
    acc ^= ((uint64_t)(n) * 0xABCDEFULL + (SALT));              \
    acc += (0x5555ULL - (n) + (SALT));                          \
    acc *= (5ull + ((n) & 7) + (SALT));                         \
    acc ^= ((uint64_t)(n) + 0x7777 + (SALT));                   \
    acc += ((uint64_t)(n) * 17 + 0x99 + (SALT));               \
    acc *= (7ull + ((n) >> 1) + (SALT));                        \
    acc ^= (0xC0FFEEULL + (n) + (SALT));                        \
    acc += ((uint64_t)(n) * 31 + 0xAB + (SALT));               \
    acc *= (9ull + ((n) & 3) + (SALT));                         \
    acc ^= (0xDEADBEEFULL + (n) * 3 + (SALT));                 \
    acc += ((uint64_t)(n) * 13 + 0xCD + (SALT));               \
    break;                                                       \
  }

#define FE_CASE4(SALT, b) \
  FE_CASE(SALT, (b) + 0)  \
  FE_CASE(SALT, (b) + 1)  \
  FE_CASE(SALT, (b) + 2)  \
  FE_CASE(SALT, (b) + 3)

#define FE_CASE16(SALT, b) \
  FE_CASE4(SALT, (b) + 0)  \
  FE_CASE4(SALT, (b) + 4)  \
  FE_CASE4(SALT, (b) + 8)  \
  FE_CASE4(SALT, (b) + 12)

#define FE_CASE256(SALT)     \
  FE_CASE16(SALT, 0)         \
  FE_CASE16(SALT, 16)        \
  FE_CASE16(SALT, 32)        \
  FE_CASE16(SALT, 48)        \
  FE_CASE16(SALT, 64)        \
  FE_CASE16(SALT, 80)        \
  FE_CASE16(SALT, 96)        \
  FE_CASE16(SALT, 112)       \
  FE_CASE16(SALT, 128)       \
  FE_CASE16(SALT, 144)       \
  FE_CASE16(SALT, 160)       \
  FE_CASE16(SALT, 176)       \
  FE_CASE16(SALT, 192)       \
  FE_CASE16(SALT, 208)       \
  FE_CASE16(SALT, 224)       \
  FE_CASE16(SALT, 240)

__attribute__((noinline)) static uint64_t fe_switch_a(
    uint64_t acc, uint8_t idx) {
  switch (idx) { FE_CASE256(0x11ULL) }
  return acc;
}

__attribute__((noinline)) static uint64_t fe_switch_b(
    uint64_t acc, uint8_t idx) {
  switch (idx) { FE_CASE256(0x22ULL) }
  return acc;
}

__attribute__((noinline)) static uint64_t fe_switch_c(
    uint64_t acc, uint8_t idx) {
  switch (idx) { FE_CASE256(0x33ULL) }
  return acc;
}

static void contendedUseFE(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kFeChaseSize; ++i) {
      feChase[i] =
          (uint32_t)((i * 2654435761u + 12345u) & (kFeChaseSize - 1));
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t acc = 0x1234567ULL + (uint64_t)t;
        uint32_t p = (uint32_t)t;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = feChase[p & (kFeChaseSize - 1)];
          uint8_t idx = (uint8_t)(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc ^= fe_switch_a(acc, idx);
              break;
            case 1:
              acc ^= fe_switch_b(acc, idx);
              break;
            default:
              acc ^= fe_switch_c(acc, idx);
              break;
          }
        }
        sink.fetch_add(acc, std::memory_order_relaxed);
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
BENCHMARK_NAMED_PARAM(contendedUseFE, 8_to_100_fe, 8, 100)

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
