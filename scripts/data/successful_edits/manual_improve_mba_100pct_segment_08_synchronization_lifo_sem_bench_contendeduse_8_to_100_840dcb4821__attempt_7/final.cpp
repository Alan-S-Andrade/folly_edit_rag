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
// I-cache footprint helpers for contendedUse_8_to_100_icache.
//
// Three large noinline 256-case switch functions. Each case performs a fixed
// number of ALU ops with case-unique literal constants so the compiler cannot
// deduplicate case bodies. Rotating among the three functions in the hot loop
// (driven by a pointer-chase index) forces the executed instruction working
// set far beyond the L1 instruction cache, raising L1i misses / lowering IPC.
// ---------------------------------------------------------------------------
#define ALU_CASE(N, S)                                                  \
  case (N):                                                             \
    acc += 0x1001ull + (uint64_t)(N) + (S)*7ull;                       \
    acc ^= 0x2A3Bull * (uint64_t)((N) + 1u) + (S);                     \
    acc *= 0x1000003ull | (uint64_t)(((N) << 1) + (S));               \
    acc -= 0xABCDull ^ (uint64_t)((N) + (S));                          \
    acc += 0x55A1ull + (uint64_t)((N)*3u + (S));                       \
    acc ^= 0x1234ull + (uint64_t)((N) + (S)*3u);                       \
    acc *= 0x9E37ull | (uint64_t)(((N) + 7u) + (S));                   \
    acc += 0xF00Dull ^ (uint64_t)(((N) << 2) + (S));                   \
    acc -= 0x0BAD1ull + (uint64_t)((N) + (S)*5u);                      \
    acc ^= 0xBEE5ull * (uint64_t)((N) + 5u + (S));                     \
    acc += 0xDEA7ull ^ (uint64_t)((N) + 11u + (S));                    \
    acc *= 0x1357ull | (uint64_t)(((N) + 3u) + (S));                   \
    acc -= 0x2468ull + (uint64_t)(((N) << 1) + (S));                   \
    acc ^= 0xCAF1ull ^ (uint64_t)((N) + (S));                          \
    acc += 0x3C3Cull + (uint64_t)((N)*5u + (S));                       \
    acc ^= 0x7777ull + (uint64_t)((N) + (S)*9u);                       \
    acc *= 0xA5A7ull | (uint64_t)(((N) + 13u) + (S));                  \
    acc += 0x9119ull ^ (uint64_t)(((N) << 3) + (S));                   \
    acc -= 0x4242ull + (uint64_t)((N) + (S)*11u);                      \
    acc ^= 0x8AAAull * (uint64_t)((N) + 17u + (S));                    \
    acc += 0x6F6Full ^ (uint64_t)((N) + 19u + (S));                    \
    acc *= 0x2B2Dull | (uint64_t)(((N) + 23u) + (S));                  \
    acc -= 0x5151ull + (uint64_t)(((N) << 2) + (S));                   \
    acc ^= 0xC1C1ull ^ (uint64_t)((N)*7u + (S));                       \
    acc += 0x33A5ull + (uint64_t)((N) + (S)*13u);                      \
    acc ^= 0x9D9Dull + (uint64_t)((N) + 29u + (S));                    \
    acc *= 0xE00Full | (uint64_t)(((N) + 31u) + (S));                  \
    acc += 0x1B1Dull ^ (uint64_t)((N) + 37u + (S));                    \
    break;

#define ICACHE_C16(b, S)                                                \
  ALU_CASE(b + 0, S) ALU_CASE(b + 1, S) ALU_CASE(b + 2, S)             \
  ALU_CASE(b + 3, S) ALU_CASE(b + 4, S) ALU_CASE(b + 5, S)             \
  ALU_CASE(b + 6, S) ALU_CASE(b + 7, S) ALU_CASE(b + 8, S)             \
  ALU_CASE(b + 9, S) ALU_CASE(b + 10, S) ALU_CASE(b + 11, S)           \
  ALU_CASE(b + 12, S) ALU_CASE(b + 13, S) ALU_CASE(b + 14, S)          \
  ALU_CASE(b + 15, S)

#define ICACHE_ALL_CASES(S)                                             \
  ICACHE_C16(0, S) ICACHE_C16(16, S) ICACHE_C16(32, S)                 \
  ICACHE_C16(48, S) ICACHE_C16(64, S) ICACHE_C16(80, S)                \
  ICACHE_C16(96, S) ICACHE_C16(112, S) ICACHE_C16(128, S)              \
  ICACHE_C16(144, S) ICACHE_C16(160, S) ICACHE_C16(176, S)             \
  ICACHE_C16(192, S) ICACHE_C16(208, S) ICACHE_C16(224, S)             \
  ICACHE_C16(240, S)

__attribute__((noinline)) static uint64_t icache_fn0(uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_ALL_CASES(1u)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t icache_fn1(uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_ALL_CASES(2u)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t icache_fn2(uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICACHE_ALL_CASES(3u)
  }
  return acc;
}

#undef ICACHE_ALL_CASES
#undef ICACHE_C16
#undef ALU_CASE

BENCHMARK(contendedUse_8_to_100_icache, iters) {
  constexpr int posters = 8;
  constexpr int waiters = 100;
  constexpr size_t kRing = 4096;
  const uint32_t n = static_cast<uint32_t>(iters);

  LifoSemImpl<std::atomic> sem;

  static std::vector<uint32_t> chase;
  static std::vector<uint8_t> payload;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  BENCHMARK_SUSPEND {
    chase.resize(kRing);
    payload.resize(kRing);
    for (size_t i = 0; i < kRing; ++i) {
      chase[i] = static_cast<uint32_t>((i * 2654435761u + 12345u) % kRing);
      payload[i] = static_cast<uint8_t>(i * 31u + 7u);
    }

    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t acc = static_cast<uint64_t>(t) + 1u;
        size_t p = static_cast<size_t>(t) % kRing;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int j = 0; j < 24; ++j) {
            p = chase[p];
            uint8_t idx = static_cast<uint8_t>(payload[p] & 0xFF);
            switch (j % 3) {
              case 0:
                acc = icache_fn0(acc, idx);
                break;
              case 1:
                acc = icache_fn1(acc, idx);
                break;
              default:
                acc = icache_fn2(acc, idx);
                break;
            }
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
  folly::doNotOptimizeAway(sink.load());
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
