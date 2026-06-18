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
// Derived benchmark: contendedUseIcache
//
// Same contended LifoSem producer/consumer structure as contendedUse, but each
// completed wait/post drives a pointer-chase load whose low byte selects one of
// 256 switch cases inside one of three large noinline functions (rotated by a
// running counter). This deliberately inflates the executed instruction
// working set to raise L1 i-cache pressure relative to the lean reference loop.
// ---------------------------------------------------------------------------

// 14 ALU ops per case; the (S) salt keeps constants unique across the three
// functions and (K) keeps them unique across the 256 cases, defeating compiler
// deduplication of case bodies.
#define ALU_CASE(S, K)                                       \
  case (K): {                                                \
    acc += (uint64_t)((K)*2654435761u + (S)*7u + 1u);        \
    acc ^= (uint64_t)((K)*40503u + (S)*3u + 7u);             \
    acc *= (uint64_t)((((K) | 1u)) + (S));                   \
    acc += (uint64_t)((K)*2246822519u + (S));                \
    acc ^= (uint64_t)(((K) << 3) + (S)*5u + 13u);            \
    acc *= (uint64_t)(((K)*31u + 101u) | 1u);                \
    acc += (uint64_t)((~(unsigned)(K)) + (S)*11u + 17u);     \
    acc ^= (uint64_t)((K)*19u + (S)*2u + 5u);                \
    acc *= (uint64_t)((((K) + 3u + (S)) | 1u));              \
    acc += (uint64_t)((K) ^ (0xABu + (S)));                  \
    acc ^= (uint64_t)((K)*13u + (S)*9u + 29u);               \
    acc *= (uint64_t)((((K) | 0x80u)) + (S));                \
    acc += (uint64_t)((K)*7u + (S)*4u + 23u);                \
    acc ^= (uint64_t)((K)*17u + (S)*6u + 31u);               \
    break;                                                   \
  }

#define C4(S, B) ALU_CASE(S, B + 0) ALU_CASE(S, B + 1) ALU_CASE(S, B + 2) ALU_CASE(S, B + 3)
#define C16(S, B) C4(S, B) C4(S, B + 4) C4(S, B + 8) C4(S, B + 12)
#define C64(S, B) C16(S, B) C16(S, B + 16) C16(S, B + 32) C16(S, B + 48)
#define C256(S) C64(S, 0) C64(S, 64) C64(S, 128) C64(S, 192)

__attribute__((noinline)) static uint64_t switchWorkA(uint8_t sw, uint64_t acc) {
  switch (sw) {
    C256(1u)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t switchWorkB(uint8_t sw, uint64_t acc) {
  switch (sw) {
    C256(2u)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t switchWorkC(uint8_t sw, uint64_t acc) {
  switch (sw) {
    C256(3u)
  }
  return acc;
}

#undef C256
#undef C64
#undef C16
#undef C4
#undef ALU_CASE

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::vector<uint32_t> chase;

  BENCHMARK_SUSPEND {
    const uint32_t kChase = 4096;
    chase.resize(kChase);
    for (uint32_t i = 0; i < kChase; ++i) {
      chase[i] = (uint32_t)((i * 2654435761u + 1u) % kChase);
    }

    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = (uint64_t)t + 1u;
        uint32_t p = (uint32_t)t % (uint32_t)chase.size();
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p];
          uint8_t idx = (uint8_t)(p & 0xFFu);
          switch (i % 3u) {
            case 0:
              acc = switchWorkA(idx, acc);
              break;
            case 1:
              acc = switchWorkB(idx, acc);
              break;
            default:
              acc = switchWorkC(idx, acc);
              break;
          }
        }
        folly::doNotOptimizeAway(acc);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &chase] {
        while (!go.load()) {
          std::this_thread::yield();
        }
        uint64_t acc = (uint64_t)t + 7u;
        uint32_t p = ((uint32_t)t * 7u + 3u) % (uint32_t)chase.size();
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          p = chase[p];
          uint8_t idx = (uint8_t)(p & 0xFFu);
          switch (i % 3u) {
            case 0:
              acc = switchWorkB(idx, acc);
              break;
            case 1:
              acc = switchWorkC(idx, acc);
              break;
            default:
              acc = switchWorkA(idx, acc);
              break;
          }
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
