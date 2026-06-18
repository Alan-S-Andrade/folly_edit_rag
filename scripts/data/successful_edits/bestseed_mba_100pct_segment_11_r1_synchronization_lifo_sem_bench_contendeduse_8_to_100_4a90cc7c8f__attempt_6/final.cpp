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
// Derived frontend-bound variant of contendedUse(8_to_100).
//
// The reference benchmark has a very compact executed instruction working set,
// which produces an IPC that is too high and an L1i miss rate that is too low.
// To grow the hot code footprint we rotate among three noinline functions, each
// containing a full 256-case switch with a fixed number of ALU ops per case.
// Each case folds the (constant) case index into the accumulator, so the cases
// do not deduplicate and the three-function rotation thrashes the icache.
//
// L1i MPKI scales roughly linearly with ALU ops per case; 14 ops/case targets
// the desired ~15 MPKI while leaving IPC near the requested value.
// ---------------------------------------------------------------------------

#define LSB_ACC_OPS(x)                                  \
  do {                                                  \
    uint64_t k = static_cast<uint64_t>(x) + LSB_SALT;   \
    acc += k * 0x9e3779b97f4a7c15ULL;                   \
    acc ^= k + 0x1ULL;                                  \
    acc *= (k | 0x3ULL);                                \
    acc -= k ^ 0xa5a5a5a5a5a5a5a5ULL;                   \
    acc += (k << 7) + 0x11ULL;                          \
    acc ^= k * 0xff51afd7ed558ccdULL;                   \
    acc += (k >> 3) ^ 0x55ULL;                          \
    acc *= 0x100000001b3ULL;                            \
    acc ^= k - 0xdeadbeefULL;                           \
    acc += k * 0x7ULL;                                  \
    acc ^= (k << 11) | 0x1ULL;                          \
    acc *= (k | 0x5ULL);                                \
    acc += 0xcafef00dULL ^ k;                           \
    acc ^= k * 0x2545f4914f6cdd1dULL;                   \
  } while (0)

#define LSB_CASE1(n)  \
  case (n):           \
    LSB_ACC_OPS(n);   \
    break;
#define LSB_CASE4(n) \
  LSB_CASE1((n)) LSB_CASE1((n) + 1) LSB_CASE1((n) + 2) LSB_CASE1((n) + 3)
#define LSB_CASE16(n) \
  LSB_CASE4((n)) LSB_CASE4((n) + 4) LSB_CASE4((n) + 8) LSB_CASE4((n) + 12)
#define LSB_CASE64(n) \
  LSB_CASE16((n)) LSB_CASE16((n) + 16) LSB_CASE16((n) + 32) LSB_CASE16((n) + 48)
#define LSB_CASE256(n) \
  LSB_CASE64((n)) LSB_CASE64((n) + 64) LSB_CASE64((n) + 128) \
      LSB_CASE64((n) + 192)

#define LSB_SALT 0x1111111111111111ULL
__attribute__((noinline)) static uint64_t frontMixA(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LSB_CASE256(0)
  }
  return acc;
}
#undef LSB_SALT

#define LSB_SALT 0x2222222222222222ULL
__attribute__((noinline)) static uint64_t frontMixB(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LSB_CASE256(0)
  }
  return acc;
}
#undef LSB_SALT

#define LSB_SALT 0x3333333333333333ULL
__attribute__((noinline)) static uint64_t frontMixC(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LSB_CASE256(0)
  }
  return acc;
}
#undef LSB_SALT

#undef LSB_CASE256
#undef LSB_CASE64
#undef LSB_CASE16
#undef LSB_CASE4
#undef LSB_CASE1
#undef LSB_ACC_OPS

static void contendedUseFront(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          uint8_t idx = static_cast<uint8_t>((acc ^ i) & 0xFF);
          switch ((i / static_cast<uint32_t>(waiters)) % 3u) {
            case 0:
              acc = frontMixA(acc, idx);
              break;
            case 1:
              acc = frontMixB(acc, idx);
              break;
            default:
              acc = frontMixC(acc, idx);
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
        uint64_t acc = static_cast<uint64_t>(t) + 7;
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          uint8_t idx = static_cast<uint8_t>((acc ^ i) & 0xFF);
          switch ((i / static_cast<uint32_t>(posters)) % 3u) {
            case 0:
              acc = frontMixA(acc, idx);
              break;
            case 1:
              acc = frontMixB(acc, idx);
              break;
            default:
              acc = frontMixC(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUseFront, 8_to_100, 8, 100)
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
