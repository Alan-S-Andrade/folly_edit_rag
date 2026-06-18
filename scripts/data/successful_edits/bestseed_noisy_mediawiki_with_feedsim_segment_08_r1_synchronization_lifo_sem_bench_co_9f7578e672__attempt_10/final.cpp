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
// Derived from contendedUse(8_to_100): same LifoSem contention pattern, but
// each worker also drives a large, distinct executed-instruction working set
// to raise L1 i-cache pressure (frontend-bound) toward the target counter
// profile. Three noinline 256-case switch blocks (each with unique literal
// constants per case) are rotated by j%3 and indexed by a pointer-chase load.
// ---------------------------------------------------------------------------

#define ALU_OPS(acc, k)                          \
  acc += (uint64_t)(k) * 2654435761ULL;          \
  acc ^= acc >> 13;                              \
  acc *= ((uint64_t)(k) | 1ULL);                 \
  acc += 0x9E3779B97F4A7C15ULL ^ (uint64_t)(k);  \
  acc ^= acc << 7;                               \
  acc -= (uint64_t)(k) * 40503ULL;               \
  acc ^= acc >> 17;                              \
  acc *= (uint64_t)((k) * 3u + 1u);              \
  acc += (uint64_t)(k) << 3;                     \
  acc ^= acc >> 11;                              \
  acc *= 0xC2B2AE3D27D4EB4FULL;                  \
  acc -= (uint64_t)(k);                          \
  acc ^= acc << 5;                               \
  acc += (uint64_t)((k) ^ 0xABCDu);

#define CASE(k, salt) \
  case (k): {         \
    ALU_OPS(acc, ((k) + (salt)))                 \
  } break;
#define C4(n, s) CASE(n, s) CASE(n + 1, s) CASE(n + 2, s) CASE(n + 3, s)
#define C16(n, s) C4(n, s) C4(n + 4, s) C4(n + 8, s) C4(n + 12, s)
#define C64(n, s) C16(n, s) C16(n + 16, s) C16(n + 32, s) C16(n + 48, s)
#define C256(n, s) C64(n, s) C64(n + 64, s) C64(n + 128, s) C64(n + 192, s)

#define DEFINE_ICACHE_BLOCK(name, salt)                                  \
  __attribute__((noinline)) static uint64_t name(                       \
      uint64_t acc, uint32_t idx) {                                      \
    switch (idx & 0xFFu) {                                              \
      C256(0, salt)                                                      \
    }                                                                    \
    return acc;                                                          \
  }

DEFINE_ICACHE_BLOCK(icacheBlock0, 0)
DEFINE_ICACHE_BLOCK(icacheBlock1, 101)
DEFINE_ICACHE_BLOCK(icacheBlock2, 202)

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static const uint32_t kChaseSize = 1024; // power of two
  static const int kInnerOps = 24;

  auto icacheWork = [](std::vector<uint32_t>& chase, uint64_t& acc,
                       uint32_t& p) {
    for (int j = 0; j < kInnerOps; ++j) {
      p = chase[p & (kChaseSize - 1)];
      uint32_t idx = p & 0xFFu;
      switch (j % 3) {
        case 0:
          acc = icacheBlock0(acc, idx);
          break;
        case 1:
          acc = icacheBlock1(acc, idx);
          break;
        default:
          acc = icacheBlock2(acc, idx);
          break;
      }
    }
  };

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        std::vector<uint32_t> chase(kChaseSize);
        for (uint32_t i = 0; i < kChaseSize; ++i) {
          chase[i] = (i * 2654435761u + (uint32_t)t * 40503u + 7u) %
              kChaseSize;
        }
        uint64_t acc = (uint64_t)t + 1;
        uint32_t p = (uint32_t)t;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          icacheWork(chase, acc, p);
        }
        folly::doNotOptimizeAway(acc);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go] {
        std::vector<uint32_t> chase(kChaseSize);
        for (uint32_t i = 0; i < kChaseSize; ++i) {
          chase[i] = (i * 2246822519u + (uint32_t)t * 32779u + 11u) %
              kChaseSize;
        }
        uint64_t acc = (uint64_t)t + 31;
        uint32_t p = (uint32_t)t + 3;
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          icacheWork(chase, acc, p);
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
