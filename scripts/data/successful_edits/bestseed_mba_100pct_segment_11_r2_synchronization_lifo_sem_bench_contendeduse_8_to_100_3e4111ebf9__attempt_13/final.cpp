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
// Derived from-scratch benchmark: contendedUseICache.
//
// Goal: keep the LifoSem contention pattern of contendedUse, but inflate the
// executed instruction working set so that L1 i-cache pressure matches the
// target profile. Three noinline functions each hold a 256-case switch with
// ~14 unique ALU ops per case; the hot loop rotates among them (j % 3) and
// indexes the switch with payload bytes from a pointer-chase load. The unique
// per-case constants prevent compiler case/function deduplication, so the
// rotation thrashes a large code footprint.
// ---------------------------------------------------------------------------

#define ICACHE_ALU(n)                                  \
  case (n): {                                          \
    a += (uint64_t)(n) * 2654435761u + (SALT);         \
    a ^= a >> 13;                                       \
    a *= (0xff51afd7ed558ccdULL + (uint64_t)(n));       \
    a += (uint64_t)(n) * 40503u + (SALT);               \
    a ^= (uint64_t)(n) * 0x27d4eb2fULL;                 \
    a -= (a << 7);                                       \
    a *= (0xc2b2ae3d27d4eb4fULL ^ (uint64_t)(n));       \
    a ^= a >> 17;                                        \
    a += ((uint64_t)(n) ^ 0xdeadbeefULL) + (SALT);      \
    a *= 0x165667b19e3779f9ULL;                          \
    a ^= (uint64_t)(n) * 31u + 7u;                       \
    a += (a << 3);                                        \
    a ^= a >> 11;                                         \
    a *= (0x85ebca6bULL + (uint64_t)(n) + (SALT));        \
    break;                                                \
  }

#define ICACHE_C1(n) ICACHE_ALU(n)
#define ICACHE_C4(n) \
  ICACHE_C1((n) + 0) ICACHE_C1((n) + 1) ICACHE_C1((n) + 2) ICACHE_C1((n) + 3)
#define ICACHE_C16(n) \
  ICACHE_C4((n) + 0) ICACHE_C4((n) + 4) ICACHE_C4((n) + 8) ICACHE_C4((n) + 12)
#define ICACHE_C64(n)                                                  \
  ICACHE_C16((n) + 0) ICACHE_C16((n) + 16) ICACHE_C16((n) + 32)        \
  ICACHE_C16((n) + 48)
#define ICACHE_C256 \
  ICACHE_C64(0) ICACHE_C64(64) ICACHE_C64(128) ICACHE_C64(192)

__attribute__((noinline)) static uint64_t icacheMix0(uint64_t a, uint8_t idx) {
#define SALT 0x9e3779b97f4a7c15ULL
  switch (idx) { ICACHE_C256 }
#undef SALT
  return a;
}

__attribute__((noinline)) static uint64_t icacheMix1(uint64_t a, uint8_t idx) {
#define SALT 0xc2b2ae3d27d4eb4fULL
  switch (idx) { ICACHE_C256 }
#undef SALT
  return a;
}

__attribute__((noinline)) static uint64_t icacheMix2(uint64_t a, uint8_t idx) {
#define SALT 0x165667b19e3779f9ULL
  switch (idx) { ICACHE_C256 }
#undef SALT
  return a;
}

static void contendedUseICache(uint32_t n, int posters, int waiters) {
  constexpr size_t kBufSize = 1u << 12; // 4096 entries, 16 KiB

  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::vector<uint32_t> chase(kBufSize);

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kBufSize; ++i) {
      chase[i] = static_cast<uint32_t>(
          (i * 2654435761u + 1013904223u) & (kBufSize - 1));
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) & (kBufSize - 1);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & (kBufSize - 1)];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = icacheMix0(acc, idx);
              break;
            case 1:
              acc = icacheMix1(acc, idx);
              break;
            default:
              acc = icacheMix2(acc, idx);
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
        uint64_t acc = static_cast<uint64_t>(t) + 7;
        uint32_t p = static_cast<uint32_t>(t * 17) & (kBufSize - 1);
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          p = chase[p & (kBufSize - 1)];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = icacheMix1(acc, idx);
              break;
            case 1:
              acc = icacheMix2(acc, idx);
              break;
            default:
              acc = icacheMix0(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUseICache, 8_to_100, 8, 100)

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
