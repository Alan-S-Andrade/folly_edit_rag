#include <folly/portability/Asm.h>
#include <folly/synchronization/LifoSem.h>
#include <folly/synchronization/NativeSemaphore.h>

#include <folly/Benchmark.h>

#include <cstdint>

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
// Derived from contendedUse(8_to_100): same contended LifoSem traffic, but
// each post/wait turn is interleaved with a large, branchy, pointer-chase
// driven code footprint to raise the executed-instruction working set
// (frontend / L1i pressure) while keeping the synchronization pattern intact.
// ---------------------------------------------------------------------------

static const std::vector<uint32_t>& icacheChaseRing() {
  static const std::vector<uint32_t> ring = [] {
    const size_t N = 4096;
    std::vector<uint32_t> v(N);
    for (size_t i = 0; i < N; ++i) {
      v[i] = (uint32_t)((i * 2654435761u + 1u) & (N - 1));
    }
    return v;
  }();
  return ring;
}

#define ICASE(n)                                                              \
  case (n): {                                                                 \
    acc += UINT64_C(0x9E3779B97F4A7C15) ^ ((uint64_t)(n) + 1u);              \
    acc ^= (acc << 13) ^ ((uint64_t)(n) * UINT64_C(0x100000001B3));          \
    acc += (acc >> 7) + ((uint64_t)(n) * UINT64_C(2654435761));              \
    acc *= (UINT64_C(1) | ((uint64_t)(n) << 1));                             \
    acc ^= (acc >> 17) + ((uint64_t)(n) * UINT64_C(40503) + CASE_SALT);      \
    acc += UINT64_C(0xFF51AFD7ED558CCD) - (uint64_t)(n);                     \
    acc ^= (acc << 5) ^ ((uint64_t)(n) * UINT64_C(0xC2B2AE3D27D4EB4F));      \
    acc += (acc >> 11) + ((uint64_t)(n) * UINT64_C(374761393));              \
    acc *= (UINT64_C(3) | ((uint64_t)(n) << 2));                             \
    acc ^= (acc >> 23) + ((uint64_t)(n) * UINT64_C(668265263) + CASE_SALT);  \
    acc += UINT64_C(0xD6E8FEB86659FD93) ^ ((uint64_t)(n) << 3);              \
    acc ^= (acc << 9) + ((uint64_t)(n) * UINT64_C(2246822519));              \
    break;                                                                    \
  }

#define ICACHE_R4(n) ICASE(n) ICASE((n) + 1) ICASE((n) + 2) ICASE((n) + 3)
#define ICACHE_R16(n) \
  ICACHE_R4(n) ICACHE_R4((n) + 4) ICACHE_R4((n) + 8) ICACHE_R4((n) + 12)
#define ICACHE_R64(n) \
  ICACHE_R16(n) ICACHE_R16((n) + 16) ICACHE_R16((n) + 32) ICACHE_R16((n) + 48)
#define ICACHE_R256 \
  ICACHE_R64(0) ICACHE_R64(64) ICACHE_R64(128) ICACHE_R64(192)

#define CASE_SALT UINT64_C(0xA17B91C3D5E7F029)
__attribute__((noinline)) static uint64_t icache_fn0(uint64_t acc, uint8_t idx) {
  switch (idx) { ICACHE_R256 }
  return acc;
}
#undef CASE_SALT

#define CASE_SALT UINT64_C(0xB28C72D4E6F8013A)
__attribute__((noinline)) static uint64_t icache_fn1(uint64_t acc, uint8_t idx) {
  switch (idx) { ICACHE_R256 }
  return acc;
}
#undef CASE_SALT

#define CASE_SALT UINT64_C(0xC39D53E5F7091B4B)
__attribute__((noinline)) static uint64_t icache_fn2(uint64_t acc, uint8_t idx) {
  switch (idx) { ICACHE_R256 }
  return acc;
}
#undef CASE_SALT

#undef ICACHE_R256
#undef ICACHE_R64
#undef ICACHE_R16
#undef ICACHE_R4
#undef ICASE

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  const std::vector<uint32_t>& ring = icacheChaseRing();
  const uint32_t mask = (uint32_t)ring.size() - 1u;

  LifoSemImpl<std::atomic> sem;
  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &ring] {
        uint64_t acc = 0x1234567ull + (uint64_t)t;
        uint32_t p = (uint32_t)t & mask;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = ring[p & mask];
          uint8_t idx = (uint8_t)(p & 0xFF);
          switch (i % 3) {
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
        folly::doNotOptimizeAway(acc);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &ring] {
        uint64_t acc = 0x89ABCDEull + (uint64_t)t;
        uint32_t p = (uint32_t)(t * 7 + 1) & mask;
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          p = ring[p & mask];
          uint8_t idx = (uint8_t)(p & 0xFF);
          switch (i % 3) {
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
