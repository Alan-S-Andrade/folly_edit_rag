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
// Derived benchmark: contendedUseHot(8_to_100)
//
// Same LifoSem contention skeleton as contendedUse, but each woken waiter
// drives a pointer-chase-indexed dispatch into one of three large
// noinline 256-case switch functions.  This deliberately inflates the hot
// instruction working set to raise L1i pressure (frontend-bound) and bring
// IPC down toward the target band.
// ---------------------------------------------------------------------------

#define LIFO_HOT_OPS(S, N)                                                  \
  acc += static_cast<uint64_t>((N) * 0x9e3779b97f4a7c15ull + (S) + 1u);     \
  acc ^= acc >> 7;                                                          \
  acc *= static_cast<uint64_t>((N) * 2u + (S) + 3u) | 1u;                   \
  acc += static_cast<uint64_t>(((N) ^ 0x5bd1e995u) + (S));                  \
  acc ^= acc << 11;                                                         \
  acc *= static_cast<uint64_t>((N) | 1u);                                   \
  acc += static_cast<uint64_t>((N) * 7u + (S) + 13u);                       \
  acc ^= acc >> 5;                                                          \
  acc *= static_cast<uint64_t>((N) * 3u + (S) + 1u) | 1u;                   \
  acc += static_cast<uint64_t>(((N) ^ 0xdeadbeefu) + (S));                  \
  acc ^= acc << 3;                                                          \
  acc *= static_cast<uint64_t>((N) + (S) + 0xabcu) | 1u;                    \
  acc += static_cast<uint64_t>((N) * 5u + (S) + 17u);                       \
  acc ^= acc >> 9;

#define LIFO_HC0(N) \
  case (N): {       \
    LIFO_HOT_OPS(0x11u, N) break;                                           \
  }
#define LIFO_HC1(N) \
  case (N): {       \
    LIFO_HOT_OPS(0x22u, N) break;                                           \
  }
#define LIFO_HC2(N) \
  case (N): {       \
    LIFO_HOT_OPS(0x33u, N) break;                                           \
  }

#define LIFO_R16(M, b)                                                      \
  M(b + 0) M(b + 1) M(b + 2) M(b + 3) M(b + 4) M(b + 5) M(b + 6) M(b + 7)   \
      M(b + 8) M(b + 9) M(b + 10) M(b + 11) M(b + 12) M(b + 13) M(b + 14)   \
          M(b + 15)
#define LIFO_R256(M)                                                        \
  LIFO_R16(M, 0)                                                            \
  LIFO_R16(M, 16) LIFO_R16(M, 32) LIFO_R16(M, 48) LIFO_R16(M, 64)           \
      LIFO_R16(M, 80) LIFO_R16(M, 96) LIFO_R16(M, 112) LIFO_R16(M, 128)     \
          LIFO_R16(M, 144) LIFO_R16(M, 160) LIFO_R16(M, 176)               \
              LIFO_R16(M, 192) LIFO_R16(M, 208) LIFO_R16(M, 224)            \
                  LIFO_R16(M, 240)

__attribute__((noinline)) static uint64_t lifoHotSwitch0(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFO_R256(LIFO_HC0)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) static uint64_t lifoHotSwitch1(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFO_R256(LIFO_HC1)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) static uint64_t lifoHotSwitch2(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFO_R256(LIFO_HC2)
    default:
      break;
  }
  return acc;
}

static uint32_t lifoChase[4096];
[[maybe_unused]] static const bool lifoChaseInit = [] {
  uint32_t x = 0x12345678u;
  for (uint32_t i = 0; i < 4096; ++i) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    lifoChase[i] = x & 4095u;
  }
  return true;
}();

static void contendedUseHot(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = 0x9e3779b9u + static_cast<uint32_t>(t);
        uint32_t p = (static_cast<uint32_t>(t) * 131u) & 4095u;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = lifoChase[p];
          uint8_t sw = static_cast<uint8_t>((lifoChase[p] ^ acc) & 0xFFu);
          switch (p % 3u) {
            case 0:
              acc = lifoHotSwitch0(acc, sw);
              break;
            case 1:
              acc = lifoHotSwitch1(acc, sw);
              break;
            default:
              acc = lifoHotSwitch2(acc, sw);
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
BENCHMARK_NAMED_PARAM(contendedUseHot, 8_to_100, 8, 100)

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
