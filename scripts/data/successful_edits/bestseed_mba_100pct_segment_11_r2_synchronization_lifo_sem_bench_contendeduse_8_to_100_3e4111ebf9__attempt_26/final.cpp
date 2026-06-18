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
// Derived benchmark: contendedUseSwitch
//
// Same LifoSem contention skeleton as contendedUse, but each post/wait turn
// runs a large, branch-heavy switch dispatched off a pointer-chase payload.
// Three noinline 256-case switch functions are rotated with j%3 to inflate the
// executed instruction working set (frontend / L1i footprint) without changing
// the synchronization behavior.  This widens hot code footprint to bring the
// L1-icache-load-miss MPKI up toward the reference target.
// ---------------------------------------------------------------------------
namespace {

#define ICACHE_ALU(acc, k)                                       \
  do {                                                           \
    acc += (uint64_t)(k) * 0x9E3779B97F4A7C15ull + SALT;         \
    acc ^= (acc >> 7) + (uint64_t)(k);                           \
    acc *= 0xD1B54A32D192ED03ull ^ (uint64_t)(k);                \
    acc += (uint64_t)(k) ^ (0xCAFEBABEull + SALT);               \
    acc ^= (acc << 13) + (uint64_t)(k);                          \
    acc -= (uint64_t)(k) * 0x100000001B3ull;                     \
    acc *= 0x2545F4914F6CDD1Dull + (uint64_t)(k);                \
    acc ^= (uint64_t)(k) * 7u + (0xDEADBEEFull ^ SALT);          \
    acc += (acc >> 11) ^ (uint64_t)(k);                          \
    acc ^= (uint64_t)(k) << 3;                                   \
    acc *= 0xFF51AFD7ED558CCDull ^ (uint64_t)(k);                \
    acc += (uint64_t)(k) * 3u + (0xABCDEFull + SALT);            \
    acc ^= (acc >> 17) + (uint64_t)(k);                          \
    acc += (uint64_t)(k) * 0xC2B2AE3D27D4EB4Full;                \
  } while (0)

#define ICASE(N) \
  case (N):      \
    ICACHE_ALU(acc, (N)); \
    break;
#define IC4(N) ICASE(N) ICASE((N) + 1) ICASE((N) + 2) ICASE((N) + 3)
#define IC16(N) IC4(N) IC4((N) + 4) IC4((N) + 8) IC4((N) + 12)
#define IC64(N) IC16(N) IC16((N) + 16) IC16((N) + 32) IC16((N) + 48)
#define IC256 IC64(0) IC64(64) IC64(128) IC64(192)

#define SALT 0x1111111111111111ull
__attribute__((noinline)) static uint64_t icacheSwitch0(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    IC256
  }
  return acc;
}
#undef SALT

#define SALT 0x2222222222222222ull
__attribute__((noinline)) static uint64_t icacheSwitch1(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    IC256
  }
  return acc;
}
#undef SALT

#define SALT 0x3333333333333333ull
__attribute__((noinline)) static uint64_t icacheSwitch2(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    IC256
  }
  return acc;
}
#undef SALT

#undef IC256
#undef IC64
#undef IC16
#undef IC4
#undef ICASE
#undef ICACHE_ALU

} // namespace

static void contendedUseSwitch(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::vector<uint32_t> chase;

  BENCHMARK_SUSPEND {
    chase.resize(4096);
    for (size_t i = 0; i < chase.size(); ++i) {
      chase[i] = (uint32_t)((i * 2654435761u + 12345u) & 4095u);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = (uint64_t)t + 1;
        uint32_t p = (uint32_t)t;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & 4095];
          uint8_t idx = (uint8_t)(p ^ (acc >> 3));
          switch ((i / (uint32_t)waiters) % 3) {
            case 0:
              acc = icacheSwitch0(acc, idx);
              break;
            case 1:
              acc = icacheSwitch1(acc, idx);
              break;
            default:
              acc = icacheSwitch2(acc, idx);
              break;
          }
        }
        folly::doNotOptimizeAway(acc);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &chase] {
        uint64_t acc = (uint64_t)t + 100;
        uint32_t p = (uint32_t)t + 7;
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          p = chase[p & 4095];
          uint8_t idx = (uint8_t)(p ^ (acc >> 5));
          switch ((i / (uint32_t)posters) % 3) {
            case 0:
              acc = icacheSwitch0(acc, idx);
              break;
            case 1:
              acc = icacheSwitch1(acc, idx);
              break;
            default:
              acc = icacheSwitch2(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUseSwitch, 8_to_100, 8, 100)

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
