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

// ---------------------------------------------------------------------------
// Derived benchmark: contendedUseIcache(8_to_100)
//
// Same contended LifoSem post/wait shape as contendedUse(8_to_100), but each
// waiter thread folds a large, deduplication-resistant code footprint into the
// hot path to widen the executed instruction working set (raise L1i MPKI and
// lower IPC).  The footprint is three noinline 256-case switch functions, each
// case running 14 ALU ops on a local accumulator using unique literal
// constants, dispatched from a pointer-chase load and rotated with j % 3.
// ---------------------------------------------------------------------------

#define LSB_CASE(c, salt)                                  \
  case (c): {                                              \
    a += (uint32_t)((c) * 2654435761u + (salt) + 0x01u);  \
    a ^= (a << 13) ^ (uint32_t)((c) + 1u);                \
    a *= (uint32_t)((c) | 1u);                             \
    a += (uint32_t)(0x85ebca6bu ^ ((c) * 3u) ^ (salt));   \
    a ^= a >> 7;                                           \
    a *= (uint32_t)(0xc2b2ae35u + ((c) << 1));            \
    a += (uint32_t)((c) * 40503u + (salt));               \
    a ^= (a << 5) ^ (uint32_t)((c) + 7u);                 \
    a *= (uint32_t)((c) * 2u + 3u);                        \
    a += (uint32_t)(0x27d4eb2fu ^ (c) ^ (salt));          \
    a ^= a >> 11;                                          \
    a *= (uint32_t)((c) | 5u);                             \
    a += (uint32_t)((c) * 7919u + (salt));                \
    a ^= (uint32_t)(((c) << 3) + 0x165667b1u + (salt));   \
    break;                                                 \
  }

#define LSB_C4(b, s) \
  LSB_CASE((b) + 0, s) LSB_CASE((b) + 1, s) LSB_CASE((b) + 2, s) LSB_CASE((b) + 3, s)
#define LSB_C16(b, s) \
  LSB_C4((b) + 0, s) LSB_C4((b) + 4, s) LSB_C4((b) + 8, s) LSB_C4((b) + 12, s)
#define LSB_C64(b, s) \
  LSB_C16((b) + 0, s) LSB_C16((b) + 16, s) LSB_C16((b) + 32, s) LSB_C16((b) + 48, s)
#define LSB_C256(s) \
  LSB_C64(0, s) LSB_C64(64, s) LSB_C64(128, s) LSB_C64(192, s)

__attribute__((noinline)) static uint32_t lifoSemSwitch0(uint32_t a, uint8_t idx) {
  switch (idx) {
    LSB_C256(0x1234567u)
  }
  return a;
}

__attribute__((noinline)) static uint32_t lifoSemSwitch1(uint32_t a, uint8_t idx) {
  switch (idx) {
    LSB_C256(0x89abcdefu)
  }
  return a;
}

__attribute__((noinline)) static uint32_t lifoSemSwitch2(uint32_t a, uint8_t idx) {
  switch (idx) {
    LSB_C256(0xfeedface0u)
  }
  return a;
}

#undef LSB_C256
#undef LSB_C64
#undef LSB_C16
#undef LSB_C4
#undef LSB_CASE

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static const std::vector<uint32_t> chase = [] {
    std::vector<uint32_t> v(4096);
    uint32_t x = 0x9e3779b9u;
    for (size_t i = 0; i < v.size(); ++i) {
      x = x * 1664525u + 1013904223u;
      v[i] = x;
    }
    return v;
  }();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint32_t a = 0x85ebca6bu ^ static_cast<uint32_t>(t);
        uint32_t p = static_cast<uint32_t>(t) * 2654435761u + 1u;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & (chase.size() - 1)];
          uint8_t idx = static_cast<uint8_t>(p & 0xFFu);
          switch ((i / static_cast<uint32_t>(waiters)) % 3u) {
            case 0:
              a = lifoSemSwitch0(a, idx);
              break;
            case 1:
              a = lifoSemSwitch1(a, idx);
              break;
            default:
              a = lifoSemSwitch2(a, idx);
              break;
          }
        }
        folly::doNotOptimizeAway(a);
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
