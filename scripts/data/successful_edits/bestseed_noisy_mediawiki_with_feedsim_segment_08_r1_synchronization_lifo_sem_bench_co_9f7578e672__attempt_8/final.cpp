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

// Icache-thrashing helper: 256-case switch with many unique-literal ALU ops.
// Three rotated noinline variants force a large hot instruction footprint while
// a pointer-chase index keeps the front-end stalled and IPC low.
#define ICASE(n, salt)                                                    \
  case (n): {                                                             \
    acc += (uint64_t)((n) * 2654435761u + (salt) + 1u);                  \
    acc ^= (uint64_t)((n) * 40499u + (salt) * 7u + 7u);                  \
    acc *= ((uint64_t)((n) * 2u + (salt)) | 1u);                         \
    acc += (uint64_t)((n) ^ ((salt) + 0xA5u));                           \
    acc ^= (acc >> 13);                                                  \
    acc -= (uint64_t)((n) * 7u + (salt) + 3u);                           \
    acc *= ((uint64_t)((n) * 3u + (salt)) | 1u);                         \
    acc += (uint64_t)((n) * 131u + (salt) + 17u);                        \
    acc ^= (uint64_t)(((n) + (salt)) << 1u);                             \
    acc += (acc << 5);                                                   \
    acc ^= (uint64_t)((n) * 19u + (salt) + 23u);                         \
    acc *= ((uint64_t)((n) + (salt) | 1u));                              \
    acc += (uint64_t)((n) * 5u + (salt) + 11u);                          \
    acc ^= (acc >> 9);                                                   \
    acc += (uint64_t)((n) * 251u + (salt) + 29u);                        \
    acc ^= (uint64_t)((n) * 13u + (salt) + 31u);                         \
    acc *= ((uint64_t)((n) * 7u + (salt)) | 3u);                         \
    acc -= (uint64_t)((n) * 37u + (salt) + 41u);                         \
    acc ^= (acc << 11);                                                  \
    acc += (uint64_t)((n) * 97u + (salt) + 43u);                         \
    break;                                                               \
  }

#define ICASE16(b, salt)                                                  \
  ICASE((b) + 0, salt) ICASE((b) + 1, salt) ICASE((b) + 2, salt)         \
  ICASE((b) + 3, salt) ICASE((b) + 4, salt) ICASE((b) + 5, salt)         \
  ICASE((b) + 6, salt) ICASE((b) + 7, salt) ICASE((b) + 8, salt)         \
  ICASE((b) + 9, salt) ICASE((b) + 10, salt) ICASE((b) + 11, salt)       \
  ICASE((b) + 12, salt) ICASE((b) + 13, salt) ICASE((b) + 14, salt)      \
  ICASE((b) + 15, salt)

#define ICASE256(salt)                                                    \
  ICASE16(0, salt) ICASE16(16, salt) ICASE16(32, salt) ICASE16(48, salt) \
  ICASE16(64, salt) ICASE16(80, salt) ICASE16(96, salt)                  \
  ICASE16(112, salt) ICASE16(128, salt) ICASE16(144, salt)               \
  ICASE16(160, salt) ICASE16(176, salt) ICASE16(192, salt)               \
  ICASE16(208, salt) ICASE16(224, salt) ICASE16(240, salt)

static __attribute__((noinline)) uint64_t icacheSwitchA(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICASE256(0x11u)
  }
  return acc;
}

static __attribute__((noinline)) uint64_t icacheSwitchB(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICASE256(0x53u)
  }
  return acc;
}

static __attribute__((noinline)) uint64_t icacheSwitchC(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
    ICASE256(0x97u)
  }
  return acc;
}

#undef ICASE256
#undef ICASE16
#undef ICASE

static const std::vector<uint8_t>& icacheChase() {
  static const std::vector<uint8_t> v = [] {
    std::vector<uint8_t> tmp(4096);
    uint32_t x = 2463534242u;
    for (auto& e : tmp) {
      x = x * 1664525u + 1013904223u;
      e = static_cast<uint8_t>((x >> 16) & 0xFFu);
    }
    return tmp;
  }();
  return v;
}

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  const auto& chase = icacheChase();
  const uint32_t mask = static_cast<uint32_t>(chase.size() - 1);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) * 2654435761u + 1u;
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = (p + acc + chase[p & mask]) & mask;
          uint8_t idx = chase[p];
          switch (j++ % 3) {
            case 0:
              acc = icacheSwitchA(acc, idx);
              break;
            case 1:
              acc = icacheSwitchB(acc, idx);
              break;
            default:
              acc = icacheSwitchC(acc, idx);
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
