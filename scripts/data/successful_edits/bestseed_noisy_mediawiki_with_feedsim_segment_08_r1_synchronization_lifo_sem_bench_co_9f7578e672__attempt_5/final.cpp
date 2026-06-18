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
// Frontend-bound (i-cache thrashing) companion to contendedUse(8_to_100).
//
// A large, unique-per-case switch body is replicated across three noinline
// functions; the hot loop rotates among them (i % 3) using a pointer-chase
// derived index. This intentionally inflates the executed instruction working
// set so that the L1 instruction cache miss rate (and the resulting frontend
// stalls) approximate the reference contended-semaphore workload.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kChaseSize = 4096; // power of two

#define ICASE(n, s)                                                      \
  case ((n)): {                                                          \
    acc ^= 0x9E3779B97F4A7C15ull * (uint64_t)((n) + (s));                \
    acc += (uint64_t)((uint32_t)((n) * 2654435761u) ^                    \
                      (uint32_t)((s) * 40503u));                         \
    acc *= (((uint64_t)(n) << 1) | 1ull);                                \
    acc -= (uint64_t)((s) + (n) * 7u + 3u);                              \
    acc ^= acc >> 7;                                                     \
    acc += (uint64_t)((n) * 11u + (s) * 17u);                            \
    acc *= 0x100000001B3ull;                                             \
    acc ^= (uint64_t)((uint32_t)(n) ^ (uint32_t)((s) << 3));             \
    acc += (uint64_t)((n) * 19u + 23u);                                  \
    acc *= (((uint64_t)(s)) | 1ull);                                     \
    acc ^= acc << 13;                                                    \
    acc += (uint64_t)((n) + (s) + 29u);                                  \
    break;                                                               \
  }

#define ICASE16(b, s)                                                    \
  ICASE((b) + 0, s) ICASE((b) + 1, s) ICASE((b) + 2, s) ICASE((b) + 3, s)   \
  ICASE((b) + 4, s) ICASE((b) + 5, s) ICASE((b) + 6, s) ICASE((b) + 7, s)   \
  ICASE((b) + 8, s) ICASE((b) + 9, s) ICASE((b) + 10, s) ICASE((b) + 11, s) \
  ICASE((b) + 12, s) ICASE((b) + 13, s) ICASE((b) + 14, s) ICASE((b) + 15, s)

#define ICASE256(s)                                                      \
  ICASE16(0, s) ICASE16(16, s) ICASE16(32, s) ICASE16(48, s)             \
  ICASE16(64, s) ICASE16(80, s) ICASE16(96, s) ICASE16(112, s)           \
  ICASE16(128, s) ICASE16(144, s) ICASE16(160, s) ICASE16(176, s)        \
  ICASE16(192, s) ICASE16(208, s) ICASE16(224, s) ICASE16(240, s)

__attribute__((noinline)) uint64_t icacheMixA(uint8_t idx, uint64_t acc) {
  switch (idx) {
    ICASE256(0x11)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) uint64_t icacheMixB(uint8_t idx, uint64_t acc) {
  switch (idx) {
    ICASE256(0x53)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) uint64_t icacheMixC(uint8_t idx, uint64_t acc) {
  switch (idx) {
    ICASE256(0xA7)
    default:
      break;
  }
  return acc;
}

#undef ICASE256
#undef ICASE16
#undef ICASE

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  std::vector<uint32_t> chase;

  BENCHMARK_SUSPEND {
    chase.resize(kChaseSize);
    for (size_t i = 0; i < kChaseSize; ++i) {
      chase[i] =
          (uint32_t)((i * 2654435761u + 1013904223u) % kChaseSize);
    }

    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink, &chase] {
        uint64_t acc = 0x12345678ull + (uint64_t)t;
        uint32_t p = (uint32_t)(t * 131u);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & (kChaseSize - 1)];
          uint8_t idx = (uint8_t)(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = icacheMixA(idx, acc);
              break;
            case 1:
              acc = icacheMixB(idx, acc);
              break;
            default:
              acc = icacheMixC(idx, acc);
              break;
          }
        }
        sink.fetch_add(acc, std::memory_order_relaxed);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &sink, &chase] {
        while (!go.load()) {
          std::this_thread::yield();
        }
        uint64_t acc = 0x9abcdef0ull + (uint64_t)t;
        uint32_t p = (uint32_t)(t * 977u);
        for (uint32_t i = t; i < n; i += posters) {
          p = chase[p & (kChaseSize - 1)];
          uint8_t idx = (uint8_t)(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = icacheMixA(idx, acc);
              break;
            case 1:
              acc = icacheMixB(idx, acc);
              break;
            default:
              acc = icacheMixC(idx, acc);
              break;
          }
          sem.post();
        }
        sink.fetch_add(acc, std::memory_order_relaxed);
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
