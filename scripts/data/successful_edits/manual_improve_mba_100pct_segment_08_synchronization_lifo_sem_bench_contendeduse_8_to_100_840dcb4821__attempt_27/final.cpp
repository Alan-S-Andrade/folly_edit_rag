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
// Derived from-scratch benchmark: contendedUseHot(8_to_100).
//
// Same LifoSem contention skeleton as contendedUse, but each woken waiter
// also rotates through three large, noinline 256-case switch functions whose
// case index comes from a pointer-chase load (payload & 0xFF). This blows up
// the executed instruction working set to raise L1-icache-load-misses_MPKI
// and pull IPC back down toward the frontend-bound target. Every case uses
// unique literal constants (derived from the case number plus a per-function
// salt) so the compiler cannot deduplicate the bodies.
// ---------------------------------------------------------------------------
namespace {

// 14 ALU ops per case, all constants derived from (n) and salt (s) so each
// expansion is unique.
#define ALU14(n, s)                              \
  acc += 0x1000u + (uint64_t)(n) + (s);          \
  acc ^= 0x2000u + (uint64_t)(n) * 3u;           \
  acc *= (2u * (uint32_t)(n) + 3u);              \
  acc += 0x3000u ^ ((uint64_t)(n) + (s));        \
  acc ^= 0x4000u + (uint64_t)(n) * 5u;           \
  acc *= (2u * (uint32_t)(n) + 5u);              \
  acc += 0x5000u + (uint64_t)(n) * 7u;           \
  acc ^= 0x6000u + ((uint64_t)(n) + 11u);        \
  acc *= (2u * (uint32_t)(n) + 7u);              \
  acc += 0x7000u + (uint64_t)(n) * 13u;          \
  acc ^= 0x8000u + (uint64_t)(n) * 17u;          \
  acc *= (2u * (uint32_t)(n) + 9u);              \
  acc += 0x9000u ^ ((uint64_t)(n) * 19u);        \
  acc ^= 0xA000u + (uint64_t)(n) * 23u + (s);

#define ICASE(n, s) \
  case (n): {       \
    ALU14(n, s)     \
  } break;

#define REP4(M, n, s) M((n) + 0, s) M((n) + 1, s) M((n) + 2, s) M((n) + 3, s)
#define REP16(M, n, s)                                              \
  REP4(M, (n) + 0, s) REP4(M, (n) + 4, s) REP4(M, (n) + 8, s)       \
      REP4(M, (n) + 12, s)
#define REP64(M, n, s)                                              \
  REP16(M, (n) + 0, s) REP16(M, (n) + 16, s) REP16(M, (n) + 32, s)  \
      REP16(M, (n) + 48, s)
#define REP256(M, n, s)                                                \
  REP64(M, (n) + 0, s) REP64(M, (n) + 64, s) REP64(M, (n) + 128, s)    \
      REP64(M, (n) + 192, s)

__attribute__((noinline)) static uint64_t icacheSwitchA(
    uint64_t acc, uint32_t payload) {
  switch (payload & 0xFFu) {
    REP256(ICASE, 0, 0x1111u)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheSwitchB(
    uint64_t acc, uint32_t payload) {
  switch (payload & 0xFFu) {
    REP256(ICASE, 0, 0x2222u)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheSwitchC(
    uint64_t acc, uint32_t payload) {
  switch (payload & 0xFFu) {
    REP256(ICASE, 0, 0x3333u)
  }
  return acc;
}

#undef REP256
#undef REP64
#undef REP16
#undef REP4
#undef ICASE
#undef ALU14

} // namespace

static void contendedUseHot(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr size_t kRing = 4096;
  std::vector<uint32_t> ring(kRing);
  for (size_t i = 0; i < kRing; ++i) {
    ring[i] = static_cast<uint32_t>((i * 2654435761u) ^ (i << 7));
  }

  std::atomic<uint64_t> sink{0};

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &ring, &sink] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t idx = static_cast<uint32_t>(t * 1009u) & (kRing - 1);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          uint32_t payload = ring[idx];
          idx = payload & (kRing - 1);
          switch (j % 3) {
            case 0:
              acc = icacheSwitchA(acc, payload);
              break;
            case 1:
              acc = icacheSwitchB(acc, payload);
              break;
            default:
              acc = icacheSwitchC(acc, payload);
              break;
          }
          ++j;
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
BENCHMARK_NAMED_PARAM(contendedUseHot, 8_to_100, 8, 100)
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
