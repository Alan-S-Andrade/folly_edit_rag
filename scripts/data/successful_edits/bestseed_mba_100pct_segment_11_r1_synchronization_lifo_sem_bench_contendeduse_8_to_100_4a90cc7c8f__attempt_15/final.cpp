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
// Derived from-scratch benchmark: a frontend-bound (I-cache thrashing) variant
// of the contended LifoSem use pattern.  Each woken waiter performs a
// pointer-chase load and then dispatches into one of three large
// __attribute__((noinline)) 256-case switch routines, rotating among them so
// that the executed instruction working set greatly exceeds the L1 icache.
// This intentionally lifts L1-icache-load-misses_MPKI and lowers IPC relative
// to the lean reference body, while keeping the same threading shape.
// ---------------------------------------------------------------------------
namespace {

struct ChaseNode {
  uint64_t payload;
  uint32_t next;
};

constexpr uint32_t kChaseRing = 8192;

std::vector<ChaseNode> makeChaseRing() {
  std::vector<ChaseNode> r(kChaseRing);
  uint32_t x = 1u;
  for (uint32_t i = 0; i < kChaseRing; ++i) {
    x = x * 1664525u + 1013904223u;
    r[i].payload = static_cast<uint64_t>(x) * 2654435761ull;
    r[i].next = x % kChaseRing;
  }
  return r;
}

const std::vector<ChaseNode> gChaseRing = makeChaseRing();

// 14 ALU ops per case, unique literal constants derived from the case index
// (n) and a per-function salt (s) to defeat compiler deduplication.
#define ICACHE_OP(n, s)                                  \
  case (n): {                                            \
    acc += static_cast<uint64_t>((n) * 2u + 1u + (s));   \
    acc ^= static_cast<uint64_t>((n) * 3u + 7u + (s));   \
    acc *= static_cast<uint64_t>(((n) | 1u) + (s));      \
    acc += static_cast<uint64_t>((n) ^ 0x5Au);           \
    acc -= static_cast<uint64_t>((n) * 5u + 3u);         \
    acc ^= static_cast<uint64_t>(((n) << 1) | 1u);       \
    acc += static_cast<uint64_t>((n) * 7u + 11u + (s));  \
    acc *= static_cast<uint64_t>(((n) + 3u) | 1u);       \
    acc ^= static_cast<uint64_t>((n) * 11u + 13u);       \
    acc += static_cast<uint64_t>((n) * 13u + 17u + (s)); \
    acc -= static_cast<uint64_t>((n) ^ 0x3Cu);           \
    acc *= static_cast<uint64_t>(((n) + 5u) | 1u);       \
    acc += static_cast<uint64_t>((n) * 17u + 19u);       \
    acc ^= static_cast<uint64_t>((n) * 19u + 23u + (s)); \
    break;                                               \
  }

#define ICACHE_16(b, s)                                                 \
  ICACHE_OP((b) + 0, s) ICACHE_OP((b) + 1, s) ICACHE_OP((b) + 2, s)     \
      ICACHE_OP((b) + 3, s) ICACHE_OP((b) + 4, s) ICACHE_OP((b) + 5, s) \
          ICACHE_OP((b) + 6, s) ICACHE_OP((b) + 7, s)                   \
              ICACHE_OP((b) + 8, s) ICACHE_OP((b) + 9, s)               \
                  ICACHE_OP((b) + 10, s) ICACHE_OP((b) + 11, s)         \
                      ICACHE_OP((b) + 12, s) ICACHE_OP((b) + 13, s)     \
                          ICACHE_OP((b) + 14, s) ICACHE_OP((b) + 15, s)

#define ICACHE_256(s)                                               \
  ICACHE_16(0, s) ICACHE_16(16, s) ICACHE_16(32, s) ICACHE_16(48, s) \
      ICACHE_16(64, s) ICACHE_16(80, s) ICACHE_16(96, s)             \
          ICACHE_16(112, s) ICACHE_16(128, s) ICACHE_16(144, s)     \
              ICACHE_16(160, s) ICACHE_16(176, s) ICACHE_16(192, s) \
                  ICACHE_16(208, s) ICACHE_16(224, s) ICACHE_16(240, s)

__attribute__((noinline)) uint64_t icacheSwitchA(int idx, uint64_t acc) {
  switch (idx) { ICACHE_256(0x11u) }
  return acc;
}

__attribute__((noinline)) uint64_t icacheSwitchB(int idx, uint64_t acc) {
  switch (idx) { ICACHE_256(0x22u) }
  return acc;
}

__attribute__((noinline)) uint64_t icacheSwitchC(int idx, uint64_t acc) {
  switch (idx) { ICACHE_256(0x33u) }
  return acc;
}

#undef ICACHE_256
#undef ICACHE_16
#undef ICACHE_OP

} // namespace

static void icacheContended(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) % kChaseRing;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          uint64_t payload = gChaseRing[p].payload;
          p = gChaseRing[p].next;
          int idx = static_cast<int>(payload & 0xFFu);
          switch (i % 3) {
            case 0:
              acc = icacheSwitchA(idx, acc);
              break;
            case 1:
              acc = icacheSwitchB(idx, acc);
              break;
            default:
              acc = icacheSwitchC(idx, acc);
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
BENCHMARK_NAMED_PARAM(icacheContended, 8_to_100, 8, 100)

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
