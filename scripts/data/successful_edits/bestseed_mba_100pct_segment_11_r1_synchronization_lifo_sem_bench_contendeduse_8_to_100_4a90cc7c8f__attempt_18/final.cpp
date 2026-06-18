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
// Derived from-scratch experiment for contendedUse(8_to_100): the contended
// LifoSem traffic is preserved, but each completed wait drives a pointer-chase
// load that feeds a wide 256-case ALU switch. Three distinct noinline switch
// functions are rotated with (i % 3) to inflate the executed instruction
// working set (L1i footprint) without disturbing the original family.
// ---------------------------------------------------------------------------
namespace {

struct ChaseNode {
  uint32_t payload;
  uint32_t next;
};

// 14 ALU ops/case, unique literal constants per case to defeat dedup.
#define ICACHE_ALU(n, s)                            \
  case (n): {                                       \
    acc += (uint64_t)((n) * 7u + (s) + 1u);         \
    acc ^= (uint64_t)((n) * 3u + (s) + 5u);         \
    acc *= (uint64_t)(((n) | 1u) + (s));            \
    acc -= (uint64_t)((n) * 2u + (s) + 9u);         \
    acc ^= (uint64_t)((n) * 5u + (s) + 3u);         \
    acc += (uint64_t)((n) * 11u + (s) + 7u);        \
    acc *= (uint64_t)((((n) * 2u) | 1u) + (s));     \
    acc -= (uint64_t)((n) + (s) + 13u);             \
    acc ^= (uint64_t)((n) * 13u + (s) + 2u);        \
    acc += (uint64_t)((n) * 17u + (s) + 4u);        \
    acc *= (uint64_t)((((n) * 3u) | 1u) + (s));     \
    acc -= (uint64_t)((n) * 7u + (s) + 8u);         \
    acc ^= (uint64_t)((n) * 19u + (s) + 6u);        \
    acc += (uint64_t)((n) * 23u + (s) + 10u);       \
    break;                                          \
  }

#define ICACHE_A4(b, s)                                       \
  ICACHE_ALU((b) + 0, s) ICACHE_ALU((b) + 1, s)               \
  ICACHE_ALU((b) + 2, s) ICACHE_ALU((b) + 3, s)
#define ICACHE_A16(b, s)                                      \
  ICACHE_A4((b) + 0, s) ICACHE_A4((b) + 4, s)                 \
  ICACHE_A4((b) + 8, s) ICACHE_A4((b) + 12, s)
#define ICACHE_A64(b, s)                                      \
  ICACHE_A16((b) + 0, s) ICACHE_A16((b) + 16, s)              \
  ICACHE_A16((b) + 32, s) ICACHE_A16((b) + 48, s)
#define ICACHE_A256(s)                                        \
  ICACHE_A64(0, s) ICACHE_A64(64, s)                          \
  ICACHE_A64(128, s) ICACHE_A64(192, s)

__attribute__((noinline)) static uint64_t icacheSwitchA(
    uint64_t acc, unsigned idx) {
  switch (idx & 0xFFu) { ICACHE_A256(101u) }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheSwitchB(
    uint64_t acc, unsigned idx) {
  switch (idx & 0xFFu) { ICACHE_A256(211u) }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheSwitchC(
    uint64_t acc, unsigned idx) {
  switch (idx & 0xFFu) { ICACHE_A256(307u) }
  return acc;
}

#undef ICACHE_A256
#undef ICACHE_A64
#undef ICACHE_A16
#undef ICACHE_A4
#undef ICACHE_ALU

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static constexpr uint32_t kRing = 1024;
  std::vector<ChaseNode> ring(kRing);

  BENCHMARK_SUSPEND {
    for (uint32_t i = 0; i < kRing; ++i) {
      ring[i].payload = (i * 2654435761u) >> 13;
      ring[i].next = (i * 2654435761u + 7u) % kRing;
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &ring] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t idx = static_cast<uint32_t>(t) % kRing;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          idx = ring[idx].next;
          unsigned p = ring[idx].payload & 0xFFu;
          switch (i % 3) {
            case 0:
              acc = icacheSwitchA(acc, p);
              break;
            case 1:
              acc = icacheSwitchB(acc, p);
              break;
            default:
              acc = icacheSwitchC(acc, p);
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

} // namespace

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
