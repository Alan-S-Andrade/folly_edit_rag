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
// Derived benchmark: contendedUseIcache
//
// Same contended LifoSem post/wait shape as contendedUse, but each woken
// waiter performs a pointer-chase load and dispatches into one of three large
// 256-case switches (rotated by j%3). This inflates the executed instruction
// working set to raise L1-icache-load-misses_MPKI (and thereby lower the
// frontend-bound IPC) without changing the synchronization behavior.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kChaseLen = 8192; // power of two

std::vector<uint32_t> makeChaseTable() {
  std::vector<uint32_t> v(kChaseLen);
  uint32_t x = 0x12345678u;
  for (size_t i = 0; i < kChaseLen; ++i) {
    x = x * 1664525u + 1013904223u;
    v[i] = x;
  }
  return v;
}

const std::vector<uint32_t> kChaseTable = makeChaseTable();

// 14 ALU ops per case with unique literal constants (per case index and per
// function SEED) so the compiler cannot deduplicate the case bodies.
#define ICACHE_CASE_OPS(n)                       \
  case (n):                                       \
    acc += (uint64_t)((n) * 7u + SEED + 1u);      \
    acc ^= (uint64_t)((n) * 3u + SEED + 2u);      \
    acc *= (uint64_t)(((n) | 1u) + SEED);         \
    acc += (uint64_t)((n) * 11u + SEED + 5u);     \
    acc -= (uint64_t)((n) * 13u + SEED + 7u);     \
    acc ^= (uint64_t)(((n) << 1) + SEED + 9u);    \
    acc *= (uint64_t)((((n) + 3u) | 1u) + SEED);  \
    acc += (uint64_t)((n) * 17u + SEED + 11u);    \
    acc ^= (uint64_t)((n) * 19u + SEED + 13u);    \
    acc -= (uint64_t)((n) * 23u + SEED + 15u);    \
    acc *= (uint64_t)((((n) + 5u) | 1u) + SEED);  \
    acc += (uint64_t)((n) * 29u + SEED + 17u);    \
    acc ^= (uint64_t)((n) * 31u + SEED + 19u);    \
    acc += (uint64_t)((n) * 37u + SEED + 21u);    \
    break;

#define ICACHE_R4(n)                                                       \
  ICACHE_CASE_OPS(n) ICACHE_CASE_OPS(n + 1) ICACHE_CASE_OPS(n + 2)         \
      ICACHE_CASE_OPS(n + 3)
#define ICACHE_R16(n)                                                      \
  ICACHE_R4(n) ICACHE_R4(n + 4) ICACHE_R4(n + 8) ICACHE_R4(n + 12)
#define ICACHE_R64(n)                                                      \
  ICACHE_R16(n) ICACHE_R16(n + 16) ICACHE_R16(n + 32) ICACHE_R16(n + 48)
#define ICACHE_R256(n)                                                     \
  ICACHE_R64(n) ICACHE_R64(n + 64) ICACHE_R64(n + 128) ICACHE_R64(n + 192)

__attribute__((noinline)) uint64_t icacheMixA(uint64_t acc, uint8_t idx) {
#define SEED 1009u
  switch (idx) { ICACHE_R256(0) }
#undef SEED
  return acc;
}

__attribute__((noinline)) uint64_t icacheMixB(uint64_t acc, uint8_t idx) {
#define SEED 2003u
  switch (idx) { ICACHE_R256(0) }
#undef SEED
  return acc;
}

__attribute__((noinline)) uint64_t icacheMixC(uint64_t acc, uint8_t idx) {
#define SEED 3001u
  switch (idx) { ICACHE_R256(0) }
#undef SEED
  return acc;
}

#undef ICACHE_R256
#undef ICACHE_R64
#undef ICACHE_R16
#undef ICACHE_R4
#undef ICACHE_CASE_OPS

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = 0x9e3779b97f4a7c15ull + t;
        uint32_t ci = static_cast<uint32_t>(t) * 2654435761u;
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          ci = kChaseTable[ci & (kChaseLen - 1)];
          uint8_t sw = static_cast<uint8_t>(ci & 0xFFu);
          switch (j % 3) {
            case 0:
              acc = icacheMixA(acc, sw);
              break;
            case 1:
              acc = icacheMixB(acc, sw);
              break;
            default:
              acc = icacheMixC(acc, sw);
              break;
          }
          ++j;
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
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100, 8, 100)
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
