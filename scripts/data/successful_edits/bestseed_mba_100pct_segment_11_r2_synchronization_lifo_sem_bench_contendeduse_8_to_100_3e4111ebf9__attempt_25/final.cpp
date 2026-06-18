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
// Derived benchmark: contendedUseMix
//
// This is a from-scratch frontend-bound variant of the contended LifoSem
// workload. Each waiter, after being released, performs a pointer-chase load
// and rotates across three large noinline 256-case switch functions. The huge
// executed-instruction working set inflates L1 i-cache pressure (raising
// L1-icache-load-misses_MPKI) which in turn lowers IPC toward the target.
//
// The switch index comes from a pointer-chase load (payload & 0xFF), and the
// j%3 rotation forces i-cache thrashing across a large code footprint. Each
// case uses unique literal constants (derived from the case index plus a
// per-function salt) to prevent compiler/linker deduplication.
// ---------------------------------------------------------------------------

#define MIX_OPS(idx, salt)                                            \
  acc += (uint32_t)((idx) * 2654435761u + 1u + (salt));              \
  acc ^= (uint32_t)(((idx) ^ 0x9e3779b9u) + (salt));                 \
  acc *= (uint32_t)(((idx) | 1u) + (salt) * 2u + 2u);                \
  acc += (uint32_t)((((idx) << 3) ^ 0x12345u) + (salt));             \
  acc ^= (uint32_t)((idx) * 7u + 3u + (salt));                       \
  acc *= (uint32_t)((((idx) + 17u) | 1u) + (salt));                  \
  acc += (uint32_t)(((idx) ^ 0xABCDu) + (salt));                     \
  acc ^= (uint32_t)((idx) * 13u + 5u + (salt));                      \
  acc *= (uint32_t)((((idx) + 31u) | 1u) + (salt));                  \
  acc += (uint32_t)((((idx) << 5) ^ 0x6789u) + (salt));              \
  acc ^= (uint32_t)((idx) * 19u + 7u + (salt));                      \
  acc *= (uint32_t)((((idx) + 41u) | 1u) + (salt));                  \
  acc += (uint32_t)(((idx) ^ 0xDEADu) + (salt));                     \
  acc ^= (uint32_t)((idx) * 23u + 11u + (salt));

#define ALU_CASE(idx, salt) \
  case (idx): {             \
    MIX_OPS(idx, salt)      \
  } break;

#define C4(b, salt)                                                    \
  ALU_CASE((b) + 0, salt) ALU_CASE((b) + 1, salt) ALU_CASE((b) + 2, salt) \
      ALU_CASE((b) + 3, salt)
#define C16(b, salt)                                                   \
  C4(b, salt) C4((b) + 4, salt) C4((b) + 8, salt) C4((b) + 12, salt)
#define C64(b, salt)                                                   \
  C16(b, salt) C16((b) + 16, salt) C16((b) + 32, salt) C16((b) + 48, salt)
#define C256(salt) C64(0, salt) C64(64, salt) C64(128, salt) C64(192, salt)

__attribute__((noinline)) static uint32_t icache_mix0(uint32_t acc, uint8_t sel) {
  switch (sel) { C256(0x11u) }
  return acc;
}

__attribute__((noinline)) static uint32_t icache_mix1(uint32_t acc, uint8_t sel) {
  switch (sel) { C256(0x22u) }
  return acc;
}

__attribute__((noinline)) static uint32_t icache_mix2(uint32_t acc, uint8_t sel) {
  switch (sel) { C256(0x33u) }
  return acc;
}

#undef C256
#undef C64
#undef C16
#undef C4
#undef ALU_CASE
#undef MIX_OPS

static void contendedUseMix(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint32_t> sink(0);

  static constexpr size_t kRing = 4096;
  std::vector<uint32_t> ring(kRing);

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kRing; ++i) {
      ring[i] = (uint32_t)((i * 2654435761u) & (kRing - 1));
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &ring, &sink] {
        uint32_t acc = (uint32_t)t + 1u;
        size_t idx = (size_t)t & (kRing - 1);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          idx = ring[idx & (kRing - 1)];
          uint8_t sel = (uint8_t)(idx & 0xFF);
          switch (j % 3) {
            case 0:
              acc = icache_mix0(acc, sel);
              break;
            case 1:
              acc = icache_mix1(acc, sel);
              break;
            default:
              acc = icache_mix2(acc, sel);
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
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1, 31, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 16_to_16, 16, 16)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_32, 32, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1000, 32, 1000)
BENCHMARK_NAMED_PARAM(contendedUseMix, 8_to_100, 8, 100)

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
