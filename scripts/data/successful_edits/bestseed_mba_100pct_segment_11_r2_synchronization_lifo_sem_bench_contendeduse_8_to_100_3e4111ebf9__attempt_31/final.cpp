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
// From-scratch frontend-bound companion for contendedUse(8_to_100).
//
// The contended LifoSem traffic above has a very small executed-instruction
// working set, which keeps L1i MPKI low and IPC high.  To exercise the
// instruction-cache subsystem under the same posters/waiters contention
// shape, each waiter rotates among three large noinline switch routines.
// Every switch has 256 cases, each case performs 14 ALU ops on a local
// accumulator using case-unique literal constants (to defeat compiler
// deduplication / identical-code-folding).  The switch index is derived from
// a pointer-chase load, so the executed code footprint is large and poorly
// reused.  ALU-ops-per-case is the primary L1i MPKI knob.
// ---------------------------------------------------------------------------

#define LIFO_ALU_CASE(C, S)                                       \
  case (C): {                                                     \
    acc += (uint64_t)((uint32_t)(C) * 2654435761u + (S) + 1u);    \
    acc ^= (uint64_t)((uint32_t)(C) ^ (0x9E3779B9u + (S)));       \
    acc *= (uint64_t)((uint32_t)(C) | 1u);                        \
    acc += (uint64_t)(((uint32_t)(C) << 3) + 7u + (S));           \
    acc ^= (uint64_t)((uint32_t)(C) * 40503u + (S));              \
    acc *= (uint64_t)((uint32_t)(C) + 0x101u);                    \
    acc += (uint64_t)((uint32_t)(C) ^ (0xABCDu + (S)));           \
    acc ^= (uint64_t)((uint32_t)(C) * 31u + 17u);                 \
    acc += (uint64_t)((uint32_t)(C) * 13u + (S));                 \
    acc *= (uint64_t)((uint32_t)(C) + 3u);                        \
    acc ^= (uint64_t)((uint32_t)(C) * 7u + 5u + (S));             \
    acc += (uint64_t)((uint32_t)(C) ^ 0x55u);                     \
    acc *= (uint64_t)((uint32_t)(C) | 0x80u);                     \
    acc ^= (uint64_t)((uint32_t)(C) * 19u + 23u + (S));           \
    break;                                                        \
  }

#define LIFO_ALU16(B, S)                                          \
  LIFO_ALU_CASE((B) + 0, S)                                       \
  LIFO_ALU_CASE((B) + 1, S)                                       \
  LIFO_ALU_CASE((B) + 2, S)                                       \
  LIFO_ALU_CASE((B) + 3, S)                                       \
  LIFO_ALU_CASE((B) + 4, S)                                       \
  LIFO_ALU_CASE((B) + 5, S)                                       \
  LIFO_ALU_CASE((B) + 6, S)                                       \
  LIFO_ALU_CASE((B) + 7, S)                                       \
  LIFO_ALU_CASE((B) + 8, S)                                       \
  LIFO_ALU_CASE((B) + 9, S)                                       \
  LIFO_ALU_CASE((B) + 10, S)                                      \
  LIFO_ALU_CASE((B) + 11, S)                                      \
  LIFO_ALU_CASE((B) + 12, S)                                      \
  LIFO_ALU_CASE((B) + 13, S)                                      \
  LIFO_ALU_CASE((B) + 14, S)                                      \
  LIFO_ALU_CASE((B) + 15, S)

#define LIFO_SWITCH_BODY(S)                                       \
  LIFO_ALU16(0, S)                                                \
  LIFO_ALU16(16, S)                                               \
  LIFO_ALU16(32, S)                                               \
  LIFO_ALU16(48, S)                                               \
  LIFO_ALU16(64, S)                                               \
  LIFO_ALU16(80, S)                                               \
  LIFO_ALU16(96, S)                                               \
  LIFO_ALU16(112, S)                                              \
  LIFO_ALU16(128, S)                                              \
  LIFO_ALU16(144, S)                                              \
  LIFO_ALU16(160, S)                                              \
  LIFO_ALU16(176, S)                                              \
  LIFO_ALU16(192, S)                                              \
  LIFO_ALU16(208, S)                                              \
  LIFO_ALU16(224, S)                                              \
  LIFO_ALU16(240, S)

FOLLY_NOINLINE static uint64_t lifoIcacheMix0(uint8_t idx, uint64_t acc) {
  switch (idx) {
    LIFO_SWITCH_BODY(0u)
    default:
      break;
  }
  return acc;
}

FOLLY_NOINLINE static uint64_t lifoIcacheMix1(uint8_t idx, uint64_t acc) {
  switch (idx) {
    LIFO_SWITCH_BODY(1103515245u)
    default:
      break;
  }
  return acc;
}

FOLLY_NOINLINE static uint64_t lifoIcacheMix2(uint8_t idx, uint64_t acc) {
  switch (idx) {
    LIFO_SWITCH_BODY(2246822519u)
    default:
      break;
  }
  return acc;
}

static uint8_t lifoChase[256];
static const bool lifoChaseInit = [] {
  for (uint32_t i = 0; i < 256; ++i) {
    lifoChase[i] = (uint8_t)((i * 167u + 13u) & 0xFFu);
  }
  return true;
}();

static void contendedSwitch(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t acc = 0x1234567u + (uint32_t)t;
        uint8_t pos = (uint8_t)t;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int j = 0; j < 24; ++j) {
            pos = lifoChase[pos];
            uint8_t k = (uint8_t)((acc ^ pos) & 0xFFu);
            switch (j % 3) {
              case 0:
                acc = lifoIcacheMix0(k, acc);
                break;
              case 1:
                acc = lifoIcacheMix1(k, acc);
                break;
              default:
                acc = lifoIcacheMix2(k, acc);
                break;
            }
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
BENCHMARK_NAMED_PARAM(contendedSwitch, 8_to_100, 8, 100)

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
