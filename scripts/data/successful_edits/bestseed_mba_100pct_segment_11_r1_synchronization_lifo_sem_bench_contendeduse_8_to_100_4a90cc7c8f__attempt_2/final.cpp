#include <folly/portability/Asm.h>
#include <folly/synchronization/LifoSem.h>
#include <folly/synchronization/NativeSemaphore.h>

#include <folly/Benchmark.h>

#include <cstdint>
#include <vector>

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

// --- Frontend-pressure helpers for contendedUseFront -------------------------
// Three large noinline switch functions create a wide hot instruction footprint
// to raise L1 icache pressure (lowering an otherwise too-high IPC). Each case
// constant-folds into a unique literal sequence to defeat compiler dedup.
#define LIFOSEM_ALU(k, salt)                       \
  do {                                             \
    uint64_t kk = (uint64_t)(k);                   \
    uint64_t s = (uint64_t)(salt);                 \
    acc += kk * 7u + 1u + s;                       \
    acc ^= kk * 3u + 2u + s;                       \
    acc *= (kk | 1u) + s;                          \
    acc += kk * 11u + 5u + s;                      \
    acc ^= kk * 13u + 7u + s;                      \
    acc *= ((kk ^ 0x5au) | 1u) + s;                \
    acc += kk * 17u + 9u + s;                      \
    acc ^= kk * 19u + 11u + s;                     \
    acc *= ((kk + 3u) | 1u) + s;                   \
    acc += kk * 23u + 13u + s;                     \
    acc ^= kk * 29u + 15u + s;                     \
    acc *= ((kk ^ 0x3cu) | 1u) + s;                \
    acc += kk * 31u + 17u + s;                     \
    acc ^= kk * 37u + 19u + s;                     \
  } while (0)

#define LIFOSEM_CASE(k, salt) \
  case (k):                   \
    LIFOSEM_ALU((k), (salt)); \
    break;

#define LIFOSEM_CASES16(base, salt)                                      \
  LIFOSEM_CASE((base) + 0, (salt)) LIFOSEM_CASE((base) + 1, (salt))      \
  LIFOSEM_CASE((base) + 2, (salt)) LIFOSEM_CASE((base) + 3, (salt))      \
  LIFOSEM_CASE((base) + 4, (salt)) LIFOSEM_CASE((base) + 5, (salt))      \
  LIFOSEM_CASE((base) + 6, (salt)) LIFOSEM_CASE((base) + 7, (salt))      \
  LIFOSEM_CASE((base) + 8, (salt)) LIFOSEM_CASE((base) + 9, (salt))      \
  LIFOSEM_CASE((base) + 10, (salt)) LIFOSEM_CASE((base) + 11, (salt))    \
  LIFOSEM_CASE((base) + 12, (salt)) LIFOSEM_CASE((base) + 13, (salt))    \
  LIFOSEM_CASE((base) + 14, (salt)) LIFOSEM_CASE((base) + 15, (salt))

#define LIFOSEM_CASES256(salt)                                           \
  LIFOSEM_CASES16(0, (salt)) LIFOSEM_CASES16(16, (salt))                 \
  LIFOSEM_CASES16(32, (salt)) LIFOSEM_CASES16(48, (salt))                \
  LIFOSEM_CASES16(64, (salt)) LIFOSEM_CASES16(80, (salt))                \
  LIFOSEM_CASES16(96, (salt)) LIFOSEM_CASES16(112, (salt))               \
  LIFOSEM_CASES16(128, (salt)) LIFOSEM_CASES16(144, (salt))              \
  LIFOSEM_CASES16(160, (salt)) LIFOSEM_CASES16(176, (salt))              \
  LIFOSEM_CASES16(192, (salt)) LIFOSEM_CASES16(208, (salt))              \
  LIFOSEM_CASES16(224, (salt)) LIFOSEM_CASES16(240, (salt))

__attribute__((noinline)) static uint64_t lifoSemThrash0(
    uint64_t acc, uint8_t idx) {
  switch (idx) { LIFOSEM_CASES256(0x1000u) }
  return acc;
}

__attribute__((noinline)) static uint64_t lifoSemThrash1(
    uint64_t acc, uint8_t idx) {
  switch (idx) { LIFOSEM_CASES256(0x2000u) }
  return acc;
}

__attribute__((noinline)) static uint64_t lifoSemThrash2(
    uint64_t acc, uint8_t idx) {
  switch (idx) { LIFOSEM_CASES256(0x3000u) }
  return acc;
}

#undef LIFOSEM_CASES256
#undef LIFOSEM_CASES16
#undef LIFOSEM_CASE
#undef LIFOSEM_ALU

static std::atomic<uint64_t> lifoSemFrontSink{0};

// Derived from contendedUse: identical contention pattern, but each waiter
// iteration drives a pointer-chase indexed 256-case switch (rotated across 3
// noinline functions) to widen the executed instruction working set, raising
// L1i misses and pulling the too-high IPC back toward target.
static void contendedUseFront(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        std::vector<uint32_t> chase(1024);
        for (uint32_t k = 0; k < chase.size(); ++k) {
          chase[k] = (k * 2654435761u + 12345u) & 1023u;
        }
        uint64_t acc = (uint64_t)t + 1u;
        uint32_t p = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p];
          uint8_t idx = (uint8_t)(p & 0xFFu);
          switch (i % 3) {
            case 0:
              acc = lifoSemThrash0(acc, idx);
              break;
            case 1:
              acc = lifoSemThrash1(acc, idx);
              break;
            default:
              acc = lifoSemThrash2(acc, idx);
              break;
          }
        }
        lifoSemFrontSink.fetch_add(acc, std::memory_order_relaxed);
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
BENCHMARK_NAMED_PARAM(contendedUseFront, 8_to_100, 8, 100)

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
