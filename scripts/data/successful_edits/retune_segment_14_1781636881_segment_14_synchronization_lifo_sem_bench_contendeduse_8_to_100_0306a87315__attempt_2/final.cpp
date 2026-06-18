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
// Derived frontend-bound variant of contendedUse(8_to_100).
//
// The reference benchmark runs with a very small executed instruction
// working set, which keeps IPC artificially high and L1i misses extremely
// low.  To match the target hardware-counter profile we inject a large,
// thrashing instruction footprint into the waiter loop: three noinline
// functions, each containing a full 256-case switch with a chain of ALU ops
// per case.  Unique literal constants (and a per-function salt) prevent the
// compiler from deduplicating either the cases or the functions, and the
// j%3 rotation forces icache thrashing across the whole code footprint.
// The switch index comes from a pointer-chase load so the branch target is
// data-dependent and unpredictable enough to exercise the frontend.
// ---------------------------------------------------------------------------

#define ALU_CASE(K, S)                                                   \
  case (K): {                                                            \
    acc += (uint64_t)(K) * 2654435761ull + (uint64_t)(S);               \
    acc ^= acc >> 13;                                                    \
    acc *= ((uint64_t)(K) | 1ull) + (uint64_t)(S);                       \
    acc += 0x9E3779B97F4A7C15ull ^ ((uint64_t)(K) + (uint64_t)(S));      \
    acc ^= acc << 7;                                                     \
    acc *= 0x85EBCA6Bull + (uint64_t)(K) * 3ull;                         \
    acc -= (uint64_t)(K) * 0xC2B2AE35ull + (uint64_t)(S);                \
    acc ^= acc >> 11;                                                    \
    acc += (acc << 3) ^ ((uint64_t)(K) + (uint64_t)(S));                 \
    acc *= 0x27D4EB2Full + (uint64_t)(K) + (uint64_t)(S);                \
    acc ^= (uint64_t)(K) << 5;                                           \
    acc += acc >> 9;                                                     \
    acc *= 0x165667B1ull + (uint64_t)(K);                                \
    acc ^= (acc << 4) + (uint64_t)(K) + (uint64_t)(S);                   \
  } break;

#define ALU_R4(n, S) \
  ALU_CASE(n, S) ALU_CASE(n + 1, S) ALU_CASE(n + 2, S) ALU_CASE(n + 3, S)
#define ALU_R16(n, S) \
  ALU_R4(n, S) ALU_R4(n + 4, S) ALU_R4(n + 8, S) ALU_R4(n + 12, S)
#define ALU_R64(n, S) \
  ALU_R16(n, S) ALU_R16(n + 16, S) ALU_R16(n + 32, S) ALU_R16(n + 48, S)
#define ALU_R256(S) \
  ALU_R64(0, S) ALU_R64(64, S) ALU_R64(128, S) ALU_R64(192, S)

__attribute__((noinline)) static uint64_t feSwitchA(uint64_t acc, int idx) {
  switch (idx & 0xFF) {
    ALU_R256(0x11)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) static uint64_t feSwitchB(uint64_t acc, int idx) {
  switch (idx & 0xFF) {
    ALU_R256(0x22)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) static uint64_t feSwitchC(uint64_t acc, int idx) {
  switch (idx & 0xFF) {
    ALU_R256(0x33)
    default:
      break;
  }
  return acc;
}

#undef ALU_R256
#undef ALU_R64
#undef ALU_R16
#undef ALU_R4
#undef ALU_CASE

static std::vector<uint32_t> makeChaseTable(size_t n) {
  std::vector<uint32_t> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = static_cast<uint32_t>((i * 2654435761ull + 12345ull) % n);
  }
  return v;
}

static void contendedUseFrontend(uint32_t n, int posters, int waiters) {
  static const std::vector<uint32_t> chase = makeChaseTable(4096);
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & 4095];
          int idx = static_cast<int>(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = feSwitchA(acc, idx);
              break;
            case 1:
              acc = feSwitchB(acc, idx);
              break;
            default:
              acc = feSwitchC(acc, idx);
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
}

BENCHMARK_DRAW_LINE();
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_1, 1, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_4, 1, 4)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_32, 1, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_1, 4, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_24, 4, 24)
BENCHMARK_NAMED_PARAM(contendedUse, 8_to_100, 8, 100)
BENCHMARK_NAMED_PARAM(contendedUseFrontend, 8_to_100_frontend, 8, 100)
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
