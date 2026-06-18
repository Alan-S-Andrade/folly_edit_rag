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

// ---------------------------------------------------------------------------
// Frontend-bound (I-cache) derived microbench for the contendedUse family.
// Three noinline 256-case switches rotated by j%3 over a pointer-chase index
// inflate the executed instruction working set so that L1i pressure (and thus
// IPC) tracks the contended-semaphore target rather than the compact baseline.
// ---------------------------------------------------------------------------
namespace {

// One switch case: many ALU ops keyed on both the case index K and a
// per-function salt S so the compiler cannot deduplicate cases or functions.
#define ICACHE_CASE(K, S)                                       \
  case (K): {                                                   \
    a += (uint32_t)(K) * 2654435761u + (S) + 0x01u;            \
    a ^= (uint32_t)(K) * 2246822519u + (S) + 0x02u;            \
    a *= ((uint32_t)(K) | 1u);                                  \
    a += (uint32_t)(K) * 374761393u + (S) + 0x03u;             \
    a ^= (uint32_t)(K) * 3266489917u + (S) + 0x04u;            \
    a += ((uint32_t)(K) << 5) ^ ((S) + 0x05u);                 \
    a ^= (uint32_t)(K) * 668265263u + (S) + 0x06u;             \
    a *= (((uint32_t)(K) << 1) | 3u);                          \
    a += (uint32_t)(K) * 40503u + (S) + 0x07u;                 \
    a ^= (uint32_t)(K) * 16777619u + (S) + 0x08u;              \
    a += (uint32_t)(K) * 2147483647u + (S) + 0x09u;            \
    a ^= ((a << 7) + (uint32_t)(K) + (S));                     \
    a += (uint32_t)(K) * 22695477u + (S) + 0x0Au;              \
    a ^= (uint32_t)(K) * 1103515245u + (S) + 0x0Bu;            \
    a *= ((uint32_t)(K) * 3u + 1u);                            \
    a += (uint32_t)(K) * 134775813u + (S) + 0x0Cu;             \
    a ^= (uint32_t)(K) * 214013u + (S) + 0x0Du;                \
    a += ((a >> 3) + (uint32_t)(K) * 5u + (S));                \
    a ^= (uint32_t)(K) * 69069u + (S) + 0x0Eu;                 \
    a += (uint32_t)(K) * 1812433253u + (S) + 0x0Fu;            \
    a *= (((uint32_t)(K) << 2) | 1u);                          \
    a ^= (uint32_t)(K) * 3935559000u + (S) + 0x10u;            \
    a += (uint32_t)(K) * 2891336453u + (S) + 0x11u;            \
    a ^= ((a << 3) + (uint32_t)(K) * 7u + (S) + 0x12u);        \
    break;                                                      \
  }

#define ICACHE_CASES_4(B, S) \
  ICACHE_CASE(B + 0, S)      \
  ICACHE_CASE(B + 1, S)      \
  ICACHE_CASE(B + 2, S)      \
  ICACHE_CASE(B + 3, S)
#define ICACHE_CASES_16(B, S)  \
  ICACHE_CASES_4(B + 0, S)     \
  ICACHE_CASES_4(B + 4, S)     \
  ICACHE_CASES_4(B + 8, S)     \
  ICACHE_CASES_4(B + 12, S)
#define ICACHE_CASES_64(B, S)   \
  ICACHE_CASES_16(B + 0, S)     \
  ICACHE_CASES_16(B + 16, S)    \
  ICACHE_CASES_16(B + 32, S)    \
  ICACHE_CASES_16(B + 48, S)
#define ICACHE_CASES_256(S)     \
  ICACHE_CASES_64(0, S)         \
  ICACHE_CASES_64(64, S)        \
  ICACHE_CASES_64(128, S)       \
  ICACHE_CASES_64(192, S)

__attribute__((noinline)) static uint32_t icache_mix_a(uint32_t idx, uint32_t a) {
  switch (idx & 0xFFu) { ICACHE_CASES_256(0x1111u) }
  return a;
}

__attribute__((noinline)) static uint32_t icache_mix_b(uint32_t idx, uint32_t a) {
  switch (idx & 0xFFu) { ICACHE_CASES_256(0x2222u) }
  return a;
}

__attribute__((noinline)) static uint32_t icache_mix_c(uint32_t idx, uint32_t a) {
  switch (idx & 0xFFu) { ICACHE_CASES_256(0x3333u) }
  return a;
}

#undef ICACHE_CASES_256
#undef ICACHE_CASES_64
#undef ICACHE_CASES_16
#undef ICACHE_CASES_4
#undef ICACHE_CASE

struct IcacheChase {
  std::vector<uint32_t> next;
  IcacheChase() {
    constexpr uint32_t N = 8192; // power of two for cheap masking
    next.resize(N);
    for (uint32_t i = 0; i < N; ++i) {
      next[i] = (i * 2654435761u + 12345u) % N;
    }
  }
};

} // namespace

BENCHMARK(contendedUse_icache_8_to_100, iters) {
  static IcacheChase chase;
  const auto& next = chase.next;
  const uint32_t mask = static_cast<uint32_t>(next.size() - 1);
  uint32_t p = 1;
  uint32_t acc = 0x12345678u;
  for (size_t i = 0; i < iters; ++i) {
    p = next[p & mask]; // dependent pointer-chase load
    uint32_t idx = p & 0xFFu;
    switch (i % 3) {
      case 0:
        acc = icache_mix_a(idx, acc);
        break;
      case 1:
        acc = icache_mix_b(idx, acc);
        break;
      default:
        acc = icache_mix_c(idx, acc);
        break;
    }
  }
  folly::doNotOptimizeAway(acc);
}

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
