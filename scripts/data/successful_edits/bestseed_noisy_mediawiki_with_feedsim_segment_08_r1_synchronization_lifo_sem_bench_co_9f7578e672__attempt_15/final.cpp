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

// ----------------------------------------------------------------------------
// Derived from-scratch benchmark for contendedUse(8_to_100).
//
// Goal: expand the executed instruction working set so that the frontend
// becomes the bottleneck (raise L1-icache-load-misses_MPKI, lower IPC). This
// uses the proven i-cache thrash pattern: three noinline 256-case switches,
// each case running a fixed number of ALU ops with unique literal constants
// (so the compiler cannot deduplicate cases or functions), driven by a
// pointer-chase load and rotated with j % 3 to maximise the hot code
// footprint.
// ----------------------------------------------------------------------------

#define ICACHE_ALU_OPS(i, s)                              \
  acc += (uint64_t)((i) * 2654435761u + (s) + 1u);        \
  acc ^= (uint64_t)((i) * 40503u + (s) + 7u);             \
  acc *= (uint64_t)(((i) ^ (s)) | 1u);                    \
  acc += (uint64_t)((i) * 0x9E3779B1u + (s));             \
  acc ^= (uint64_t)(((i) << 3) + (s) + 13u);              \
  acc *= (uint64_t)(((i) + (s)) | 1u);                    \
  acc += (uint64_t)((i) * 7u + (s) + 3u);                 \
  acc ^= (uint64_t)((i) * 11u + (s) + 5u);                \
  acc *= (uint64_t)((((i) ^ 0x5Au) + (s)) | 1u);          \
  acc += (uint64_t)((i) * 17u + (s) + 19u);               \
  acc ^= (uint64_t)((i) * 23u + (s) + 29u);               \
  acc *= (uint64_t)((((i) + 31u) ^ (s)) | 1u);            \
  acc += (uint64_t)((i) * 37u + (s) + 41u);               \
  acc ^= (uint64_t)((i) * 43u + (s) + 47u);

#define ICACHE_CASE_A(i) \
  case (i): {            \
    ICACHE_ALU_OPS(i, 0x11u) break; \
  }
#define ICACHE_CASE_B(i) \
  case (i): {            \
    ICACHE_ALU_OPS(i, 0x22u) break; \
  }
#define ICACHE_CASE_C(i) \
  case (i): {            \
    ICACHE_ALU_OPS(i, 0x33u) break; \
  }

#define ICACHE_REP16(M, base)                                  \
  M(base + 0) M(base + 1) M(base + 2) M(base + 3)              \
  M(base + 4) M(base + 5) M(base + 6) M(base + 7)              \
  M(base + 8) M(base + 9) M(base + 10) M(base + 11)            \
  M(base + 12) M(base + 13) M(base + 14) M(base + 15)

#define ICACHE_REP256(M)                                       \
  ICACHE_REP16(M, 0) ICACHE_REP16(M, 16) ICACHE_REP16(M, 32)   \
  ICACHE_REP16(M, 48) ICACHE_REP16(M, 64) ICACHE_REP16(M, 80)  \
  ICACHE_REP16(M, 96) ICACHE_REP16(M, 112)                     \
  ICACHE_REP16(M, 128) ICACHE_REP16(M, 144)                    \
  ICACHE_REP16(M, 160) ICACHE_REP16(M, 176)                    \
  ICACHE_REP16(M, 192) ICACHE_REP16(M, 208)                    \
  ICACHE_REP16(M, 224) ICACHE_REP16(M, 240)

__attribute__((noinline)) static uint64_t icacheMixA(uint8_t idx, uint64_t acc) {
  switch (idx) { ICACHE_REP256(ICACHE_CASE_A) }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheMixB(uint8_t idx, uint64_t acc) {
  switch (idx) { ICACHE_REP256(ICACHE_CASE_B) }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheMixC(uint8_t idx, uint64_t acc) {
  switch (idx) { ICACHE_REP256(ICACHE_CASE_C) }
  return acc;
}

BENCHMARK_DRAW_LINE();
BENCHMARK(contendedUse_icache_8_to_100, iters) {
  const size_t kN = 4096;
  std::vector<uint32_t> chase;
  BENCHMARK_SUSPEND {
    chase.resize(kN);
    for (size_t i = 0; i < kN; ++i) {
      chase[i] = static_cast<uint32_t>(i);
    }
    // Pseudo-random shuffle to defeat the hardware prefetcher.
    for (size_t i = kN - 1; i > 0; --i) {
      size_t j = (i * 2654435761u + 12345u) % (i + 1);
      std::swap(chase[i], chase[j]);
    }
  }

  uint64_t acc = 0;
  uint32_t cur = 0;
  for (size_t i = 0; i < iters; ++i) {
    cur = chase[cur % kN];
    uint8_t payload = static_cast<uint8_t>(cur & 0xFF);
    switch (i % 3) {
      case 0:
        acc = icacheMixA(payload, acc);
        break;
      case 1:
        acc = icacheMixB(payload, acc);
        break;
      default:
        acc = icacheMixC(payload, acc);
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
