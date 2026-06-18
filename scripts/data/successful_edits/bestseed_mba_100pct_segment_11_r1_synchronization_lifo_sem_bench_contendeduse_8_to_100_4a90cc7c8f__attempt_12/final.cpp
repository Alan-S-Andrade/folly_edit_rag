#include <folly/portability/Asm.h>
#include <folly/synchronization/LifoSem.h>
#include <folly/synchronization/NativeSemaphore.h>

#include <folly/Benchmark.h>

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
// Derived frontend-bound variant of contendedUse(8_to_100).
//
// The reference benchmark runs hot in a tiny instruction working set, giving an
// IPC well above the desired envelope. To shift the workload toward being
// frontend-bound (raising L1i miss rate and lowering IPC) we interleave the
// semaphore traffic with calls into a large, branchy code footprint: three
// noinline 256-case switch functions, each case performing several ALU ops on
// a local accumulator with unique literal constants (to defeat the optimizer's
// case deduplication). The switch index comes from a pointer-chase load, and
// the three functions are rotated in the hot loop so the executed instruction
// stream thrashes the L1 instruction cache.
// ---------------------------------------------------------------------------

#define ICACHE_C4(M, b) M(b) M((b) + 1) M((b) + 2) M((b) + 3)
#define ICACHE_C16(M, b) \
  ICACHE_C4(M, b)        \
  ICACHE_C4(M, (b) + 4) ICACHE_C4(M, (b) + 8) ICACHE_C4(M, (b) + 12)
#define ICACHE_C256(M)                                                       \
  ICACHE_C16(M, 0) ICACHE_C16(M, 16) ICACHE_C16(M, 32) ICACHE_C16(M, 48)     \
      ICACHE_C16(M, 64) ICACHE_C16(M, 80) ICACHE_C16(M, 96)                  \
          ICACHE_C16(M, 112) ICACHE_C16(M, 128) ICACHE_C16(M, 144)           \
              ICACHE_C16(M, 160) ICACHE_C16(M, 176) ICACHE_C16(M, 192)       \
                  ICACHE_C16(M, 208) ICACHE_C16(M, 224) ICACHE_C16(M, 240)

__attribute__((noinline)) static uint64_t icacheMixA(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
#define ICACHE_CASE_A(n)                       \
  case (n):                                    \
    acc += (idx + 0x1000ULL + (n));            \
    acc ^= (acc << 3) ^ ((n) * 7u + 11u);      \
    acc *= (0x100000001b3ULL + (n));           \
    acc -= ((n) ^ 0xABCDull);                  \
    acc += (acc >> 5);                         \
    acc ^= ((n) * 2654435761u);                \
    acc += (((n) << 2) | 1u);                  \
    acc *= (3u + (n));                         \
    acc ^= (acc << 11);                        \
    acc += (0xDEADull ^ (n));                  \
    acc -= ((n) * 13u);                        \
    acc ^= (acc >> 7);                         \
    acc += ((n) * 17u + 3u);                   \
    acc *= (1u + ((n) & 0x3F));                \
    break;
    ICACHE_C256(ICACHE_CASE_A)
#undef ICACHE_CASE_A
  }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheMixB(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
#define ICACHE_CASE_B(n)                       \
  case (n):                                    \
    acc ^= (idx * 0x2545F4914F6CDD1DULL + (n));\
    acc += (acc << 5) + ((n) * 3u + 7u);       \
    acc *= (0xFF51AFD7ED558CCDULL + (n));      \
    acc ^= ((n) + 0x5BD1ull);                  \
    acc -= (acc >> 9);                         \
    acc += ((n) * 40503u);                     \
    acc ^= (((n) << 1) | 3u);                  \
    acc *= (5u + (n));                         \
    acc += (acc << 13);                        \
    acc ^= (0xBEEFull + (n));                  \
    acc += ((n) * 19u);                        \
    acc -= (acc >> 3);                         \
    acc ^= ((n) * 23u + 5u);                   \
    acc *= (2u + ((n) & 0x7F));                \
    break;
    ICACHE_C256(ICACHE_CASE_B)
#undef ICACHE_CASE_B
  }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheMixC(
    uint64_t acc, uint8_t idx) {
  switch (idx) {
#define ICACHE_CASE_C(n)                          \
  case (n):                                       \
    acc += (idx ^ (0xC2B2AE3D27D4EB4FULL + (n))); \
    acc ^= (acc << 9) + ((n) * 9u + 13u);         \
    acc *= (0xC6A4A7935BD1E995ULL + (n));         \
    acc += ((n) ^ 0x1357ull);                     \
    acc ^= (acc >> 11);                           \
    acc -= ((n) * 65599u);                        \
    acc += (((n) << 3) | 7u);                     \
    acc *= (7u + (n));                            \
    acc ^= (acc << 17);                           \
    acc += (0xF00Dull ^ (n));                     \
    acc -= ((n) * 29u);                           \
    acc ^= (acc >> 6);                            \
    acc += ((n) * 31u + 9u);                      \
    acc *= (3u + ((n) & 0x3F));                   \
    break;
    ICACHE_C256(ICACHE_CASE_C)
#undef ICACHE_CASE_C
  }
  return acc;
}

static uint64_t icacheBurst(uint64_t acc, std::vector<uint32_t>& chase,
                            uint32_t& p) {
  for (int j = 0; j < 3; ++j) {
    p = chase[p];
    uint8_t idx = static_cast<uint8_t>(p ^ (acc >> 8));
    switch (j % 3) {
      case 0:
        acc = icacheMixA(acc, idx);
        break;
      case 1:
        acc = icacheMixB(acc, idx);
        break;
      default:
        acc = icacheMixC(acc, idx);
        break;
    }
  }
  return acc;
}

static void icacheContendedUse(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        std::vector<uint32_t> chase(4096);
        for (size_t k = 0; k < chase.size(); ++k) {
          chase[k] =
              static_cast<uint32_t>((k * 2654435761u + 12345u) % chase.size());
        }
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          acc = icacheBurst(acc, chase, p);
        }
        folly::doNotOptimizeAway(acc);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go] {
        std::vector<uint32_t> chase(4096);
        for (size_t k = 0; k < chase.size(); ++k) {
          chase[k] =
              static_cast<uint32_t>((k * 40503u + 7u) % chase.size());
        }
        uint64_t acc = static_cast<uint64_t>(t) + 101;
        uint32_t p = 0;
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          acc = icacheBurst(acc, chase, p);
        }
        folly::doNotOptimizeAway(acc);
      });
    }
  }

  go.store(true);
  for (auto& thr : threads) {
    thr.join();
  }
}

BENCHMARK_DRAW_LINE();
BENCHMARK_NAMED_PARAM(icacheContendedUse, 8_to_100, 8, 100)

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
