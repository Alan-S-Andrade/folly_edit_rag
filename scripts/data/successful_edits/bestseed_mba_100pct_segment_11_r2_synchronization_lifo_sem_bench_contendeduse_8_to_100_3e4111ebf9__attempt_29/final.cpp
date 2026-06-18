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
// Derived benchmark: contendedUseMix
//
// Same LifoSem contention structure as contendedUse, but each waiter feeds a
// pointer-chase load into one of three large noinline switch dispatchers,
// rotated by i%3.  The three dispatchers together form a large hot
// instruction working set that is intentionally hard to keep resident in the
// L1 instruction cache, raising L1i MPKI and lowering IPC toward the target.
// ---------------------------------------------------------------------------
namespace {

static inline uint64_t rotl64(uint64_t x, unsigned r) {
  return (x << (r & 63)) | (x >> ((64 - r) & 63));
}

// 14 ALU ops per case; FUNC_K folds a per-function constant into the encoding
// so the three dispatchers do not get deduplicated by the compiler.
#define ALU_OPS(n)                                              \
  acc += (uint64_t)(n) * 0x9e3779b97f4a7c15ull + (FUNC_K);     \
  acc ^= (acc >> 7) ^ ((uint64_t)(n) << 11);                   \
  acc *= ((uint64_t)(n) | 1ull) + (FUNC_K);                    \
  acc -= (uint64_t)(n) * 0x100000001b3ull;                     \
  acc ^= rotl64(acc, ((n) & 31));                              \
  acc += (uint64_t)((n) ^ 0x5bd1e995u) * 3u;                   \
  acc *= 0xff51afd7ed558ccdull ^ (FUNC_K);                     \
  acc ^= (acc >> 13);                                          \
  acc += (uint64_t)(n) + (FUNC_K);                             \
  acc ^= (uint64_t)(n) * 7919ull;                              \
  acc *= 0xc4ceb9fe1a85ec53ull;                                \
  acc ^= (acc << 5) + (uint64_t)(n);                           \
  acc += (uint64_t)((n) * 2246822519u);                        \
  acc ^= (FUNC_K) * 0x2545f4914f6cdd1dull;

#define ALU_CASE(n) \
  case (n): {       \
    ALU_OPS(n)      \
  } break;

#define R1(b) ALU_CASE(b)
#define R4(b) R1(b) R1((b) + 1) R1((b) + 2) R1((b) + 3)
#define R16(b) R4(b) R4((b) + 4) R4((b) + 8) R4((b) + 12)
#define R64(b) R16(b) R16((b) + 16) R16((b) + 32) R16((b) + 48)
#define R256 R64(0) R64(64) R64(128) R64(192)

#define FUNC_K 0x123456789abcdef1ull
__attribute__((noinline)) static uint64_t icacheMix0(uint64_t payload) {
  uint64_t acc = payload + FUNC_K;
  switch (payload & 0xFF) {
    R256
  }
  return acc;
}
#undef FUNC_K

#define FUNC_K 0xa5a5a5a5deadbeefull
__attribute__((noinline)) static uint64_t icacheMix1(uint64_t payload) {
  uint64_t acc = payload ^ FUNC_K;
  switch (payload & 0xFF) {
    R256
  }
  return acc;
}
#undef FUNC_K

#define FUNC_K 0x0f1e2d3c4b5a6978ull
__attribute__((noinline)) static uint64_t icacheMix2(uint64_t payload) {
  uint64_t acc = payload * 3ull + FUNC_K;
  switch (payload & 0xFF) {
    R256
  }
  return acc;
}
#undef FUNC_K

#undef R256
#undef R64
#undef R16
#undef R4
#undef R1
#undef ALU_CASE
#undef ALU_OPS

} // namespace

static void contendedUseMix(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        std::vector<uint64_t> chain(256);
        for (size_t k = 0; k < chain.size(); ++k) {
          chain[k] = (k * 2654435761ull) ^ (uint64_t(k) << 17) ^ 0x9e3779b9ull;
        }
        uint64_t acc = 0;
        uint32_t idx = uint32_t(t);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          uint64_t payload = chain[idx & 0xFF];
          idx = uint32_t(payload & 0xFF);
          switch (i % 3) {
            case 0:
              acc += icacheMix0(payload);
              break;
            case 1:
              acc += icacheMix1(payload);
              break;
            default:
              acc += icacheMix2(payload);
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
