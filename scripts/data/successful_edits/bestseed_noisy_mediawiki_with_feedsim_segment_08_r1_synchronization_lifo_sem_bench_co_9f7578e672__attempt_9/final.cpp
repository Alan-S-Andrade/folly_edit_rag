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

// Frontend / i-cache pressure companion derived from contendedUse(8_to_100).
// Three large noinline 256-case switch dispatchers (unique literal constants
// per case to defeat compiler deduplication) are rotated via j%3 so that the
// executed instruction working set is large enough to thrash L1i, while a
// dependent pointer-chase load selects the switch index.
namespace {

#define LIFOSEM_ALU_CASE(n)                                       \
  case (n): {                                                     \
    acc += 0x9E3779B97F4A7C15ull + (uint64_t)(n)*2654435761ull;   \
    acc ^= (acc >> 23);                                           \
    acc *= 0x100000001B3ull;                                      \
    acc += ((uint64_t)(n) << 7) ^ 0xABCDEFull;                    \
    acc ^= (acc << 17);                                           \
    acc -= (uint64_t)(n)*0xC2B2AE35ull;                           \
    acc *= ((uint64_t)(n) | 1ull);                                \
    acc ^= (acc >> 11) + (uint64_t)(n);                           \
    acc += (uint64_t)(n)*7919ull;                                 \
    acc ^= 0xDEADBEEF00ull + (uint64_t)(n);                       \
    acc *= 0x2545F4914F6CDD1Dull;                                 \
    acc += (acc >> 29) ^ (uint64_t)(n);                           \
    acc ^= ((uint64_t)(n) << 13) + 0x1234567ull;                  \
    acc -= (uint64_t)(n)*104729ull;                               \
    acc += (uint64_t)(n)*0xFFF1ull;                               \
    acc ^= (acc << 5) + 0x55AAull;                                \
    acc *= ((uint64_t)(n) + 3ull);                                \
    acc -= (acc >> 7) ^ (uint64_t)(n);                            \
    acc += 0x1010101ull * (uint64_t)(n);                          \
    acc ^= (acc >> 19) + 0xBEEFull + LIFOSEM_SALT;                \
    break;                                                        \
  }

#define LIFOSEM_CASES_4(b)                                        \
  LIFOSEM_ALU_CASE((b) + 0)                                       \
  LIFOSEM_ALU_CASE((b) + 1)                                       \
  LIFOSEM_ALU_CASE((b) + 2)                                       \
  LIFOSEM_ALU_CASE((b) + 3)

#define LIFOSEM_CASES_16(b)                                       \
  LIFOSEM_CASES_4((b) + 0)                                        \
  LIFOSEM_CASES_4((b) + 4)                                        \
  LIFOSEM_CASES_4((b) + 8)                                        \
  LIFOSEM_CASES_4((b) + 12)

#define LIFOSEM_CASES_256                                         \
  LIFOSEM_CASES_16(0)                                             \
  LIFOSEM_CASES_16(16)                                            \
  LIFOSEM_CASES_16(32)                                            \
  LIFOSEM_CASES_16(48)                                            \
  LIFOSEM_CASES_16(64)                                            \
  LIFOSEM_CASES_16(80)                                            \
  LIFOSEM_CASES_16(96)                                            \
  LIFOSEM_CASES_16(112)                                           \
  LIFOSEM_CASES_16(128)                                           \
  LIFOSEM_CASES_16(144)                                           \
  LIFOSEM_CASES_16(160)                                           \
  LIFOSEM_CASES_16(176)                                           \
  LIFOSEM_CASES_16(192)                                           \
  LIFOSEM_CASES_16(208)                                           \
  LIFOSEM_CASES_16(224)                                           \
  LIFOSEM_CASES_16(240)

__attribute__((noinline)) static uint64_t lifoSemIcacheA(
    uint64_t acc, unsigned idx) {
#define LIFOSEM_SALT 0xA5A5A5A5A5A5A5A5ull
  switch (idx) {
    LIFOSEM_CASES_256
    default:
      break;
  }
#undef LIFOSEM_SALT
  return acc;
}

__attribute__((noinline)) static uint64_t lifoSemIcacheB(
    uint64_t acc, unsigned idx) {
#define LIFOSEM_SALT 0x3C3C3C3C3C3C3C3Cull
  switch (idx) {
    LIFOSEM_CASES_256
    default:
      break;
  }
#undef LIFOSEM_SALT
  return acc;
}

__attribute__((noinline)) static uint64_t lifoSemIcacheC(
    uint64_t acc, unsigned idx) {
#define LIFOSEM_SALT 0x7E7E7E7E7E7E7E7Eull
  switch (idx) {
    LIFOSEM_CASES_256
    default:
      break;
  }
#undef LIFOSEM_SALT
  return acc;
}

} // namespace

BENCHMARK_DRAW_LINE();
BENCHMARK(contendedUse_icache_8_to_100, iters) {
  constexpr uint32_t kSize = 4096;
  std::vector<uint32_t> chase(kSize);
  uint64_t acc = 0;
  BENCHMARK_SUSPEND {
    for (uint32_t i = 0; i < kSize; ++i) {
      chase[i] = (i + 1577u) & (kSize - 1);
    }
  }
  uint32_t p = 0;
  for (size_t j = 0; j < iters; ++j) {
    p = chase[p];
    unsigned idx = p & 0xFFu;
    switch (j % 3) {
      case 0:
        acc = lifoSemIcacheA(acc, idx);
        break;
      case 1:
        acc = lifoSemIcacheB(acc, idx);
        break;
      default:
        acc = lifoSemIcacheC(acc, idx);
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
