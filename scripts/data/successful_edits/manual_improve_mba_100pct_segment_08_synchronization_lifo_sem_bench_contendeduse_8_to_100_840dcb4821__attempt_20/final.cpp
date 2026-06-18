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

// --------------------------------------------------------------------------
// Derived from contendedUse(8_to_100): same LifoSem contention shape, but the
// waiter loop is augmented with a large, branchy, code-heavy switch payload so
// that the executed instruction working set (L1i footprint) grows. Three
// noinline 256-case switches are rotated by j%3 and indexed by a pointer-chase
// derived byte, forcing icache thrashing across a wide code footprint.
// --------------------------------------------------------------------------
namespace {

struct LsNode {
  uint32_t next;
  uint32_t payload;
};

// 20 unique ALU ops per case; (n) and (k) seed unique literals so the compiler
// cannot deduplicate cases or the three functions.
#define LIFOSEM_OPS(n, k)                              \
  acc += ((uint64_t)(n) * 2654435761ull + (k) + 1ull); \
  acc ^= ((uint64_t)(n) * 40503ull + (k) * 7ull);      \
  acc *= (((uint64_t)(n) << 1) | 1ull);                \
  acc -= ((uint64_t)(n) * 17ull + (k) * 3ull);         \
  acc ^= (((uint64_t)(n) + 13ull) * 31ull + (k));      \
  acc += (((uint64_t)(n) ^ 0x5Aull) * 3ull);           \
  acc *= ((uint64_t)(n) * 5ull + 3ull);                \
  acc ^= ((uint64_t)(n) * 0x9E3779B9ull + (k));        \
  acc += ((uint64_t)(n) * 11ull + 0x7ull);             \
  acc ^= ((uint64_t)(n) * 23ull + (k) * 5ull);         \
  acc -= ((uint64_t)(n) * 29ull + 1ull);               \
  acc *= (((uint64_t)(n) | 1ull) + 2ull);              \
  acc ^= ((uint64_t)(n) * 37ull + (k));                \
  acc += ((uint64_t)(n) * 41ull + 0x3ull);             \
  acc ^= ((uint64_t)(n) * 43ull + (k) * 9ull);         \
  acc -= ((uint64_t)(n) * 47ull + 0x5ull);             \
  acc += ((uint64_t)(n) * 53ull + (k) * 11ull);        \
  acc ^= ((uint64_t)(n) * 0x85EBCA77ull + (k));        \
  acc *= ((uint64_t)(n) * 3ull + 7ull);                \
  acc -= ((uint64_t)(n) * 59ull + 0x9ull);

#define LS_C1(n, k) \
  case (n): {       \
    LIFOSEM_OPS((n), (k)) break;                       \
  }
#define LS_C4(n, k) \
  LS_C1((n) + 0, k) LS_C1((n) + 1, k) LS_C1((n) + 2, k) LS_C1((n) + 3, k)
#define LS_C16(n, k) \
  LS_C4((n) + 0, k) LS_C4((n) + 4, k) LS_C4((n) + 8, k) LS_C4((n) + 12, k)
#define LS_C64(n, k)                                                 \
  LS_C16((n) + 0, k) LS_C16((n) + 16, k) LS_C16((n) + 32, k)         \
      LS_C16((n) + 48, k)
#define LS_C256(n, k)                                                \
  LS_C64((n) + 0, k) LS_C64((n) + 64, k) LS_C64((n) + 128, k)        \
      LS_C64((n) + 192, k)

__attribute__((noinline)) uint64_t lifoSwitch0(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LS_C256(0, 1)
  }
  return acc;
}

__attribute__((noinline)) uint64_t lifoSwitch1(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LS_C256(0, 2)
  }
  return acc;
}

__attribute__((noinline)) uint64_t lifoSwitch2(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LS_C256(0, 3)
  }
  return acc;
}

#undef LS_C256
#undef LS_C64
#undef LS_C16
#undef LS_C4
#undef LS_C1
#undef LIFOSEM_OPS

void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        std::vector<LsNode> chain(257);
        for (uint32_t k = 0; k < chain.size(); ++k) {
          chain[k].next = (k * 131u + 17u) % chain.size();
          chain[k].payload = k * 2654435761u + 12345u;
        }
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int j = 0; j < 3; ++j) {
            p = chain[p].next;
            uint8_t idx = static_cast<uint8_t>(chain[p].payload & 0xFFu);
            switch (j % 3) {
              case 0:
                acc = lifoSwitch0(acc, idx);
                break;
              case 1:
                acc = lifoSwitch1(acc, idx);
                break;
              default:
                acc = lifoSwitch2(acc, idx);
                break;
            }
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

} // namespace

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
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100_icache, 8, 100)

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
