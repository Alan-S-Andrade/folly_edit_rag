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
// Frontend-bound (I-cache pressure) variant of contendedUse.
//
// The three noinline functions below each contain a 256-case switch with a
// long dependent ALU chain per case.  Each case uses case-unique immediate
// constants (derived from the case index), and each function uses a distinct
// "base" literal, so neither case bodies nor whole functions are deduplicated
// by the compiler / linker ICF.  Rotating among the three functions with i%3
// in the hot loop, indexed by a pointer-chase load masked to 0xFF, inflates
// the executed instruction working set far past L1i capacity.  This raises
// L1-icache-load-misses_MPKI and lowers IPC into the target band.
// ---------------------------------------------------------------------------
namespace {

static constexpr size_t kFeRing = 1u << 12;

#define FE_OP(off, mul)                          \
  acc += k * (uint64_t)(mul) + (uint64_t)(off);  \
  acc ^= acc >> 29;                              \
  acc *= 0x2545F4914F6CDD1Dull;                  \
  acc ^= acc << 17;

#define FE_BODY(n)                  \
  {                                 \
    uint64_t k = (uint64_t)(n) + base; \
    FE_OP((n) + 1u, 0x9E37u)        \
    FE_OP((n) + 3u, 0x85EBu)        \
    FE_OP((n) + 5u, 0xC2B2u)        \
    FE_OP((n) + 7u, 0x27D4u)        \
    FE_OP((n) + 11u, 0x1656u)       \
    FE_OP((n) + 13u, 0x7F4Au)       \
    FE_OP((n) + 17u, 0xB97Fu)       \
    acc += k;                       \
  }

#define CASE1(n) \
  case (n):      \
    FE_BODY(n)   \
    break;
#define CASE4(n) CASE1(n) CASE1((n) + 1) CASE1((n) + 2) CASE1((n) + 3)
#define CASE16(n) CASE4(n) CASE4((n) + 4) CASE4((n) + 8) CASE4((n) + 12)
#define CASE64(n) \
  CASE16(n) CASE16((n) + 16) CASE16((n) + 32) CASE16((n) + 48)
#define CASE256(n) \
  CASE64(n) CASE64((n) + 64) CASE64((n) + 128) CASE64((n) + 192)

__attribute__((noinline)) uint64_t fe_switch0(uint64_t acc, uint32_t idx) {
  const uint64_t base = 0x1111u;
  switch (idx & 0xFFu) {
    CASE256(0)
  }
  return acc;
}

__attribute__((noinline)) uint64_t fe_switch1(uint64_t acc, uint32_t idx) {
  const uint64_t base = 0x2222u;
  switch (idx & 0xFFu) {
    CASE256(0)
  }
  return acc;
}

__attribute__((noinline)) uint64_t fe_switch2(uint64_t acc, uint32_t idx) {
  const uint64_t base = 0x3333u;
  switch (idx & 0xFFu) {
    CASE256(0)
  }
  return acc;
}

#undef CASE256
#undef CASE64
#undef CASE16
#undef CASE4
#undef CASE1
#undef FE_BODY
#undef FE_OP

} // namespace

static void contendedUseFE(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::vector<uint32_t> ring(kFeRing);

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kFeRing; ++i) {
      ring[i] = (uint32_t)((i * 1103515245u + 12345u) & (kFeRing - 1));
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &ring] {
        uint64_t acc = (uint64_t)t + 1;
        uint32_t idx = (uint32_t)(t * 2654435761u) & (kFeRing - 1);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          idx = ring[idx & (kFeRing - 1)];
          switch (i % 3) {
            case 0:
              acc = fe_switch0(acc, idx);
              break;
            case 1:
              acc = fe_switch1(acc, idx);
              break;
            default:
              acc = fe_switch2(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUseFE, 8_to_100_icache, 8, 100)
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
