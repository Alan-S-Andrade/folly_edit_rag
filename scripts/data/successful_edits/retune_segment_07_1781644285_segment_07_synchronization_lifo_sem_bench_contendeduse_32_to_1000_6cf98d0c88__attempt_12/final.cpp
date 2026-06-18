#include <folly/portability/Asm.h>
#include <folly/synchronization/LifoSem.h>
#include <folly/synchronization/NativeSemaphore.h>

#include <folly/Benchmark.h>

#include <array>
#include <cstdint>

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

// Pointer-chase payload table used to derive the switch index for the
// hot-loop code-footprint generators below. It is initialized once at static
// init time with a pseudo-random self-referencing permutation so that the
// per-iteration index load is a genuine data-dependent (pointer-chase) load.
static const std::array<uint8_t, 256> kLifoSemChase = [] {
  std::array<uint8_t, 256> a{};
  uint32_t x = 0x12345678u;
  for (int i = 0; i < 256; ++i) {
    x = (x * 1664525u) + 1013904223u;
    a[i] = static_cast<uint8_t>((x >> 16) & 0xffu);
  }
  return a;
}();

// Frontend (Tier T1) footprint generators. Each function is a noinline
// 256-case switch with a fixed block of straight-line ALU ops per case. Every
// case derives its literal constants from the case index so the compiler
// cannot deduplicate cases, and the three functions use distinct per-function
// salts so identical-code-folding cannot collapse them. Rotating among the
// three via j%3 in the hot loop forces a large instruction working set through
// the L1 instruction cache, raising L1i load-miss MPKI toward target while
// keeping the work straight-line (IPC-protective).
#define LSEM_ALU(X)                                                  \
  acc = acc + (uint32_t)(0x1000003u * (uint32_t)(X) + LSEM_SALT);    \
  acc = acc ^ (acc >> 7);                                            \
  acc = acc * (uint32_t)(2u * (uint32_t)(X) + 1u);                   \
  acc = acc + (uint32_t)(0x85ebca6bu ^ (uint32_t)(X));               \
  acc = acc ^ (acc << 13);                                           \
  acc = acc - (uint32_t)(0xc2b2ae35u + (uint32_t)(X));               \
  acc = acc * (uint32_t)(4u * (uint32_t)(X) + 3u);                   \
  acc = acc ^ (acc >> 11);                                           \
  acc = acc + (uint32_t)(0x27d4eb2fu * (uint32_t)(X));               \
  acc = acc ^ (acc << 9);                                            \
  acc = acc * (uint32_t)(6u * (uint32_t)(X) + 5u);                   \
  acc = acc ^ (acc >> 15);                                           \
  acc = acc + (uint32_t)(0x165667b1u ^ (uint32_t)((X) << 3));

#define LSEM_CASE(X) \
  case (X): {        \
    LSEM_ALU(X)      \
    break;           \
  }
#define LSEM_CASES4(B) \
  LSEM_CASE((B) + 0) LSEM_CASE((B) + 1) LSEM_CASE((B) + 2) LSEM_CASE((B) + 3)
#define LSEM_CASES16(B)                                            \
  LSEM_CASES4((B) + 0) LSEM_CASES4((B) + 4) LSEM_CASES4((B) + 8)   \
      LSEM_CASES4((B) + 12)
#define LSEM_CASES64(B)                                                \
  LSEM_CASES16((B) + 0) LSEM_CASES16((B) + 16) LSEM_CASES16((B) + 32)  \
      LSEM_CASES16((B) + 48)
#define LSEM_CASES256                                                   \
  LSEM_CASES64(0) LSEM_CASES64(64) LSEM_CASES64(128) LSEM_CASES64(192)

#define LSEM_DEFINE_FN(NAME)                                       \
  __attribute__((noinline)) static uint32_t NAME(                 \
      uint32_t acc, uint8_t idx) {                                \
    switch (idx) {                                                \
      LSEM_CASES256                                               \
      default:                                                    \
        acc ^= 0xdeadbeefu;                                       \
        break;                                                    \
    }                                                             \
    return acc;                                                   \
  }

#define LSEM_SALT 0x9e3779b9u
LSEM_DEFINE_FN(lifoSemSwitchA)
#undef LSEM_SALT
#define LSEM_SALT 0x85ebca6bu
LSEM_DEFINE_FN(lifoSemSwitchB)
#undef LSEM_SALT
#define LSEM_SALT 0xc2b2ae35u
LSEM_DEFINE_FN(lifoSemSwitchC)
#undef LSEM_SALT

// Nearby variant of contendedUse: same contended wait/post structure and the
// same operation volume, but each hot-loop iteration first drives a large,
// rotating instruction footprint through the icache.
//
// Attempt 12 corrective patch (single lever: frontend code footprint):
// the per-iteration scalar mixer is replaced by a pointer-chase-indexed call
// into one of three noinline 256-case switch generators (rotated via j%3).
// This inflates the executed instruction working set to raise the primary
// Tier T1 frontend miss (L1-icache-load-misses_MPKI) toward target while the
// per-case work stays straight-line to protect IPC. The op count per case
// (13) is tuned against the documented L1i-MPKI-vs-ops slope.
static void contendedUseBiasedSelect(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint32_t acc = 0x9e3779b9u ^ static_cast<uint32_t>(t);
        uint32_t p = static_cast<uint32_t>(t) * 2654435761u;
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          uint8_t idx = kLifoSemChase[p & 0xFFu];
          switch (j % 3u) {
            case 0:
              acc = lifoSemSwitchA(acc, idx);
              break;
            case 1:
              acc = lifoSemSwitchB(acc, idx);
              break;
            default:
              acc = lifoSemSwitchC(acc, idx);
              break;
          }
          p = idx;
          ++j;
          sem.wait();
        }
        folly::doNotOptimizeAway(acc);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go] {
        while (!go.load()) {
          std::this_thread::yield();
        }
        uint32_t acc = 0x85ebca6bu ^ static_cast<uint32_t>(t);
        uint32_t p = static_cast<uint32_t>(t) * 40503u + 0x9e3779b9u;
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += posters) {
          uint8_t idx = kLifoSemChase[p & 0xFFu];
          switch (j % 3u) {
            case 0:
              acc = lifoSemSwitchB(acc, idx);
              break;
            case 1:
              acc = lifoSemSwitchC(acc, idx);
              break;
            default:
              acc = lifoSemSwitchA(acc, idx);
              break;
          }
          p = idx;
          ++j;
          sem.post();
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

#undef LSEM_ALU
#undef LSEM_CASE
#undef LSEM_CASES4
#undef LSEM_CASES16
#undef LSEM_CASES64
#undef LSEM_CASES256
#undef LSEM_DEFINE_FN

BENCHMARK_DRAW_LINE();
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_1, 1, 1)
BENCHMARK_NAMED_PARAM(contendedUseBiasedSelect, 32_to_1000, 32, 1000)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_4, 1, 4)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_32, 1, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_1, 4, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_24, 4, 24)
BENCHMARK_NAMED_PARAM(contendedUse, 8_to_100, 8, 100)
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
