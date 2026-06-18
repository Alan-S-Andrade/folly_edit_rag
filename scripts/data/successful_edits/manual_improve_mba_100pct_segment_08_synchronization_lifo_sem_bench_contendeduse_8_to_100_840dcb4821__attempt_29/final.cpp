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

// --- Derived frontend-bound (I-cache) variant of contendedUse(8_to_100) ---
//
// This benchmark keeps the original contended LifoSem post/wait structure but
// inflates the executed instruction working set by rotating among three large
// noinline 256-case switches in the poster hot loop. The switch index is taken
// from a pointer-chase load so it cannot be hoisted/predicted, forcing the
// instruction stream to thrash the L1 instruction cache.

#define ALU_A(acc, k)                          \
  do {                                         \
    (acc) += 0x9E3779B1u + (uint32_t)(k);      \
    (acc) ^= (acc) << 13;                      \
    (acc) *= 0x85EBCA77u;                      \
    (acc) ^= (acc) >> 7;                       \
    (acc) += 0xC2B2AE3Du * (uint32_t)(k);      \
    (acc) ^= (acc) << 17;                      \
    (acc) *= 0x27D4EB2Fu;                      \
    (acc) -= 0x165667B1u + (uint32_t)(k);      \
    (acc) ^= (acc) >> 11;                      \
    (acc) += (uint32_t)(k) * 2654435761u;      \
    (acc) *= 0x9E3779B1u | 1u;                 \
    (acc) ^= (acc) >> 15;                      \
  } while (0)

#define ALU_B(acc, k)                          \
  do {                                         \
    (acc) ^= 0x7FB5D329u + (uint32_t)(k);      \
    (acc) *= 0xA24BAED5u | 1u;                 \
    (acc) += (acc) >> 9;                        \
    (acc) ^= (acc) << 12;                       \
    (acc) -= 0x4A7F1B3Du * (uint32_t)(k);      \
    (acc) *= 0xD3E1F59Bu;                      \
    (acc) ^= (acc) >> 14;                       \
    (acc) += 0x68E31DA4u ^ (uint32_t)(k);      \
    (acc) *= 0xB5297A4Du | 1u;                 \
    (acc) ^= (acc) << 5;                        \
    (acc) += (uint32_t)(k) * 40503u;           \
    (acc) ^= (acc) >> 16;                       \
  } while (0)

#define ALU_C(acc, k)                          \
  do {                                         \
    (acc) += 0x1B873593u ^ (uint32_t)(k);      \
    (acc) *= 0xCC9E2D51u;                      \
    (acc) ^= (acc) << 11;                       \
    (acc) -= 0x3243F6A9u + (uint32_t)(k);      \
    (acc) *= 0xFD7046C5u | 1u;                 \
    (acc) ^= (acc) >> 8;                        \
    (acc) += 0x8BADF00Du * (uint32_t)(k);      \
    (acc) ^= (acc) << 6;                        \
    (acc) *= 0x2545F491u | 1u;                 \
    (acc) += (acc) >> 13;                       \
    (acc) ^= (uint32_t)(k) * 0x9E3779B9u;      \
    (acc) *= 0x6C078965u | 1u;                 \
  } while (0)

#define CASE_BLK(MAC, k) \
  case (k):              \
    MAC(acc, (k));       \
    break;

#define CASES16(MAC, b)                                                      \
  CASE_BLK(MAC, (b) + 0) CASE_BLK(MAC, (b) + 1) CASE_BLK(MAC, (b) + 2)        \
      CASE_BLK(MAC, (b) + 3) CASE_BLK(MAC, (b) + 4) CASE_BLK(MAC, (b) + 5)    \
          CASE_BLK(MAC, (b) + 6) CASE_BLK(MAC, (b) + 7)                       \
              CASE_BLK(MAC, (b) + 8) CASE_BLK(MAC, (b) + 9)                   \
                  CASE_BLK(MAC, (b) + 10) CASE_BLK(MAC, (b) + 11)             \
                      CASE_BLK(MAC, (b) + 12) CASE_BLK(MAC, (b) + 13)         \
                          CASE_BLK(MAC, (b) + 14) CASE_BLK(MAC, (b) + 15)

#define CASES256(MAC)                                                        \
  CASES16(MAC, 0) CASES16(MAC, 16) CASES16(MAC, 32) CASES16(MAC, 48)         \
      CASES16(MAC, 64) CASES16(MAC, 80) CASES16(MAC, 96) CASES16(MAC, 112)   \
          CASES16(MAC, 128) CASES16(MAC, 144) CASES16(MAC, 160)              \
              CASES16(MAC, 176) CASES16(MAC, 192) CASES16(MAC, 208)          \
                  CASES16(MAC, 224) CASES16(MAC, 240)

__attribute__((noinline)) static uint32_t alu_switch_a(
    uint32_t acc, uint32_t idx) {
  switch (idx & 0xFFu) { CASES256(ALU_A) }
  return acc;
}

__attribute__((noinline)) static uint32_t alu_switch_b(
    uint32_t acc, uint32_t idx) {
  switch (idx & 0xFFu) { CASES256(ALU_B) }
  return acc;
}

__attribute__((noinline)) static uint32_t alu_switch_c(
    uint32_t acc, uint32_t idx) {
  switch (idx & 0xFFu) { CASES256(ALU_C) }
  return acc;
}

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  static const std::vector<uint32_t> chase = [] {
    std::vector<uint32_t> v(1024);
    for (size_t i = 0; i < v.size(); ++i) {
      v[i] = static_cast<uint32_t>(i * 2654435761u + 12345u);
    }
    return v;
  }();

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
        uint32_t acc = static_cast<uint32_t>(t) + 1u;
        uint32_t p = static_cast<uint32_t>(t);
        for (uint32_t i = t; i < n; i += posters) {
          p = chase[p & 1023u];
          uint32_t idx = p & 0xFFu;
          switch (i % 3u) {
            case 0:
              acc = alu_switch_a(acc, idx);
              break;
            case 1:
              acc = alu_switch_b(acc, idx);
              break;
            default:
              acc = alu_switch_c(acc, idx);
              break;
          }
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

BENCHMARK_DRAW_LINE();
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_1, 1, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_4, 1, 4)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_32, 1, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_1, 4, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_24, 4, 24)
BENCHMARK_NAMED_PARAM(contendedUse, 8_to_100, 8, 100)
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100, 8, 100)
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
