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
// Derived frontend/icache-pressure variant of contendedUse(8_to_100).
//
// The three noinline functions below each contain a 256-case switch with a
// long sequence of ALU ops per case, using per-case unique literal constants
// to defeat compiler case-deduplication.  A function-level seed makes the
// three bodies distinct so identical-code-folding cannot merge them.  The hot
// loop rotates among the three with i%3, indexing each switch with a byte
// derived from a pointer-chase load, producing a large executed instruction
// working set that thrashes the L1 instruction cache.
// ---------------------------------------------------------------------------

#define LSB_OP(i)                                                              \
  case ((i)): {                                                               \
    acc += 0x9E3779B97F4A7C15ull + (uint64_t)(i) * 0x2545F4914F6CDD1Dull +    \
        kSeed;                                                                 \
    acc ^= acc >> 7;                                                          \
    acc *= 0x100000001B3ull + (uint64_t)(i) * 3u;                            \
    acc += (0xC2B2AE3D27D4EB4Full ^ ((uint64_t)(i) * 7u)) + kSeed;            \
    acc ^= acc << 11;                                                        \
    acc -= 0x165667B19E3779F9ull + (uint64_t)(i) * 11u;                      \
    acc *= 0xFF51AFD7ED558CCDull ^ ((uint64_t)(i) * 13u);                    \
    acc ^= acc >> 5;                                                         \
    acc += 0xD6E8FEB86659FD93ull + (uint64_t)(i) * 17u + kSeed;              \
    acc *= 0xCBF29CE484222325ull ^ ((uint64_t)(i) * 19u);                    \
    acc ^= 0x9DDFEA08EB382D69ull + (uint64_t)(i) * 23u;                      \
    acc += acc << 3;                                                         \
    acc *= 0x00000100000001B3ull + (uint64_t)(i) * 29u;                      \
    acc ^= acc >> 9;                                                         \
  } break;

#define LSB_OP16(b)                                                           \
  LSB_OP((b) + 0)                                                             \
  LSB_OP((b) + 1)                                                             \
  LSB_OP((b) + 2)                                                             \
  LSB_OP((b) + 3)                                                             \
  LSB_OP((b) + 4)                                                             \
  LSB_OP((b) + 5)                                                             \
  LSB_OP((b) + 6)                                                             \
  LSB_OP((b) + 7)                                                             \
  LSB_OP((b) + 8)                                                             \
  LSB_OP((b) + 9)                                                             \
  LSB_OP((b) + 10)                                                            \
  LSB_OP((b) + 11)                                                            \
  LSB_OP((b) + 12)                                                            \
  LSB_OP((b) + 13)                                                            \
  LSB_OP((b) + 14)                                                            \
  LSB_OP((b) + 15)

#define LSB_SWITCH_BODY                                                        \
  switch (idx & 0xFFu) {                                                      \
    LSB_OP16(0)                                                               \
    LSB_OP16(16)                                                              \
    LSB_OP16(32)                                                              \
    LSB_OP16(48)                                                              \
    LSB_OP16(64)                                                              \
    LSB_OP16(80)                                                              \
    LSB_OP16(96)                                                              \
    LSB_OP16(112)                                                             \
    LSB_OP16(128)                                                             \
    LSB_OP16(144)                                                             \
    LSB_OP16(160)                                                             \
    LSB_OP16(176)                                                             \
    LSB_OP16(192)                                                             \
    LSB_OP16(208)                                                             \
    LSB_OP16(224)                                                             \
    LSB_OP16(240)                                                             \
  }

__attribute__((noinline)) static uint64_t lsbMix0(uint64_t acc, uint32_t idx) {
  const uint64_t kSeed = 0x1111111111111111ull;
  LSB_SWITCH_BODY
  return acc;
}

__attribute__((noinline)) static uint64_t lsbMix1(uint64_t acc, uint32_t idx) {
  const uint64_t kSeed = 0x2222222222222222ull;
  LSB_SWITCH_BODY
  return acc;
}

__attribute__((noinline)) static uint64_t lsbMix2(uint64_t acc, uint32_t idx) {
  const uint64_t kSeed = 0x3333333333333333ull;
  LSB_SWITCH_BODY
  return acc;
}

#undef LSB_SWITCH_BODY
#undef LSB_OP16
#undef LSB_OP

static constexpr size_t kMixBufSize = 4096;
static uint32_t mixChase[kMixBufSize];
static uint8_t mixPayload[kMixBufSize];

static void contendedUseMix(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kMixBufSize; ++i) {
      mixChase[i] =
          (uint32_t)((i * 2654435761u + 1013904223u) % kMixBufSize);
      mixPayload[i] = (uint8_t)((i * 131u + 7u) & 0xFFu);
    }
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t acc = (uint64_t)t + 1;
        uint32_t idx = (uint32_t)((t * 2654435761u) % kMixBufSize);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          idx = mixChase[idx];
          uint32_t p = mixPayload[idx];
          switch (i % 3) {
            case 0:
              acc = lsbMix0(acc, p);
              break;
            case 1:
              acc = lsbMix1(acc, p);
              break;
            default:
              acc = lsbMix2(acc, p);
              break;
          }
        }
        sink.fetch_add(acc, std::memory_order_relaxed);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go, &sink] {
        while (!go.load()) {
          std::this_thread::yield();
        }
        uint64_t acc = (uint64_t)t + 0x9E37u;
        uint32_t idx = (uint32_t)((t * 40503u + 12345u) % kMixBufSize);
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          idx = mixChase[idx];
          uint32_t p = mixPayload[idx];
          switch (i % 3) {
            case 0:
              acc = lsbMix1(acc, p);
              break;
            case 1:
              acc = lsbMix2(acc, p);
              break;
            default:
              acc = lsbMix0(acc, p);
              break;
          }
        }
        sink.fetch_add(acc, std::memory_order_relaxed);
      });
    }
  }

  go.store(true);
  for (auto& thr : threads) {
    thr.join();
  }
  folly::doNotOptimizeAway(sink.load());
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
