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
// Frontend-bound derived workload for contendedUse(8_to_100).
//
// The reference benchmark retires too few distinct instructions and reuses a
// tiny code footprint, which keeps IPC well above target.  To force frontend /
// L1-icache pressure we add three large, noinline 256-case switch functions
// (each case carries unique literal constants so the compiler cannot fold
// them) and rotate among them with j % 3 in the hot loop, indexing each switch
// with the low byte of a pointer-chase load.  This blows up the executed
// instruction working set far beyond the L1 instruction cache.
// ---------------------------------------------------------------------------

#define FE_OPS_A(N)                                                        \
  a += 0x9E3779B97F4A7C15ULL + (uint64_t)(N) * 0x01000193ULL;             \
  a ^= 0xC2B2AE3D27D4EB4FULL ^ ((uint64_t)(N) << 1);                       \
  a *= (0x100000001B3ULL | ((uint64_t)(N) << 1));                          \
  a += (a >> 7) + (uint64_t)(N) * 7919ULL;                                 \
  a ^= (a << 3) ^ ((uint64_t)(N) * 31u + 17u);                            \
  a -= 0xDEADBEEFCAFEULL - (uint64_t)(N) * 13u;                            \
  a *= 0xFF51AFD7ED558CCDULL ^ ((uint64_t)(N) + 3u);                       \
  a += 0xABCDEF0123456789ULL + (uint64_t)(N) * 5u;                         \
  a ^= (a >> 11) + (uint64_t)(N) * 1234567u;                              \
  a *= (((uint64_t)(N) * 99u) | 1u);                                       \
  a += 0x55AA55AA55AA55AAULL ^ ((uint64_t)(N) << 2);                       \
  a ^= 0x0F0F0F0F0F0F0F0FULL + (uint64_t)(N) * 3u;                         \
  a -= (a >> 5) ^ ((uint64_t)(N) * 7u);                                    \
  a *= (0xBF58476D1CE4E5B9ULL | ((uint64_t)(N) + 1u));                     \
  a += (a << 5) + (uint64_t)(N) * 65537u;                                  \
  a ^= 0x94D049BB133111EBULL ^ ((uint64_t)(N) * 11u);                      \
  a += 0x2545F4914F6CDD1DULL + (uint64_t)(N) * 19u;                        \
  a *= (((uint64_t)(N) << 3) | 1u);

#define FE_OPS_B(N)                                                        \
  a ^= 0x6A09E667F3BCC908ULL + (uint64_t)(N) * 0x85EBCA77ULL;             \
  a += 0xBB67AE8584CAA73BULL ^ ((uint64_t)(N) << 2);                       \
  a *= (0x27D4EB2F165667C5ULL | ((uint64_t)(N) << 1));                     \
  a -= (a >> 9) + (uint64_t)(N) * 6151ULL;                                 \
  a ^= (a << 4) ^ ((uint64_t)(N) * 37u + 23u);                            \
  a += 0xFEEDFACE1234ULL - (uint64_t)(N) * 17u;                            \
  a *= 0xC6A4A7935BD1E995ULL ^ ((uint64_t)(N) + 7u);                       \
  a -= 0x0123456789ABCDEFULL + (uint64_t)(N) * 11u;                        \
  a ^= (a >> 13) + (uint64_t)(N) * 2654435u;                              \
  a *= (((uint64_t)(N) * 131u) | 1u);                                      \
  a += 0xA5A5A5A5A5A5A5A5ULL ^ ((uint64_t)(N) << 3);                       \
  a ^= 0x3333333333333333ULL + (uint64_t)(N) * 9u;                         \
  a -= (a >> 6) ^ ((uint64_t)(N) * 5u);                                    \
  a *= (0xD6E8FEB86659FD93ULL | ((uint64_t)(N) + 5u));                     \
  a += (a << 7) + (uint64_t)(N) * 49157u;                                  \
  a ^= 0xB492B66FBE98F273ULL ^ ((uint64_t)(N) * 13u);                      \
  a += 0x9FB21C651E98DF25ULL + (uint64_t)(N) * 29u;                        \
  a *= (((uint64_t)(N) << 4) | 1u);

#define FE_OPS_C(N)                                                        \
  a += 0x510E527FADE682D1ULL ^ (uint64_t)(N) * 0xCC9E2D51ULL;             \
  a ^= 0x9B05688C2B3E6C1FULL + ((uint64_t)(N) << 1);                       \
  a *= (0x1B873593ULL | ((uint64_t)(N) << 2));                             \
  a -= (a >> 8) + (uint64_t)(N) * 3257ULL;                                 \
  a ^= (a << 5) ^ ((uint64_t)(N) * 41u + 19u);                            \
  a += 0xCAFEB0BA1357ULL - (uint64_t)(N) * 23u;                            \
  a *= 0x2127599BF4325C37ULL ^ ((uint64_t)(N) + 11u);                      \
  a -= 0xFEDCBA9876543210ULL + (uint64_t)(N) * 7u;                         \
  a ^= (a >> 15) + (uint64_t)(N) * 40503u;                                \
  a *= (((uint64_t)(N) * 167u) | 1u);                                      \
  a += 0x5C5C5C5C5C5C5C5CULL ^ ((uint64_t)(N) << 4);                       \
  a ^= 0x7777777777777777ULL + (uint64_t)(N) * 15u;                        \
  a -= (a >> 4) ^ ((uint64_t)(N) * 3u);                                    \
  a *= (0x8EBC6AF09C88C6E3ULL | ((uint64_t)(N) + 9u));                     \
  a += (a << 6) + (uint64_t)(N) * 24593u;                                  \
  a ^= 0x589965CC75374CC3ULL ^ ((uint64_t)(N) * 17u);                      \
  a += 0x1D8E4E27C47D124FULL + (uint64_t)(N) * 31u;                        \
  a *= (((uint64_t)(N) << 5) | 1u);

#define FE_C16(B, OPS)                                                     \
  case (B) + 0: OPS((B) + 0) break;                                        \
  case (B) + 1: OPS((B) + 1) break;                                        \
  case (B) + 2: OPS((B) + 2) break;                                        \
  case (B) + 3: OPS((B) + 3) break;                                        \
  case (B) + 4: OPS((B) + 4) break;                                        \
  case (B) + 5: OPS((B) + 5) break;                                        \
  case (B) + 6: OPS((B) + 6) break;                                        \
  case (B) + 7: OPS((B) + 7) break;                                        \
  case (B) + 8: OPS((B) + 8) break;                                        \
  case (B) + 9: OPS((B) + 9) break;                                        \
  case (B) + 10: OPS((B) + 10) break;                                      \
  case (B) + 11: OPS((B) + 11) break;                                      \
  case (B) + 12: OPS((B) + 12) break;                                      \
  case (B) + 13: OPS((B) + 13) break;                                      \
  case (B) + 14: OPS((B) + 14) break;                                      \
  case (B) + 15: OPS((B) + 15) break;

#define FE_SWITCH256(OPS)                                                  \
  FE_C16(0, OPS) FE_C16(16, OPS) FE_C16(32, OPS) FE_C16(48, OPS)           \
  FE_C16(64, OPS) FE_C16(80, OPS) FE_C16(96, OPS) FE_C16(112, OPS)         \
  FE_C16(128, OPS) FE_C16(144, OPS) FE_C16(160, OPS) FE_C16(176, OPS)      \
  FE_C16(192, OPS) FE_C16(208, OPS) FE_C16(224, OPS) FE_C16(240, OPS)

__attribute__((noinline)) static uint64_t feSwitchA(uint64_t a, uint32_t idx) {
  switch (idx & 0xFFu) {
    FE_SWITCH256(FE_OPS_A)
  }
  return a;
}

__attribute__((noinline)) static uint64_t feSwitchB(uint64_t a, uint32_t idx) {
  switch (idx & 0xFFu) {
    FE_SWITCH256(FE_OPS_B)
  }
  return a;
}

__attribute__((noinline)) static uint64_t feSwitchC(uint64_t a, uint32_t idx) {
  switch (idx & 0xFFu) {
    FE_SWITCH256(FE_OPS_C)
  }
  return a;
}

static const std::vector<uint32_t>& feChase() {
  static const std::vector<uint32_t> chase = [] {
    std::vector<uint32_t> c(4096);
    for (size_t i = 0; i < c.size(); ++i) {
      c[i] = static_cast<uint32_t>((i * 2654435761u) ^ (i << 7) ^ (i >> 3));
    }
    return c;
  }();
  return chase;
}

static void contendedUseFrontend(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  const std::vector<uint32_t>& chase = feChase();
  const uint32_t mask = static_cast<uint32_t>(chase.size() - 1);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t a = 0x12345678ULL + static_cast<uint64_t>(t);
        uint32_t p = chase[static_cast<uint32_t>(t) & mask];
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (int j = 0; j < 24; ++j) {
            uint32_t idx = p & 0xFFu;
            switch (j % 3) {
              case 0:
                a = feSwitchA(a, idx);
                break;
              case 1:
                a = feSwitchB(a, idx);
                break;
              default:
                a = feSwitchC(a, idx);
                break;
            }
            p = chase[(p ^ static_cast<uint32_t>(a)) & mask];
          }
        }
        folly::doNotOptimizeAway(a);
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
BENCHMARK_NAMED_PARAM(contendedUseFrontend, 8_to_100_fe, 8, 100)

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
