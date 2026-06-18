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

// --- Derived benchmark: contendedUse with an oversized executed-instruction
// working set to raise frontend (L1 icache) pressure and lower IPC.
// Three noinline 256-case switch functions, each case with 14 unique-literal
// ALU ops, are rotated by j%3 and indexed by a pointer-chase payload byte.
#define LSB_REP16(M, B)                                                    \
  M(B + 0) M(B + 1) M(B + 2) M(B + 3) M(B + 4) M(B + 5) M(B + 6) M(B + 7)  \
  M(B + 8) M(B + 9) M(B + 10) M(B + 11) M(B + 12) M(B + 13) M(B + 14)      \
  M(B + 15)

#define LSB_REP256(M)                                                      \
  LSB_REP16(M, 0) LSB_REP16(M, 16) LSB_REP16(M, 32) LSB_REP16(M, 48)       \
  LSB_REP16(M, 64) LSB_REP16(M, 80) LSB_REP16(M, 96) LSB_REP16(M, 112)     \
  LSB_REP16(M, 128) LSB_REP16(M, 144) LSB_REP16(M, 160) LSB_REP16(M, 176)  \
  LSB_REP16(M, 192) LSB_REP16(M, 208) LSB_REP16(M, 224) LSB_REP16(M, 240)

#define LSB_CASE_A(i)                                                      \
  case (i): {                                                              \
    acc += 0x9E3779B97F4A7C15ULL + (uint64_t)(i)*0x100000001B3ULL;         \
    acc ^= (uint64_t)(i)*0xFF51AFD7ED558CCDULL;                            \
    acc *= ((uint64_t)(i) | 1ULL);                                         \
    acc += (uint64_t)(i) ^ 0x00C0FFEEULL;                                  \
    acc -= (uint64_t)(i)*7ULL + 11ULL;                                     \
    acc ^= ((uint64_t)(i) << 7) + 3ULL;                                    \
    acc += (uint64_t)(i)*1315423911ULL;                                    \
    acc *= (3ULL + ((uint64_t)(i)&7ULL));                                  \
    acc ^= (uint64_t)(i)*0x2545F4914F6CDD1DULL;                            \
    acc += (uint64_t)(i) + 0xDEADBEEFULL;                                  \
    acc -= (uint64_t)(i)*5ULL + 17ULL;                                     \
    acc ^= (uint64_t)(i)*0x85EBCA6BULL;                                    \
    acc += (uint64_t)(i)*0xC2B2AE3D27D4EB4FULL;                            \
    acc *= (1ULL + ((uint64_t)(i)&15ULL));                                 \
    break;                                                                 \
  }

#define LSB_CASE_B(i)                                                      \
  case (i): {                                                              \
    acc ^= 0xA0761D6478BD642FULL + (uint64_t)(i)*0x9E3779B1ULL;            \
    acc += (uint64_t)(i)*0xE7037ED1A0B428DBULL;                            \
    acc *= ((uint64_t)(i) | 3ULL);                                        \
    acc -= (uint64_t)(i) ^ 0x0BADF00DULL;                                  \
    acc += (uint64_t)(i)*13ULL + 23ULL;                                    \
    acc ^= ((uint64_t)(i) << 5) + 9ULL;                                    \
    acc -= (uint64_t)(i)*2654435761ULL;                                    \
    acc *= (5ULL + ((uint64_t)(i)&7ULL));                                  \
    acc ^= (uint64_t)(i)*0x8EBC6AF09C88C6E3ULL;                            \
    acc += (uint64_t)(i) + 0xFEEDFACEULL;                                  \
    acc -= (uint64_t)(i)*9ULL + 29ULL;                                     \
    acc ^= (uint64_t)(i)*0xC2B2AE35ULL;                                    \
    acc += (uint64_t)(i)*0x589965CC75374CC3ULL;                            \
    acc *= (1ULL + ((uint64_t)(i)&31ULL));                                 \
    break;                                                                 \
  }

#define LSB_CASE_C(i)                                                      \
  case (i): {                                                              \
    acc += 0x27D4EB2F165667C5ULL ^ (uint64_t)(i)*0x9E3779ULL;              \
    acc ^= (uint64_t)(i)*0x94D049BB133111EBULL;                            \
    acc *= ((uint64_t)(i) | 7ULL);                                        \
    acc += (uint64_t)(i) ^ 0x1337C0DEULL;                                  \
    acc -= (uint64_t)(i)*19ULL + 31ULL;                                    \
    acc ^= ((uint64_t)(i) << 9) + 13ULL;                                   \
    acc += (uint64_t)(i)*40503ULL;                                         \
    acc *= (7ULL + ((uint64_t)(i)&7ULL));                                  \
    acc ^= (uint64_t)(i)*0xD6E8FEB86659FD93ULL;                            \
    acc += (uint64_t)(i) + 0xCAFEBABEULL;                                  \
    acc -= (uint64_t)(i)*11ULL + 37ULL;                                    \
    acc ^= (uint64_t)(i)*0x165667B1ULL;                                    \
    acc += (uint64_t)(i)*0xFF51AFD7ED558CCDULL;                            \
    acc *= (1ULL + ((uint64_t)(i)&63ULL));                                 \
    break;                                                                 \
  }

__attribute__((noinline)) static uint64_t lsbIcacheWorkA(
    uint64_t acc, uint8_t idx) {
  switch (idx) { LSB_REP256(LSB_CASE_A) }
  return acc;
}

__attribute__((noinline)) static uint64_t lsbIcacheWorkB(
    uint64_t acc, uint8_t idx) {
  switch (idx) { LSB_REP256(LSB_CASE_B) }
  return acc;
}

__attribute__((noinline)) static uint64_t lsbIcacheWorkC(
    uint64_t acc, uint8_t idx) {
  switch (idx) { LSB_REP256(LSB_CASE_C) }
  return acc;
}

static const std::vector<uint8_t>& lsbChasePayload() {
  static const std::vector<uint8_t> chase = [] {
    std::vector<uint8_t> v(8192);
    uint64_t x = 0x243F6A8885A308D3ULL;
    for (size_t i = 0; i < v.size(); ++i) {
      x = x * 6364136223846793005ULL + 1442695040888963407ULL;
      v[i] = static_cast<uint8_t>((x >> 33) & 0xFF);
    }
    return v;
  }();
  return chase;
}

static void contendedUseWork(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  const std::vector<uint8_t>& chase = lsbChasePayload();
  const size_t mask = chase.size() - 1;

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        size_t p = static_cast<size_t>(t) * 2654435761u;
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = (p * 6364136223846793005ULL + acc) & mask;
          uint8_t idx = static_cast<uint8_t>(chase[p] & 0xFF);
          switch (j % 3) {
            case 0:
              acc = lsbIcacheWorkA(acc, idx);
              break;
            case 1:
              acc = lsbIcacheWorkB(acc, idx);
              break;
            default:
              acc = lsbIcacheWorkC(acc, idx);
              break;
          }
          ++j;
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
BENCHMARK_NAMED_PARAM(contendedUseWork, 8_to_100_icache, 8, 100)

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
