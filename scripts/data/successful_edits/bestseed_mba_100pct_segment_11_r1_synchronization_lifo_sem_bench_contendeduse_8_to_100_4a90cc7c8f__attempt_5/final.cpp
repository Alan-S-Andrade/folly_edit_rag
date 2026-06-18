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
// Frontend-pressure variant of contendedUse. The waiter threads execute a
// large, branchy code footprint (three noinline 256-case switches selected by
// a pointer-chase payload) so that the executed instruction working set no
// longer fits comfortably in the L1 instruction cache. This intentionally
// lowers IPC toward the frontend-bound target by raising I-cache pressure.
// ---------------------------------------------------------------------------

// 14 ALU ops per case; (i) makes each case's literals unique to defeat
// compiler case-deduplication, (s) makes each of the three functions distinct
// to defeat identical-code-folding.
#define ALU_OPS(i, s)                                            \
  a += (uint64_t)((i) * 2654435761ull + (s) * 1000003ull + 1ull); \
  a ^= (uint64_t)((i) * 40503ull + (s) * 7919ull + 7ull);         \
  a *= (uint64_t)(((i) | 3ull) + (s) + 11ull);                    \
  a += (uint64_t)(((i) ^ 0x5bd1e995ull) + (s));                   \
  a ^= a >> 13;                                                    \
  a *= (uint64_t)((i) * 0x85ebca6bull + (s) * 13ull + 17ull);     \
  a += (uint64_t)(((i) << 3) + (s) * 3ull + 29ull);               \
  a ^= (uint64_t)((i) * 0xc2b2ae35ull + (s) * 19ull + 53ull);     \
  a *= (uint64_t)((((i) + 1ull) | 1ull) + (s));                   \
  a += (uint64_t)((i) * 99991ull + (s) * 23ull + 101ull);         \
  a ^= a << 7;                                                     \
  a *= (uint64_t)((i) * 0x27d4eb2full + (s) * 29ull + 131ull);    \
  a += (uint64_t)(((i) ^ 0xdeadbeefull) + (s) * 31ull);           \
  a ^= (uint64_t)((i) * 7ull + (s) * 37ull + 197ull);

#define ALU_CASE_S(i, s) \
  case (i): {            \
    ALU_OPS(i, s)        \
    break;               \
  }
#define ALU_CASE0(i) ALU_CASE_S(i, 1ull)
#define ALU_CASE1(i) ALU_CASE_S(i, 2ull)
#define ALU_CASE2(i) ALU_CASE_S(i, 3ull)

#define REP16(M, b)                                           \
  M((b) + 0) M((b) + 1) M((b) + 2) M((b) + 3) M((b) + 4)      \
      M((b) + 5) M((b) + 6) M((b) + 7) M((b) + 8) M((b) + 9)  \
          M((b) + 10) M((b) + 11) M((b) + 12) M((b) + 13)     \
              M((b) + 14) M((b) + 15)

#define REP256(M)                                                          \
  REP16(M, 0) REP16(M, 16) REP16(M, 32) REP16(M, 48) REP16(M, 64)          \
      REP16(M, 80) REP16(M, 96) REP16(M, 112) REP16(M, 128) REP16(M, 144)  \
          REP16(M, 160) REP16(M, 176) REP16(M, 192) REP16(M, 208)          \
              REP16(M, 224) REP16(M, 240)

__attribute__((noinline)) static uint64_t aluSwitch0(uint64_t a, uint8_t idx) {
  switch (idx) { REP256(ALU_CASE0) }
  return a;
}

__attribute__((noinline)) static uint64_t aluSwitch1(uint64_t a, uint8_t idx) {
  switch (idx) { REP256(ALU_CASE1) }
  return a;
}

__attribute__((noinline)) static uint64_t aluSwitch2(uint64_t a, uint8_t idx) {
  switch (idx) { REP256(ALU_CASE2) }
  return a;
}

__attribute__((noinline)) static uint64_t frontendWork(
    const uint32_t* chase, uint32_t mask, uint32_t p, uint64_t a) {
  for (int j = 0; j < 16; ++j) {
    p = chase[p & mask]; // pointer-chase load
    uint8_t idx = (uint8_t)(p & 0xFF);
    switch (j % 3) {
      case 0:
        a = aluSwitch0(a, idx);
        break;
      case 1:
        a = aluSwitch1(a, idx);
        break;
      default:
        a = aluSwitch2(a, idx);
        break;
    }
  }
  return a ^ p;
}

static void contendedUseFront(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  static const std::vector<uint32_t> chase = [] {
    std::vector<uint32_t> v(4096);
    for (uint32_t i = 0; i < v.size(); ++i) {
      v[i] = (i * 2654435761u + 1013904223u) & 4095u;
    }
    return v;
  }();
  const uint32_t mask = (uint32_t)chase.size() - 1u;

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &sink] {
        uint64_t a = (uint64_t)t + 1;
        uint32_t p = (uint32_t)t * 2654435761u;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          a = frontendWork(chase.data(), mask, p ^ (uint32_t)a, a);
          p = (uint32_t)a;
        }
        sink.fetch_add(a, std::memory_order_relaxed);
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
  folly::doNotOptimizeAway(sink.load());
}

BENCHMARK_DRAW_LINE();
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_1, 1, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_4, 1, 4)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_32, 1, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_1, 4, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_24, 4, 24)
BENCHMARK_NAMED_PARAM(contendedUse, 8_to_100, 8, 100)
BENCHMARK_NAMED_PARAM(contendedUseFront, 8_to_100, 8, 100)
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
