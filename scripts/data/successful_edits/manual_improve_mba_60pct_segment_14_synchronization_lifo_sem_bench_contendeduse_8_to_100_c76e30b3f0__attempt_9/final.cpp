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
// Frontend/i-cache footprint helper for the derived icache variant below.
// Three noinline functions, each a fully-populated 256-case switch with a
// fixed number of ALU ops per case using unique literal constants (so the
// compiler cannot deduplicate cases). Rotating among the three functions with
// j % 3 in the hot loop, indexed by a pointer-chase payload, inflates the
// executed instruction working set and drives up L1-icache load misses.
// ---------------------------------------------------------------------------
namespace {

struct ChaseNode {
  ChaseNode* next;
  uint64_t payload;
};

#define WDL_ALU_BODY(K, S)                                                     \
  case (K): {                                                                  \
    acc += (uint64_t)0x9E3779B97F4A7C15ull + ((uint64_t)(K)*0x100000001b3ull); \
    acc ^= (acc >> 7) ^ ((uint64_t)(K) << 1);                                  \
    acc += (acc << 3);                                                         \
    acc *= ((uint64_t)(K) | 1u);                                              \
    acc ^= (acc >> 11);                                                        \
    acc += ((uint64_t)(K) ^ 0xABCDEFu) + (uint64_t)(S);                        \
    acc -= (acc << 5);                                                         \
    acc ^= ((uint64_t)(K)*2654435761u);                                        \
    acc += (acc >> 3);                                                         \
    acc *= 0x2545F4914F6CDD1Dull;                                             \
    acc ^= ((uint64_t)(K) + (uint64_t)(S));                                    \
    acc += (acc << 7);                                                         \
    acc ^= (acc >> 13);                                                        \
    acc += (uint64_t)(K)*7u + 3u;                                              \
    break;                                                                     \
  }

#define WDL_ALU_A(K) WDL_ALU_BODY(K, 0x1111111111111111ull)
#define WDL_ALU_B(K) WDL_ALU_BODY(K, 0x2222222222222222ull)
#define WDL_ALU_C(K) WDL_ALU_BODY(K, 0x3333333333333333ull)

#define WDL_REP16(M, B)                                                        \
  M((B) + 0) M((B) + 1) M((B) + 2) M((B) + 3) M((B) + 4) M((B) + 5)            \
      M((B) + 6) M((B) + 7) M((B) + 8) M((B) + 9) M((B) + 10) M((B) + 11)      \
          M((B) + 12) M((B) + 13) M((B) + 14) M((B) + 15)

#define WDL_REP256(M)                                                          \
  WDL_REP16(M, 0) WDL_REP16(M, 16) WDL_REP16(M, 32) WDL_REP16(M, 48)           \
      WDL_REP16(M, 64) WDL_REP16(M, 80) WDL_REP16(M, 96) WDL_REP16(M, 112)     \
          WDL_REP16(M, 128) WDL_REP16(M, 144) WDL_REP16(M, 160)                \
              WDL_REP16(M, 176) WDL_REP16(M, 192) WDL_REP16(M, 208)            \
                  WDL_REP16(M, 224) WDL_REP16(M, 240)

__attribute__((noinline)) uint64_t wdlIcacheSwitchA(uint64_t acc, uint8_t idx) {
  switch (idx) { WDL_REP256(WDL_ALU_A) }
  return acc;
}

__attribute__((noinline)) uint64_t wdlIcacheSwitchB(uint64_t acc, uint8_t idx) {
  switch (idx) { WDL_REP256(WDL_ALU_B) }
  return acc;
}

__attribute__((noinline)) uint64_t wdlIcacheSwitchC(uint64_t acc, uint8_t idx) {
  switch (idx) { WDL_REP256(WDL_ALU_C) }
  return acc;
}

} // namespace

BENCHMARK(contendedUse_8_to_100_icache, iters) {
  constexpr size_t kRing = 4096;
  std::vector<ChaseNode> nodes(kRing);
  BENCHMARK_SUSPEND {
    for (size_t i = 0; i < kRing; ++i) {
      nodes[i].payload = i * 0x9E3779B97F4A7C15ull + 0xD1B54A32D192ED03ull;
      // stride 7 is coprime to kRing (power of two) => single full cycle.
      nodes[i].next = &nodes[(i * 7 + 1) % kRing];
    }
  }

  ChaseNode* p = &nodes[0];
  uint64_t acc = 0x12345678ull;
  for (size_t i = 0; i < iters; ++i) {
    uint8_t idx = static_cast<uint8_t>((p->payload ^ acc) & 0xFF);
    switch (i % 3) {
      case 0:
        acc = wdlIcacheSwitchA(acc, idx);
        break;
      case 1:
        acc = wdlIcacheSwitchB(acc, idx);
        break;
      default:
        acc = wdlIcacheSwitchC(acc, idx);
        break;
    }
    p = p->next;
  }
  folly::doNotOptimizeAway(acc);
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
