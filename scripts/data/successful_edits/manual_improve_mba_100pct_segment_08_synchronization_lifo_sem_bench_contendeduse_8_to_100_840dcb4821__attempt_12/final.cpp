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
// Derived benchmark: contendedUseICache(8_to_100)
//
// This variant keeps the original contended LifoSem usage pattern but injects
// a large hot-code footprint into each poster/waiter iteration to raise the
// L1 instruction-cache miss rate (and thus lower the otherwise-too-high IPC).
// Three noinline functions, each with a 256-case switch of ALU ops on a local
// accumulator, are rotated via j%3. The switch index comes from the low byte
// of a pointer-chase load so the case selection is data dependent and the
// instruction working set cannot be hoisted or folded.
// ---------------------------------------------------------------------------
namespace {

struct ChaseNode {
  ChaseNode* next;
  uint64_t payload;
};

std::atomic<uint64_t> gIcacheSink{0};

// 14 ALU ops per case; every op references the case constant K (and the
// function salt S) so the compiler cannot deduplicate cases or functions.
#define ALU_OPS(K, S)                                            \
  a += (uint64_t)(K) * 0x9e3779b97f4a7c15ull + (uint64_t)(S);    \
  a ^= a >> 7;                                                   \
  a *= 0x2545f4914f6cdd1dull + (uint64_t)(K);                    \
  a += (uint64_t)((K) * 31u + 7u);                               \
  a ^= a << 11;                                                  \
  a -= (uint64_t)(K) ^ 0xabcdu;                                  \
  a *= 3u + (uint64_t)(K);                                       \
  a += a >> 3;                                                   \
  a ^= (uint64_t)(K) * 12345u + (uint64_t)(S);                   \
  a += 0x55u + (uint64_t)(K);                                    \
  a *= 0x100000001b3ull ^ (uint64_t)(K);                         \
  a ^= a >> 5;                                                   \
  a += (uint64_t)(K) * 777u + (uint64_t)(S);                     \
  a *= 5u + (uint64_t)(K);

#define ALU_CASE(K, S) \
  case (K): {          \
    ALU_OPS(K, S)      \
  } break;

#define ALU_CASES16(B, S)                                                 \
  ALU_CASE((B) + 0, S) ALU_CASE((B) + 1, S) ALU_CASE((B) + 2, S)          \
      ALU_CASE((B) + 3, S) ALU_CASE((B) + 4, S) ALU_CASE((B) + 5, S)      \
          ALU_CASE((B) + 6, S) ALU_CASE((B) + 7, S) ALU_CASE((B) + 8, S)  \
              ALU_CASE((B) + 9, S) ALU_CASE((B) + 10, S)                  \
                  ALU_CASE((B) + 11, S) ALU_CASE((B) + 12, S)             \
                      ALU_CASE((B) + 13, S) ALU_CASE((B) + 14, S)         \
                          ALU_CASE((B) + 15, S)

#define ALU_CASES256(S)                                                   \
  ALU_CASES16(0, S) ALU_CASES16(16, S) ALU_CASES16(32, S)                 \
      ALU_CASES16(48, S) ALU_CASES16(64, S) ALU_CASES16(80, S)            \
          ALU_CASES16(96, S) ALU_CASES16(112, S) ALU_CASES16(128, S)      \
              ALU_CASES16(144, S) ALU_CASES16(160, S) ALU_CASES16(176, S) \
                  ALU_CASES16(192, S) ALU_CASES16(208, S)                 \
                      ALU_CASES16(224, S) ALU_CASES16(240, S)

__attribute__((noinline)) static uint64_t icacheWorkA(uint64_t a, uint8_t idx) {
  switch (idx) {
    ALU_CASES256(0xA001u)
  }
  return a;
}

__attribute__((noinline)) static uint64_t icacheWorkB(uint64_t a, uint8_t idx) {
  switch (idx) {
    ALU_CASES256(0xB002u)
  }
  return a;
}

__attribute__((noinline)) static uint64_t icacheWorkC(uint64_t a, uint8_t idx) {
  switch (idx) {
    ALU_CASES256(0xC003u)
  }
  return a;
}

#undef ALU_CASES256
#undef ALU_CASES16
#undef ALU_CASE
#undef ALU_OPS

} // namespace

static void contendedUseICache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  constexpr size_t kRing = 1024;

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        std::vector<ChaseNode> ring(kRing);
        for (size_t k = 0; k < kRing; ++k) {
          ring[k].next = &ring[(k * 7 + 13) % kRing];
          ring[k].payload = k * 2654435761u + (uint64_t)t;
        }
        ChaseNode* p = &ring[0];
        uint64_t acc = 0x12345u + (uint64_t)t;
        uint64_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = p->next;
          uint8_t idx = (uint8_t)(p->payload & 0xFF);
          switch (j++ % 3) {
            case 0:
              acc = icacheWorkA(acc, idx);
              break;
            case 1:
              acc = icacheWorkB(acc, idx);
              break;
            default:
              acc = icacheWorkC(acc, idx);
              break;
          }
        }
        gIcacheSink.fetch_add(acc, std::memory_order_relaxed);
      });
    }
    for (int t = 0; t < posters; ++t) {
      threads.emplace_back([=, &sem, &go] {
        std::vector<ChaseNode> ring(kRing);
        for (size_t k = 0; k < kRing; ++k) {
          ring[k].next = &ring[(k * 7 + 13) % kRing];
          ring[k].payload = k * 2654435761u + (uint64_t)t + 99u;
        }
        ChaseNode* p = &ring[0];
        uint64_t acc = 0x67890u + (uint64_t)t;
        uint64_t j = 0;
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          sem.post();
          p = p->next;
          uint8_t idx = (uint8_t)(p->payload & 0xFF);
          switch (j++ % 3) {
            case 0:
              acc = icacheWorkA(acc, idx);
              break;
            case 1:
              acc = icacheWorkB(acc, idx);
              break;
            default:
              acc = icacheWorkC(acc, idx);
              break;
          }
        }
        gIcacheSink.fetch_add(acc, std::memory_order_relaxed);
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
BENCHMARK_NAMED_PARAM(contendedUseICache, 8_to_100, 8, 100)

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
