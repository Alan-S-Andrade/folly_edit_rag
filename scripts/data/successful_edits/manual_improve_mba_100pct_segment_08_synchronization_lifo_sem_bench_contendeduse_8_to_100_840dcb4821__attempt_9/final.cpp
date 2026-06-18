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
// Derived frontend-bound experiment for contendedUse(8_to_100).
//
// The original contended LifoSem path is heavily backend/latency bound and
// keeps the executed instruction working set very compact, which produces an
// artificially high IPC and a very low L1 icache miss rate.  To reproduce a
// frontend-bound profile we drive a large, deduplication-resistant code
// footprint: three independent noinline functions, each holding a 256-case
// switch with many unique ALU ops per case.  The hot loop rotates among the
// three functions (j % 3) and selects the switch arm from a pointer-chased
// payload byte (payload & 0xFF), forcing the instruction stream to thrash the
// L1 instruction cache.
// ---------------------------------------------------------------------------
namespace {

#define ALU_CASE(n, S)                                            \
  case (n): {                                                     \
    uint64_t k = (uint64_t)(n);                                   \
    acc += k * 2654435761ull + 0x1u + (S);                        \
    acc ^= k * 40503ull + 0x7u;                                   \
    acc *= ((k | 1u) * 2246822519ull);                            \
    acc += k ^ 0x9e3779b9ull;                                     \
    acc ^= (acc >> 13) + k;                                       \
    acc *= (((k * 3u) + 0x85ebca6bull) | 1u);                     \
    acc += (k << 3) ^ 0xc2b2ae35ull;                              \
    acc ^= (acc << 7) + k + (S);                                  \
    acc *= ((k + 17u) | 1u);                                      \
    acc += 0x27d4eb2full ^ (k * 5u);                              \
    acc ^= (acc >> 11) + (k * 7u);                                \
    acc *= (((k * 11u) + 3u) | 1u);                               \
    acc += (k * 13u) ^ 0xff51afd7ull;                             \
    acc ^= (acc >> 9) + (k * 17u);                                \
    acc *= (((k * 19u) + 23u) | 1u);                              \
    acc += (k * 29u) ^ 0xc4ceb9fe1a85ec53ull;                     \
    break;                                                        \
  }

#define CC4(n, S) \
  ALU_CASE(n, S) ALU_CASE((n) + 1, S) ALU_CASE((n) + 2, S) ALU_CASE((n) + 3, S)
#define CC16(n, S) \
  CC4(n, S) CC4((n) + 4, S) CC4((n) + 8, S) CC4((n) + 12, S)
#define CC64(n, S) \
  CC16(n, S) CC16((n) + 16, S) CC16((n) + 32, S) CC16((n) + 48, S)
#define CC256(S) \
  CC64(0, S) CC64(64, S) CC64(128, S) CC64(192, S)

__attribute__((noinline)) uint64_t icacheSwitchA(uint8_t idx, uint64_t acc) {
  switch (idx) { CC256(0x1111u) }
  return acc;
}

__attribute__((noinline)) uint64_t icacheSwitchB(uint8_t idx, uint64_t acc) {
  switch (idx) { CC256(0x2222u) }
  return acc;
}

__attribute__((noinline)) uint64_t icacheSwitchC(uint8_t idx, uint64_t acc) {
  switch (idx) { CC256(0x3333u) }
  return acc;
}

#undef CC256
#undef CC64
#undef CC16
#undef CC4
#undef ALU_CASE

} // namespace

BENCHMARK(contendedUse_switch_8_to_100, iters) {
  constexpr size_t kN = 4096;
  static std::vector<uint32_t> chase;
  static std::vector<uint8_t> payload;

  BENCHMARK_SUSPEND {
    if (chase.empty()) {
      chase.resize(kN);
      payload.resize(kN);
      uint64_t s = 0x9e3779b97f4a7c15ull;
      auto rnd = [&]() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return s >> 33;
      };
      for (size_t i = 0; i < kN; ++i) {
        chase[i] = static_cast<uint32_t>(rnd() % kN);
        payload[i] = static_cast<uint8_t>(rnd());
      }
    }
  }

  uint64_t acc = 0;
  uint32_t p = 0;
  for (size_t i = 0; i < iters; ++i) {
    p = chase[p];
    uint8_t idx = static_cast<uint8_t>(payload[p] & 0xFF);
    switch (i % 3) {
      case 0:
        acc = icacheSwitchA(idx, acc);
        break;
      case 1:
        acc = icacheSwitchB(idx, acc);
        break;
      default:
        acc = icacheSwitchC(idx, acc);
        break;
    }
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
