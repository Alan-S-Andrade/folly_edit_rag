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

// Derived from contendedUse(8_to_100): identical contention shape, but each
// waiter wakeup drives a pointer-chased index into one of three large
// 256-case switch dispatchers. The three rotating noinline dispatchers, each
// with many unique-constant ALU ops per case, inflate the executed
// instruction working set to raise the frontend / L1i pressure that the plain
// LifoSem contention loop lacks.
namespace {

constexpr size_t kChainLen = 8192;

std::vector<uint32_t> g_chain = [] {
  std::vector<uint32_t> v(kChainLen);
  uint32_t x = 2463534242u;
  for (size_t i = 0; i < kChainLen; ++i) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    v[i] = x;
  }
  return v;
}();

#define LSB_REP16(M, b)                                                       \
  M((b) + 0) M((b) + 1) M((b) + 2) M((b) + 3) M((b) + 4) M((b) + 5)           \
      M((b) + 6) M((b) + 7) M((b) + 8) M((b) + 9) M((b) + 10) M((b) + 11)     \
          M((b) + 12) M((b) + 13) M((b) + 14) M((b) + 15)

#define LSB_REP256(M)                                                         \
  LSB_REP16(M, 0) LSB_REP16(M, 16) LSB_REP16(M, 32) LSB_REP16(M, 48)          \
      LSB_REP16(M, 64) LSB_REP16(M, 80) LSB_REP16(M, 96) LSB_REP16(M, 112)    \
          LSB_REP16(M, 128) LSB_REP16(M, 144) LSB_REP16(M, 160)               \
              LSB_REP16(M, 176) LSB_REP16(M, 192) LSB_REP16(M, 208)           \
                  LSB_REP16(M, 224) LSB_REP16(M, 240)

#define LSB_CASE_A(i)                                                         \
  case (i): {                                                                 \
    acc += (uint64_t)(i)*0x9E3779B97F4A7C15ull;                              \
    acc ^= acc >> 23;                                                         \
    acc *= 0xBF58476D1CE4E5B9ull + (uint64_t)(i);                            \
    acc ^= (uint64_t)(i)*0xD6E8FEB86659FD93ull;                             \
    acc += acc << 11;                                                         \
    acc -= (uint64_t)(i) ^ 0xC2B2AE3D27D4EB4Full;                            \
    acc ^= acc >> 17;                                                         \
    acc += (uint64_t)(i)*0x165667B19E3779F9ull;                             \
    acc *= 0x27D4EB2F165667C5ull ^ (uint64_t)(i);                            \
    acc ^= acc << 5;                                                          \
    acc += (uint64_t)(i) + 0x94D049BB133111EBull;                            \
    acc ^= acc >> 29;                                                         \
    acc *= 0x2545F4914F6CDD1Dull + (uint64_t)(i);                            \
    acc ^= (uint64_t)(i)*0x9FB21C651E98DF25ull;                             \
    acc += acc << 7;                                                          \
    acc -= (uint64_t)(i) ^ 0xFF51AFD7ED558CCDull;                            \
  } break;

#define LSB_CASE_B(i)                                                         \
  case (i): {                                                                 \
    acc ^= (uint64_t)(i)*0xA0761D6478BD642Full;                             \
    acc += acc << 13;                                                         \
    acc *= 0xE7037ED1A0B428DBull + (uint64_t)(i);                            \
    acc ^= acc >> 19;                                                         \
    acc -= (uint64_t)(i) ^ 0x8EBC6AF09C88C6E3ull;                            \
    acc += (uint64_t)(i)*0x589965CC75374CC3ull;                             \
    acc ^= acc << 9;                                                          \
    acc *= 0x1D8E4E27C47D124Full ^ (uint64_t)(i);                            \
    acc += (uint64_t)(i) + 0x4CF5AD432745937Full;                            \
    acc ^= acc >> 31;                                                         \
    acc -= (uint64_t)(i)*0x9E3779B185EBCA87ull;                             \
    acc += acc << 3;                                                          \
    acc ^= (uint64_t)(i) ^ 0xC4CEB9FE1A85EC53ull;                            \
    acc *= 0x2127599BF4325C37ull + (uint64_t)(i);                            \
    acc ^= acc >> 11;                                                         \
    acc += (uint64_t)(i)*0x880355F21E6D1965ull;                             \
  } break;

#define LSB_CASE_C(i)                                                         \
  case (i): {                                                                 \
    acc *= 0xF1357AEA2E62A9C5ull + (uint64_t)(i);                            \
    acc ^= acc >> 27;                                                         \
    acc += (uint64_t)(i)*0xACAACAACAC0FFEE1ull;                             \
    acc ^= acc << 15;                                                         \
    acc -= (uint64_t)(i) ^ 0x6A09E667F3BCC909ull;                            \
    acc += (uint64_t)(i)*0xBB67AE8584CAA73Bull;                             \
    acc ^= acc >> 7;                                                          \
    acc *= 0x3C6EF372FE94F82Bull ^ (uint64_t)(i);                            \
    acc += acc << 17;                                                         \
    acc ^= (uint64_t)(i) + 0xA54FF53A5F1D36F1ull;                            \
    acc -= (uint64_t)(i)*0x510E527FADE682D1ull;                             \
    acc ^= acc >> 21;                                                         \
    acc += (uint64_t)(i) ^ 0x9B05688C2B3E6C1Full;                            \
    acc *= 0x1F83D9ABFB41BD6Bull + (uint64_t)(i);                            \
    acc ^= acc << 25;                                                         \
    acc += (uint64_t)(i)*0x5BE0CD19137E2179ull;                             \
  } break;

__attribute__((noinline)) uint64_t lsb_switch_a(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LSB_REP256(LSB_CASE_A)
  }
  return acc;
}

__attribute__((noinline)) uint64_t lsb_switch_b(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LSB_REP256(LSB_CASE_B)
  }
  return acc;
}

__attribute__((noinline)) uint64_t lsb_switch_c(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LSB_REP256(LSB_CASE_C)
  }
  return acc;
}

#undef LSB_CASE_A
#undef LSB_CASE_B
#undef LSB_CASE_C
#undef LSB_REP256
#undef LSB_REP16

} // namespace

static void contendedUseSwitch(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        size_t p = (static_cast<size_t>(t) * 2654435761u) % kChainLen;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          uint32_t payload = g_chain[p];
          p = payload % kChainLen;
          uint8_t idx = static_cast<uint8_t>(payload & 0xFF);
          switch (i % 3) {
            case 0:
              acc = lsb_switch_a(acc, idx);
              break;
            case 1:
              acc = lsb_switch_b(acc, idx);
              break;
            default:
              acc = lsb_switch_c(acc, idx);
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
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1, 31, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 16_to_16, 16, 16)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_32, 32, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1000, 32, 1000)
BENCHMARK_NAMED_PARAM(contendedUseSwitch, 8_to_100, 8, 100)

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
