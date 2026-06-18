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

// --- Derived from contendedUse(8_to_100): adds a large hot-code footprint to
// raise the executed instruction working set (frontend / L1i pressure) while
// keeping the same contended LifoSem post/wait structure. The three noinline
// 256-case switch functions are rotated with j%3 and indexed by a pointer-chase
// payload byte so the compiler cannot prune or deduplicate the code.
namespace {

constexpr uint32_t kChaseSize = 8192u;
uint32_t kNext[kChaseSize];

bool initChase() {
  uint64_t x = 0x9E3779B97F4A7C15ull;
  for (uint32_t i = 0; i < kChaseSize; ++i) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    kNext[i] = static_cast<uint32_t>(x % kChaseSize);
  }
  return true;
}
bool kChaseInit = initChase();

#define R16(M, b)                                                            \
  M(b + 0) M(b + 1) M(b + 2) M(b + 3) M(b + 4) M(b + 5) M(b + 6) M(b + 7)    \
      M(b + 8) M(b + 9) M(b + 10) M(b + 11) M(b + 12) M(b + 13) M(b + 14)    \
          M(b + 15)
#define R256(M)                                                             \
  R16(M, 0) R16(M, 16) R16(M, 32) R16(M, 48) R16(M, 64) R16(M, 80)          \
      R16(M, 96) R16(M, 112) R16(M, 128) R16(M, 144) R16(M, 160)            \
          R16(M, 176) R16(M, 192) R16(M, 208) R16(M, 224) R16(M, 240)

#define OPS_A(i)                                                            \
  case (i): {                                                               \
    acc += (uint64_t)(i) * 2654435761ull + 0x9E3779B1ull;                   \
    acc ^= (uint64_t)(i) * 40503ull + 0x01000193ull;                        \
    acc += (uint64_t)(i) << (((i) & 15) + 1);                               \
    acc *= ((uint64_t)(i) | 3ull);                                          \
    acc ^= (uint64_t)(i) * 2246822519ull;                                   \
    acc += (uint64_t)(i) * 3266489917ull;                                   \
    acc -= (uint64_t)(i) * 668265263ull;                                    \
    acc ^= (uint64_t)(i) * 374761393ull;                                    \
    acc += (uint64_t)(i) ^ 0x5BD1E995ull;                                   \
    acc *= 0x100000001B3ull;                                                \
    acc ^= acc >> 7;                                                        \
    acc += (uint64_t)(i) * 9176ull + 11ull;                                 \
    acc -= (uint64_t)(i) * 4099ull;                                         \
    acc ^= (uint64_t)(i) * 2147483647ull;                                   \
    break;                                                                  \
  }

#define OPS_B(i)                                                            \
  case (i): {                                                               \
    acc ^= (uint64_t)(i) * 2891336453ull + 0x27D4EB2Full;                   \
    acc += (uint64_t)(i) * 0x85EBCA6Bull + 0x165667B1ull;                   \
    acc -= (uint64_t)(i) << (((i) & 7) + 2);                                \
    acc *= ((uint64_t)(i) | 5ull);                                          \
    acc ^= (uint64_t)(i) * 1640531527ull;                                   \
    acc += (uint64_t)(i) * 4256249ull;                                      \
    acc -= (uint64_t)(i) * 741103597ull;                                    \
    acc ^= (uint64_t)(i) * 0xC2B2AE35ull;                                   \
    acc += (uint64_t)(i) ^ 0xCC9E2D51ull;                                   \
    acc *= 0x9E3779B97F4A7C15ull;                                           \
    acc ^= acc >> 11;                                                       \
    acc += (uint64_t)(i) * 6151ull + 17ull;                                 \
    acc -= (uint64_t)(i) * 3079ull;                                         \
    acc ^= (uint64_t)(i) * 1000000007ull;                                   \
    break;                                                                  \
  }

#define OPS_C(i)                                                            \
  case (i): {                                                               \
    acc += (uint64_t)(i) * 0x1B873593ull + 0xE6546B64ull;                   \
    acc ^= (uint64_t)(i) * 0xFF51AFD7ull + 0xC4CEB9FEull;                   \
    acc += (uint64_t)(i) << (((i) & 31) + 1);                               \
    acc *= ((uint64_t)(i) | 7ull);                                          \
    acc ^= (uint64_t)(i) * 0xD6E8FEB8ull;                                   \
    acc += (uint64_t)(i) * 2129725531ull;                                   \
    acc -= (uint64_t)(i) * 433494437ull;                                    \
    acc ^= (uint64_t)(i) * 0x9DDFEA08ull;                                   \
    acc += (uint64_t)(i) ^ 0x6C62272Eull;                                   \
    acc *= 0xBF58476D1CE4E5B9ull;                                           \
    acc ^= acc >> 13;                                                       \
    acc += (uint64_t)(i) * 7919ull + 23ull;                                 \
    acc -= (uint64_t)(i) * 5147ull;                                         \
    acc ^= (uint64_t)(i) * 2038074743ull;                                   \
    break;                                                                  \
  }

__attribute__((noinline)) uint64_t icache_fn0(uint8_t idx, uint64_t acc) {
  switch (idx) { R256(OPS_A) }
  return acc;
}

__attribute__((noinline)) uint64_t icache_fn1(uint8_t idx, uint64_t acc) {
  switch (idx) { R256(OPS_B) }
  return acc;
}

__attribute__((noinline)) uint64_t icache_fn2(uint8_t idx, uint64_t acc) {
  switch (idx) { R256(OPS_C) }
  return acc;
}

#undef OPS_A
#undef OPS_B
#undef OPS_C
#undef R256
#undef R16

std::atomic<uint64_t> icacheSink{0};

} // namespace

static void icacheContendedUse(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = 0x12345ull + static_cast<uint64_t>(t) * 0x9E3779B1ull;
        uint32_t chase =
            (static_cast<uint32_t>(t) * 2654435761u) & (kChaseSize - 1);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          chase = kNext[chase];
          uint8_t p = static_cast<uint8_t>(chase & 0xFFu);
          switch (j % 3) {
            case 0:
              acc = icache_fn0(p, acc);
              break;
            case 1:
              acc = icache_fn1(p, acc);
              break;
            default:
              acc = icache_fn2(p, acc);
              break;
          }
          ++j;
        }
        icacheSink.fetch_add(acc, std::memory_order_relaxed);
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
BENCHMARK_NAMED_PARAM(icacheContendedUse, 8_to_100, 8, 100)

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
