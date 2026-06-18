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
// Derived from contendedUse(8_to_100): same semaphore contention pattern, but
// each waiter wakeup drives a large, branch-dispatched code footprint so that
// the executed instruction working set overflows the L1 instruction cache.
// The work is built from three noinline 256-case switches (unique per-case
// literals to defeat ICF/dedup) rotated by j%3 and indexed by the low byte of
// a pointer-chase load.
// ---------------------------------------------------------------------------
namespace {

#define ALU_CASE(SALT, N)                                              \
  case (N): {                                                          \
    acc += (uint64_t)((N) * 0x9e3779b1ull + (SALT) + 0x01ull);        \
    acc ^= (uint64_t)((N) * 0x85ebca77ull + (SALT) + 0x02ull);        \
    acc *= (uint64_t)(((N) | 1u) + (SALT));                           \
    acc -= (uint64_t)((N) * 0xc2b2ae35ull + (SALT) + 0x03ull);        \
    acc ^= (uint64_t)(((N) << 5) + (SALT) + 0x04ull);                 \
    acc += (uint64_t)((N) * 0x27d4eb2full + (SALT) + 0x05ull);        \
    acc *= (uint64_t)(((N) & 0x7f) | 1u);                             \
    acc ^= (uint64_t)((N) * 0x165667b1ull + (SALT) + 0x06ull);        \
    acc -= (uint64_t)((N) * 0xff51afd7ull + (SALT) + 0x07ull);        \
    acc += (uint64_t)(((N) << 3) + (SALT) + 0x08ull);                 \
    acc ^= (uint64_t)((N) * 0xd6e8feb1ull + (SALT) + 0x09ull);        \
    acc *= (uint64_t)(((N) ^ 0x5bd1e995u) | 1u);                      \
    acc -= (uint64_t)((N) * 0xcc9e2d51ull + (SALT) + 0x0aull);        \
    acc += (uint64_t)((N) * 0x1b873593ull + (SALT) + 0x0bull);        \
    acc ^= (uint64_t)(((N) << 7) + (SALT) + 0x0cull);                 \
    acc *= (uint64_t)(((N) | 3u));                                    \
    acc ^= (uint64_t)((N) * 0xe6546b64ull + (SALT) + 0x0dull);        \
    acc += (uint64_t)((N) + (SALT) + 0x0eull);                        \
    break;                                                            \
  }

#define ALU_CASE_A(N) ALU_CASE(0x1111u, N)
#define ALU_CASE_B(N) ALU_CASE(0x2222u, N)
#define ALU_CASE_C(N) ALU_CASE(0x3333u, N)

#define REP16(M, B)                                                    \
  M(B + 0) M(B + 1) M(B + 2) M(B + 3) M(B + 4) M(B + 5) M(B + 6)       \
      M(B + 7) M(B + 8) M(B + 9) M(B + 10) M(B + 11) M(B + 12)         \
          M(B + 13) M(B + 14) M(B + 15)

#define REP256(M)                                                      \
  REP16(M, 0) REP16(M, 16) REP16(M, 32) REP16(M, 48) REP16(M, 64)      \
      REP16(M, 80) REP16(M, 96) REP16(M, 112) REP16(M, 128)            \
          REP16(M, 144) REP16(M, 160) REP16(M, 176) REP16(M, 192)      \
              REP16(M, 208) REP16(M, 224) REP16(M, 240)

__attribute__((__noinline__)) uint64_t icacheSwitchA(uint8_t idx, uint64_t acc) {
  switch (idx) { REP256(ALU_CASE_A) }
  return acc;
}

__attribute__((__noinline__)) uint64_t icacheSwitchB(uint8_t idx, uint64_t acc) {
  switch (idx) { REP256(ALU_CASE_B) }
  return acc;
}

__attribute__((__noinline__)) uint64_t icacheSwitchC(uint8_t idx, uint64_t acc) {
  switch (idx) { REP256(ALU_CASE_C) }
  return acc;
}

#undef REP256
#undef REP16
#undef ALU_CASE_A
#undef ALU_CASE_B
#undef ALU_CASE_C
#undef ALU_CASE

std::vector<uint64_t> makeChaseBuffer(size_t len) {
  std::vector<uint64_t> v(len);
  for (size_t i = 0; i < len; ++i) {
    v[i] = (i * 2654435761ull + 12345ull) % len;
  }
  return v;
}

void icacheBurst(const std::vector<uint64_t>& chase, size_t steps, uint64_t seed) {
  uint64_t acc = seed;
  size_t p = seed % chase.size();
  for (size_t j = 0; j < steps; ++j) {
    p = chase[p]; // pointer-chase load
    uint8_t idx = (uint8_t)(p & 0xFFu);
    switch (j % 3) {
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

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;
  static const std::vector<uint64_t> chase = makeChaseBuffer(8192);

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          icacheBurst(chase, 24, i + 1);
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
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100, 8, 100)

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
