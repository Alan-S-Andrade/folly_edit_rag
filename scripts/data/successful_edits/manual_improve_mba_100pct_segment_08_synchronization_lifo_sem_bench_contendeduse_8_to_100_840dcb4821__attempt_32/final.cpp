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

// ---------------------------------------------------------------------------
// Derived from contendedUse(8_to_100): same contended LifoSem post/wait
// topology, but each waiter feeds a pointer-chase byte into one of three
// large 256-case switch functions (rotated j%3) to expand the executed hot
// code footprint and raise L1-icache pressure / lower IPC toward target.
// ---------------------------------------------------------------------------
namespace {

// 14 unique-constant ALU ops per case; salt differentiates the three
// functions so the compiler cannot deduplicate bodies.
#define LSB_CASE_OPS(i, s)                          \
  case (i):                                          \
    a += 0x9e3779b9u ^ ((i) + (s));                  \
    a ^= ((i) * 2654435761u + (s));                  \
    a *= (((i) << 1) | 1u);                           \
    a += ((i) ^ 0x5bd1e995u) + (s);                  \
    a ^= ((i) * 40503u + (s));                       \
    a *= ((((i) + (s)) | 1u));                       \
    a += ((i) * 2246822519u);                        \
    a ^= ((i) + 0x165667b1u + (s));                  \
    a *= ((((i) ^ (s)) | 1u));                       \
    a += ((i) * 3266489917u + (s));                  \
    a ^= (((i) << 3) + (s));                         \
    a *= ((((i) + 7u) | 1u));                        \
    a += ((i) * 668265263u + (s));                   \
    a ^= ((i) * 374761393u + (s));                   \
    break;

#define LSB_R1(b, s) LSB_CASE_OPS(b, s)
#define LSB_R4(b, s) \
  LSB_R1(b, s) LSB_R1((b) + 1, s) LSB_R1((b) + 2, s) LSB_R1((b) + 3, s)
#define LSB_R16(b, s) \
  LSB_R4(b, s) LSB_R4((b) + 4, s) LSB_R4((b) + 8, s) LSB_R4((b) + 12, s)
#define LSB_R64(b, s) \
  LSB_R16(b, s) LSB_R16((b) + 16, s) LSB_R16((b) + 32, s) LSB_R16((b) + 48, s)
#define LSB_R256(b, s) \
  LSB_R64(b, s) LSB_R64((b) + 64, s) LSB_R64((b) + 128, s) LSB_R64((b) + 192, s)

__attribute__((noinline)) uint64_t icacheSwitch0(uint64_t a, uint8_t idx) {
  switch (idx) { LSB_R256(0, 0x11u) }
  return a;
}
__attribute__((noinline)) uint64_t icacheSwitch1(uint64_t a, uint8_t idx) {
  switch (idx) { LSB_R256(0, 0x22u) }
  return a;
}
__attribute__((noinline)) uint64_t icacheSwitch2(uint64_t a, uint8_t idx) {
  switch (idx) { LSB_R256(0, 0x33u) }
  return a;
}

#undef LSB_R256
#undef LSB_R64
#undef LSB_R16
#undef LSB_R4
#undef LSB_R1
#undef LSB_CASE_OPS

std::vector<uint32_t> makeChaseRing(size_t len) {
  std::vector<uint32_t> v(len);
  for (size_t i = 0; i < len; ++i) {
    v[i] = static_cast<uint32_t>((i * 2654435761u + 1u) & (len - 1));
  }
  return v;
}

} // namespace

static void contendedUseSwitch(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  static const std::vector<uint32_t> chase = makeChaseRing(4096);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t a = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t);
        const uint32_t mask = static_cast<uint32_t>(chase.size() - 1);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p & mask];
          uint8_t idx = static_cast<uint8_t>(p & 0xFFu);
          switch (i % 3) {
            case 0:
              a = icacheSwitch0(a, idx);
              break;
            case 1:
              a = icacheSwitch1(a, idx);
              break;
            default:
              a = icacheSwitch2(a, idx);
              break;
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
