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
// Frontend-bound (I-cache footprint) companion to contendedUse(8_to_100).
//
// This benchmark deliberately grows the executed instruction working set so
// that the hot path stresses the L1 instruction cache.  It pairs the same
// LifoSem contended post/wait traffic with a rotation over three large
// noinline 256-case switch functions, each case running several ALU ops with
// unique literal constants (prevents compiler dedup / ICF).  The switch index
// comes from a pointer-chase load so the path is not predictable.
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t kLsbChaseSize = 8192;
constexpr uint32_t kLsbChaseMask = kLsbChaseSize - 1;

std::vector<uint32_t>& lifoChaseBuffer() {
  static std::vector<uint32_t> buf = [] {
    std::vector<uint32_t> v(kLsbChaseSize);
    uint32_t x = 0x12345678u;
    for (uint32_t i = 0; i < kLsbChaseSize; ++i) {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      v[i] = x & kLsbChaseMask;
    }
    return v;
  }();
  return buf;
}

#define LSB_REP16(M, b)                                                     \
  M((b) + 0) M((b) + 1) M((b) + 2) M((b) + 3) M((b) + 4) M((b) + 5)         \
      M((b) + 6) M((b) + 7) M((b) + 8) M((b) + 9) M((b) + 10) M((b) + 11)   \
          M((b) + 12) M((b) + 13) M((b) + 14) M((b) + 15)
#define LSB_REP256(M)                                                       \
  LSB_REP16(M, 0) LSB_REP16(M, 16) LSB_REP16(M, 32) LSB_REP16(M, 48)        \
      LSB_REP16(M, 64) LSB_REP16(M, 80) LSB_REP16(M, 96) LSB_REP16(M, 112)  \
          LSB_REP16(M, 128) LSB_REP16(M, 144) LSB_REP16(M, 160)             \
              LSB_REP16(M, 176) LSB_REP16(M, 192) LSB_REP16(M, 208)         \
                  LSB_REP16(M, 224) LSB_REP16(M, 240)

#define LSB_CASE(n)                                                         \
  case (n): {                                                               \
    acc += (uint32_t)(n) * 2654435761u + LSB_SALT;                          \
    acc ^= ((uint32_t)(n) << 5) ^ (LSB_SALT * 7u + 0x9E3779B9u);            \
    acc *= (2u * (uint32_t)(n) + 1u);                                       \
    acc -= (uint32_t)(n) * 40503u + (LSB_SALT ^ 0xC2B2AE35u);               \
    acc += ((uint32_t)(n) ^ 0x0000ABCDu) + LSB_SALT;                        \
    acc ^= (uint32_t)(n) * 2246822519u;                                     \
    acc *= 0x85EBCA77u;                                                      \
    acc += (uint32_t)(n) * 3266489917u + LSB_SALT;                          \
    acc ^= acc >> 15;                                                       \
    acc += 0x27D4EB2Fu ^ ((uint32_t)(n) + LSB_SALT);                        \
    acc *= (3u * (uint32_t)(n) + 1u);                                       \
    acc -= (LSB_SALT ^ (uint32_t)(n)) + 0x165667B1u;                        \
    acc ^= (uint32_t)(n) * 668265263u;                                      \
    acc += LSB_SALT + (uint32_t)(n);                                        \
    break;                                                                  \
  }

FOLLY_NOINLINE uint64_t lifoSwitchA(uint64_t acc, uint32_t idx) {
#define LSB_SALT 0x01234567u
  switch (idx & 0xFFu) { LSB_REP256(LSB_CASE) }
#undef LSB_SALT
  return acc;
}

FOLLY_NOINLINE uint64_t lifoSwitchB(uint64_t acc, uint32_t idx) {
#define LSB_SALT 0x89ABCDEFu
  switch (idx & 0xFFu) { LSB_REP256(LSB_CASE) }
#undef LSB_SALT
  return acc;
}

FOLLY_NOINLINE uint64_t lifoSwitchC(uint64_t acc, uint32_t idx) {
#define LSB_SALT 0x0F1E2D3Cu
  switch (idx & 0xFFu) { LSB_REP256(LSB_CASE) }
#undef LSB_SALT
  return acc;
}

#undef LSB_CASE
#undef LSB_REP256
#undef LSB_REP16

} // namespace

static void contendedUseICache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;
  auto& chase = lifoChaseBuffer();

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chase] {
        uint64_t acc = static_cast<uint64_t>(t) + 1u;
        uint32_t p = (static_cast<uint32_t>(t) * 2654435761u) & kLsbChaseMask;
        uint32_t j = static_cast<uint32_t>(t);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p];
          uint32_t payload = p & 0xFFu;
          switch (j % 3u) {
            case 0:
              acc = lifoSwitchA(acc, payload);
              break;
            case 1:
              acc = lifoSwitchB(acc, payload);
              break;
            default:
              acc = lifoSwitchC(acc, payload);
              break;
          }
          ++j;
        }
        doNotOptimizeAway(acc);
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
BENCHMARK_NAMED_PARAM(contendedUseICache, 8_to_100, 8, 100)
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
