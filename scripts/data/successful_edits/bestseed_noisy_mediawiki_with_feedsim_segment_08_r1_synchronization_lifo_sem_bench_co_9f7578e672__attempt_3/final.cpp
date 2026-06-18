#include <cstdint>

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
// Frontend/i-cache pressure derivative of contendedUse(8_to_100).
//
// The 3 noinline functions below each contain a 256-case switch with 16 unique
// ALU ops per case.  Rotating across them with (i % 3) while indexing each
// switch with the low byte of a pointer-chased value spreads the executed
// instruction working set across a large code footprint, raising
// L1-icache-load-misses_MPKI and lowering IPC into the target band.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kChaseSize = 4096;
constexpr uint32_t kChaseMask = kChaseSize - 1;

uint32_t* makeChaseBuffer() {
  static uint32_t buf[kChaseSize];
  for (size_t i = 0; i < kChaseSize; ++i) {
    buf[i] = static_cast<uint32_t>((i * 2654435761ULL + 12345ULL) & kChaseMask);
  }
  return buf;
}

uint32_t* gChase = makeChaseBuffer();

#define LIFOSEM_ICACHE_CASE(i)                                            \
  case (i): {                                                             \
    acc += (static_cast<uint64_t>(i) * 2654435761ULL + SALT + 1ULL);      \
    acc ^= (static_cast<uint64_t>(i) * 40503ULL + 6789ULL);               \
    acc *= (static_cast<uint64_t>(i) | 1ULL);                             \
    acc += (static_cast<uint64_t>(i) ^ (0x9e3779b97f4a7c15ULL + SALT));   \
    acc -= (static_cast<uint64_t>(i) * 2246822519ULL + 11ULL);            \
    acc ^= ((static_cast<uint64_t>(i) << 7) + 0x165667b1ULL);             \
    acc += (static_cast<uint64_t>(i) * 3266489917ULL + 3ULL);             \
    acc *= (static_cast<uint64_t>(i) | 3ULL);                             \
    acc ^= (static_cast<uint64_t>(i) * 668265263ULL + SALT);              \
    acc += (static_cast<uint64_t>(i) ^ 0xff51afd7ed558ccdULL);            \
    acc -= (static_cast<uint64_t>(i) * 374761393ULL + 5ULL);              \
    acc ^= ((static_cast<uint64_t>(i) << 13) + 0x27d4eb2fULL);            \
    acc += (static_cast<uint64_t>(i) * 2654435761ULL + 7ULL);             \
    acc *= (static_cast<uint64_t>(i) | 5ULL);                             \
    acc ^= (static_cast<uint64_t>(i) * 2246822519ULL + SALT);             \
    acc += (static_cast<uint64_t>(i) ^ 0xc2b2ae3d27d4eb4fULL);            \
    break;                                                                \
  }

#define LIFOSEM_ICACHE_C2(b) \
  LIFOSEM_ICACHE_CASE(b) LIFOSEM_ICACHE_CASE((b) + 1)
#define LIFOSEM_ICACHE_C4(b) \
  LIFOSEM_ICACHE_C2(b) LIFOSEM_ICACHE_C2((b) + 2)
#define LIFOSEM_ICACHE_C8(b) \
  LIFOSEM_ICACHE_C4(b) LIFOSEM_ICACHE_C4((b) + 4)
#define LIFOSEM_ICACHE_C16(b) \
  LIFOSEM_ICACHE_C8(b) LIFOSEM_ICACHE_C8((b) + 8)
#define LIFOSEM_ICACHE_C32(b) \
  LIFOSEM_ICACHE_C16(b) LIFOSEM_ICACHE_C16((b) + 16)
#define LIFOSEM_ICACHE_C64(b) \
  LIFOSEM_ICACHE_C32(b) LIFOSEM_ICACHE_C32((b) + 32)
#define LIFOSEM_ICACHE_C128(b) \
  LIFOSEM_ICACHE_C64(b) LIFOSEM_ICACHE_C64((b) + 64)
#define LIFOSEM_ICACHE_C256(b) \
  LIFOSEM_ICACHE_C128(b) LIFOSEM_ICACHE_C128((b) + 128)

#define SALT 0x1111111111111111ULL
__attribute__((noinline)) uint64_t lifoSemIcacheFn0(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFOSEM_ICACHE_C256(0)
  }
  return acc;
}
#undef SALT

#define SALT 0x2222222222222222ULL
__attribute__((noinline)) uint64_t lifoSemIcacheFn1(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFOSEM_ICACHE_C256(0)
  }
  return acc;
}
#undef SALT

#define SALT 0x3333333333333333ULL
__attribute__((noinline)) uint64_t lifoSemIcacheFn2(uint64_t acc, uint8_t idx) {
  switch (idx) {
    LIFOSEM_ICACHE_C256(0)
  }
  return acc;
}
#undef SALT

} // namespace

static void contendedUseFe(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = (static_cast<uint32_t>(t) * 2654435761u) & kChaseMask;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = gChase[p];
          uint8_t idx = static_cast<uint8_t>(p & 0xFF);
          switch (i % 3) {
            case 0:
              acc = lifoSemIcacheFn0(acc, idx);
              break;
            case 1:
              acc = lifoSemIcacheFn1(acc, idx);
              break;
            default:
              acc = lifoSemIcacheFn2(acc, idx);
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

BENCHMARK_NAMED_PARAM(contendedUseFe, 8_to_100_icache, 8, 100)

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
