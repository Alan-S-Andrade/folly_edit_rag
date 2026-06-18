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
// Derived benchmark: contendedUseSwitch(8_to_100)
//
// Frontend-bound (icache) variant of contendedUse. Each waiter, before every
// LifoSem wait, performs a pointer-chase load and rotates among three large
// noinline 256-case switch routines (selected via j%3). The switch index is
// the chased payload & 0xFF. The huge, unique-constant code footprint of the
// three routines inflates the executed instruction working set to raise
// L1-icache-load-misses and bring IPC down toward target.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kChaseSize = 4096;

struct ChaseNode {
  uint32_t next;
  uint8_t payload;
};

static std::vector<ChaseNode>& chaseBuffer() {
  static std::vector<ChaseNode> buf = [] {
    std::vector<ChaseNode> v(kChaseSize);
    for (size_t i = 0; i < kChaseSize; ++i) {
      v[i].next = static_cast<uint32_t>((i * 2654435761u + 12345u) % kChaseSize);
      v[i].payload = static_cast<uint8_t>(i * 31u + 7u);
    }
    return v;
  }();
  return buf;
}

// 24 ALU ops per case, each with unique literal constants (K is unique per
// case, S is unique per routine) so the compiler cannot deduplicate the code.
#define LIFOSEM_ALU(K, S)                                                 \
  acc += static_cast<uint64_t>((K) * 2654435761u + (S) + 0x9E3779B9u);    \
  acc ^= static_cast<uint64_t>((K) * 2246822519u + (S) + 0x85EBCA77u);    \
  acc *= static_cast<uint64_t>((((K) ^ (S)) << 1) | 1u);                  \
  acc += static_cast<uint64_t>(((K) + (S)) ^ 0xC2B2AE35u);                \
  acc ^= static_cast<uint64_t>((K) * 374761393u + (S) + 0x27D4EB2Fu);     \
  acc -= static_cast<uint64_t>((K) * 668265263u + (S) + 0x165667B1u);     \
  acc *= static_cast<uint64_t>((((K) * 5u) ^ (S)) | 3u);                  \
  acc ^= static_cast<uint64_t>((K) + (S) + 0x9E3779B1u);                  \
  acc += static_cast<uint64_t>((K) * 2654435761u + (S) * 7u);             \
  acc ^= static_cast<uint64_t>(((K) << 3) + (S));                         \
  acc *= static_cast<uint64_t>((((K) * 7u) ^ (S)) | 1u);                  \
  acc += static_cast<uint64_t>((K) * 40503u + (S) + 12345u);              \
  acc ^= static_cast<uint64_t>((K) * 0x9E3779u + (S) + 0xABCDu);          \
  acc -= static_cast<uint64_t>((K) * 99991u + (S) + 0x5151u);             \
  acc *= static_cast<uint64_t>((((K) + 13u) ^ (S)) | 5u);                 \
  acc += static_cast<uint64_t>((K) * 0xDEADu + (S) + 0xBEEFu);            \
  acc ^= static_cast<uint64_t>((K) * 0xFEEDu + (S) + 0xFACEu);            \
  acc += static_cast<uint64_t>((K) * 22695477u + (S) + 1u);               \
  acc *= static_cast<uint64_t>((((K) * 3u) ^ (S)) | 7u);                  \
  acc ^= static_cast<uint64_t>((K) * 1103515245u + (S) + 12347u);         \
  acc += static_cast<uint64_t>((K) * 134775813u + (S) + 1u);              \
  acc -= static_cast<uint64_t>((K) * 214013u + (S) + 2531011u);           \
  acc *= static_cast<uint64_t>((((K) ^ 0x55u) + (S)) | 1u);               \
  acc ^= static_cast<uint64_t>((K) * 16807u + (S) + 0x7FFFu);

#define LIFOSEM_REP16(M, B)                                               \
  M(B + 0) M(B + 1) M(B + 2) M(B + 3) M(B + 4) M(B + 5) M(B + 6)          \
  M(B + 7) M(B + 8) M(B + 9) M(B + 10) M(B + 11) M(B + 12) M(B + 13)      \
  M(B + 14) M(B + 15)

#define LIFOSEM_REP256(M)                                                 \
  LIFOSEM_REP16(M, 0) LIFOSEM_REP16(M, 16) LIFOSEM_REP16(M, 32)           \
  LIFOSEM_REP16(M, 48) LIFOSEM_REP16(M, 64) LIFOSEM_REP16(M, 80)          \
  LIFOSEM_REP16(M, 96) LIFOSEM_REP16(M, 112) LIFOSEM_REP16(M, 128)        \
  LIFOSEM_REP16(M, 144) LIFOSEM_REP16(M, 160) LIFOSEM_REP16(M, 176)       \
  LIFOSEM_REP16(M, 192) LIFOSEM_REP16(M, 208) LIFOSEM_REP16(M, 224)       \
  LIFOSEM_REP16(M, 240)

#define LIFOSEM_CASE_A(K) case (K): { LIFOSEM_ALU(K, 0x11u) break; }
#define LIFOSEM_CASE_B(K) case (K): { LIFOSEM_ALU(K, 0x29u) break; }
#define LIFOSEM_CASE_C(K) case (K): { LIFOSEM_ALU(K, 0x53u) break; }

__attribute__((noinline)) static uint64_t lifoSwitchA(uint8_t idx, uint64_t acc) {
  switch (idx) {
    LIFOSEM_REP256(LIFOSEM_CASE_A)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t lifoSwitchB(uint8_t idx, uint64_t acc) {
  switch (idx) {
    LIFOSEM_REP256(LIFOSEM_CASE_B)
  }
  return acc;
}

__attribute__((noinline)) static uint64_t lifoSwitchC(uint8_t idx, uint64_t acc) {
  switch (idx) {
    LIFOSEM_REP256(LIFOSEM_CASE_C)
  }
  return acc;
}

#undef LIFOSEM_CASE_A
#undef LIFOSEM_CASE_B
#undef LIFOSEM_CASE_C
#undef LIFOSEM_REP256
#undef LIFOSEM_REP16
#undef LIFOSEM_ALU

} // namespace

static void contendedUseSwitch(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  auto& buf = chaseBuffer();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &buf] {
        uint64_t acc = 0x12345678u + static_cast<uint64_t>(t);
        uint32_t p = static_cast<uint32_t>((t * 131u) % kChaseSize);
        for (uint32_t i = t; i < n; i += waiters) {
          for (int j = 0; j < 3; ++j) {
            p = buf[p].next;
            uint8_t sw = static_cast<uint8_t>(buf[p].payload & 0xFF);
            switch (j % 3) {
              case 0:
                acc = lifoSwitchA(sw, acc);
                break;
              case 1:
                acc = lifoSwitchB(sw, acc);
                break;
              default:
                acc = lifoSwitchC(sw, acc);
                break;
            }
          }
          sem.wait();
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
