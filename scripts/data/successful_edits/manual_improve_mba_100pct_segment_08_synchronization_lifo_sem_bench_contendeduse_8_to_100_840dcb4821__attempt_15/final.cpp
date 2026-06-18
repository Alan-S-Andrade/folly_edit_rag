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

// Frontend-bound icache-pressure variant derived from contendedUse(8_to_100).
// Each waiter, after taking the semaphore, runs one of three large 256-case
// switch functions selected by a rotation index. The combined code footprint
// of the three noinline functions is large enough to thrash the L1 instruction
// cache, lowering IPC into the contended-use target range.
namespace {

#define ALU_CASE(K)                                                   \
  case (K): {                                                         \
    acc += UINT64_C(0x9E3779B97F4A7C15) * (uint64_t)((K) + SALT);     \
    acc ^= (uint64_t)((K) * 2654435761u + SALT);                      \
    acc *= (uint64_t)(((K) | 1u) + SALT);                             \
    acc += (uint64_t)((K) * 2246822519u);                             \
    acc ^= (uint64_t)((K) * 3266489917u + SALT);                      \
    acc *= UINT64_C(0xFF51AFD7ED558CCD);                              \
    acc ^= acc >> 29;                                                 \
    acc += (uint64_t)((K) ^ (0x5bd1e995u + SALT));                    \
    acc *= UINT64_C(0xC4CEB9FE1A85EC53);                              \
    acc ^= acc >> 32;                                                 \
    acc -= (uint64_t)((K) * 374761393u + SALT);                       \
    acc = (acc << 7) | (acc >> 57);                                   \
    acc += (uint64_t)((K) * 668265263u);                              \
    acc *= UINT64_C(0xD6E8FEB86659FD93);                              \
    break;                                                            \
  }

#define ALU_C4(b) ALU_CASE(b) ALU_CASE((b) + 1) ALU_CASE((b) + 2) ALU_CASE((b) + 3)
#define ALU_C16(b) ALU_C4(b) ALU_C4((b) + 4) ALU_C4((b) + 8) ALU_C4((b) + 12)
#define ALU_C64(b) ALU_C16(b) ALU_C16((b) + 16) ALU_C16((b) + 32) ALU_C16((b) + 48)
#define ALU_C256 ALU_C64(0) ALU_C64(64) ALU_C64(128) ALU_C64(192)

#define SALT 0x1111u
__attribute__((noinline)) static uint64_t icacheSwitchA(uint64_t acc, unsigned idx) {
  switch (idx & 0xFFu) {
    ALU_C256
  }
  return acc;
}
#undef SALT

#define SALT 0x2222u
__attribute__((noinline)) static uint64_t icacheSwitchB(uint64_t acc, unsigned idx) {
  switch (idx & 0xFFu) {
    ALU_C256
  }
  return acc;
}
#undef SALT

#define SALT 0x3333u
__attribute__((noinline)) static uint64_t icacheSwitchC(uint64_t acc, unsigned idx) {
  switch (idx & 0xFFu) {
    ALU_C256
  }
  return acc;
}
#undef SALT

#undef ALU_C256
#undef ALU_C64
#undef ALU_C16
#undef ALU_C4
#undef ALU_CASE

} // namespace

static void contendedSwitch(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        std::vector<uint32_t> chase(1024);
        for (size_t k = 0; k < chase.size(); ++k) {
          chase[k] = (uint32_t)((k * 2654435761u + t * 40503u + 1u) % chase.size());
        }
        uint64_t acc = 0x12345u + (uint64_t)t;
        uint32_t p = (uint32_t)(t & (chase.size() - 1));
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = chase[p];
          unsigned payload = (unsigned)((chase[p] ^ acc) & 0xFFu);
          switch ((i + t) % 3) {
            case 0:
              acc = icacheSwitchA(acc, payload);
              break;
            case 1:
              acc = icacheSwitchB(acc, payload);
              break;
            default:
              acc = icacheSwitchC(acc, payload);
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
BENCHMARK_NAMED_PARAM(contendedSwitch, 8_to_100, 8, 100)

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
