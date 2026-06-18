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
// Derived benchmark: contendedUsePerturb
//
// This variant keeps the exact LifoSem contention pattern of contendedUse but
// adds a large, hard-to-cache instruction footprint executed by the waiters.
// Three noinline functions, each containing a 256-case switch with 14 ALU ops
// per case (unique literal constants to defeat compiler deduplication), are
// rotated via j%3. The switch index comes from a pointer-chase load so the
// branch target is data-dependent. This intentionally raises the executed
// instruction working set to push L1-icache-load-misses_MPKI up and bring the
// frontend-bound IPC down toward the target band.
// ---------------------------------------------------------------------------

#define PERTURB_CASE(K, S)                                         \
  case (K): {                                                      \
    acc += (uint64_t)((K) * 2654435761u + (S) + 0x9e3779b9u);      \
    acc ^= ((uint64_t)(K) << 13) ^ (uint64_t)(S);                  \
    acc *= (uint64_t)((K) | 1u) + (uint64_t)(S);                   \
    acc += (uint64_t)((K) ^ (0x5bd1e995u + (S)));                  \
    acc ^= acc >> 7;                                               \
    acc *= 0x100000001b3ull ^ (uint64_t)((K) + (S));               \
    acc += (uint64_t)((K) * 3u + 7u + (S));                        \
    acc ^= (uint64_t)(((K) << 5) + 0x12345u + (S));                \
    acc *= (uint64_t)((K) + 0xABCu + (S));                         \
    acc += (uint64_t)((K) ^ (0xdeadu + (S)));                      \
    acc ^= (uint64_t)(K) * 7919u + (uint64_t)(S);                  \
    acc *= (uint64_t)((K) | 0x40u) + (uint64_t)(S);                \
    acc += (uint64_t)(((K) << 3) ^ (0xBEEFu + (S)));               \
    acc ^= (uint64_t)((K) + 0x77u + (S));                          \
    break;                                                         \
  }

#define PC1(K, S) PERTURB_CASE(K, S)
#define PC4(K, S) PC1(K, S) PC1((K) + 1, S) PC1((K) + 2, S) PC1((K) + 3, S)
#define PC16(K, S) \
  PC4(K, S) PC4((K) + 4, S) PC4((K) + 8, S) PC4((K) + 12, S)
#define PC64(K, S) \
  PC16(K, S) PC16((K) + 16, S) PC16((K) + 32, S) PC16((K) + 48, S)
#define PC256(K, S) \
  PC64(K, S) PC64((K) + 64, S) PC64((K) + 128, S) PC64((K) + 192, S)

static __attribute__((noinline)) uint64_t perturbSwitchA(
    uint64_t acc, uint32_t idx) {
  switch (idx & 0xFFu) { PC256(0, 0x1111u) }
  return acc;
}

static __attribute__((noinline)) uint64_t perturbSwitchB(
    uint64_t acc, uint32_t idx) {
  switch (idx & 0xFFu) { PC256(0, 0x2222u) }
  return acc;
}

static __attribute__((noinline)) uint64_t perturbSwitchC(
    uint64_t acc, uint32_t idx) {
  switch (idx & 0xFFu) { PC256(0, 0x3333u) }
  return acc;
}

static std::vector<uint32_t> makePerturbChain() {
  std::vector<uint32_t> chase(256);
  for (uint32_t i = 0; i < 256; ++i) {
    chase[i] = (i * 167u + 13u) & 0xFFu;
  }
  return chase;
}

static void contendedUsePerturb(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  static const std::vector<uint32_t> chase = makePerturbChain();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem] {
        uint64_t acc = 0x9e3779b97f4a7c15ull + static_cast<uint64_t>(t);
        uint32_t idx = static_cast<uint32_t>(t) & 0xFFu;
        uint64_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          idx = chase[idx & 0xFFu];
          uint32_t payload = idx & 0xFFu;
          switch (j % 3) {
            case 0:
              acc = perturbSwitchA(acc, payload);
              break;
            case 1:
              acc = perturbSwitchB(acc, payload);
              break;
            default:
              acc = perturbSwitchC(acc, payload);
              break;
          }
          ++j;
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
BENCHMARK_NAMED_PARAM(contendedUsePerturb, 8_to_100_perturb, 8, 100)
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
