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
// Frontend-bound derived variant of contendedUse.
//
// This intentionally inflates the executed-instruction working set (L1i
// footprint) so that IPC drops toward the contended target. The proven
// pattern: three noinline functions, each a 256-case switch with many ALU
// ops per case using unique literal constants (to defeat compiler
// deduplication). The hot loop rotates among the three functions with a
// j % 3 selector and indexes the switch via a pointer-chase payload byte.
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t kFeSize = 4096; // power of two

struct FeNode {
  uint32_t next;
  uint32_t payload;
};

static inline uint32_t feRotl(uint32_t x, uint32_t r) {
  r &= 31u;
  return (x << r) | (x >> ((32u - r) & 31u));
}

const std::vector<FeNode>& feChain() {
  static const std::vector<FeNode> chain = [] {
    std::vector<FeNode> v(kFeSize);
    for (uint32_t i = 0; i < kFeSize; ++i) {
      v[i].next = (i * 2654435761u + 0x9E3779B9u);
      v[i].payload = (i * 40503u + 12345u) ^ (i << 7);
    }
    return v;
  }();
  return chain;
}

// 14 ALU ops per case; each op mixes the case index (i) and a per-function
// salt (s) so every case body is a unique constant pattern.
#define FE_CASE(i, s)                                          \
  case (i): {                                                  \
    acc += (uint32_t)(i) * 2654435761u + (s) + 1u;             \
    acc ^= feRotl(acc, (uint32_t)((i) & 31) + 1u);             \
    acc *= ((uint32_t)(i) | 1u);                               \
    acc += (acc >> 7) ^ ((uint32_t)(i) + (s));                 \
    acc -= (uint32_t)(i) * 7u + (s);                           \
    acc ^= feRotl(acc, (uint32_t)(((i) >> 2) & 31) + 3u);      \
    acc += (uint32_t)(i) * 40503u;                             \
    acc ^= 0x9E3779B9u + (s) - (uint32_t)(i);                  \
    acc *= 0x85EBCA77u ^ (uint32_t)(i);                        \
    acc += feRotl(acc, ((s) & 15) + 5u);                       \
    acc ^= (uint32_t)(i) * 0xC2B2AE35u;                        \
    acc -= 0x27D4EB2Fu + (uint32_t)(i);                        \
    acc += (acc << 3) ^ ((s) + (uint32_t)(i));                 \
    acc ^= feRotl(acc, ((uint32_t)(i) ^ (s)) & 31);            \
    break;                                                     \
  }

#define FE_C4(b, s) FE_CASE(b + 0, s) FE_CASE(b + 1, s) FE_CASE(b + 2, s) FE_CASE(b + 3, s)
#define FE_C16(b, s) FE_C4(b + 0, s) FE_C4(b + 4, s) FE_C4(b + 8, s) FE_C4(b + 12, s)
#define FE_C64(b, s) FE_C16(b + 0, s) FE_C16(b + 16, s) FE_C16(b + 32, s) FE_C16(b + 48, s)
#define FE_C256(s) FE_C64(0, s) FE_C64(64, s) FE_C64(128, s) FE_C64(192, s)

__attribute__((noinline)) uint32_t feSwitch0(uint32_t acc, uint32_t idx) {
  switch (idx & 0xFFu) {
    FE_C256(0x1111u)
  }
  return acc;
}

__attribute__((noinline)) uint32_t feSwitch1(uint32_t acc, uint32_t idx) {
  switch (idx & 0xFFu) {
    FE_C256(0x2222u)
  }
  return acc;
}

__attribute__((noinline)) uint32_t feSwitch2(uint32_t acc, uint32_t idx) {
  switch (idx & 0xFFu) {
    FE_C256(0x3333u)
  }
  return acc;
}

#undef FE_C256
#undef FE_C64
#undef FE_C16
#undef FE_C4
#undef FE_CASE

std::atomic<uint64_t> frontendSink{0};

} // namespace

static void contendedUseFrontend(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);

  const std::vector<FeNode>& chain = feChain();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &chain] {
        uint32_t acc = 0x12345u + static_cast<uint32_t>(t);
        uint32_t pos = static_cast<uint32_t>(t);
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          for (uint32_t k = 0; k < 24; ++k) {
            pos = chain[pos & (kFeSize - 1)].next;
            uint32_t idx = chain[pos & (kFeSize - 1)].payload & 0xFFu;
            switch ((i + k) % 3u) {
              case 0:
                acc = feSwitch0(acc, idx);
                break;
              case 1:
                acc = feSwitch1(acc, idx);
                break;
              default:
                acc = feSwitch2(acc, idx);
                break;
            }
          }
        }
        frontendSink.fetch_add(acc, std::memory_order_relaxed);
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
BENCHMARK_NAMED_PARAM(contendedUseFrontend, 8_to_100_fe, 8, 100)

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
