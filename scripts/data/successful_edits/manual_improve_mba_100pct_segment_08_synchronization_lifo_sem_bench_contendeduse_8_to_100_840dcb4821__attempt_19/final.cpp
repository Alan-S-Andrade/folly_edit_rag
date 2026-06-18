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
// Derived benchmark: contendedUseIcache(8_to_100)
//
// Frontend-bound variant of contendedUse(8_to_100). The waiter threads run a
// pointer-chase load whose payload selects a case in one of three large
// noinline 256-case switch functions, rotated by i % 3. The unique per-case
// literal constants defeat compiler deduplication, producing a large hot
// instruction working set that raises L1-icache-load-misses and lowers IPC.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kChaseLen = 4096;

struct ChaseNode {
  uint32_t next;
  uint32_t payload;
};

static std::vector<ChaseNode>& chaseRing() {
  static std::vector<ChaseNode> ring = [] {
    std::vector<ChaseNode> r(kChaseLen);
    for (size_t i = 0; i < kChaseLen; ++i) {
      r[i].next = static_cast<uint32_t>((i * 2654435761ull + 1ull) % kChaseLen);
      r[i].payload = static_cast<uint32_t>(i * 40503ull + 7ull);
    }
    return r;
  }();
  return ring;
}

// 16 ALU ops per case; constants vary with the case index (i) and a
// per-function seed (S) so no two cases are identical.
#define ALU_OPS(i, S)                                          \
  acc += (uint64_t)((i) * 6364136223846793005ull + (S) + 1ull); \
  acc ^= (uint64_t)((i) * 1442695040888963407ull + (S));        \
  acc *= (uint64_t)((((i) + (S)) << 1) | 1ull);                 \
  acc += (uint64_t)((i) ^ ((S) * 2654435761ull));               \
  acc ^= (uint64_t)((i) + 0x9e3779b9ull + (S));                 \
  acc += (uint64_t)((i) * 40503ull + (S));                      \
  acc *= (uint64_t)((((i) ^ (S)) << 1) | 1ull);                 \
  acc ^= (uint64_t)((i) * 2246822519ull + (S));                 \
  acc += (uint64_t)((i) * 3266489917ull + (S));                 \
  acc ^= (uint64_t)((i) * 668265263ull + (S));                  \
  acc += (uint64_t)((i) * 374761393ull + (S));                  \
  acc *= (uint64_t)(((((i) + 3ull) ^ (S)) << 1) | 1ull);        \
  acc ^= (uint64_t)((i) * 2090056519ull + (S));                 \
  acc += (uint64_t)((i) * 3812015801ull + (S));                 \
  acc ^= (uint64_t)((i) + 0x85ebca6bull + (S));                 \
  acc += (uint64_t)((i) * 0xc2b2ae35ull + (S));

#define LSC(i, S) \
  case (i): {     \
    ALU_OPS((i), (S)) break; \
  }
#define LSC4(b, S) LSC((b) + 0, S) LSC((b) + 1, S) LSC((b) + 2, S) LSC((b) + 3, S)
#define LSC16(b, S) \
  LSC4((b) + 0, S) LSC4((b) + 4, S) LSC4((b) + 8, S) LSC4((b) + 12, S)
#define LSC64(b, S) \
  LSC16((b) + 0, S) LSC16((b) + 16, S) LSC16((b) + 32, S) LSC16((b) + 48, S)
#define LSC256(S) \
  LSC64(0, S) LSC64(64, S) LSC64(128, S) LSC64(192, S)

FOLLY_NOINLINE uint64_t icacheSwitchA(uint64_t acc, uint8_t idx) {
  switch (idx) { LSC256(0x11u) }
  return acc;
}

FOLLY_NOINLINE uint64_t icacheSwitchB(uint64_t acc, uint8_t idx) {
  switch (idx) { LSC256(0x22u) }
  return acc;
}

FOLLY_NOINLINE uint64_t icacheSwitchC(uint64_t acc, uint8_t idx) {
  switch (idx) { LSC256(0x33u) }
  return acc;
}

#undef LSC256
#undef LSC64
#undef LSC16
#undef LSC4
#undef LSC
#undef ALU_OPS

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;
  auto& ring = chaseRing();

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  std::atomic<uint64_t> sink(0);

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &ring, &sink] {
        uint64_t acc = static_cast<uint64_t>(t) + 1;
        uint32_t p = static_cast<uint32_t>(t) % kChaseLen;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          p = ring[p].next;
          uint8_t idx = static_cast<uint8_t>(ring[p].payload & 0xFFu);
          switch (i % 3) {
            case 0:
              acc = icacheSwitchA(acc, idx);
              break;
            case 1:
              acc = icacheSwitchB(acc, idx);
              break;
            default:
              acc = icacheSwitchC(acc, idx);
              break;
          }
        }
        sink.fetch_add(acc, std::memory_order_relaxed);
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
  folly::doNotOptimizeAway(sink.load());
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
