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

// Derived from contendedUse(8_to_100). The contended LifoSem traffic on its
// own keeps the executed instruction working set tiny, so this variant grows
// the hot code footprint by routing each woken waiter through one of three
// large noinline 256-case switches (rotated via a per-iteration index). The
// switch selector comes from a pointer-chase load, which prevents the loads
// and the taken switch arm from being hoisted or specialized. Each case body
// runs a fixed sequence of ALU ops on a local accumulator using per-case
// unique literal constants so the compiler cannot deduplicate the arms; this
// inflates the I-cache footprint to better match the frontend-bound target.
namespace {

constexpr size_t kChaseSize = 4096;

struct ChaseNode {
  uint32_t next;
  uint32_t payload;
};

std::vector<ChaseNode>& chaseBuffer() {
  static std::vector<ChaseNode> buf = [] {
    std::vector<ChaseNode> v(kChaseSize);
    for (size_t i = 0; i < kChaseSize; ++i) {
      v[i].next = static_cast<uint32_t>((i * 2654435761ull + 1ull) % kChaseSize);
      v[i].payload = static_cast<uint32_t>(i * 1103515245ull + 12345ull);
    }
    return v;
  }();
  return buf;
}

#define R16(M, b)                                                         \
  M(b + 0) M(b + 1) M(b + 2) M(b + 3) M(b + 4) M(b + 5) M(b + 6)          \
      M(b + 7) M(b + 8) M(b + 9) M(b + 10) M(b + 11) M(b + 12) M(b + 13)  \
          M(b + 14) M(b + 15)
#define R256(M)                                                           \
  R16(M, 0) R16(M, 16) R16(M, 32) R16(M, 48) R16(M, 64) R16(M, 80)        \
      R16(M, 96) R16(M, 112) R16(M, 128) R16(M, 144) R16(M, 160)          \
          R16(M, 176) R16(M, 192) R16(M, 208) R16(M, 224) R16(M, 240)

#define ALU_CASE_A(n)              \
  case (n): {                      \
    uint64_t k = (uint64_t)(n) + 1ull; \
    acc += k * 2654435761ull;      \
    acc ^= k * 40503ull;           \
    acc *= (k | 1ull);             \
    acc += k ^ 0x5bd1e995ull;      \
    acc ^= acc >> 13;              \
    acc *= 0x9e3779b1ull + k;      \
    acc += k * 3ull;               \
    acc ^= k * 5ull;               \
    acc *= 0x85ebca6bull + k;      \
    acc += k * 7ull;               \
    acc ^= k << 3;                 \
    acc *= 2ull + k;               \
    acc += k * 11ull;              \
    acc ^= 0xc2b2ae35ull * k;      \
  } break;

#define ALU_CASE_B(n)              \
  case (n): {                      \
    uint64_t k = (uint64_t)(n) + 3ull; \
    acc ^= k * 0x100000001b3ull;   \
    acc += k * 2246822519ull;      \
    acc *= (k | 3ull);             \
    acc ^= k + 0x27d4eb2full;      \
    acc += acc << 7;               \
    acc *= 0xcc9e2d51ull + k;      \
    acc ^= k * 13ull;              \
    acc += k * 17ull;              \
    acc *= 0x1b873593ull + k;      \
    acc ^= k * 19ull;              \
    acc += k >> 2;                 \
    acc *= 5ull + k;               \
    acc ^= k * 23ull;              \
    acc += 0xff51afd7ull * k;      \
  } break;

#define ALU_CASE_C(n)              \
  case (n): {                      \
    uint64_t k = (uint64_t)(n) + 5ull; \
    acc += k * 3266489917ull;      \
    acc ^= k * 668265263ull;       \
    acc *= (k | 5ull);             \
    acc += k ^ 0xeb44accbull;      \
    acc ^= acc >> 11;              \
    acc *= 0x165667b1ull + k;      \
    acc += k * 29ull;              \
    acc ^= k * 31ull;              \
    acc *= 0x9e3779b9ull + k;      \
    acc += k * 37ull;              \
    acc ^= k << 5;                 \
    acc *= 7ull + k;               \
    acc += k * 41ull;              \
    acc ^= 0xed558ccdull * k;      \
  } break;

__attribute__((noinline)) uint64_t icacheSwitchA(uint64_t acc, uint32_t idx) {
  switch (idx & 0xFFu) {
    R256(ALU_CASE_A)
  }
  return acc;
}

__attribute__((noinline)) uint64_t icacheSwitchB(uint64_t acc, uint32_t idx) {
  switch (idx & 0xFFu) {
    R256(ALU_CASE_B)
  }
  return acc;
}

__attribute__((noinline)) uint64_t icacheSwitchC(uint64_t acc, uint32_t idx) {
  switch (idx & 0xFFu) {
    R256(ALU_CASE_C)
  }
  return acc;
}

#undef ALU_CASE_A
#undef ALU_CASE_B
#undef ALU_CASE_C
#undef R256
#undef R16

} // namespace

static void contendedUseIcache(uint32_t n, int posters, int waiters) {
  LifoSemImpl<std::atomic> sem;

  std::vector<std::thread> threads;
  std::atomic<bool> go(false);
  auto& buf = chaseBuffer();

  BENCHMARK_SUSPEND {
    for (int t = 0; t < waiters; ++t) {
      threads.emplace_back([=, &sem, &buf] {
        uint64_t acc = static_cast<uint64_t>(t) + 1ull;
        uint32_t cur = static_cast<uint32_t>((t * 131u) % kChaseSize);
        uint32_t j = 0;
        for (uint32_t i = t; i < n; i += waiters) {
          sem.wait();
          uint32_t payload = buf[cur].payload;
          cur = buf[cur].next;
          switch (j % 3u) {
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
BENCHMARK_NAMED_PARAM(contendedUseIcache, 8_to_100, 8, 100)
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
