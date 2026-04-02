/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <folly/portability/Asm.h>
#include <folly/synchronization/LifoSem.h>
#include <folly/synchronization/NativeSemaphore.h>

#include <folly/Benchmark.h>

#include <numeric>
#include <random>
#include <vector>

using namespace folly;

BENCHMARK(lifo_sem_pingpong, iters) {
  LifoSem a;
  LifoSem b; // Added missing declaration for 'b'
  b.post();
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

namespace {
struct ChainNode {
  ChainNode* next;
  char pad[56];
};

constexpr size_t kHotLen = 16384;
constexpr size_t kColdLen = 2u << 0;

alignas(64) static ChainNode hotNodes[kHotLen];
static ChainNode* coldNodes;

static void initChain(ChainNode* nodes, size_t len) {
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), 0u);
  std::mt19937_64 rng(42);
  for (size_t i = len - 1; i > 0; --i) {
    std::swap(perm[i], perm[rng() % (i + 1)]);
  }
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
}

#define EXPANDED_ALU_OPS_BODY(v_acc, c_val, base_val, case_id)                 \
  v_acc = ((v_acc * 3) ^ ((uint64_t)c_val + base_val + case_id + 0));         \
  v_acc = ((v_acc + 5) & ((uint64_t)c_val - base_val + case_id + 1));         \
  v_acc = ((v_acc * 7) + ((uint64_t)c_val | (base_val + case_id + 2)));       \
  v_acc = ((v_acc ^ 11) * ((uint64_t)c_val + base_val + case_id + 3));        \
  v_acc = ((v_acc - 13) >> (((uint64_t)c_val % 31) + 1)); /* avoid zero shift */ \
  v_acc = ((v_acc + 17) ^ ((uint64_t)c_val * base_val + case_id + 4));        \
  v_acc = ((v_acc * 19) + ((uint64_t)c_val & (base_val + case_id + 5)));      \
  v_acc = ((v_acc ^ 23) - ((uint64_t)c_val + base_val + case_id + 6));        \
  v_acc = ((v_acc + 29) * ((uint64_t)c_val | (base_val + case_id + 7)));      \
  v_acc = ((v_acc - 31) & ((uint64_t)c_val + base_val + case_id + 8));        \
  v_acc = ((v_acc * 37) ^ ((uint64_t)c_val - base_val + case_id + 9));        \
  v_acc = ((v_acc + 41) >> (((uint64_t)c_val % 31) + 1));                     \
  v_acc = ((v_acc * 43) + ((uint64_t)c_val ^ (base_val + case_id + 10)));    \
                                                                               \
  /* New block of operations to increase I-cache footprint */                  \
  v_acc = ((v_acc * 53) ^ ((uint64_t)c_val + base_val + case_id + 11));       \
  v_acc = ((v_acc + 59) & ((uint64_t)c_val - base_val + case_id + 12));       \
  v_acc = ((v_acc * 61) + ((uint64_t)c_val | (base_val + case_id + 13)));     \
  v_acc = ((v_acc ^ 67) * ((uint64_t)c_val + base_val + case_id + 14));       \
  v_acc = ((v_acc - 71) >> (((uint64_t)c_val % 31) + 1));                     \
  v_acc = ((v_acc + 73) ^ ((uint64_t)c_val * base_val + case_id + 15));       \
  v_acc = ((v_acc * 79) + ((uint64_t)c_val & (base_val + case_id + 16)));     \
  v_acc = ((v_acc ^ 83) - ((uint64_t)c_val + base_val + case_id + 17));       \
  v_acc = ((v_acc + 89) * ((uint64_t)c_val | (base_val + case_id + 18)));     \
  v_acc = ((v_acc - 97) & ((uint64_t)c_val + base_val + case_id + 19));       \
  v_acc = ((v_acc * 101) ^ ((uint64_t)c_val - base_val + case_id + 20));      \
  v_acc = ((v_acc + 103) >> (((uint64_t)c_val % 31) + 1));                    \
  v_acc = ((v_acc * 107) + ((uint64_t)c_val ^ (base_val + case_id + 21)));   \
                                                                               \
  /* Third block of operations to further increase I-cache footprint */        \
  v_acc = ((v_acc * 109) ^ ((uint64_t)c_val + base_val + case_id + 22));      \
  v_acc = ((v_acc + 113) & ((uint64_t)c_val - base_val + case_id + 23));      \
  v_acc = ((v_acc * 127) + ((uint64_t)c_val | (base_val + case_id + 24)));    \
  v_acc = ((v_acc ^ 131) * ((uint64_t)c_val + base_val + case_id + 25));       \
  v_acc = ((v_acc - 137) >> (((uint64_t)c_val % 31) + 1));                    \
  v_acc = ((v_acc + 139) ^ ((uint64_t)c_val * base_val + case_id + 26));      \
  v_acc = ((v_acc * 149) + ((uint64_t)c_val & (base_val + case_id + 27)));    \
  v_acc = ((v_acc ^ 151) - ((uint64_t)c_val + base_val + case_id + 28));      \
  v_acc = ((v_acc + 157) * ((uint64_t)c_val | (base_val + case_id + 29)));    \
  v_acc = ((v_acc - 163) & ((uint64_t)c_val + base_val + case_id + 30));      \
  v_acc = ((v_acc * 167) ^ ((uint64_t)c_val - base_val + case_id + 31));      \
  v_acc = ((v_acc + 173) >> (((uint64_t)c_val % 31) + 1));                    \
  v_acc = ((v_acc * 179) + ((uint64_t)c_val ^ (base_val + case_id + 32)));    \
                                                                               \
  /* Fourth block of operations to further increase I-cache footprint */        \
  v_acc = ((v_acc * 181) ^ ((uint64_t)c_val + base_val + case_id + 33));      \
  v_acc = ((v_acc + 191) & ((uint64_t)c_val - base_val + case_id + 34));      \
  v_acc = ((v_acc * 193) + ((uint64_t)c_val | (base_val + case_id + 35)));    \
  v_acc = ((v_acc ^ 197) * ((uint64_t)c_val + base_val + case_id + 36));      \
  v_acc = ((v_acc - 199) >> (((uint64_t)c_val % 31) + 1));                    \
  v_acc = ((v_acc + 211) ^ ((uint64_t)c_val * base_val + case_id + 37));      \
  v_acc = ((v_acc * 223) + ((uint64_t)c_val & (base_val + case_id + 38)));    \
  v_acc = ((v_acc ^ 227) - ((uint64_t)c_val + base_val + case_id + 39));      \
  v_acc = ((v_acc + 229) * ((uint64_t)c_val | (base_val + case_id + 40)));    \
  v_acc = ((v_acc - 233) & ((uint64_t)c_val + base_val + case_id + 41));      \
  v_acc = ((v_acc * 239) + ((uint64_t)c_val ^ (base_val + case_id + 42)));   \
                                                                               \
  /* Fifth block of operations to further increase I-cache footprint */         \
  v_acc = ((v_acc * 241) ^ ((uint64_t)c_val + base_val + case_id + 43));      \
  v_acc = ((v_acc + 251) & ((uint64_t)c_val - base_val + case_id + 44));      \
  v_acc = ((v_acc * 257) + ((uint64_t)c_val | (base_val + case_id + 45)));    \
  v_acc = ((v_acc ^ 263) * ((uint64_t)c_val + base_val + case_id + 46));      \
  v_acc = ((v_acc - 269) >> (((uint64_t)c_val % 31) + 1));                    \
  v_acc = ((v_acc + 271) ^ ((uint64_t)c_val * base_val + case_id + 47));      \
  v_acc = ((v_acc * 277) + ((uint64_t)c_val & (base_val + case_id + 48)));    \
  v_acc = ((v_acc ^ 281) - ((uint64_t)c_val + base_val + case_id + 49));      \
  v_acc = ((v_acc + 283) * ((uint64_t)c_val | (base_val + case_id + 50)));    \
  v_acc = ((v_acc - 293) & ((uint64_t)c_val + base_val + case_id + 51));      \
  v_acc = ((v_acc * 307) + ((uint64_t)c_val ^ (base_val + case_id + 52)));   \
                                                                               \
  /* Sixth block of operations to further increase I-cache footprint */         \
  v_acc = ((v_acc * 311) ^ ((uint64_t)c_val + base_val + case_id + 53));      \
  v_acc = ((v_acc + 313) & ((uint64_t)c_val - base_val + case_id + 54));      \
  v_acc = ((v_acc * 317) + ((uint64_t)c_val | (base_val + case_id + 55)));    \
  v_acc = ((v_acc ^ 331) * ((uint64_t)c_val + base_val + case_id + 56));      \
  v_acc = ((v_acc - 337) >> (((uint64_t)c_val % 31) + 1));                    \
  v_acc = ((v_acc + 347) ^ ((uint64_t)c_val * base_val + case_id + 57));      \
  v_acc = ((v_acc * 349) + ((uint64_t)c_val & (base_val + case_id + 58)));    \
  v_acc = ((v_acc ^ 353) - ((uint64_t)c_val + base_val + case_id + 59));      \
  v_acc = ((v_acc + 359) * ((uint64_t)c_val | (base_val + case_id + 60)));    \
  v_acc = ((v_acc - 367) & ((uint64_t)c_val + base_val + case_id + 61));      \
  v_acc = ((v_acc * 373) + ((uint64_t)c_val ^ (base_val + case_id + 62)));   \
                                                                               \
  /* Seventh block of operations to further increase I-cache footprint */       \
  v_acc = ((v_acc * 379) ^ ((uint64_t)c_val + base_val + case_id + 63));      \
  v_acc = ((v_acc + 383) & ((uint64_t)c_val - base_val + case_id + 64));      \
  v_acc = ((v_acc * 389) + ((uint64_t)c_val | (base_val + case_id + 65)));    \
  v_acc = ((v_acc ^ 397) * ((uint64_t)c_val + base_val + case_id + 66));      \
  v_acc = ((v_acc - 401) >> (((uint64_t)c_val % 31) + 1));                    \
  v_acc = ((v_acc + 409) ^ ((uint64_t)c_val * base_val + case_id + 67));      \
  v_acc = ((v_acc * 419) + ((uint64_t)c_val & (base_val + case_id + 68)));    \
  v_acc = ((v_acc ^ 421) - ((uint64_t)c_val + base_val + case_id + 69));      \
  v_acc = ((v_acc + 431) * ((uint64_t)c_val | (base_val + case_id + 70)));    \
  v_acc = ((v_acc - 433) & ((uint64_t)c_val + base_val + case_id + 71));      \
  v_acc = ((v_acc * 439) + ((uint64_t)c_val ^ (base_val + case_id + 72)));   \
                                                                               \
  /* Eighth block of operations to further increase I-cache footprint */        \
  v_acc = ((v_acc * 443) ^ ((uint64_t)c_val + base_val + case_id + 73));      \
  v_acc = ((v_acc + 449) & ((uint64_t)c_val - base_val + case_id + 74));      \
  v_acc = ((v_acc * 457) + ((uint64_t)c_val | (base_val + case_id + 75)));    \
  v_acc = ((v_acc ^ 461) * ((uint64_t)c_val + base_val + case_id + 76));      \
  v_acc = ((v_acc - 463) >> (((uint64_t)c_val % 31) + 1));                    \
  v_acc = ((v_acc + 467) ^ ((uint64_t)c_val * base_val + case_id + 77));      \
  v_acc = ((v_acc * 479) + ((uint64_t)c_val & (base_val + case_id + 78)));    \
  v_acc = ((v_acc ^ 487) - ((uint64_t)c_val + base_val + case_id + 79));      \
  v_acc = ((v_acc + 491) * ((uint64_t)c_val | (base_val + case_id + 80)));    \
  v_acc = ((v_acc - 499) & ((uint64_t)c_val + base_val + case_id + 81));      \
  v_acc = ((v_acc * 503) + ((uint64_t)c_val ^ (base_val + case_id + 82)));   \
                                                                               \
  /* Ninth block of operations to further increase I-cache footprint */         \
  v_acc = ((v_acc * 509) ^ ((uint64_t)c_val + base_val + case_id + 83));      \
  v_acc = ((v_acc + 521) & ((uint64_t)c_val - base_val + case_id + 84));      \
  v_acc = ((v_acc * 523) + ((uint64_t)c_val | (base_val + case_id + 85)));    \
  v_acc = ((v_acc ^ 541) * ((uint64_t)c_val + base_val + case_id + 86));      \
  v_acc = ((v_acc - 547) >> (((uint64_t)c_val % 31) + 1));                    \
  v_acc = ((v_acc + 557) ^ ((uint64_t)c_val * base_val + case_id + 87));      \
  v_acc = ((v_acc * 563) + ((uint64_t)c_val & (base_val + case_id + 88)));    \
  v_acc = ((v_acc ^ 569) - ((uint64_t)c_val + base_val + case_id + 89));      \
  v_acc = ((v_acc + 571) * ((uint64_t)c_val | (base_val + case_id + 90)));    \
  v_acc = ((v_acc - 577) & ((uint64_t)c_val + base_val + case_id + 91));      \
  v_acc = ((v_acc * 587) + ((uint64_t)c_val ^ (base_val + case_id + 92)));

#define CASES_8(v, c, base)                                                    \
  case base + 0:                                                               \
    EXPANDED_ALU_OPS_BODY(v, c, base, 0);                                      \
    break;                                                                     \
  case base + 1:                                                               \
    EXPANDED_ALU_OPS_BODY(v, c, base, 1);                                      \
    break;                                                                     \
  case base + 2:                                                               \
    EXPANDED_ALU_OPS_BODY(v, c, base, 2);                                      \
    break;                                                                     \
  case base + 3:                                                               \
    EXPANDED_ALU_OPS_BODY(v, c, base, 3);                                      \
    break;                                                                     \
  case base + 4:                                                               \
    EXPANDED_ALU_OPS_BODY(v, c, base, 4);                                      \
    break;                                                                     \
  case base + 5:                                                               \
    EXPANDED_ALU_OPS_BODY(v, c, base, 5);                                      \
    break;                                                                     \
  case base + 6:                                                               \
    EXPANDED_ALU_OPS_BODY(v, c, base, 6);                                      \
    break;                                                                     \
  case base + 7:                                                               \
    EXPANDED_ALU_OPS_BODY(v, c, base, 7);                                      \
    break;

#define CASES_256(v, c)                                                        \
  CASES_8(v, c, 0)                                                             \
  CASES_8(v, c, 8) CASES_8(v, c, 16) CASES_8(v, c, 24) CASES_8(v, c, 32)       \
      CASES_8(v, c, 40) CASES_8(v, c, 48) CASES_8(v, c, 56)                     \
          CASES_8(v, c, 64) CASES_8(v, c, 72) CASES_8(v, c, 80)                 \
              CASES_8(v, c, 88) CASES_8(v, c, 96) CASES_8(v, c, 104)            \
                  CASES_8(v, c, 112) CASES_8(v, c, 120) CASES_8(v, c, 128)      \
                      CASES_8(v, c, 136) CASES_8(v, c, 144)                     \
                          CASES_8(v, c, 152) CASES_8(v, c, 160)                 \
                              CASES_8(v, c, 168) CASES_8(v, c, 176)             \
                                  CASES_8(v, c, 184) CASES_8(v, c, 192)         \
                                      CASES_8(v, c, 200) CASES_8(v, c, 208)     \
                                          CASES_8(v, c, 216)                   \
                                              CASES_8(v, c, 224)               \
                                                  CASES_8(v, c, 232)           \
                                                      CASES_8(v, c, 240)       \
                                                          CASES_8(v, c, 248)

__attribute__((noinline)) uint64_t switchA(uint64_t v, int c) {
  switch (c) {
    CASES_256(v, c);
    default:;
  }
  return v;
}
__attribute__((noinline)) uint64_t switchB(uint64_t v, int c) {
  switch (c) {
    CASES_256(v, c * 3);
    default:;
  }
  return v;
}
__attribute__((noinline)) uint64_t switchC(uint64_t v, int c) {
  switch (c) {
    CASES_256(v, c * 5);
    default:;
  }
  return v;
}

struct DepWorkInitializer {
  DepWorkInitializer() {
    coldNodes = new ChainNode[kColdLen];
    initChain(hotNodes, kHotLen);
    initChain(coldNodes, kColdLen);
  }
  ~DepWorkInitializer() { delete[] coldNodes; }
};
static DepWorkInitializer initializer;

static void contendedUse_v2(uint32_t n, int posters, int waiters) {
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
        ChainNode* hotPos = &hotNodes[t % kHotLen];
        ChainNode* coldPos = &coldNodes[t % kColdLen];
        uint64_t acc = t;

        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          hotPos = hotPos->next;
          uint64_t payload = (uint64_t)(uintptr_t)hotPos;
          switch (i % 3) {
            case 0:
              acc = switchA(acc, payload & 0xFF);
              break;
            case 1:
              acc = switchB(acc, payload & 0xFF);
              break;
            case 2:
              acc = switchC(acc, payload & 0xFF);
              break;
          }

          auto coldMask = -uint64_t(i % 8 == 0);
          coldPos = (ChainNode*)(((uintptr_t)coldPos->next & coldMask) |
                                 ((uintptr_t)coldPos & ~coldMask));
          sem.post();
        }
        folly::doNotOptimizeAway(acc);
        folly::doNotOptimizeAway(hotPos);
        folly::doNotOptimizeAway(coldPos);
      });
    }
  }

  go.store(true);
  for (auto& thr : threads) {
    thr.join();
  }
}
} // namespace

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

// NEW BENCHMARK FUNCTION: contendedUse_v3, a variant of contendedUse with minimal added work
static void contendedUse_v3(uint32_t n, int posters, int waiters) {
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
        uint64_t acc = t; // Added for minimal work
        while (!go.load()) {
          std::this_thread::yield();
        }
        for (uint32_t i = t; i < n; i += posters) {
          // Minimal ALU work to materially change hot path
          acc = (acc * 3 + i) ^ (t + 5);
          sem.post();
        }
        folly::doNotOptimizeAway(acc); // Ensure acc is not optimized away
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
// NEW BENCHMARK REGISTRATION: contendedUse_v3, adjacent to 1_to_1
BENCHMARK_NAMED_PARAM(contendedUse_v3, 32_to_1000_minimal_work, 32, 1000)
BENCHMARK_NAMED_PARAM(contendedUse_v2, 32_to_1000_L1i, 32, 1000) // Added variant for L1i pressure with 32 posters and 1000 waiters
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_4, 1, 4)
BENCHMARK_NAMED_PARAM(contendedUse, 1_to_32, 1, 32)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_1, 4, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 4_to_24, 4, 24)
BENCHMARK_NAMED_PARAM(contendedUse, 8_to_100, 8, 100)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_1, 31, 1)
BENCHMARK_NAMED_PARAM(contendedUse, 16_to_16, 16, 16)
BENCHMARK_NAMED_PARAM(contendedUse, 32_to_32, 32, 32)
BENCHMARK_NAMED_PARAM(contendedUse_v2, 32_to_32_v2, 32, 32)
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
