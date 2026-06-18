#include <fmt/format.h>

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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include <folly/Benchmark.h>
#include <folly/Conv.h>
#include <folly/Format.h>
#include <folly/Function.h>
#include <folly/Random.h>
#include <folly/hash/Hash.h>
#include <folly/init/Init.h>
#include <folly/portability/GFlags.h>

#include <folly/container/F14Map.h>

using namespace folly;

// Depending the max load factor and the rehashing policy, the map size could
// affect benchmark results quite a bit. For example, a map that's just rehashed
// could have a lower probability of collisions, but the rehash time would get
// counted toward the operation that triggered the rehash.
DEFINE_int32(
    map_size_min,
    11,
    "min number of entries to benchmark each map with (inclusive)");
DEFINE_int32(
    map_size_max,
    100000,
    "max number of entries to benchmark each map with (inclusive)");
DEFINE_int32(
    map_size_step, 32, "multiplier for each benchmark between iterations");

//////// Key related preparation ////////

namespace {
static const std::string kPadding("0123456789012345678901234567890123");
struct NonSSOString : public std::string {
  template <typename... Args>
  explicit NonSSOString(Args&&... args)
      : std::string(std::string(std::forward<Args>(args)...) + kPadding) {}

  NonSSOString(NonSSOString const&) = default;
  NonSSOString& operator=(NonSSOString const&) = default;
  NonSSOString(NonSSOString&&) = default;
  NonSSOString& operator=(NonSSOString&&) = default;
};
} // namespace

namespace std {
template <>
struct hash<NonSSOString> {
  size_t operator()(const NonSSOString& s) const { return hash<string>()(s); }
};

#ifdef __GLIBCXX__
template <>
struct __is_fast_hash<hash<NonSSOString>> : public std::false_type {};

static_assert(
    __cache_default<string, hash<string>>::value ==
        __cache_default<NonSSOString, hash<NonSSOString>>::value,
    "To draw a fair comparison, the policy of whether to cache hash codes "
    "for NonSSOString keys should be the same as for std::string keys.");
#endif
} // namespace std

namespace folly {
template <typename K>
struct IsAvalanchingHasher<std::hash<NonSSOString>, K> : std::true_type {};
} // namespace folly

const int kSalt = 0x619abc7e;

template <typename K>
K keyGen(uint32_t key) {
  return folly::to<K>(hash::jenkins_rev_mix32(key ^ kSalt));
}

template <>
NonSSOString keyGen<NonSSOString>(uint32_t key) {
  return NonSSOString(to<std::string>(key));
}

template <class K>
std::vector<K>& keyList(int /*max*/ = 0) {
  static std::vector<K> keys;
  return keys;
}

template <class K>
void prepare(size_t max) {
  auto& keys = keyList<K>();
  if (keys.size() < max) {
    for (auto key = keys.size(); key < max; ++key) {
      keys.push_back(keyGen<K>(key));
    }
  }
}

template <class K>
const K& key(int index) {
  return keyList<K>()[index];
}

template <typename TArray>
TArray value(int i) {
  // quick and dirty hack, since we don't test for correctness, ignore any
  // narrowing that might happen.
  return {static_cast<typename TArray::value_type>(i)};
}

template <template <class, class> class Map, class K, class V, class Test>
void benchmarkFilledMap(int runs, int size, const Test& test, int div = 1) {
  BenchmarkSuspender braces;
  Map<K, V> map(size);
  prepare<K>(size);
  for (int i = 0; i < size; i += div) {
    auto& k = key<K>(i);
    map.insert(std::pair<K, V>(k, value<V>(i * 3)));
  }
  folly::makeUnpredictable(map);
  braces.dismissing([&] {
    for (int r = 0; r < runs; ++r) {
      test(map);
    }
  });
}

static auto& getRNG() {
  static std::mt19937 rng{42};
  return rng;
}

template <template <class, class> class Map, class K, class V, class Test>
void benchmarkManyFilledMapsByKey(
    int runs, int size, const Test& test, int div = 1) {
  BenchmarkSuspender braces;
  Map<K, V> maps[64];
  std::vector<K> toInsert;
  prepare<K>(size);
  for (int i = 0; i < size; ++i) {
    auto& k = key<K>(i);
    toInsert.push_back(k);
  }
  for (int i = 0; i * div < size; ++i) {
    maps[i % 64].insert(std::pair<K, V>(toInsert[i], value<V>(i * 3)));
  }
  std::shuffle(toInsert.begin(), toInsert.end(), getRNG());
  folly::makeUnpredictable(maps);
  folly::makeUnpredictable(toInsert);
  braces.dismissing([&] {
    for (int r = 0; r < runs; ++r) {
      for (int i = 0; i < size; i += div) {
        test(maps[i % 64], toInsert[i]);
      }
    }
  });
}

template <template <class, class> class Map, class K, class V, class Test>
void benchmarkFilledMapByKey(
    int runs, int size, const Test& test, int div = 1) {
  BenchmarkSuspender braces;
  Map<K, V> map(size);
  std::vector<K> toInsert;
  prepare<K>(size);
  for (int i = 0; i < size; ++i) {
    auto& k = key<K>(i);
    toInsert.push_back(k);
  }
  for (int i = 0; i * div < size; ++i) {
    map.insert(std::pair<K, V>(toInsert[i], value<V>(i * 3)));
  }
  std::shuffle(toInsert.begin(), toInsert.end(), getRNG());
  folly::makeUnpredictable(map);
  folly::makeUnpredictable(toInsert);
  braces.dismissing([&] {
    for (int r = 0; r < runs; ++r) {
      for (int i = 0; i < size; i += div) {
        test(map, toInsert[i]);
      }
    }
  });
}

template <
    template <class, class> class Map,
    class K,
    class V,
    class... Args,
    class Test>
void benchmarkFromEmptyArgs(int runs, int size, Test test, Args... args) {
  BenchmarkSuspender braces;
  prepare<K>(size);
  for (int r = 0; r < runs; ++r) {
    Map<K, V> map{args...};
    folly::makeUnpredictable(map);
    braces.dismissing([&] {
      for (int i = 0; i < size; ++i) {
        test(map, i);
      }
    });
  }
}

template <template <class, class> class Map, class K, class V>
void benchmarkInsert(int runs, int size) {
  benchmarkFromEmptyArgs<Map, K, V>(
      runs,
      size,
      [](Map<K, V>& m, int i) {
        m.insert(std::pair<K, V>(key<K>(i), value<V>(i)));
      },
      unsigned(size));
}

template <template <class, class> class Map, class K, class V>
void benchmarkInsertGrow(int runs, int size) {
  benchmarkFromEmptyArgs<Map, K, V>(runs, size, [](Map<K, V>& m, int i) {
    m.insert(std::pair<K, V>(key<K>(i), value<V>(i)));
  });
}

template <template <class, class> class Map, class K, class V>
void benchmarkInsertSqBr(int runs, int size) {
  benchmarkFromEmptyArgs<Map, K, V>(runs, size, [](Map<K, V>& m, int i) {
    m[key<K>(i)] = value<V>(i);
  });
}

template <template <class, class> class Map, class K, class V>
void benchmarkFind(int runs, int size) {
  int x = 0;
  benchmarkFilledMapByKey<Map, K, V>(
      runs,
      size,
      [&](Map<K, V>& m, const K& key) {
        auto found = m.find(key);
        if (found != m.end()) {
          x ^= found->second[0];
          folly::doNotOptimizeAway(x);
        }
      },
      2);
}

template <template <class, class> class Map, class K, class V>
void benchmarkManyFind(int runs, int size) {
  int x = 0;
  benchmarkManyFilledMapsByKey<Map, K, V>(
      runs,
      size,
      [&](Map<K, V>& m, const K& key) {
        auto found = m.find(key);
        if (found != m.end()) {
          x ^= found->second[0];
          folly::doNotOptimizeAway(x);
        }
      },
      2);
}

template <template <class, class> class Map, class K, class V>
void benchmarkSqBrFind(int runs, int size) {
  int x = 0;
  benchmarkFilledMapByKey<Map, K, V>(
      runs,
      size,
      [&](Map<K, V>& m, const K& key) {
        x ^= m[key][0];
        folly::doNotOptimizeAway(x);
      },
      2);
}

template <template <class, class> class Map, class K, class V>
void benchmarkErase(int runs, int size) {
  for (int i = 0; i < runs; ++i) {
    benchmarkFilledMapByKey<Map, K, V>(
        1, size, [&](Map<K, V>& m, const K& key) { m.erase(key); }, 2);
  }
}

template <template <class, class> class Map, class K, class V>
void benchmarkDtor(int runs, int size) {
  for (int i = 0; i < runs; ++i) {
    benchmarkFilledMap<Map, K, V>(
        1,
        size,
        [&](Map<K, V>& m) {
          Map<K, V> toDestruct;
          swap(m, toDestruct);
        },
        2);
  }
}

template <template <class, class> class Map, class K, class V>
void benchmarkClear(int runs, int size) {
  for (int i = 0; i < runs; ++i) {
    benchmarkFilledMap<Map, K, V>(1, size, [&](Map<K, V>& m) { m.clear(); }, 1);
  }
}

template <template <class, class> class Map, class K, class V>
void benchmarkCopyCtor(int runs, int size) {
  Map<K, V> n;
  for (int i = 0; i < runs; ++i) {
    benchmarkFilledMap<Map, K, V>(1, size, [&](Map<K, V>& m) { n = m; }, 1);
    folly::doNotOptimizeAway(n);
  }
}

template <template <class, class> class Map, class K, class V>
void benchmarkIter(int runs, int size) {
  int x = 0;
  benchmarkFilledMap<Map, K, V>(runs, size, [&](Map<K, V>& m) {
    for (auto& entry : m) {
      x ^= entry.second[0];
    }
  });
  folly::doNotOptimizeAway(x);
}

template <template <class, class> class Map, class K, class V>
void benchmarkSparseIter(int runs, int size) {
  int x = 0;
  benchmarkFilledMap<Map, K, V>(
      runs,
      size,
      [&](Map<K, V>& m) {
        for (auto& entry : m) {
          x ^= entry.second[0];
        }
      },
      8);
  folly::doNotOptimizeAway(x);
}

template <template <class, class> class Map, class K, class V>
void benchmarkCtorWithCapacity(int runs, int size) {
  for (int ii = 0; ii < runs; ++ii) {
    Map<K, V> map(size);
    folly::doNotOptimizeAway(map);
  }
}

template <class K, class V>
using unord = std::unordered_map<K, V>;
template <class K, class V>
using f14val = F14ValueMap<K, V>;
template <class K, class V>
using f14node = F14NodeMap<K, V>;
template <class K, class V>
using f14vec = F14VectorMap<K, V>;

//////// Derived local variant of %CtorWithCapacity f14val<uint64_t, a[1]>[11264]
//
// This variant keeps the original capacity-construction work but augments the
// timed hot path with a pointer-chasing + indirect-branch frontend stressor.
// The chase walks a permuted Hamiltonian cycle that overflows L1d (and a much
// larger cold cycle that overflows the LLC), and rotates among several large
// noinline switch functions whose case index comes from a data-dependent load.
// This shifts the benchmark from compute-bound toward the memory/branch-bound
// profile of the target workload (lower IPC, higher i-cache/branch MPKI).
namespace {

struct ChainNode {
  ChainNode* next;
  char pad[56]; // 64 bytes per node
};

constexpr size_t kHotLen = 16384; // 1 MB: exceeds L1d, fits L2
constexpr size_t kColdLen = 1u << 21; // 128 MB: exceeds LLC

ChainNode* g_hotNodes = nullptr;
ChainNode* g_coldNodes = nullptr;
uint64_t g_chaseAcc = 0x9e3779b97f4a7c15ULL;

void initChain(ChainNode* nodes, size_t len, uint64_t seed) {
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), size_t(0));
  std::mt19937_64 rng(seed);
  for (size_t i = len - 1; i > 0; --i) {
    std::swap(perm[i], perm[rng() % (i + 1)]);
  }
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
}

bool initChains() {
  g_hotNodes = new ChainNode[kHotLen];
  g_coldNodes = new ChainNode[kColdLen];
  initChain(g_hotNodes, kHotLen, 42);
  initChain(g_coldNodes, kColdLen, 1337);
  return true;
}

// 256 distinct switch cases, each with several ALU ops on a local accumulator
// using case-unique literal constants (prevents compiler/linker dedup). The
// per-function salt makes the functions distinct so ICF cannot merge them,
// forcing i-cache thrashing across a large code footprint.
//
// Tuning note (attempt 23, primary L1i-MPKI lever: temporal-reuse dilution).
// Attempt 22 added a 75-iteration Collatz inner loop to chase branch-misses,
// but that loop is a tiny, heavily-reused code footprint that dominated the
// retired-instruction stream (~88% of instructions came from a body that stays
// resident in L1i). Because L1-icache-load-misses_MPKI is misses-per-kilo-
// *instruction*, that compute-dense, low-miss loop diluted the metric down to
// ~4.2 MPKI (target ~28.3) and simultaneously pushed IPC far too high (1.42 vs
// 0.88). The diagnostician re-prioritized to the i-cache playbook, whose root
// cause here is "excessive temporal reuse keeping the executed instruction
// working set too compact." The single corrective lever this attempt is the
// removal of that Collatz dilution loop so the large rotated-switch footprint
// (8 distinct noinline 256-case functions) once again dominates the retired
// stream. With the dense loop gone the per-iteration instruction count drops by
// roughly the same factor that the MPKI was diluted, restoring L1i MPKI toward
// the target and lowering IPC into the acceptable band. No other lever (switch
// width, ops-per-case, chase geometry) is touched this attempt.
#define ALU_CASE(n, s)                                       \
  case (n): {                                                \
    uint64_t k = (uint64_t)(n);                              \
    v += (0x9E3779B97F4A7C15ULL ^ (k * 0x1111ULL)) + (s);    \
    v ^= (v << 7) + (k * 3 + 1 + (s));                       \
    v *= 0x100000001B3ULL + (k * 2);                         \
    v -= (k ^ 0xABCDEFULL) ^ (uint64_t)(s);                  \
    v ^= (v >> 11) ^ (k * 5);                                \
    v += (k << 3) | 1ULL;                                    \
    v *= 0x2545F4914F6CDD1DULL;                              \
    v ^= (k + 0x55ULL) + (s);                                \
    v += (k * 0x2222ULL) ^ (0xDEADBEEFULL + (s));            \
    v ^= (v << 13) + (k * 7 + 3 + (s));                      \
    v *= 0xFF51AFD7ED558CCDULL + (k * 4);                    \
    v -= (k ^ 0x123456ULL) + (uint64_t)(s);                  \
    v ^= (v >> 17) ^ (k * 9);                                \
    v += (k << 5) | 0x3ULL;                                  \
    break;                                                   \
  }

#define C4(b, s) \
  ALU_CASE((b) + 0, s) ALU_CASE((b) + 1, s) ALU_CASE((b) + 2, s) ALU_CASE((b) + 3, s)
#define C16(b, s) \
  C4((b) + 0, s) C4((b) + 4, s) C4((b) + 8, s) C4((b) + 12, s)
#define C64(b, s) \
  C16((b) + 0, s) C16((b) + 16, s) C16((b) + 32, s) C16((b) + 48, s)
#define C256(s) C64(0, s) C64(64, s) C64(128, s) C64(192, s)

#define DEFINE_SWITCH_FN(name, salt)                       \
  __attribute__((noinline)) uint64_t name(uint64_t v, int c) { \
    switch (c & 0xFF) {                                    \
      C256(salt)                                           \
      default:                                             \
        v += (salt);                                       \
        break;                                             \
    }                                                      \
    return v;                                              \
  }

DEFINE_SWITCH_FN(switchA, 0x11ULL)
DEFINE_SWITCH_FN(switchB, 0x22ULL)
DEFINE_SWITCH_FN(switchC, 0x33ULL)
DEFINE_SWITCH_FN(switchD, 0x44ULL)
DEFINE_SWITCH_FN(switchE, 0x55ULL)
DEFINE_SWITCH_FN(switchF, 0x66ULL)
DEFINE_SWITCH_FN(switchG, 0x77ULL)
DEFINE_SWITCH_FN(switchH, 0x88ULL)

#undef DEFINE_SWITCH_FN
#undef C256
#undef C64
#undef C16
#undef C4
#undef ALU_CASE

int benchmarkCtorWithCapacityChase(int iters) {
  static const bool inited = initChains();
  (void)inited;
  constexpr size_t kSize = 11264;
  constexpr int kInner = 2048;
  // Branch-entropy lever (attempt 29, primary T0 IPC / branch-misses_MPKI).
  // IPC was too high (1.42 vs 0.88) and branch-misses_MPKI too low (1.99 vs
  // 5.99): the predictor was learning the hot path too easily because the only
  // unpredictable branch was the single 8-way switch dispatch, whose density
  // relative to the 14-op ALU case bodies is small. This attempt adds a short,
  // tiny-footprint Collatz branch chain seeded by the cache-missing
  // pointer-chase load value. Each Collatz step is a data-dependent
  // taken/not-taken branch (~35% misprediction) at ~2 instructions, so it
  // injects branch entropy and pipeline flushes (raising branch-misses_MPKI and
  // pulling IPC down) without expanding the code footprint or perturbing the
  // i-cache working set. kCollatz is the single tuned lever this attempt; the
  // switch width, ops-per-case, and chase geometry are unchanged.
  constexpr int kCollatz = 28;

  uint64_t acc = g_chaseAcc + 0x12345ULL;
  ChainNode* hotPos = g_hotNodes + (acc % kHotLen);
  ChainNode* coldPos = g_coldNodes + (acc % kColdLen);

  for (int ii = 0; ii < iters; ++ii) {
    // Preserve the original benchmark's capacity-construction work.
    F14ValueMap<uint64_t, std::array<uint8_t, 1>> map(kSize);
    folly::doNotOptimizeAway(map);

    for (int j = 0; j < kInner; ++j) {
      // Dependent pointer-chase load (L1d/L2 miss).
      hotPos = hotPos->next;
      uint64_t payload = (uint64_t)(uintptr_t)hotPos;

      // Nodes are 64-byte sized; shift past the low alignment bits so the
      // permuted node index drives a full 0..255 case distribution and spreads
      // execution across all 256 cases of each switch.
      int sidx = (int)((payload >> 6) & 0xFF);

      // The data-dependent dispatch below selects among 8 distinct noinline
      // 256-case switch functions, keeping the executed code footprint large
      // and the dispatch branch unpredictable.
      switch ((payload >> 14) % 8) {
        case 0:
          acc = switchA(acc ^ payload, sidx);
          break;
        case 1:
          acc = switchB(acc ^ payload, sidx);
          break;
        case 2:
          acc = switchC(acc ^ payload, sidx);
          break;
        case 3:
          acc = switchD(acc ^ payload, sidx);
          break;
        case 4:
          acc = switchE(acc ^ payload, sidx);
          break;
        case 5:
          acc = switchF(acc ^ payload, sidx);
          break;
        case 6:
          acc = switchG(acc ^ payload, sidx);
          break;
        default:
          acc = switchH(acc ^ payload, sidx);
          break;
      }

      // Branch-entropy injection: a tiny Collatz chain seeded by the
      // cache-missing pointer-chase load value. Each step is an unpredictable
      // data-dependent branch (~35% misprediction) in a ~2-instruction body,
      // so it raises branch-misses_MPKI and inserts pipeline flushes that lower
      // IPC without enlarging the i-cache footprint.
      uint64_t cz = (payload ^ acc) | 1ULL;
      for (int c = 0; c < kCollatz; ++c) {
        if (cz & 1) {
          cz = 3 * cz + 1;
        } else {
          cz >>= 1;
        }
        if (cz == 0) {
          cz = payload | 1ULL;
        }
      }
      acc ^= cz;

      // Cold-chain advance every 4 steps via a branchless mask so the LLC miss
      // density stays high without perturbing the branch predictor.
      uint64_t coldMask = -(uint64_t)((j & 3) == 0);
      uintptr_t nextp = (uintptr_t)coldPos->next;
      uintptr_t curp = (uintptr_t)coldPos;
      coldPos = (ChainNode*)((nextp & coldMask) | (curp & ~coldMask));
      acc ^= (uint64_t)(uintptr_t)coldPos;
    }
  }

  g_chaseAcc = acc;
  folly::doNotOptimizeAway(acc);
  folly::doNotOptimizeAway(g_chaseAcc);
  return iters;
}

} // namespace

void runAllHashMapTests() {
  using std::map;
  using std::string;
  std::vector<string> testOrder;
  map<size_t,
      map<string,
          map<string, map<string, map<string, std::function<int(int)>>>>>>
      tests;

#define Z(test, map, key, value_size)                                          \
  for (auto size = FLAGS_map_size_min; size <= FLAGS_map_size_max;             \
       size *= FLAGS_map_size_step) {                                          \
    auto value = fmt::format("a[{}]", #value_size);                            \
    tests[size][#test][#key][value][#map] = [=](int iters) {                   \
      benchmark##test<map, key, std::array<uint8_t, value_size>>(iters, size); \
      return iters;                                                            \
    };                                                                         \
  }

#define Y(test, map)             \
  Z(test, map, uint64_t, 1)      \
  Z(test, map, std::string, 1)   \
  Z(test, map, NonSSOString, 1)  \
  Z(test, map, uint64_t, 128)    \
  Z(test, map, std::string, 128) \
  Z(test, map, NonSSOString, 128)

#define X(test)               \
  testOrder.push_back(#test); \
  Y(test, unord)              \
  Y(test, f14val)             \
  Y(test, f14node)            \
  Y(test, f14vec)

  X(Insert);
  X(InsertSqBr);
  X(InsertGrow);
  X(Find);
  X(ManyFind);
  X(SqBrFind);
  X(Erase);
  X(Iter);
  X(SparseIter);
  X(Clear);
  X(Dtor);
  X(CopyCtor);
  X(CtorWithCapacity);

#undef X
#undef Y
#undef Z
  for (auto& size : tests) {
    for (auto& test : testOrder) {
      for (auto& key : size.second[test]) {
        for (auto& value : key.second) {
          bool isBaseline = true;
          for (auto& map : value.second) {
            addBenchmark(
                __FILE__,
                fmt::format(
                    "{}{} {:>8}<{}, {}>[{}]",
                    isBaseline ? "" : "%",
                    test,
                    map.first,
                    key.first,
                    value.first,
                    size.first),
                std::move(map.second));
            isBaseline = false;
          }
          addBenchmark(__FILE__, "-", [](int iters) { return iters; });
        }
      }
    }
  }

  // One derived local variant of %CtorWithCapacity f14val<uint64_t, a[1]>[11264]
  // that augments the timed hot path with a permuted pointer-chase + indirect
  // switch stressor. Registered as a single concrete benchmark.
  addBenchmark(
      __FILE__,
      "%CtorWithCapacityChase f14val<uint64_t, a[1]>[11264]",
      std::function<int(int)>(
          [](int iters) { return benchmarkCtorWithCapacityChase(iters); }));
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  runAllHashMapTests();
  folly::runBenchmarks();
  return 0;
}
