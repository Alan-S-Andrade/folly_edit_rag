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

//////// Pointer-chasing + branch-perturbed variant of CtorWithCapacity ////////
//
// This local variant of the f14val<uint64_t, a[1]> CtorWithCapacity benchmark
// augments the timed hot path with a large permuted pointer-chase (dependent
// loads exceeding L1d/LLC) and rotating indirect 256-case switch functions
// (frontend / branch-predictor pressure). This deliberately shifts the
// previously compute-bound profile toward a memory/branch-bound one.
namespace {

struct ChainNode {
  ChainNode* next;
  char pad[48];
};

constexpr size_t kHotChaseLen = 16384; // ~1 MB, exceeds L1d, fits L2
constexpr size_t kColdChaseLen = size_t(2) << 20; // ~128 MB, exceeds LLC

void initChase(ChainNode* nodes, size_t len) {
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), size_t(0));
  std::mt19937_64 rng(42);
  for (size_t i = len - 1; i > 0; --i) {
    std::swap(perm[i], perm[rng() % (i + 1)]);
  }
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
}

#define CHASE_CASE4(n)                                                       \
  case (n) + 0:                                                              \
    v += 0x1234u + ((n) + 0);                                                \
    v ^= 0x9e3779b9u;                                                        \
    v = (v << 3) | (v >> 61);                                                \
    break;                                                                   \
  case (n) + 1:                                                              \
    v -= 0x5678u + ((n) + 1);                                                \
    v ^= 0xabcd1234u;                                                        \
    v = (v << 5) | (v >> 59);                                                \
    break;                                                                   \
  case (n) + 2:                                                              \
    v += 0x13579bdfu + ((n) + 2);                                            \
    v ^= 0x2468ace0u;                                                        \
    v = (v << 7) | (v >> 57);                                                \
    break;                                                                   \
  case (n) + 3:                                                              \
    v ^= 0x0f0f0f0fu + ((n) + 3);                                            \
    v += 0xdeadbeefu;                                                        \
    v = (v << 11) | (v >> 53);                                               \
    break;
#define CHASE_CASE16(n) \
  CHASE_CASE4(n) CHASE_CASE4((n) + 4) CHASE_CASE4((n) + 8) CHASE_CASE4((n) + 12)
#define CHASE_CASE64(n)                                       \
  CHASE_CASE16(n) CHASE_CASE16((n) + 16) CHASE_CASE16((n) + 32) \
      CHASE_CASE16((n) + 48)
#define CHASE_CASE256 \
  CHASE_CASE64(0) CHASE_CASE64(64) CHASE_CASE64(128) CHASE_CASE64(192)

__attribute__((noinline)) uint64_t chaseSwitchA(uint64_t v, int c) {
  v += 0x111u;
  switch (c & 0xFF) { CHASE_CASE256 }
  return v;
}

__attribute__((noinline)) uint64_t chaseSwitchB(uint64_t v, int c) {
  v ^= 0x222u;
  switch (c & 0xFF) { CHASE_CASE256 }
  return v * 0x100000001b3ull;
}

__attribute__((noinline)) uint64_t chaseSwitchC(uint64_t v, int c) {
  v -= 0x333u;
  switch (c & 0xFF) { CHASE_CASE256 }
  return v + 0x9e3779b97f4a7c15ull;
}

#undef CHASE_CASE256
#undef CHASE_CASE64
#undef CHASE_CASE16
#undef CHASE_CASE4

int benchmarkCtorWithCapacityChase(int runs, int size) {
  static ChainNode* hotNodes = nullptr;
  static ChainNode* coldNodes = nullptr;
  if (hotNodes == nullptr) {
    hotNodes = new ChainNode[kHotChaseLen];
    coldNodes = new ChainNode[kColdChaseLen];
    initChase(hotNodes, kHotChaseLen);
    initChase(coldNodes, kColdChaseLen);
  }

  uint64_t acc = 0;
  ChainNode* hotPos = &hotNodes[0];
  ChainNode* hotPos2 = &hotNodes[kHotChaseLen / 2];
  ChainNode* coldPos = &coldNodes[0];

  for (int ii = 0; ii < runs; ++ii) {
    f14val<uint64_t, std::array<uint8_t, 1>> map(size);
    folly::doNotOptimizeAway(map);

    for (size_t j = 0; j < 24; ++j) {
      hotPos = hotPos->next; // dependent L1d-missing load
      hotPos2 = hotPos2->next; // second independent walker
      uint64_t payload =
          (uint64_t)(uintptr_t)hotPos ^ ((uint64_t)(uintptr_t)hotPos2 >> 6);
      int sel = static_cast<int>(payload & 0xFF);
      switch (j % 3) { // indirect-ish rotation across 3 hot switch funcs
        case 0:
          acc = chaseSwitchA(acc, sel);
          break;
        case 1:
          acc = chaseSwitchB(acc, sel);
          break;
        default:
          acc = chaseSwitchC(acc, sel);
          break;
      }
      // Branchless cold-chain advance every 8th step -> sparse LLC misses.
      uint64_t coldMask = 0u - uint64_t((j & 7) == 0);
      coldPos = (ChainNode*)(((uintptr_t)coldPos->next & coldMask) |
                             ((uintptr_t)coldPos & ~coldMask));
      acc ^= (uint64_t)(uintptr_t)coldPos;
    }
  }

  folly::doNotOptimizeAway(acc);
  folly::doNotOptimizeAway(hotPos);
  folly::doNotOptimizeAway(hotPos2);
  folly::doNotOptimizeAway(coldPos);
  return runs;
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

  // One extra local variant in the CtorWithCapacity f14val<uint64_t, a[1]>
  // [11264] family: same constructor work, but the timed hot path is augmented
  // with permuted pointer-chasing and rotating 256-case switch functions to
  // move the profile from compute-bound toward memory/branch-bound.
  addBenchmark(
      __FILE__,
      "%CtorWithCapacityChase f14val<uint64_t, a[1]>[11264]",
      [](int iters) { return benchmarkCtorWithCapacityChase(iters, 11264); });
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::gflags::SetCommandLineOptionWithMode(
      "bm_max_iters", "100000", folly::gflags::SET_FLAG_IF_DEFAULT);
  folly::gflags::SetCommandLineOptionWithMode(
      "bm_min_iters", "10000", folly::gflags::SET_FLAG_IF_DEFAULT);
  folly::gflags::SetCommandLineOptionWithMode(
      "bm_max_secs", "1", folly::gflags::SET_FLAG_IF_DEFAULT);
  LOG(INFO) << "Preparing benchmark...";
  runAllHashMapTests();
  LOG(INFO) << "Running benchmark, which could take tens of minutes...";
  runBenchmarks();
  return 0;
}
