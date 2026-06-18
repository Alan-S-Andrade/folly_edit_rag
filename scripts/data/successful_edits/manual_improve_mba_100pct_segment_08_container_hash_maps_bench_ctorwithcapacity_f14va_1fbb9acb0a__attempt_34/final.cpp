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
#include <functional>
#include <map>
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

// ---------------------------------------------------------------------------
// Derived benchmark for %CtorWithCapacity f14val<uint64_t, a[1]>[11264].
//
// The reference benchmark is extremely frontend-light (IPC far above target),
// so this derived variant injects a large hot-code footprint to raise
// L1-icache-load-misses_MPKI and bring IPC down. The proven structural pattern
// is used: three noinline rotor functions, each holding a 256-case switch with
// many ALU ops per case using unique literal constants (to defeat compiler
// deduplication), indexed by a pointer-chase payload byte and rotated with
// i%3 in the hot loop.
//
// Refinement (branch-predictor entropy playbook): the indirect/jump-table
// switch alone leaves branch-misses_MPKI far too low (predictor learns the
// hot path too easily), which keeps IPC above target. We add a compact Collatz
// branch sequence per inner-loop iteration, seeded with the cache-missing
// pointer-chase value, to inject reliable, data-dependent conditional branch
// mispredictions with minimal additional icache footprint. N_COLLATZ controls
// the branch-miss MPKI lever.
// ---------------------------------------------------------------------------

#define N_COLLATZ 75

#define ICACHE_CASE(n)                                            \
  case (n):                                                       \
    acc += 0x9E3779B97F4A7C15ULL ^ ((n) * 0x100000001B3ULL);      \
    acc ^= (acc << 13) + ((n) * 0xA24BAED4963EE407ULL);           \
    acc *= 0xD6E8FEB86659FD93ULL + (n);                           \
    acc -= ((n) * 0xC2B2AE3D27D4EB4FULL) ^ (acc >> 7);            \
    acc += (acc << 3) ^ ((n) * 0x165667B19E3779F9ULL);            \
    acc ^= 0x27D4EB2F165667C5ULL - ((n) * 3ULL);                  \
    acc *= 0x9E3779B185EBCA87ULL + ((n) * 5ULL);                  \
    acc += (acc >> 11) ^ ((n) * 0xFF51AFD7ED558CCDULL);           \
    acc -= ((n) << 7) ^ (acc << 5);                               \
    acc ^= (acc >> 17) + ((n) * 0xBF58476D1CE4E5B9ULL);           \
    acc += 0x94D049BB133111EBULL ^ ((n) * 7ULL);                  \
    acc *= 0x2545F4914F6CDD1DULL + ((n) * 9ULL);                  \
    acc -= (acc << 9) ^ ((n) * 0x9FB21C651E98DF25ULL);            \
    acc ^= (acc >> 23) + ((n) * 11ULL);                           \
    acc += ((n) * 0xEB44ACCAB455D165ULL) ^ (acc << 2);            \
    acc *= 0x589965CC75374CC3ULL + ((n) * 13ULL);                 \
    break;

#define IC4(n) \
  ICACHE_CASE(n) ICACHE_CASE((n) + 1) ICACHE_CASE((n) + 2) ICACHE_CASE((n) + 3)
#define IC16(n) IC4(n) IC4((n) + 4) IC4((n) + 8) IC4((n) + 12)
#define IC64(n) IC16(n) IC16((n) + 16) IC16((n) + 32) IC16((n) + 48)
#define IC256 IC64(0) IC64(64) IC64(128) IC64(192)

__attribute__((noinline)) static uint64_t icacheRotor0(
    uint64_t acc, uint8_t idx) {
  acc ^= 0x1111111111111111ULL;
  switch (idx) {
    IC256
  }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheRotor1(
    uint64_t acc, uint8_t idx) {
  acc ^= 0x2222222222222222ULL;
  switch (idx) {
    IC256
  }
  return acc;
}

__attribute__((noinline)) static uint64_t icacheRotor2(
    uint64_t acc, uint8_t idx) {
  acc ^= 0x3333333333333333ULL;
  switch (idx) {
    IC256
  }
  return acc;
}

template <template <class, class> class Map, class K, class V>
void benchmarkCtorWithCapacityIcache(int runs, int size) {
  std::vector<uint32_t> chase(size);
  for (int i = 0; i < size; ++i) {
    chase[i] = static_cast<uint32_t>(
        (static_cast<uint64_t>(i) * 2654435761ULL + 12345ULL) % size);
  }
  folly::makeUnpredictable(chase);
  uint64_t acc = 0;
  uint32_t p = 0;
  for (int ii = 0; ii < runs; ++ii) {
    Map<K, V> map(size);
    folly::doNotOptimizeAway(map);
    for (int i = 0; i < size; ++i) {
      p = chase[p % static_cast<uint32_t>(size)];
      uint8_t idx = static_cast<uint8_t>(p & 0xFFu);
      switch (i % 3) {
        case 0:
          acc = icacheRotor0(acc, idx);
          break;
        case 1:
          acc = icacheRotor1(acc, idx);
          break;
        default:
          acc = icacheRotor2(acc, idx);
          break;
      }
      // Compact Collatz branch sequence to raise branch-misses_MPKI and pull
      // IPC down toward target. The seed mixes the cache-missing pointer-chase
      // value so the per-iteration conditional branch is hard to predict; the
      // odd-bit set avoids a trivial zero fixpoint that would be predictable.
      uint64_t x = (acc ^ (static_cast<uint64_t>(p) * 0x9E3779B97F4A7C15ULL)) | 1ULL;
      for (int c = 0; c < N_COLLATZ; ++c) {
        if (x & 1ULL) {
          x = 3ULL * x + 1ULL;
        } else {
          x >>= 1;
        }
      }
      acc += x;
    }
    folly::doNotOptimizeAway(acc);
  }
  folly::doNotOptimizeAway(acc);
}

template <class K, class V>
using unord = std::unordered_map<K, V>;
template <class K, class V>
using f14val = F14ValueMap<K, V>;
template <class K, class V>
using f14node = F14NodeMap<K, V>;
template <class K, class V>
using f14vec = F14VectorMap<K, V>;

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

  // Derived icache-thrashing variant of the CtorWithCapacity reference.
  addBenchmark(
      __FILE__,
      "%CtorWithCapacityIcache   f14val<uint64_t, a[1]>[11264]",
      [](int iters) {
        benchmarkCtorWithCapacityIcache<f14val, uint64_t, std::array<uint8_t, 1>>(
            iters, 11264);
        return iters;
      });
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
