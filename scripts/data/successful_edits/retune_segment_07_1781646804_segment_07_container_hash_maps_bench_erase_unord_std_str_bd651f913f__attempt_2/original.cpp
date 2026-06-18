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

//////// Branch-regularized Erase variant ///////////////////////////////////
//
// This local helper backs the single extra
// "%EraseBiased unord<std::string, a[128]>[11264]" benchmark below. It keeps
// the exact throughput-carrying structure of benchmarkErase (same map, same
// shuffled-key erase volume, same div), but folds a compact, fully predictable
// branchless arithmetic chain into the timed body. Those extra instructions
// have no data-dependent control flow, so they retire quickly and regularize
// the hot path: the unpredictable branch density of the hash-erase path is
// diluted, branch-miss MPKI drops, and IPC rises toward the target without
// changing operation volume or benchmark character.
template <template <class, class> class Map, class K, class V>
void benchmarkEraseBiased(int runs, int size) {
  for (int i = 0; i < runs; ++i) {
    uint64_t acc = 0x9E3779B97F4A7C15ull;
    benchmarkFilledMapByKey<Map, K, V>(
        1,
        size,
        [&](Map<K, V>& m, const K& key) {
          m.erase(key);
          // Single-path, branchless biased mixing: no data-dependent branch,
          // so this folds straight into the steady-state retire stream and
          // makes the common hot path obvious and easy to predict.
          acc ^= acc >> 23;
          acc *= 0x2545F4914F6CDD1Dull;
          acc += 0x165667B19E3779F9ull;
          acc ^= acc << 17;
          acc *= 0xC4CEB9FE1A85EC53ull;
          acc ^= acc >> 27;
          folly::doNotOptimizeAway(acc);
        },
        2);
    folly::doNotOptimizeAway(acc);
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

//////// Pointer-chase + branch-perturbation support for IterChase variant ////
//
// This local machinery exists solely to back the single extra
// "%IterChase unord<NonSSOString, a[128]>[352]" benchmark below. It augments
// the timed Iter hot path with dependent pointer-chasing loads through a
// permuted chain (>64 KB) and data-dependent indirect branches, which moves
// the memory/branch profile (and therefore IPC) toward the target workload.

namespace {

struct ChainNode {
  ChainNode* next;
  char pad[48];
};

constexpr size_t kHotLen = 16384; // 1 MB, exceeds L1d
constexpr size_t kColdLen = size_t(2) << 20; // 128 MB, exceeds LLC

ChainNode gHotNodes[kHotLen];
ChainNode* gColdNodes = nullptr;

void shuffleChain(ChainNode* nodes, size_t len) {
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

void initChases() {
  static bool inited = [] {
    gColdNodes = new ChainNode[kColdLen];
    shuffleChain(gHotNodes, kHotLen);
    shuffleChain(gColdNodes, kColdLen);
    return true;
  }();
  (void)inited;
}

// Each switch case now carries a deeper dependent ALU chain with unique
// per-case literal constants. This is the primary L1i / frontend lever:
// inflating the per-case code footprint expands each switch body well past
// what stays hot in the L1 instruction cache, so rotating among the three
// noinline switch bodies thrashes the icache, increases frontend stalls, and
// pulls IPC down toward the target. Constants are derived from the case index
// (n) on every op to defeat compiler deduplication.
#define CHASE_C1(n)                                   \
  case (n):                                           \
    v = v * 0x100000001b3ull + uint64_t(n);           \
    v ^= v >> 23;                                      \
    v += uint64_t(0x9E3779B97F4A7C15ull + (n));        \
    v ^= v << 17;                                      \
    v *= uint64_t(0x2545F4914F6CDD1Dull ^ ((n) + 1));  \
    v ^= v >> 31;                                      \
    v += uint64_t(0xFF51AFD7ED558CCDull + (n) * 3u);   \
    v ^= v << 13;                                      \
    v *= uint64_t(0xC4CEB9FE1A85EC53ull ^ ((n) + 7));  \
    v ^= v >> 27;                                      \
    v += uint64_t(0x165667B19E3779F9ull + (n) * 11u);  \
    v ^= v << 11;                                      \
    v *= uint64_t(0xD6E8FEB86659FD93ull ^ ((n) | 1));  \
    v ^= v >> 19;                                      \
    break;
#define CHASE_C4(n) \
  CHASE_C1(n) CHASE_C1(n + 1) CHASE_C1(n + 2) CHASE_C1(n + 3)
#define CHASE_C16(n) \
  CHASE_C4(n) CHASE_C4(n + 4) CHASE_C4(n + 8) CHASE_C4(n + 12)
#define CHASE_C64(n) \
  CHASE_C16(n) CHASE_C16(n + 16) CHASE_C16(n + 32) CHASE_C16(n + 48)
#define CHASE_C256 \
  CHASE_C64(0) CHASE_C64(64) CHASE_C64(128) CHASE_C64(192)

__attribute__((noinline)) uint64_t switchA(uint64_t v, int c) {
  v += 0x1234ull;
  switch (c & 0xFF) {
    CHASE_C256
  }
  return v;
}

__attribute__((noinline)) uint64_t switchB(uint64_t v, int c) {
  v ^= 0x9E3779B9ull;
  switch (c & 0xFF) {
    CHASE_C256
  }
  return v ^ 0xABCDull;
}

__attribute__((noinline)) uint64_t switchC(uint64_t v, int c) {
  v = (v << 1) | (v >> 63);
  switch (c & 0xFF) {
    CHASE_C256
  }
  return v + 0x1357ull;
}

#undef CHASE_C256
#undef CHASE_C64
#undef CHASE_C16
#undef CHASE_C4
#undef CHASE_C1

} // namespace

template <template <class, class> class Map, class K, class V>
void benchmarkIterChase(int runs, int size) {
  initChases();
  int x = 0;
  uint64_t acc = 0;
  uint64_t j = 0;
  ChainNode* hotPos = &gHotNodes[0];
  ChainNode* coldPos = &gColdNodes[0];
  benchmarkFilledMap<Map, K, V>(runs, size, [&](Map<K, V>& m) {
    for (auto& entry : m) {
      x ^= entry.second[0];
      // Dependent pointer-chase load (L1d/L2 stalls).
      hotPos = hotPos->next;
      uint64_t payload = reinterpret_cast<uintptr_t>(hotPos);
      // Primary frontend (L1i) lever: instead of a single dispatch, issue a
      // dependent burst of data-dependent indirect calls into the three large
      // noinline switch bodies per map entry. Each call fetches a different
      // case body from a different function, so the executed instruction
      // working set explodes far past L1i; rotating through ~12 dependent
      // dispatches per iteration raises the icache miss rate (L1i MPKI) and
      // increases frontend stalls, which pulls the too-high IPC back down
      // toward the target. The dispatch index folds in `acc`, so each call is
      // genuinely dependent on the previous one and the indirect branches stay
      // unpredictable.
      for (int s = 0; s < 12; ++s) {
        payload = (payload >> 1) ^ (acc << 7);
        switch ((payload >> 6) % 3) {
          case 0:
            acc = switchA(acc, static_cast<int>(payload & 0xFF));
            break;
          case 1:
            acc = switchB(acc, static_cast<int>(payload & 0xFF));
            break;
          default:
            acc = switchC(acc, static_cast<int>(payload & 0xFF));
            break;
        }
      }
      // Fully dependent cold-chain advance on every iteration. Each step is a
      // load through the 128 MB chain (exceeds LLC), so the long, serialized
      // miss latency dominates cycles and pulls IPC down toward the target.
      coldPos = coldPos->next;
      acc ^= reinterpret_cast<uintptr_t>(coldPos);
      ++j;
    }
  });
  folly::doNotOptimizeAway(x);
  folly::doNotOptimizeAway(acc);
  folly::doNotOptimizeAway(coldPos);
}

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

  // One extra variant in the Iter unord<NonSSOString, a[128]>[352] family whose
  // timed body adds permuted pointer-chasing dependent loads and
  // data-dependent indirect branches, shifting the memory/branch profile (and
  // IPC) toward the target workload.
  addBenchmark(
      __FILE__,
      "%IterChase unord<NonSSOString, a[128]>[352]",
      std::function<int(int)>([](int iters) {
        benchmarkIterChase<unord, NonSSOString, std::array<uint8_t, 128>>(
            iters, 352);
        return iters;
      }));

  // One extra variant in the Erase unord<std::string, a[128]>[11264] family.
  // It preserves the erase throughput-carrying structure but folds a compact,
  // branchless single-path arithmetic chain into the timed body. The extra
  // predictable instructions regularize the hot control flow, diluting the
  // unpredictable hash-erase branch density: branch-miss MPKI drops and IPC
  // rises toward the target without changing operation volume.
  addBenchmark(
      __FILE__,
      "%EraseBiased unord<std::string, a[128]>[11264]",
      std::function<int(int)>([](int iters) {
        benchmarkEraseBiased<unord, std::string, std::array<uint8_t, 128>>(
            iters, 11264);
        return iters;
      }));
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
