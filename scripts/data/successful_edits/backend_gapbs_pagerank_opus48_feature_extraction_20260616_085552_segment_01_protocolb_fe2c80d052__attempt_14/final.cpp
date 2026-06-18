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

#include <thrift/lib/cpp2/frozen/FrozenUtil.h>
#include <thrift/lib/cpp2/protocol/Object.h>
#include <thrift/lib/cpp2/protocol/Serializer.h>
#include <thrift/lib/cpp2/test/Structs.h>

#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <folly/Optional.h>

#include <random>
#include <vector>

using namespace apache::thrift;
using namespace thrift::benchmark;

template <class>
struct SerializerTraits;
template <class ReaderType, class WriterType>
struct SerializerTraits<Serializer<ReaderType, WriterType>> {
  using Reader = ReaderType;
  using Writer = WriterType;
};

template <class T>
using GetReader = typename SerializerTraits<T>::Reader;
template <class T>
using GetWriter = typename SerializerTraits<T>::Writer;

struct FrozenSerializer {
  template <class T>
  static void serialize(const T& obj, folly::IOBufQueue* out) {
    out->append(folly::IOBuf::fromString(frozen::freezeToString(obj)));
  }
  template <class T>
  static size_t deserialize(folly::IOBuf* iobuf, T& t) {
    auto view = frozen::mapFrozen<T>(iobuf->coalesce());
    t = view.thaw();
    return 0;
  }
};

enum class SerializerMethod {
  Codegen,
  Object,
};

// The benckmark is to measure single struct use case, the iteration here is
// more like a benchmark artifact, so avoid doing optimizationon iteration
// usecase in this benchmark (e.g. move string definition out of while loop)

template <
    SerializerMethod kSerializerMethod,
    typename Serializer,
    typename Struct>
void writeBench(::benchmark::State& state) {
  auto strct = create<Struct>();
  protocol::Object obj;
  if constexpr (kSerializerMethod == SerializerMethod::Object) {
    folly::IOBufQueue q;
    Serializer::serialize(strct, &q);
    obj = protocol::parseObject<GetReader<Serializer>>(*q.move());
  }

  folly::IOBufQueue q;
  for (auto _ : state) {
    if constexpr (kSerializerMethod == SerializerMethod::Object) {
      protocol::serializeObject<GetWriter<Serializer>>(obj, q);
      ::benchmark::DoNotOptimize(q);
    } else {
      Serializer::serialize(strct, &q);
      ::benchmark::DoNotOptimize(q);
    }

    // Reuse the queue across iterations to avoid allocating a new buffer for
    // each struct (which would dominate the measurement), but keep only the
    // tail to avoid unbounded growth.
    if (auto head = q.front(); head && head->isChained()) {
      q.append(q.move()->prev()->unlink());
    }
  }
}

template <
    SerializerMethod kSerializerMethod,
    typename Serializer,
    typename Struct>
void readBench(::benchmark::State& state) {
  auto strct = create<Struct>();
  folly::IOBufQueue q;
  Serializer::serialize(strct, &q);
  auto buf = q.move();
  // coalesce the IOBuf chain to test fast path
  buf->coalesce();

  for (auto _ : state) {
    if constexpr (kSerializerMethod == SerializerMethod::Object) {
      auto obj = protocol::parseObject<GetReader<Serializer>>(*buf);
      ::benchmark::DoNotOptimize(obj);
    } else {
      Struct data;
      Serializer::deserialize(buf.get(), data);
      ::benchmark::DoNotOptimize(data);
    }
  }
}

constexpr SerializerMethod getSerializerMethod(std::string_view prefix) {
  return prefix == "" || prefix == "OpEncode" ? SerializerMethod::Codegen
      : prefix == "Object"
      ? SerializerMethod::Object
      : throw std::invalid_argument(std::string(prefix) + " is invalid");
}

// clang-format off
#define X1(Prefix, proto, rdwr, bench, benchprefix)                        \
  static void Prefix##proto##Protocol_##rdwr##_##bench(                    \
      ::benchmark::State& state) {                                         \
    rdwr##Bench<                                                           \
        getSerializerMethod(#Prefix),                                      \
        proto##Serializer,                                                 \
        benchprefix##bench>(state);                                        \
  }                                                                        \
  BENCHMARK(Prefix##proto##Protocol_##rdwr##_##bench);

#define X2(Prefix, proto, bench)  \
  X1(Prefix, proto, write, bench,) \
  X1(Prefix, proto, read, bench,)

#define OpEncodeX2(Prefix, proto, bench)  \
  X1(Prefix, proto, write, bench, Op) \
  X1(Prefix, proto, read, bench, Op)

#define APPLY(M, Prefix, proto)        \
  M(Prefix, proto, Empty)              \
  M(Prefix, proto, SmallInt)           \
  M(Prefix, proto, BigInt)             \
  M(Prefix, proto, SmallString)        \
  M(Prefix, proto, BigString)          \
  M(Prefix, proto, BigBinary)          \
  M(Prefix, proto, LargeBinary)        \
  M(Prefix, proto, Mixed)              \
  M(Prefix, proto, MixedUnion)         \
  M(Prefix, proto, MixedByte)           \
  M(Prefix, proto, MixedShort)         \
  M(Prefix, proto, MixedInt)           \
  M(Prefix, proto, MixedBigInt)        \
  M(Prefix, proto, LargeMixed)         \
  M(Prefix, proto, LargeMixedSparse)   \
  M(Prefix, proto, SmallListInt)       \
  M(Prefix, proto, BigListByte)        \
  M(Prefix, proto, BigListShort)       \
  M(Prefix, proto, BigListInt)         \
  M(Prefix, proto, BigListBigInt)      \
  M(Prefix, proto, BigListFloat)       \
  M(Prefix, proto, BigListDouble)      \
  M(Prefix, proto, BigListMixed)       \
  M(Prefix, proto, BigListMixedByte)   \
  M(Prefix, proto, BigListMixedShort)  \
  M(Prefix, proto, BigListMixedInt)    \
  M(Prefix, proto, BigListMixedBigInt) \
  M(Prefix, proto, LargeListMixed)     \
  M(Prefix, proto, LargeSetInt)        \
  M(Prefix, proto, UnorderedSetInt)    \
  M(Prefix, proto, SortedVecSetInt)    \
  M(Prefix, proto, LargeMapInt)        \
  M(Prefix, proto, LargeMapMixed)      \
  M(Prefix, proto, LargeUnorderedMapMixed)        \
  M(Prefix, proto, LargeSortedVecMapMixed)        \
  M(Prefix, proto, UnorderedMapInt)    \
  M(Prefix, proto, NestedMap)          \
  M(Prefix, proto, SortedVecNestedMap) \
  M(Prefix, proto, ComplexStruct)      \
  M(Prefix, proto, ComplexUnion)

#define X(Prefix, proto) APPLY(X2, Prefix, proto)              

#define OpEncodeX(Prefix, proto) APPLY(OpEncodeX2, Prefix, proto)              

// NOLINTBEGIN(facebook-avoid-non-const-global-variables)
X(, Binary)
X(, Compact)
X(, SimpleJSON)
X(, JSON)
X(, Frozen)
X(Object, Binary)
X(Object, Compact)
OpEncodeX(OpEncode, Binary)
OpEncodeX(OpEncode, Compact)
// clang-format on

// Derived from FrozenProtocol_write_UnorderedSetInt: the bare frozen write of
// this struct is too instruction-dense (IPC too high, L1d misses too low). To
// move the working set out of L1d while staying inside L2, interleave the same
// frozen write with a Fisher-Yates shuffled pointer-chase over a 1 MB hot
// chain (16384 nodes x 64 bytes). The dependent loads serialize on L1d misses
// (but hit L2), lowering IPC and raising L1-dcache-load-misses_MPKI.
namespace {

struct ChainNode {
  ChainNode* next;
  char pad[48];
};

ChainNode* buildHotChain() {
  constexpr size_t kNodes = 16384; // 1 MB working set, fits in L2.
  static std::vector<ChainNode> nodes(kNodes);
  std::vector<size_t> order(kNodes);
  for (size_t i = 0; i < kNodes; ++i) {
    order[i] = i;
  }
  std::mt19937 rng(0x9E3779B9u);
  for (size_t i = kNodes - 1; i > 0; --i) {
    std::uniform_int_distribution<size_t> dist(0, i);
    std::swap(order[i], order[dist(rng)]);
  }
  for (size_t i = 0; i + 1 < kNodes; ++i) {
    nodes[order[i]].next = &nodes[order[i + 1]];
  }
  nodes[order[kNodes - 1]].next = &nodes[order[0]];
  return &nodes[order[0]];
}

} // namespace

static void FrozenProtocol_write_UnorderedSetInt_ChainWalk(
    ::benchmark::State& state) {
  auto strct = create<UnorderedSetInt>();
  ChainNode* p = buildHotChain();

  folly::IOBufQueue q;
  for (auto _ : state) {
    FrozenSerializer::serialize(strct, &q);
    ::benchmark::DoNotOptimize(q);

    // Dependent pointer-chase through the 1 MB hot chain. Each step is a load
    // that misses L1d but hits L2, generating reliable L1d misses.
    for (int i = 0; i < 128; ++i) {
      p = p->next;
    }
    ::benchmark::DoNotOptimize(p);

    if (auto head = q.front(); head && head->isChained()) {
      q.append(q.move()->prev()->unlink());
    }
  }
}
BENCHMARK(FrozenProtocol_write_UnorderedSetInt_ChainWalk);

BENCHMARK_MAIN();
// NOLINTEND(facebook-avoid-non-const-global-variables)
