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

#include <glog/logging.h>
#include <folly/Benchmark.h>
#include <folly/BenchmarkUtil.h>
#include <folly/Optional.h>
#include <folly/init/Init.h>
#include <folly/portability/GFlags.h>

#include <numeric>
#include <random>

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
void writeBench(size_t iters) {
  folly::BenchmarkSuspender susp;
  auto strct = create<Struct>();
  protocol::Object obj;
  if constexpr (kSerializerMethod == SerializerMethod::Object) {
    folly::IOBufQueue q;
    Serializer::serialize(strct, &q);
    obj = protocol::parseObject<GetReader<Serializer>>(*q.move());
  }
  susp.dismiss();

  folly::IOBufQueue q;
  while (iters--) {
    if constexpr (kSerializerMethod == SerializerMethod::Object) {
      protocol::serializeObject<GetWriter<Serializer>>(obj, q);
      folly::doNotOptimizeAway(q);
    } else {
      Serializer::serialize(strct, &q);
      folly::doNotOptimizeAway(q);
    }

    // Reuse the queue across iterations to avoid allocating a new buffer for
    // each struct (which would dominate the measurement), but keep only the
    // tail to avoid unbounded growth.
    if (auto head = q.front(); head && head->isChained()) {
      q.append(q.move()->prev()->unlink());
    }
  }
  susp.rehire();
}

template <
    SerializerMethod kSerializerMethod,
    typename Serializer,
    typename Struct>
void readBench(size_t iters) {
  folly::BenchmarkSuspender susp;
  auto strct = create<Struct>();
  folly::IOBufQueue q;
  Serializer::serialize(strct, &q);
  auto buf = q.move();
  // coalesce the IOBuf chain to test fast path
  buf->coalesce();
  susp.dismiss();

  while (iters--) {
    if constexpr (kSerializerMethod == SerializerMethod::Object) {
      auto obj = protocol::parseObject<GetReader<Serializer>>(*buf);
      folly::doNotOptimizeAway(obj);
    } else {
      Struct data;
      Serializer::deserialize(buf.get(), data);
      folly::doNotOptimizeAway(data);
    }
  }
  susp.rehire();
}

constexpr SerializerMethod getSerializerMethod(std::string_view prefix) {
  return prefix == "" || prefix == "OpEncode" ? SerializerMethod::Codegen
      : prefix == "Object"
      ? SerializerMethod::Object
      : throw std::invalid_argument(std::string(prefix) + " is invalid");
}

#define X1(Prefix, proto, rdwr, bench, benchprefix)            \
  BENCHMARK(Prefix##proto##Protocol_##rdwr##_##bench, iters) { \
    rdwr##Bench<                                               \
        getSerializerMethod(#Prefix),                          \
        proto##Serializer,                                     \
        benchprefix##bench>(iters);                            \
  }

// clang-format off
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
  M(Prefix, proto, MixedInt)           \
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
  M(Prefix, proto, BigListMixedInt)    \
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

X(, Binary)
X(, Compact)
X(, SimpleJSON)
X(, JSON)
X(, Frozen)
X(Object, Binary)
X(Object, Compact)
OpEncodeX(OpEncode, Binary)
OpEncodeX(OpEncode, Compact)

namespace {

struct ChaseNode_LargeSetInt_v1 {
  ChaseNode_LargeSetInt_v1* next;
  int64_t payload;
};

std::vector<ChaseNode_LargeSetInt_v1> chase_chain_LargeSetInt_v1;
ChaseNode_LargeSetInt_v1* chase_head_LargeSetInt_v1 = nullptr;

void setup_chase_chain_LargeSetInt_v1(size_t size) {
  if (chase_head_LargeSetInt_v1) {
    return;
  }
  chase_chain_LargeSetInt_v1.resize(size);
  std::vector<int> indices(size);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(
      indices.begin(), indices.end(), std::default_random_engine(123));

  for (size_t i = 0; i < size; ++i) {
    chase_chain_LargeSetInt_v1[indices[i]].next =
        &chase_chain_LargeSetInt_v1[indices[(i + 1) % size]];
    chase_chain_LargeSetInt_v1[indices[i]].payload =
        i * 37 + (i & 1 ? 1 : -1) * i * i * 13;
  }
  chase_head_LargeSetInt_v1 = &chase_chain_LargeSetInt_v1[0];
}

template <int N>
__attribute__((noinline)) int64_t
large_set_int_v1_perturb(int64_t acc, int8_t input) {
  // Using different constants in each template instantiation to prevent
  // code deduplication by the compiler/linker.
  switch (input) {
#define V1_CASE_BLOCK(i)                                                       \
  case i:                                                                      \
    acc += (i * (0x1010101 + N * 0x11));                                        \
    acc ^= (i * (0x2020202 + N * 0x22));                                        \
    acc *= (i * (0x3030303 + N * 0x33));                                        \
    acc += (i * (0x4040404 + N * 0x44));                                        \
    acc ^= (i * (0x5050505 + N * 0x55));                                        \
    acc *= (i * (0x6060606 + N * 0x66));                                        \
    acc += (i * (0x7070707 + N * 0x77));                                        \
    acc ^= (i * (0x8080808 + N * 0x88));                                        \
    acc *= (i * (0x9090909 + N * 0x99));                                        \
    acc += (i * (0xA0A0A0A + N * 0xAA));                                        \
    acc ^= (i * (0xB0B0B0B + N * 0xBB));                                        \
    acc *= (i * (0xC0C0C0C + N * 0xCC));                                        \
    acc += (i * (0xD0D0D0D + N * 0xDD));                                        \
    acc ^= (i * (0xE0E0E0E + N * 0xEE));                                        \
    break;

#define V1_P_16(p)                                                             \
  V1_CASE_BLOCK(p + 0)                                                         \
  V1_CASE_BLOCK(p + 1)                                                         \
  V1_CASE_BLOCK(p + 2)                                                         \
  V1_CASE_BLOCK(p + 3)                                                         \
  V1_CASE_BLOCK(p + 4)                                                         \
  V1_CASE_BLOCK(p + 5)                                                         \
  V1_CASE_BLOCK(p + 6)                                                         \
  V1_CASE_BLOCK(p + 7)                                                         \
  V1_CASE_BLOCK(p + 8)                                                         \
  V1_CASE_BLOCK(p + 9)                                                         \
  V1_CASE_BLOCK(p + 10)                                                        \
  V1_CASE_BLOCK(p + 11)                                                        \
  V1_CASE_BLOCK(p + 12)                                                        \
  V1_CASE_BLOCK(p + 13)                                                        \
  V1_CASE_BLOCK(p + 14)                                                        \
  V1_CASE_BLOCK(p + 15)
#define V1_P_64(p)                                                             \
  V1_P_16(p) V1_P_16(p + 16) V1_P_16(p + 32) V1_P_16(p + 48)
#define V1_P_256() V1_P_64(0) V1_P_64(64) V1_P_64(128) V1_P_64(192)

    V1_P_256()

#undef V1_P_256
#undef V1_P_64
#undef V1_P_16
#undef V1_CASE_BLOCK
  }
  return acc;
}
} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v1, iters) {
  folly::BenchmarkSuspender susp;
  using Serializer = CompactSerializer;
  using Struct = LargeSetInt;
  auto strct = create<Struct>();
  folly::IOBufQueue q;
  Serializer::serialize(strct, &q);
  auto buf = q.move();
  buf->coalesce();

  constexpr size_t kChaseChainSize = 1024;
  setup_chase_chain_LargeSetInt_v1(kChaseChainSize);
  auto* chaser = chase_head_LargeSetInt_v1;
  int64_t acc = 0;
  int j = 0;

  susp.dismiss();

  while (iters--) {
    auto obj = protocol::parseObject<GetReader<Serializer>>(*buf);
    folly::doNotOptimizeAway(obj);

    chaser = chaser->next;
    int8_t switch_input = chaser->payload & 0xFF;
    switch ((j++) % 3) {
      case 0:
        acc = large_set_int_v1_perturb<0>(acc, switch_input);
        break;
      case 1:
        acc = large_set_int_v1_perturb<1>(acc, switch_input);
        break;
      case 2:
        acc = large_set_int_v1_perturb<2>(acc, switch_input);
        break;
    }
  }

  susp.rehire();
  folly::doNotOptimizeAway(acc);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
