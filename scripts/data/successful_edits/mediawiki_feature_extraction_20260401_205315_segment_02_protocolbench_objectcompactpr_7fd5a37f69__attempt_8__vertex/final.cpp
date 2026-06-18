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
#include <folly/Random.h>
#include <folly/init/Init.h>
#include <folly/portability/GFlags.h>

#include <algorithm>
#include <numeric>
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

struct Node_L1i_v1 {
  Node_L1i_v1* next;
  uint64_t payload;
};

template <int F>
__attribute__((noinline)) uint64_t do_work_L1i_v1(uint64_t acc, uint8_t idx) {
  constexpr int offset = F * 256 * 28;
  switch (idx) {
#define CASE_BLOCK_L1i_v1(i)                                                   \
  case i:                                                                      \
    acc += (i * 28 + 1 + offset);                                              \
    acc ^= (i * 28 + 2 + offset);                                              \
    acc *= (i * 28 + 3 + offset);                                              \
    acc += (i * 28 + 4 + offset);                                              \
    acc ^= (i * 28 + 5 + offset);                                              \
    acc *= (i * 28 + 6 + offset);                                              \
    acc += (i * 28 + 7 + offset);                                              \
    acc ^= (i * 28 + 8 + offset);                                              \
    acc *= (i * 28 + 9 + offset);                                              \
    acc += (i * 28 + 10 + offset);                                             \
    acc ^= (i * 28 + 11 + offset);                                             \
    acc *= (i * 28 + 12 + offset);                                             \
    acc += (i * 28 + 13 + offset);                                             \
    acc ^= (i * 28 + 14 + offset);                                             \
    acc *= (i * 28 + 15 + offset);                                             \
    acc += (i * 28 + 16 + offset);                                             \
    acc ^= (i * 28 + 17 + offset);                                             \
    acc *= (i * 28 + 18 + offset);                                             \
    acc += (i * 28 + 19 + offset);                                             \
    acc ^= (i * 28 + 20 + offset);                                             \
    acc *= (i * 28 + 21 + offset);                                             \
    acc += (i * 28 + 22 + offset);                                             \
    acc ^= (i * 28 + 23 + offset);                                             \
    acc *= (i * 28 + 24 + offset);                                             \
    acc += (i * 28 + 25 + offset);                                             \
    acc ^= (i * 28 + 26 + offset);                                             \
    acc *= (i * 28 + 27 + offset);                                             \
    acc += (i * 28 + 28 + offset);                                             \
    break;

#define CASES_16_L1i_v1(i)                                                     \
  CASE_BLOCK_L1i_v1(i + 0)                                                     \
  CASE_BLOCK_L1i_v1(i + 1)                                                     \
  CASE_BLOCK_L1i_v1(i + 2)                                                     \
  CASE_BLOCK_L1i_v1(i + 3)                                                     \
  CASE_BLOCK_L1i_v1(i + 4)                                                     \
  CASE_BLOCK_L1i_v1(i + 5)                                                     \
  CASE_BLOCK_L1i_v1(i + 6)                                                     \
  CASE_BLOCK_L1i_v1(i + 7)                                                     \
  CASE_BLOCK_L1i_v1(i + 8)                                                     \
  CASE_BLOCK_L1i_v1(i + 9)                                                     \
  CASE_BLOCK_L1i_v1(i + 10)                                                    \
  CASE_BLOCK_L1i_v1(i + 11)                                                    \
  CASE_BLOCK_L1i_v1(i + 12)                                                    \
  CASE_BLOCK_L1i_v1(i + 13)                                                    \
  CASE_BLOCK_L1i_v1(i + 14)                                                    \
  CASE_BLOCK_L1i_v1(i + 15)

    CASES_16_L1i_v1(0)
    CASES_16_L1i_v1(16)
    CASES_16_L1i_v1(32)
    CASES_16_L1i_v1(48)
    CASES_16_L1i_v1(64)
    CASES_16_L1i_v1(80)
    CASES_16_L1i_v1(96)
    CASES_16_L1i_v1(112)
    CASES_16_L1i_v1(128)
    CASES_16_L1i_v1(144)
    CASES_16_L1i_v1(160)
    CASES_16_L1i_v1(176)
    CASES_16_L1i_v1(192)
    CASES_16_L1i_v1(208)
    CASES_16_L1i_v1(224)
    CASES_16_L1i_v1(240)

#undef CASE_BLOCK_L1i_v1
#undef CASES_16_L1i_v1
  }
  return acc;
}

} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_L1i_v1, iters) {
  folly::BenchmarkSuspender susp;

  constexpr size_t num_nodes = 65536;
  static std::vector<Node_L1i_v1> nodes(num_nodes);
  static bool initialized = false;

  if (!initialized) {
    std::vector<int> indices(num_nodes);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(1);
    std::shuffle(indices.begin(), indices.end(), g);

    for (size_t i = 0; i < num_nodes; ++i) {
      nodes[indices[i]].next = &nodes[indices[(i + 1) % num_nodes]];
      nodes[indices[i]].payload = folly::Random::rand64(g);
    }
    initialized = true;
  }

  Node_L1i_v1* current = &nodes[0];
  uint64_t acc = 0;

  susp.dismiss();

  for (size_t j = 0; j < iters; ++j) {
    uint8_t idx = current->payload & 0xFF;
    switch (j % 3) {
      case 0:
        acc = do_work_L1i_v1<0>(acc, idx);
        break;
      case 1:
        acc = do_work_L1i_v1<1>(acc, idx);
        break;
      case 2:
        acc = do_work_L1i_v1<2>(acc, idx);
        break;
    }
    current = current->next;
  }

  folly::doNotOptimizeAway(acc);
  folly::doNotOptimizeAway(current);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
