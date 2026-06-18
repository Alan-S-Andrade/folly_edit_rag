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

struct NodeV9 {
  NodeV9* next;
  uint8_t value;
  uint8_t padding[55];
};

constexpr int kOpsPerCaseV9 = 30;

#define V9_ALU_OPS_3(acc, base) \
  (acc) += (base) + 0; \
  (acc) ^= (base) + 1; \
  (acc) *= (base) + 2;

#define V9_MAKE_CASE(i, offset, acc) \
  case i: { \
    constexpr int base = (i * kOpsPerCaseV9) + (offset); \
    V9_ALU_OPS_3(acc, base + 0) \
    V9_ALU_OPS_3(acc, base + 3) \
    V9_ALU_OPS_3(acc, base + 6) \
    V9_ALU_OPS_3(acc, base + 9) \
    V9_ALU_OPS_3(acc, base + 12) \
    V9_ALU_OPS_3(acc, base + 15) \
    V9_ALU_OPS_3(acc, base + 18) \
    V9_ALU_OPS_3(acc, base + 21) \
    V9_ALU_OPS_3(acc, base + 24) \
    V9_ALU_OPS_3(acc, base + 27) \
    break; \
  }

#define V9_GEN_CASES_16(start, offset, acc) \
  V9_MAKE_CASE(start + 0, offset, acc) \
  V9_MAKE_CASE(start + 1, offset, acc) \
  V9_MAKE_CASE(start + 2, offset, acc) \
  V9_MAKE_CASE(start + 3, offset, acc) \
  V9_MAKE_CASE(start + 4, offset, acc) \
  V9_MAKE_CASE(start + 5, offset, acc) \
  V9_MAKE_CASE(start + 6, offset, acc) \
  V9_MAKE_CASE(start + 7, offset, acc) \
  V9_MAKE_CASE(start + 8, offset, acc) \
  V9_MAKE_CASE(start + 9, offset, acc) \
  V9_MAKE_CASE(start + 10, offset, acc) \
  V9_MAKE_CASE(start + 11, offset, acc) \
  V9_MAKE_CASE(start + 12, offset, acc) \
  V9_MAKE_CASE(start + 13, offset, acc) \
  V9_MAKE_CASE(start + 14, offset, acc) \
  V9_MAKE_CASE(start + 15, offset, acc)

#define V9_GEN_ALL_CASES(offset, acc) \
  V9_GEN_CASES_16(0, offset, acc) \
  V9_GEN_CASES_16(16, offset, acc) \
  V9_GEN_CASES_16(32, offset, acc) \
  V9_GEN_CASES_16(48, offset, acc) \
  V9_GEN_CASES_16(64, offset, acc) \
  V9_GEN_CASES_16(80, offset, acc) \
  V9_GEN_CASES_16(96, offset, acc) \
  V9_GEN_CASES_16(112, offset, acc) \
  V9_GEN_CASES_16(128, offset, acc) \
  V9_GEN_CASES_16(144, offset, acc) \
  V9_GEN_CASES_16(160, offset, acc) \
  V9_GEN_CASES_16(176, offset, acc) \
  V9_GEN_CASES_16(192, offset, acc) \
  V9_GEN_CASES_16(208, offset, acc) \
  V9_GEN_CASES_16(224, offset, acc) \
  V9_GEN_CASES_16(240, offset, acc)

__attribute__((noinline)) int64_t v9_switch_0(uint8_t val, int64_t acc) {
  switch (val) { V9_GEN_ALL_CASES(0, acc) }
  return acc;
}
__attribute__((noinline)) int64_t v9_switch_1(uint8_t val, int64_t acc) {
  switch (val) { V9_GEN_ALL_CASES(256 * kOpsPerCaseV9, acc) }
  return acc;
}
__attribute__((noinline)) int64_t v9_switch_2(uint8_t val, int64_t acc) {
  switch (val) { V9_GEN_ALL_CASES(256 * kOpsPerCaseV9 * 2, acc) }
  return acc;
}

#undef V9_ALU_OPS_3
#undef V9_MAKE_CASE
#undef V9_GEN_CASES_16
#undef V9_GEN_ALL_CASES
} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v9, iters) {
  folly::BenchmarkSuspender susp;
  constexpr size_t num_nodes = 4096;
  auto nodes_uptr = std::make_unique<NodeV9[]>(num_nodes);
  NodeV9* nodes = nodes_uptr.get();
  for (size_t i = 0; i < num_nodes; ++i) {
    nodes[i].next = &nodes[(i + 1) % num_nodes];
    nodes[i].value = static_cast<uint8_t>(folly::Random::rand32());
  }
  NodeV9* current = &nodes[0];
  int64_t acc = 0;
  susp.dismiss();
  for (size_t j = 0; j < iters; ++j) {
    uint8_t val = current->value;
    switch (j % 3) {
      case 0:
        acc = v9_switch_0(val, acc);
        break;
      case 1:
        acc = v9_switch_1(val, acc);
        break;
      case 2:
        acc = v9_switch_2(val, acc);
        break;
    }
    current = current->next;
  }
  folly::doNotOptimizeAway(acc);
  susp.rehire();
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
