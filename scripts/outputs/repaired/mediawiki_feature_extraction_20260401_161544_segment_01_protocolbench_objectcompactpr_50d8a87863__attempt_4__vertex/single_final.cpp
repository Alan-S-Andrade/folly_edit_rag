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

struct ICacheBenchNode {
  ICacheBenchNode* next;
  uint64_t payload;
};

#define ALU_OPS_28(acc, i, op1, op2, salt) \
    acc = (acc op1 (i * 28 * 2 + 0 + salt)) op2 (i * 28 * 2 + 1 + salt); \
    acc = (acc op1 (i * 28 * 2 + 2 + salt)) op2 (i * 28 * 2 + 3 + salt); \
    acc = (acc op1 (i * 28 * 2 + 4 + salt)) op2 (i * 28 * 2 + 5 + salt); \
    acc = (acc op1 (i * 28 * 2 + 6 + salt)) op2 (i * 28 * 2 + 7 + salt); \
    acc = (acc op1 (i * 28 * 2 + 8 + salt)) op2 (i * 28 * 2 + 9 + salt); \
    acc = (acc op1 (i * 28 * 2 + 10 + salt)) op2 (i * 28 * 2 + 11 + salt); \
    acc = (acc op1 (i * 28 * 2 + 12 + salt)) op2 (i * 28 * 2 + 13 + salt); \
    acc = (acc op1 (i * 28 * 2 + 14 + salt)) op2 (i * 28 * 2 + 15 + salt); \
    acc = (acc op1 (i * 28 * 2 + 16 + salt)) op2 (i * 28 * 2 + 17 + salt); \
    acc = (acc op1 (i * 28 * 2 + 18 + salt)) op2 (i * 28 * 2 + 19 + salt); \
    acc = (acc op1 (i * 28 * 2 + 20 + salt)) op2 (i * 28 * 2 + 21 + salt); \
    acc = (acc op1 (i * 28 * 2 + 22 + salt)) op2 (i * 28 * 2 + 23 + salt); \
    acc = (acc op1 (i * 28 * 2 + 24 + salt)) op2 (i * 28 * 2 + 25 + salt); \
    acc = (acc op1 (i * 28 * 2 + 26 + salt)) op2 (i * 28 * 2 + 27 + salt);


#define CASE_ITEM(i, op1, op2, salt) case i: ALU_OPS_28(acc, i, op1, op2, salt); break;

#define CASES_16(start, op1, op2, salt) \
    CASE_ITEM(start + 0, op1, op2, salt) \
    CASE_ITEM(start + 1, op1, op2, salt) \
    CASE_ITEM(start + 2, op1, op2, salt) \
    CASE_ITEM(start + 3, op1, op2, salt) \
    CASE_ITEM(start + 4, op1, op2, salt) \
    CASE_ITEM(start + 5, op1, op2, salt) \
    CASE_ITEM(start + 6, op1, op2, salt) \
    CASE_ITEM(start + 7, op1, op2, salt) \
    CASE_ITEM(start + 8, op1, op2, salt) \
    CASE_ITEM(start + 9, op1, op2, salt) \
    CASE_ITEM(start + 10, op1, op2, salt) \
    CASE_ITEM(start + 11, op1, op2, salt) \
    CASE_ITEM(start + 12, op1, op2, salt) \
    CASE_ITEM(start + 13, op1, op2, salt) \
    CASE_ITEM(start + 14, op1, op2, salt) \
    CASE_ITEM(start + 15, op1, op2, salt)

#define DEFINE_SWITCH_FUNC(func_name, op1, op2, salt) \
__attribute__((noinline)) \
uint64_t func_name(uint64_t acc, uint8_t val) { \
    switch (val) { \
    CASES_16(0, op1, op2, salt) \
    CASES_16(16, op1, op2, salt) \
    CASES_16(32, op1, op2, salt) \
    CASES_16(48, op1, op2, salt) \
    CASES_16(64, op1, op2, salt) \
    CASES_16(80, op1, op2, salt) \
    CASES_16(96, op1, op2, salt) \
    CASES_16(112, op1, op2, salt) \
    CASES_16(128, op1, op2, salt) \
    CASES_16(144, op1, op2, salt) \
    CASES_16(160, op1, op2, salt) \
    CASES_16(176, op1, op2, salt) \
    CASES_16(192, op1, op2, salt) \
    CASES_16(208, op1, op2, salt) \
    CASES_16(224, op1, op2, salt) \
    CASES_16(240, op1, op2, salt) \
    default: break; \
    } \
    return acc; \
}

DEFINE_SWITCH_FUNC(icache_func_1, +, ^, 0)
DEFINE_SWITCH_FUNC(icache_func_2, *, +, 20000)
DEFINE_SWITCH_FUNC(icache_func_3, ^, *, 40000)

} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_icache_v1, iters) {
    folly::BenchmarkSuspender susp;

    constexpr size_t num_nodes = 256 * 1024; // 4MB of nodes
    std::vector<ICacheBenchNode> nodes(num_nodes);
    std::vector<uint32_t> indices(num_nodes);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    for (size_t i = 0; i < num_nodes; ++i) {
        nodes[indices[i]].next = &nodes[indices[(i + 1) % num_nodes]];
        nodes[indices[i]].payload = g();
    }

    uint64_t (*funcs[])(uint64_t, uint8_t) = {
      icache_func_1, icache_func_2, icache_func_3
    };

    ICacheBenchNode* p = &nodes[0];
    uint64_t acc = 0;

    susp.dismiss();

    size_t j = 0;
    while (iters--) {
        p = p->next;
        acc = funcs[j++ % 3](acc, p->payload & 0xFF);
    }

    folly::doNotOptimizeAway(acc);
    folly::doNotOptimizeAway(p);
    susp.rehire();
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
