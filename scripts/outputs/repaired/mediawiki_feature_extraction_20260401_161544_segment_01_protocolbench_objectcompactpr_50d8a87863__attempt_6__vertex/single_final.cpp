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

struct Node {
  Node* next;
  int payload;
};

// Common macros for generating switch cases with lots of ALU ops.
#define GEN_CASE(i, base_const1, base_const2) \
    case i: \
        acc = (acc * (base_const1 + 4*i)) + (base_const2 + i); \
        acc = (acc ^ (base_const1 + 20 + 4*i)) - (base_const2 + 20 + i); \
        acc = (acc * (base_const1 + 40 + 4*i)) + (base_const2 + 40 + i); \
        acc = (acc ^ (base_const1 + 60 + 4*i)) - (base_const2 + 60 + i); \
        acc = (acc * (base_const1 + 80 + 4*i)) + (base_const2 + 80 + i); \
        acc = (acc ^ (base_const1 + 100 + 4*i)) - (base_const2 + 100 + i); \
        acc = (acc * (base_const1 + 120 + 4*i)) + (base_const2 + 120 + i); \
        acc = (acc ^ (base_const1 + 140 + 4*i)) - (base_const2 + 140 + i); \
        acc = (acc * (base_const1 + 160 + 4*i)) + (base_const2 + 160 + i); \
        acc = (acc ^ (base_const1 + 180 + 4*i)) - (base_const2 + 180 + i); \
        acc = (acc * (base_const1 + 200 + 4*i)) + (base_const2 + 200 + i); \
        acc = (acc ^ (base_const1 + 220 + 4*i)) - (base_const2 + 220 + i); \
        acc = (acc * (base_const1 + 240 + 4*i)) + (base_const2 + 240 + i); \
        acc = (acc ^ (base_const1 + 260 + 4*i)) - (base_const2 + 260 + i); \
        acc = (acc * (base_const1 + 4*i)) + (base_const2 + i + 1); \
        acc = (acc ^ (base_const1 + 20 + 4*i)) - (base_const2 + 20 + i + 1); \
        acc = (acc * (base_const1 + 40 + 4*i)) + (base_const2 + 40 + i + 1); \
        acc = (acc ^ (base_const1 + 60 + 4*i)) - (base_const2 + 60 + i + 1); \
        acc = (acc * (base_const1 + 80 + 4*i)) + (base_const2 + 80 + i + 1); \
        acc = (acc ^ (base_const1 + 100 + 4*i)) - (base_const2 + 100 + i + 1); \
        acc = (acc * (base_const1 + 120 + 4*i)) + (base_const2 + 120 + i + 1); \
        acc = (acc ^ (base_const1 + 140 + 4*i)) - (base_const2 + 140 + i + 1); \
        acc = (acc * (base_const1 + 160 + 4*i)) + (base_const2 + 160 + i + 1); \
        acc = (acc ^ (base_const1 + 180 + 4*i)) - (base_const2 + 180 + i + 1); \
        acc = (acc * (base_const1 + 200 + 4*i)) + (base_const2 + 200 + i + 1); \
        acc = (acc ^ (base_const1 + 220 + 4*i)) - (base_const2 + 220 + i + 1); \
        acc = (acc * (base_const1 + 240 + 4*i)) + (base_const2 + 240 + i + 1); \
        acc = (acc ^ (base_const1 + 260 + 4*i)) - (base_const2 + 260 + i + 1); \
        break;

#define GEN_CASES_16(i, base_const1, base_const2) \
    GEN_CASE(i+0, base_const1, base_const2) GEN_CASE(i+1, base_const1, base_const2) \
    GEN_CASE(i+2, base_const1, base_const2) GEN_CASE(i+3, base_const1, base_const2) \
    GEN_CASE(i+4, base_const1, base_const2) GEN_CASE(i+5, base_const1, base_const2) \
    GEN_CASE(i+6, base_const1, base_const2) GEN_CASE(i+7, base_const1, base_const2) \
    GEN_CASE(i+8, base_const1, base_const2) GEN_CASE(i+9, base_const1, base_const2) \
    GEN_CASE(i+10, base_const1, base_const2) GEN_CASE(i+11, base_const1, base_const2) \
    GEN_CASE(i+12, base_const1, base_const2) GEN_CASE(i+13, base_const1, base_const2) \
    GEN_CASE(i+14, base_const1, base_const2) GEN_CASE(i+15, base_const1, base_const2)

#define GEN_CASES_256(base_const1, base_const2) \
    GEN_CASES_16(0, base_const1, base_const2) GEN_CASES_16(16, base_const1, base_const2) \
    GEN_CASES_16(32, base_const1, base_const2) GEN_CASES_16(48, base_const1, base_const2) \
    GEN_CASES_16(64, base_const1, base_const2) GEN_CASES_16(80, base_const1, base_const2) \
    GEN_CASES_16(96, base_const1, base_const2) GEN_CASES_16(112, base_const1, base_const2) \
    GEN_CASES_16(128, base_const1, base_const2) GEN_CASES_16(144, base_const1, base_const2) \
    GEN_CASES_16(160, base_const1, base_const2) GEN_CASES_16(176, base_const1, base_const2) \
    GEN_CASES_16(192, base_const1, base_const2) GEN_CASES_16(208, base_const1, base_const2) \
    GEN_CASES_16(224, base_const1, base_const2) GEN_CASES_16(240, base_const1, base_const2)

__attribute__((noinline)) int churn_code_1(int acc, int index) {
  switch (index) {
    GEN_CASES_256(101, 11)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) int churn_code_2(int acc, int index) {
  switch (index) {
    GEN_CASES_256(201, 211)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) int churn_code_3(int acc, int index) {
  switch (index) {
    GEN_CASES_256(301, 311)
    default:
      break;
  }
  return acc;
}

#undef GEN_CASE
#undef GEN_CASES_16
#undef GEN_CASES_256

} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v6, iters) {
    folly::BenchmarkSuspender susp;

    constexpr size_t kNumNodes = 4096;
    std::vector<Node> nodes(kNumNodes);
    // Use a PRNG to initialize next pointers to avoid trivial-to-predict
    // access patterns.
    uint32_t seed = 12345;
    auto lcg = [&]() {
      seed = (1103515245 * seed + 12345) & 0x7fffffff;
      return seed;
    };
    for (size_t i = 0; i < kNumNodes; ++i) {
        nodes[i].next = &nodes[lcg() % kNumNodes];
        nodes[i].payload = lcg();
    }
    Node* current = &nodes[0];
    int acc = 1;

    susp.dismiss();

    for (size_t i = 0; i < iters; ++i) {
        int index = current->payload & 0xFF;
        switch (i % 3) {
            case 0:
                acc = churn_code_1(acc, index);
                break;
            case 1:
                acc = churn_code_2(acc, index);
                break;
            case 2:
                acc = churn_code_3(acc, index);
                break;
        }
        current = current->next;
    }

    folly::doNotOptimizeAway(acc);
    folly::doNotOptimizeAway(current);
    susp.rehire();
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
