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

#include <cstdint>
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

#define CASE_BLOCK_V9(i, prime) \
    case i: \
        acc += (i * 37 + 1 * prime); acc ^= (i * 41 + 2 * prime); acc *= (i * 43 + 3 * prime); \
        acc += (i * 47 + 5 * prime); acc ^= (i * 53 + 7 * prime); acc *= (i * 59 + 11 * prime); \
        acc += (i * 61 + 13 * prime); acc ^= (i * 67 + 17 * prime); acc *= (i * 71 + 19 * prime); \
        acc += (i * 73 + 23 * prime); acc ^= (i * 79 + 29 * prime); acc *= (i * 83 + 31 * prime); \
        acc += (i * 89 + 37 * prime); acc ^= (i * 97 + 41 * prime); acc *= (i * 101 + 43 * prime); \
        acc += (i * 103 + 47 * prime); acc ^= (i * 107 + 53 * prime); acc *= (i * 109 + 59 * prime); \
        acc += (i * 113 + 61 * prime); acc ^= (i * 127 + 67 * prime); acc *= (i * 131 + 71 * prime); \
        acc += (i * 137 + 73 * prime); acc ^= (i * 139 + 79 * prime); acc *= (i * 149 + 83 * prime); \
        acc += (i * 151 + 89 * prime); acc ^= (i * 157 + 97 * prime); acc *= (i * 163 + 101 * prime); \
        break;

#define GEN_CASES_16_V9(base, prime) \
    CASE_BLOCK_V9(base + 0, prime) CASE_BLOCK_V9(base + 1, prime) CASE_BLOCK_V9(base + 2, prime) CASE_BLOCK_V9(base + 3, prime) \
    CASE_BLOCK_V9(base + 4, prime) CASE_BLOCK_V9(base + 5, prime) CASE_BLOCK_V9(base + 6, prime) CASE_BLOCK_V9(base + 7, prime) \
    CASE_BLOCK_V9(base + 8, prime) CASE_BLOCK_V9(base + 9, prime) CASE_BLOCK_V9(base + 10, prime) CASE_BLOCK_V9(base + 11, prime) \
    CASE_BLOCK_V9(base + 12, prime) CASE_BLOCK_V9(base + 13, prime) CASE_BLOCK_V9(base + 14, prime) CASE_BLOCK_V9(base + 15, prime)

#define GEN_CASES_256_V9(prime) \
    GEN_CASES_16_V9(0, prime) GEN_CASES_16_V9(16, prime) GEN_CASES_16_V9(32, prime) GEN_CASES_16_V9(48, prime) \
    GEN_CASES_16_V9(64, prime) GEN_CASES_16_V9(80, prime) GEN_CASES_16_V9(96, prime) GEN_CASES_16_V9(112, prime) \
    GEN_CASES_16_V9(128, prime) GEN_CASES_16_V9(144, prime) GEN_CASES_16_V9(160, prime) GEN_CASES_16_V9(176, prime) \
    GEN_CASES_16_V9(192, prime) GEN_CASES_16_V9(208, prime) GEN_CASES_16_V9(224, prime) GEN_CASES_16_V9(240, prime)

namespace {
__attribute__((noinline))
int func1_v9(char payload, int acc) {
    switch (static_cast<uint8_t>(payload)) {
        GEN_CASES_256_V9(257)
    }
    return acc;
}

__attribute__((noinline))
int func2_v9(char payload, int acc) {
    switch (static_cast<uint8_t>(payload)) {
        GEN_CASES_256_V9(263)
    }
    return acc;
}

__attribute__((noinline))
int func3_v9(char payload, int acc) {
    switch (static_cast<uint8_t>(payload)) {
        GEN_CASES_256_V9(269)
    }
    return acc;
}
} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v9, iters) {
    folly::BenchmarkSuspender susp;
    struct Node {
        Node* next;
        char payload[64 - sizeof(Node*)];
    };
    constexpr size_t kNumNodes = 256;
    std::vector<Node> nodes(kNumNodes);
    for(size_t i = 0; i < kNumNodes; ++i) {
        nodes[i].next = &nodes[((i * 17) + 13) % kNumNodes];
        for (size_t j = 0; j < sizeof(nodes[i].payload); ++j) {
            nodes[i].payload[j] = (i * 31 + j * 17) & 0xFF;
        }
    }

    Node* p = &nodes[0];
    volatile int acc = 0;
    susp.dismiss();

    for (size_t i = 0; i < iters; ++i) {
        char payload = p->payload[i % sizeof(p->payload)];
        switch (i % 3) {
            case 0:
                acc = func1_v9(payload, acc);
                break;
            case 1:
                acc = func2_v9(payload, acc);
                break;
            case 2:
                acc = func3_v9(payload, acc);
                break;
        }
        p = p->next;
    }
    folly::doNotOptimizeAway(acc);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
