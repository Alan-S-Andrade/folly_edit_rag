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

#include <algorithm>
#include <cstdint>
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

// clang-format on

// clang-format off
namespace {

#define ALU_OP_SEQUENCE_1(acc, i, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28) \
    acc ^= (i * 31 + c1);  acc += (i * 31 + c2);  acc *= (i * 31 + c3);  acc ^= (i * 31 + c4);  \
    acc += (i * 31 + c5);  acc *= (i * 31 + c6);  acc ^= (i * 31 + c7);  acc += (i * 31 + c8);  \
    acc *= (i * 31 + c9);  acc ^= (i * 31 + c10); acc += (i * 31 + c11); acc *= (i * 31 + c12); \
    acc ^= (i * 31 + c13); acc += (i * 31 + c14); acc *= (i * 31 + c15); acc ^= (i * 31 + c16); \
    acc += (i * 31 + c17); acc *= (i * 31 + c18); acc ^= (i * 31 + c19); acc += (i * 31 + c20); \
    acc *= (i * 31 + c21); acc ^= (i * 31 + c22); acc += (i * 31 + c23); acc *= (i * 31 + c24); \
    acc ^= (i * 31 + c25); acc += (i * 31 + c26); acc *= (i * 31 + c27); acc ^= (i * 31 + c28);

#define ALU_OP_SEQUENCE_2(acc, i, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28) \
    acc += (i * 37 + c1);  acc ^= (i * 37 + c2);  acc *= (i * 37 + c3);  acc += (i * 37 + c4);  \
    acc *= (i * 37 + c5);  acc ^= (i * 37 + c6);  acc += (i * 37 + c7);  acc *= (i * 37 + c8);  \
    acc ^= (i * 37 + c9);  acc += (i * 37 + c10); acc *= (i * 37 + c11); acc += (i * 37 + c12); \
    acc *= (i * 37 + c13); acc ^= (i * 37 + c14); acc += (i * 37 + c15); acc *= (i * 37 + c16); \
    acc ^= (i * 37 + c17); acc += (i * 37 + c18); acc *= (i * 37 + c19); acc += (i * 37 + c20); \
    acc *= (i * 37 + c21); acc ^= (i * 37 + c22); acc += (i * 37 + c23); acc *= (i * 37 + c24); \
    acc ^= (i * 37 + c25); acc += (i * 37 + c26); acc *= (i * 37 + c27); acc += (i * 37 + c28);

#define ALU_OP_SEQUENCE_3(acc, i, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28) \
    acc *= (i * 41 + c1);  acc += (i * 41 + c2);  acc ^= (i * 41 + c3);  acc *= (i * 41 + c4);  \
    acc += (i * 41 + c5);  acc ^= (i * 41 + c6);  acc *= (i * 41 + c7);  acc += (i * 41 + c8);  \
    acc ^= (i * 41 + c9);  acc *= (i * 41 + c10); acc += (i * 41 + c11); acc ^= (i * 41 + c12); \
    acc *= (i * 41 + c13); acc += (i * 41 + c14); acc ^= (i * 41 + c15); acc *= (i * 41 + c16); \
    acc += (i * 41 + c17); acc ^= (i * 41 + c18); acc *= (i * 41 + c19); acc += (i * 41 + c20); \
    acc ^= (i * 41 + c21); acc *= (i * 41 + c22); acc += (i * 41 + c23); acc ^= (i * 41 + c24); \
    acc *= (i * 41 + c25); acc += (i * 41 + c26); acc ^= (i * 41 + c27); acc *= (i * 41 + c28);

#define SWITCH_CASE(i, acc, ALU_MACRO) \
    case i: \
        ALU_MACRO(acc, i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28); \
        break;

#define SWITCH_CHUNK_8(start_idx, acc, ALU_MACRO) \
    SWITCH_CASE(start_idx + 0, acc, ALU_MACRO) \
    SWITCH_CASE(start_idx + 1, acc, ALU_MACRO) \
    SWITCH_CASE(start_idx + 2, acc, ALU_MACRO) \
    SWITCH_CASE(start_idx + 3, acc, ALU_MACRO) \
    SWITCH_CASE(start_idx + 4, acc, ALU_MACRO) \
    SWITCH_CASE(start_idx + 5, acc, ALU_MACRO) \
    SWITCH_CASE(start_idx + 6, acc, ALU_MACRO) \
    SWITCH_CASE(start_idx + 7, acc, ALU_MACRO)

#define SWITCH_CHUNK_32(start_idx, acc, ALU_MACRO) \
    SWITCH_CHUNK_8(start_idx + 0, acc, ALU_MACRO) \
    SWITCH_CHUNK_8(start_idx + 8, acc, ALU_MACRO) \
    SWITCH_CHUNK_8(start_idx + 16, acc, ALU_MACRO) \
    SWITCH_CHUNK_8(start_idx + 24, acc, ALU_MACRO)

#define SWITCH_BLOCK_256(acc, idx, ALU_MACRO) \
    switch (idx) { \
        SWITCH_CHUNK_32(0, acc, ALU_MACRO) \
        SWITCH_CHUNK_32(32, acc, ALU_MACRO) \
        SWITCH_CHUNK_32(64, acc, ALU_MACRO) \
        SWITCH_CHUNK_32(96, acc, ALU_MACRO) \
        SWITCH_CHUNK_32(128, acc, ALU_MACRO) \
        SWITCH_CHUNK_32(160, acc, ALU_MACRO) \
        SWITCH_CHUNK_32(192, acc, ALU_MACRO) \
        SWITCH_CHUNK_32(224, acc, ALU_MACRO) \
        default: break; \
    }

__attribute__((noinline))
int64_t icache_func1(int64_t acc, uint8_t idx) {
    SWITCH_BLOCK_256(acc, idx, ALU_OP_SEQUENCE_1)
    return acc;
}
__attribute__((noinline))
int64_t icache_func2(int64_t acc, uint8_t idx) {
    SWITCH_BLOCK_256(acc, idx, ALU_OP_SEQUENCE_2)
    return acc;
}
__attribute__((noinline))
int64_t icache_func3(int64_t acc, uint8_t idx) {
    SWITCH_BLOCK_256(acc, idx, ALU_OP_SEQUENCE_3)
    return acc;
}

#undef ALU_OP_SEQUENCE_1
#undef ALU_OP_SEQUENCE_2
#undef ALU_OP_SEQUENCE_3
#undef SWITCH_CASE
#undef SWITCH_CHUNK_8
#undef SWITCH_CHUNK_32
#undef SWITCH_BLOCK_256

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_icache_v1, iters) {
    folly::BenchmarkSuspender susp;

    constexpr size_t chase_size = 64 * 1024;
    struct Node {
        Node* next;
        uint64_t payload;
    };
    std::vector<Node> nodes(chase_size);
    std::vector<size_t> p(chase_size);
    std::iota(p.begin(), p.end(), 0);

    std::mt19937 g(12345);
    std::shuffle(p.begin(), p.end(), g);

    for(size_t i = 0; i < chase_size; ++i) {
        nodes[p[i]].next = &nodes[p[(i + 1) % chase_size]];
        nodes[p[i]].payload = g();
    }

    Node* current = &nodes[p[0]];
    int64_t acc = 0;

    using Serializer = CompactSerializer;
    using Struct = LargeSetInt;
    auto strct = create<Struct>();
    folly::IOBufQueue q;
    Serializer::serialize(strct, &q);
    auto buf = q.move();
    buf->coalesce();

    susp.dismiss();

    for (size_t i = 0; i < iters; ++i) {
        current = current->next;
        uint8_t switch_idx = current->payload & 0xFF;

        switch (i % 3) {
            case 0: acc = icache_func1(acc, switch_idx); break;
            case 1: acc = icache_func2(acc, switch_idx); break;
            case 2: acc = icache_func3(acc, switch_idx); break;
        }
    }

    susp.rehire();
    folly::doNotOptimizeAway(acc);
}

} // anonymous namespace
// clang-format on

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
