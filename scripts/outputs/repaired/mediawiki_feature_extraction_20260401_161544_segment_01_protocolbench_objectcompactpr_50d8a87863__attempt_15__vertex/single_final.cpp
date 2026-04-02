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

#define ALU_OP(acc, prime, xor_const) acc = acc * prime + (acc ^ xor_const)

#define ALU_OPS_CORE(acc, base_xor_const, prime_offset) \
    ALU_OP(acc, 1000003 + prime_offset, base_xor_const + 1); \
    ALU_OP(acc, 1000033 + prime_offset, base_xor_const + 2); \
    ALU_OP(acc, 1000037 + prime_offset, base_xor_const + 3); \
    ALU_OP(acc, 1000039 + prime_offset, base_xor_const + 4); \
    ALU_OP(acc, 1000081 + prime_offset, base_xor_const + 5); \
    ALU_OP(acc, 1000099 + prime_offset, base_xor_const + 6); \
    ALU_OP(acc, 1000117 + prime_offset, base_xor_const + 7); \
    ALU_OP(acc, 1000121 + prime_offset, base_xor_const + 8); \
    ALU_OP(acc, 1000151 + prime_offset, base_xor_const + 9); \
    ALU_OP(acc, 1000159 + prime_offset, base_xor_const + 10); \
    ALU_OP(acc, 1000177 + prime_offset, base_xor_const + 11); \
    ALU_OP(acc, 1000183 + prime_offset, base_xor_const + 12); \
    ALU_OP(acc, 1000193 + prime_offset, base_xor_const + 13); \
    ALU_OP(acc, 1000197 + prime_offset, base_xor_const + 14); \
    ALU_OP(acc, 1000211 + prime_offset, base_xor_const + 15); \
    ALU_OP(acc, 1000213 + prime_offset, base_xor_const + 16); \
    ALU_OP(acc, 1000231 + prime_offset, base_xor_const + 17); \
    ALU_OP(acc, 1000249 + prime_offset, base_xor_const + 18); \
    ALU_OP(acc, 1000253 + prime_offset, base_xor_const + 19); \
    ALU_OP(acc, 1000271 + prime_offset, base_xor_const + 20); \
    ALU_OP(acc, 1000273 + prime_offset, base_xor_const + 21); \
    ALU_OP(acc, 1000289 + prime_offset, base_xor_const + 22); \
    ALU_OP(acc, 1000291 + prime_offset, base_xor_const + 23); \
    ALU_OP(acc, 1000303 + prime_offset, base_xor_const + 24); \
    ALU_OP(acc, 1000313 + prime_offset, base_xor_const + 25); \
    ALU_OP(acc, 1000333 + prime_offset, base_xor_const + 26); \
    ALU_OP(acc, 1000343 + prime_offset, base_xor_const + 27); \
    ALU_OP(acc, 1000363 + prime_offset, base_xor_const + 28)

#define CASES_16(i, acc, prime_offset) \
    case i + 0: ALU_OPS_CORE(acc, (i+0) << 8, prime_offset); break; \
    case i + 1: ALU_OPS_CORE(acc, (i+1) << 8, prime_offset); break; \
    case i + 2: ALU_OPS_CORE(acc, (i+2) << 8, prime_offset); break; \
    case i + 3: ALU_OPS_CORE(acc, (i+3) << 8, prime_offset); break; \
    case i + 4: ALU_OPS_CORE(acc, (i+4) << 8, prime_offset); break; \
    case i + 5: ALU_OPS_CORE(acc, (i+5) << 8, prime_offset); break; \
    case i + 6: ALU_OPS_CORE(acc, (i+6) << 8, prime_offset); break; \
    case i + 7: ALU_OPS_CORE(acc, (i+7) << 8, prime_offset); break; \
    case i + 8: ALU_OPS_CORE(acc, (i+8) << 8, prime_offset); break; \
    case i + 9: ALU_OPS_CORE(acc, (i+9) << 8, prime_offset); break; \
    case i + 10: ALU_OPS_CORE(acc, (i+10) << 8, prime_offset); break; \
    case i + 11: ALU_OPS_CORE(acc, (i+11) << 8, prime_offset); break; \
    case i + 12: ALU_OPS_CORE(acc, (i+12) << 8, prime_offset); break; \
    case i + 13: ALU_OPS_CORE(acc, (i+13) << 8, prime_offset); break; \
    case i + 14: ALU_OPS_CORE(acc, (i+14) << 8, prime_offset); break; \
    case i + 15: ALU_OPS_CORE(acc, (i+15) << 8, prime_offset); break

#define SWITCH_FUNC(prime_offset) \
    CASES_16(0, acc, prime_offset); \
    CASES_16(16, acc, prime_offset); \
    CASES_16(32, acc, prime_offset); \
    CASES_16(48, acc, prime_offset); \
    CASES_16(64, acc, prime_offset); \
    CASES_16(80, acc, prime_offset); \
    CASES_16(96, acc, prime_offset); \
    CASES_16(112, acc, prime_offset); \
    CASES_16(128, acc, prime_offset); \
    CASES_16(144, acc, prime_offset); \
    CASES_16(160, acc, prime_offset); \
    CASES_16(176, acc, prime_offset); \
    CASES_16(192, acc, prime_offset); \
    CASES_16(208, acc, prime_offset); \
    CASES_16(224, acc, prime_offset); \
    CASES_16(240, acc, prime_offset)

__attribute__((noinline)) void thrash_icache_1(const char* p, long long& acc) {
    switch ((unsigned char)*p) {
        SWITCH_FUNC(0);
    }
}
__attribute__((noinline)) void thrash_icache_2(const char* p, long long& acc) {
    switch ((unsigned char)*p) {
        SWITCH_FUNC(100);
    }
}
__attribute__((noinline)) void thrash_icache_3(const char* p, long long& acc) {
    switch ((unsigned char)*p) {
        SWITCH_FUNC(200);
    }
}

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_ipc_v15, iters) {
    folly::BenchmarkSuspender susp;

    // Setup from reference benchmark to get some data
    auto strct = create<LargeSetInt>();
    folly::IOBufQueue q;
    CompactSerializer::serialize(strct, &q);
    auto buf = q.move();
    buf->coalesce();
    const uint8_t* data_ptr = buf->data();
    size_t data_size = buf->length();

    constexpr size_t kNumNodes = 16 * 1024;
    struct Node {
        Node* next;
        char payload;
    };
    std::vector<Node> nodes(kNumNodes);

    for (size_t i = 0; i < kNumNodes; ++i) {
        nodes[i].payload = data_ptr[i % data_size];
    }

    std::vector<size_t> indices(kNumNodes);
    for(size_t i = 0; i < kNumNodes; ++i) {
      indices[i] = i;
    }

    size_t rand_state = kNumNodes;
    for (size_t i = kNumNodes - 1; i > 0; --i) {
        rand_state = (static_cast<size_t>(1664525) * rand_state + 1013904223);
        size_t j = rand_state % (i + 1);
        auto tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    for (size_t i = 0; i < kNumNodes - 1; ++i) {
        nodes[indices[i]].next = &nodes[indices[i+1]];
    }
    nodes[indices[kNumNodes - 1]].next = &nodes[indices[0]];

    Node* current = &nodes[0];
    long long acc = 0;

    void (*funcs[])(const char*, long long&) = {
      thrash_icache_1, thrash_icache_2, thrash_icache_3
    };

    susp.dismiss();

    for (size_t i = 0; i < iters; ++i) {
        funcs[i % 3](&current->payload, acc);
        current = current->next;
    }

    folly::doNotOptimizeAway(acc);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
