/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may in compliance with the License.
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

#include <memory>
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

#define F_ALU_GROUP(acc, c) \
  acc += c;                 \
  acc ^= (c + 1);           \
  acc *= (c - 1);           \
  acc -= (c + 2);           \
  acc |= (c + 3);

#define CASE_BLOCK(i, base, acc)   \
  case i:                          \
    F_ALU_GROUP(acc, base + 0 * 5);  \
    F_ALU_GROUP(acc, base + 1 * 5);  \
    F_ALU_GROUP(acc, base + 2 * 5);  \
    F_ALU_GROUP(acc, base + 3 * 5);  \
    F_ALU_GROUP(acc, base + 4 * 5);  \
    F_ALU_GROUP(acc, base + 5 * 5);  \
    break;

#define CASES_16(i_start, base_start, acc) \
  CASE_BLOCK(i_start + 0, base_start + 0 * 30, acc)   \
  CASE_BLOCK(i_start + 1, base_start + 1 * 30, acc)   \
  CASE_BLOCK(i_start + 2, base_start + 2 * 30, acc)   \
  CASE_BLOCK(i_start + 3, base_start + 3 * 30, acc)   \
  CASE_BLOCK(i_start + 4, base_start + 4 * 30, acc)   \
  CASE_BLOCK(i_start + 5, base_start + 5 * 30, acc)   \
  CASE_BLOCK(i_start + 6, base_start + 6 * 30, acc)   \
  CASE_BLOCK(i_start + 7, base_start + 7 * 30, acc)   \
  CASE_BLOCK(i_start + 8, base_start + 8 * 30, acc)   \
  CASE_BLOCK(i_start + 9, base_start + 9 * 30, acc)   \
  CASE_BLOCK(i_start + 10, base_start + 10 * 30, acc) \
  CASE_BLOCK(i_start + 11, base_start + 11 * 30, acc) \
  CASE_BLOCK(i_start + 12, base_start + 12 * 30, acc) \
  CASE_BLOCK(i_start + 13, base_start + 13 * 30, acc) \
  CASE_BLOCK(i_start + 14, base_start + 14 * 30, acc) \
  CASE_BLOCK(i_start + 15, base_start + 15 * 30, acc)

#define ALL_CASES(base, acc) \
  CASES_16(0, base, acc) \
  CASES_16(16, base + 16 * 30, acc) \
  CASES_16(32, base + 32 * 30, acc) \
  CASES_16(48, base + 48 * 30, acc) \
  CASES_16(64, base + 64 * 30, acc) \
  CASES_16(80, base + 80 * 30, acc) \
  CASES_16(96, base + 96 * 30, acc) \
  CASES_16(112, base + 112 * 30, acc) \
  CASES_16(128, base + 128 * 30, acc) \
  CASES_16(144, base + 144 * 30, acc) \
  CASES_16(160, base + 160 * 30, acc) \
  CASES_16(176, base + 176 * 30, acc) \
  CASES_16(192, base + 192 * 30, acc) \
  CASES_16(208, base + 208 * 30, acc) \
  CASES_16(224, base + 224 * 30, acc) \
  CASES_16(240, base + 240 * 30, acc)

__attribute__((noinline)) int large_switch_func_v3_1(volatile const char* data) {
    int acc = 0;
    switch (*data & 0xff) {
        ALL_CASES(1000, acc);
    }
    return acc;
}
__attribute__((noinline)) int large_switch_func_v3_2(volatile const char* data) {
    int acc = 0;
    switch (*data & 0xff) {
        ALL_CASES(1000 + 256 * 30 * 2, acc);
    }
    return acc;
}
__attribute__((noinline)) int large_switch_func_v3_3(volatile const char* data) {
    int acc = 0;
    switch (*data & 0xff) {
        ALL_CASES(1000 + 256 * 30 * 4, acc);
    }
    return acc;
}

#undef F_ALU_GROUP
#undef CASE_BLOCK
#undef CASES_16
#undef ALL_CASES

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v3, iters) {
    folly::BenchmarkSuspender susp;

    struct Node {
        Node* next;
        char payload[64];
    };

    constexpr size_t num_nodes = 256;
    std::vector<Node> nodes(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
        nodes[i].next = &nodes[(i + 1) % num_nodes];
        for (int j = 0; j < 64; ++j) {
            nodes[i].payload[j] = (i * 13 + j * 7) & 0xff;
        }
    }

    volatile Node* p = &nodes[0];
    long acc = 0;

    susp.dismiss();

    for (size_t i = 0; i < iters; ++i) {
        p = p->next;
        switch (i % 3) {
            case 0:
                acc += large_switch_func_v3_1(p->payload);
                break;
            case 1:
                acc += large_switch_func_v3_2(p->payload);
                break;
            case 2:
                acc += large_switch_func_v3_3(p->payload);
                break;
        }
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
