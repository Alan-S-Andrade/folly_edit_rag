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

namespace {
__attribute__((noinline)) int icache_churn_0(int x, int y) {
  int acc = y;
  switch (x) {
#define CASE_BLOCK(i, p1, p2, p3, p4)                                     \
  case i:                                                                \
    acc = (acc * (i * 3 + p1)) ^ (acc + (i * 5 + p2));                    \
    acc = (acc * (i * 7 + p1 + 2)) + (acc ^ (i * 11 + p2 + 4));           \
    acc = (acc * (i * 13 + p1 + 6)) ^ (acc + (i * 17 + p2 + 6));          \
    acc = (acc * (i * 19 + p1 + 8)) + (acc ^ (i * 23 + p3));              \
    acc = (acc * (i * 29 + p1 + 10)) ^ (acc + (i * 31 + p3 + 2));         \
    acc = (acc * (i * 37 + p1 + 12)) + (acc ^ (i * 41 + p4));             \
    acc = (acc * (i * 43 + p1 + 14)) ^ (acc + (i * 47 + p4 + 4));         \
    break;
#define C1(i, ...) CASE_BLOCK(i, __VA_ARGS__)
#define C2(i, ...) \
  C1(i, __VA_ARGS__) \
  C1(i + 1, __VA_ARGS__)
#define C4(i, ...) \
  C2(i, __VA_ARGS__) \
  C2(i + 2, __VA_ARGS__)
#define C8(i, ...) \
  C4(i, __VA_ARGS__) \
  C4(i + 4, __VA_ARGS__)
#define C16(i, ...) \
  C8(i, __VA_ARGS__) \
  C8(i + 8, __VA_ARGS__)
#define C32(i, ...) \
  C16(i, __VA_ARGS__) \
  C16(i + 16, __VA_ARGS__)
#define C64(i, ...) \
  C32(i, __VA_ARGS__) \
  C32(i + 32, __VA_ARGS__)
#define C128(i, ...) \
  C64(i, __VA_ARGS__) \
  C64(i + 64, __VA_ARGS__)
#define C256(i, ...) \
  C128(i, __VA_ARGS__) \
  C128(i + 128, __VA_ARGS__)
    C256(0, 101, 13, 19, 37)
#undef CASE_BLOCK
#undef C1
#undef C2
#undef C4
#undef C8
#undef C16
#undef C32
#undef C64
#undef C128
#undef C256
  }
  return acc;
}

__attribute__((noinline)) int icache_churn_1(int x, int y) {
  int acc = y;
  switch (x) {
#define CASE_BLOCK(i, p1, p2, p3, p4)                                     \
  case i:                                                                \
    acc = (acc * (i * 3 + p1)) ^ (acc + (i * 5 + p2));                    \
    acc = (acc * (i * 7 + p1 + 2)) + (acc ^ (i * 11 + p2 + 4));           \
    acc = (acc * (i * 13 + p1 + 6)) ^ (acc + (i * 17 + p2 + 6));          \
    acc = (acc * (i * 19 + p1 + 8)) + (acc ^ (i * 23 + p3));              \
    acc = (acc * (i * 29 + p1 + 10)) ^ (acc + (i * 31 + p3 + 2));         \
    acc = (acc * (i * 37 + p1 + 12)) + (acc ^ (i * 41 + p4));             \
    acc = (acc * (i * 43 + p1 + 14)) ^ (acc + (i * 47 + p4 + 4));         \
    break;
#define C1(i, ...) CASE_BLOCK(i, __VA_ARGS__)
#define C2(i, ...) \
  C1(i, __VA_ARGS__) \
  C1(i + 1, __VA_ARGS__)
#define C4(i, ...) \
  C2(i, __VA_ARGS__) \
  C2(i + 2, __VA_ARGS__)
#define C8(i, ...) \
  C4(i, __VA_ARGS__) \
  C4(i + 4, __VA_ARGS__)
#define C16(i, ...) \
  C8(i, __VA_ARGS__) \
  C8(i + 8, __VA_ARGS__)
#define C32(i, ...) \
  C16(i, __VA_ARGS__) \
  C16(i + 16, __VA_ARGS__)
#define C64(i, ...) \
  C32(i, __VA_ARGS__) \
  C32(i + 32, __VA_ARGS__)
#define C128(i, ...) \
  C64(i, __VA_ARGS__) \
  C64(i + 64, __VA_ARGS__)
#define C256(i, ...) \
  C128(i, __VA_ARGS__) \
  C128(i + 128, __VA_ARGS__)
    C256(0, 137, 43, 59, 67)
#undef CASE_BLOCK
#undef C1
#undef C2
#undef C4
#undef C8
#undef C16
#undef C32
#undef C64
#undef C128
#undef C256
  }
  return acc;
}

__attribute__((noinline)) int icache_churn_2(int x, int y) {
  int acc = y;
  switch (x) {
#define CASE_BLOCK(i, p1, p2, p3, p4)                                     \
  case i:                                                                \
    acc = (acc * (i * 3 + p1)) ^ (acc + (i * 5 + p2));                    \
    acc = (acc * (i * 7 + p1 + 2)) + (acc ^ (i * 11 + p2 + 4));           \
    acc = (acc * (i * 13 + p1 + 6)) ^ (acc + (i * 17 + p2 + 6));          \
    acc = (acc * (i * 19 + p1 + 8)) + (acc ^ (i * 23 + p3));              \
    acc = (acc * (i * 29 + p1 + 10)) ^ (acc + (i * 31 + p3 + 2));         \
    acc = (acc * (i * 37 + p1 + 12)) + (acc ^ (i * 41 + p4));             \
    acc = (acc * (i * 43 + p1 + 14)) ^ (acc + (i * 47 + p4 + 4));         \
    break;
#define C1(i, ...) CASE_BLOCK(i, __VA_ARGS__)
#define C2(i, ...) \
  C1(i, __VA_ARGS__) \
  C1(i + 1, __VA_ARGS__)
#define C4(i, ...) \
  C2(i, __VA_ARGS__) \
  C2(i + 2, __VA_ARGS__)
#define C8(i, ...) \
  C4(i, __VA_ARGS__) \
  C4(i + 4, __VA_ARGS__)
#define C16(i, ...) \
  C8(i, __VA_ARGS__) \
  C8(i + 8, __VA_ARGS__)
#define C32(i, ...) \
  C16(i, __VA_ARGS__) \
  C16(i + 16, __VA_ARGS__)
#define C64(i, ...) \
  C32(i, __VA_ARGS__) \
  C32(i + 32, __VA_ARGS__)
#define C128(i, ...) \
  C64(i, __VA_ARGS__) \
  C64(i + 64, __VA_ARGS__)
#define C256(i, ...) \
  C128(i, __VA_ARGS__) \
  C128(i + 128, __VA_ARGS__)
    C256(0, 173, 73, 89, 101)
#undef CASE_BLOCK
#undef C1
#undef C2
#undef C4
#undef C8
#undef C16
#undef C32
#undef C64
#undef C128
#undef C256
  }
  return acc;
}

using ChurnFunc = int (*)(int, int);
ChurnFunc churn_funcs[] = {&icache_churn_0, &icache_churn_1, &icache_churn_2};

struct ChaseNode {
  ChaseNode* next;
  int payload;
};

std::vector<ChaseNode> chase_nodes;
ChaseNode* chase_head = nullptr;

void setup_chase(size_t size) {
  if (chase_head) {
    return;
  }
  chase_nodes.resize(size);
  for (size_t i = 0; i < size; ++i) {
    chase_nodes[i].next = &chase_nodes[(i + 1) % size];
    chase_nodes[i].payload = i * 37;
  }
  chase_head = &chase_nodes[0];
}
} // namespace

template <
    SerializerMethod kSerializerMethod,
    typename Serializer,
    typename Struct>
void writeBench_IcacheSaturate(size_t iters) {
  folly::BenchmarkSuspender susp;
  auto strct = create<Struct>();
  protocol::Object obj;
  if constexpr (kSerializerMethod == SerializerMethod::Object) {
    folly::IOBufQueue q;
    Serializer::serialize(strct, &q);
    obj = protocol::parseObject<GetReader<Serializer>>(*q.move());
  }

  constexpr size_t chase_size = 256;
  setup_chase(chase_size);
  auto* current_node = chase_head;
  int acc = 0;

  susp.dismiss();

  folly::IOBufQueue q;
  for (size_t j = 0; j < iters; j++) {
    if constexpr (kSerializerMethod == SerializerMethod::Object) {
      protocol::serializeObject<GetWriter<Serializer>>(obj, q);
      folly::doNotOptimizeAway(q);
    } else {
      Serializer::serialize(strct, &q);
      folly::doNotOptimizeAway(q);
    }

    current_node = current_node->next;
    acc = churn_funcs[j % 3](current_node->payload & 0xFF, acc);

    // Reuse the queue across iterations to avoid allocating a new buffer for
    // each struct (which would dominate the measurement), but keep only the
    // tail to avoid unbounded growth.
    if (auto head = q.front(); head && head->isChained()) {
      q.append(q.move()->prev()->unlink());
    }
  }
  folly::doNotOptimizeAway(acc);
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

BENCHMARK(OpEncodeBinaryProtocol_write_SortedVecSetInt_IcacheSaturate, iters) {
  writeBench_IcacheSaturate<
      getSerializerMethod("OpEncode"),
      BinarySerializer,
      OpSortedVecSetInt>(iters);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
