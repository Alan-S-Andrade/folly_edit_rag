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

#define ALU_OPS_26_OFFSET(acc, i, offset) \
  acc += (i) * 26 + 1 + offset;           \
  acc ^= (i) * 26 + 2 + offset;           \
  acc += (i) * 26 + 3 + offset;           \
  acc ^= (i) * 26 + 4 + offset;           \
  acc += (i) * 26 + 5 + offset;           \
  acc ^= (i) * 26 + 6 + offset;           \
  acc += (i) * 26 + 7 + offset;           \
  acc ^= (i) * 26 + 8 + offset;           \
  acc += (i) * 26 + 9 + offset;           \
  acc ^= (i) * 26 + 10 + offset;          \
  acc += (i) * 26 + 11 + offset;          \
  acc ^= (i) * 26 + 12 + offset;          \
  acc += (i) * 26 + 13 + offset;          \
  acc ^= (i) * 26 + 14 + offset;          \
  acc += (i) * 26 + 15 + offset;          \
  acc ^= (i) * 26 + 16 + offset;          \
  acc += (i) * 26 + 17 + offset;          \
  acc ^= (i) * 26 + 18 + offset;          \
  acc += (i) * 26 + 19 + offset;          \
  acc ^= (i) * 26 + 20 + offset;          \
  acc += (i) * 26 + 21 + offset;          \
  acc ^= (i) * 26 + 22 + offset;          \
  acc += (i) * 26 + 23 + offset;          \
  acc ^= (i) * 26 + 24 + offset;          \
  acc += (i) * 26 + 25 + offset;          \
  acc ^= (i) * 26 + 26 + offset

#define C(n, offset)                           \
  case n:                                      \
    ALU_OPS_26_OFFSET(acc, n, offset); \
    break;
#define C4(n, offset) C(n, offset) C(n + 1, offset) C(n + 2, offset) C(n + 3, offset)
#define C16(n, offset) \
  C4(n, offset) C4(n + 4, offset) C4(n + 8, offset) C4(n + 12, offset)
#define C64(n, offset)                                                    \
  C16(n, offset) C16(n + 16, offset) C16(n + 32, offset) C16(n + 48, \
                                                                    offset)
#define C256(n, offset)                                                 \
  C64(n, offset) C64(n + 64, offset) C64(n + 128, offset) C64(n + 192, \
                                                                      offset)

__attribute__((noinline)) static int64_t alu_worker1(int64_t acc, uint8_t val) {
  switch (val) {
    C256(0, 0);
  }
  return acc;
}

__attribute__((noinline)) static int64_t alu_worker2(int64_t acc, uint8_t val) {
  switch (val) {
    C256(0, 256 * 26);
  }
  return acc;
}

__attribute__((noinline)) static int64_t alu_worker3(int64_t acc, uint8_t val) {
  switch (val) {
    C256(0, 256 * 26 * 2);
  }
  return acc;
}

#undef C
#undef C4
#undef C16
#undef C64
#undef C256
#undef ALU_OPS_26_OFFSET

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v6, iters) {
  folly::BenchmarkSuspender susp;

  struct ChaseNode {
    uint8_t payload;
    ChaseNode* next;
  };
  constexpr size_t kNumNodes = 4096;
  auto nodes = std::make_unique<ChaseNode[]>(kNumNodes);
  for (size_t i = 0; i < kNumNodes; ++i) {
    nodes[i].payload = (i * 13 + 7) & 0xFF;
    nodes[i].next = &nodes[(i + 1) % kNumNodes];
  }

  ChaseNode* p = &nodes[0];
  int64_t acc = 0;

  susp.dismiss();

  for (size_t i = 0; i < iters; ++i) {
    uint8_t val = p->payload;
    switch (i % 3) {
      case 0:
        acc = alu_worker1(acc, val);
        break;
      case 1:
        acc = alu_worker2(acc, val);
        break;
      case 2:
        acc = alu_worker3(acc, val);
        break;
    }
    p = p->next;
  }

  susp.rehire();
  folly::doNotOptimizeAway(acc);
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

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
