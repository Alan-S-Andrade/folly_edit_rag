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

#define OP_CHUNK_1(acc, i, offset) \
  acc += (i * 3 + offset);         \
  acc ^= (acc >> 13);              \
  acc *= (i * 5 + offset + 1);     \
  acc -= (i * 7 + offset + 2);     \
  acc ^= (acc >> 17)

#define CASES_BODY_1(acc, i)        \
  OP_CHUNK_1(acc, i, i * 4);        \
  OP_CHUNK_1(acc, i, i * 4 + 1000); \
  OP_CHUNK_1(acc, i, i * 4 + 2000); \
  OP_CHUNK_1(acc, i, i * 4 + 3000); \
  OP_CHUNK_1(acc, i, i * 4 + 4000); \
  OP_CHUNK_1(acc, i, i * 4 + 5000)

__attribute__((noinline)) static volatile int64_t heavy_alu_1(int64_t val) {
  int64_t acc = val;
  switch (val & 0xFF) {
#define CASES_1(i)       \
  case i: {              \
    CASES_BODY_1(acc, i); \
    break;               \
  }
#define CASES_16(i)   \
  CASES_1(i)          \
  CASES_1(i + 1)      \
  CASES_1(i + 2)      \
  CASES_1(i + 3)      \
  CASES_1(i + 4)      \
  CASES_1(i + 5)      \
  CASES_1(i + 6)      \
  CASES_1(i + 7)      \
  CASES_1(i + 8)      \
  CASES_1(i + 9)      \
  CASES_1(i + 10)     \
  CASES_1(i + 11)     \
  CASES_1(i + 12)     \
  CASES_1(i + 13)     \
  CASES_1(i + 14)     \
  CASES_1(i + 15)
#define CASES_256     \
  CASES_16(0)         \
  CASES_16(16)        \
  CASES_16(32)        \
  CASES_16(48)        \
  CASES_16(64)        \
  CASES_16(80)        \
  CASES_16(96)        \
  CASES_16(112)       \
  CASES_16(128)       \
  CASES_16(144)       \
  CASES_16(160)       \
  CASES_16(176)       \
  CASES_16(192)       \
  CASES_16(208)       \
  CASES_16(224)       \
  CASES_16(240)
    CASES_256
#undef CASES_1
#undef CASES_16
#undef CASES_256
  default:
    break;
  }
  return acc;
}
#undef OP_CHUNK_1
#undef CASES_BODY_1

#define OP_CHUNK_2(acc, i, offset) \
  acc += (i * 11 + offset);        \
  acc ^= (acc >> 11);              \
  acc *= (i * 13 + offset + 1);    \
  acc -= (i * 17 + offset + 2);    \
  acc ^= (acc >> 19)

#define CASES_BODY_2(acc, i)        \
  OP_CHUNK_2(acc, i, i * 4);        \
  OP_CHUNK_2(acc, i, i * 4 + 1000); \
  OP_CHUNK_2(acc, i, i * 4 + 2000); \
  OP_CHUNK_2(acc, i, i * 4 + 3000); \
  OP_CHUNK_2(acc, i, i * 4 + 4000); \
  OP_CHUNK_2(acc, i, i * 4 + 5000)

__attribute__((noinline)) static volatile int64_t heavy_alu_2(int64_t val) {
  int64_t acc = val;
  switch (val & 0xFF) {
#define CASES_1(i)       \
  case i: {              \
    CASES_BODY_2(acc, i); \
    break;               \
  }
#define CASES_16(i)   \
  CASES_1(i)          \
  CASES_1(i + 1)      \
  CASES_1(i + 2)      \
  CASES_1(i + 3)      \
  CASES_1(i + 4)      \
  CASES_1(i + 5)      \
  CASES_1(i + 6)      \
  CASES_1(i + 7)      \
  CASES_1(i + 8)      \
  CASES_1(i + 9)      \
  CASES_1(i + 10)     \
  CASES_1(i + 11)     \
  CASES_1(i + 12)     \
  CASES_1(i + 13)     \
  CASES_1(i + 14)     \
  CASES_1(i + 15)
#define CASES_256     \
  CASES_16(0)         \
  CASES_16(16)        \
  CASES_16(32)        \
  CASES_16(48)        \
  CASES_16(64)        \
  CASES_16(80)        \
  CASES_16(96)        \
  CASES_16(112)       \
  CASES_16(128)       \
  CASES_16(144)       \
  CASES_16(160)       \
  CASES_16(176)       \
  CASES_16(192)       \
  CASES_16(208)       \
  CASES_16(224)       \
  CASES_16(240)
    CASES_256
#undef CASES_1
#undef CASES_16
#undef CASES_256
  default:
    break;
  }
  return acc;
}
#undef OP_CHUNK_2
#undef CASES_BODY_2

#define OP_CHUNK_3(acc, i, offset) \
  acc += (i * 23 + offset);        \
  acc ^= (acc >> 7);               \
  acc *= (i * 29 + offset + 1);    \
  acc -= (i * 31 + offset + 2);    \
  acc ^= (acc >> 23)

#define CASES_BODY_3(acc, i)        \
  OP_CHUNK_3(acc, i, i * 4);        \
  OP_CHUNK_3(acc, i, i * 4 + 1000); \
  OP_CHUNK_3(acc, i, i * 4 + 2000); \
  OP_CHUNK_3(acc, i, i * 4 + 3000); \
  OP_CHUNK_3(acc, i, i * 4 + 4000); \
  OP_CHUNK_3(acc, i, i * 4 + 5000)

__attribute__((noinline)) static volatile int64_t heavy_alu_3(int64_t val) {
  int64_t acc = val;
  switch (val & 0xFF) {
#define CASES_1(i)       \
  case i: {              \
    CASES_BODY_3(acc, i); \
    break;               \
  }
#define CASES_16(i)   \
  CASES_1(i)          \
  CASES_1(i + 1)      \
  CASES_1(i + 2)      \
  CASES_1(i + 3)      \
  CASES_1(i + 4)      \
  CASES_1(i + 5)      \
  CASES_1(i + 6)      \
  CASES_1(i + 7)      \
  CASES_1(i + 8)      \
  CASES_1(i + 9)      \
  CASES_1(i + 10)     \
  CASES_1(i + 11)     \
  CASES_1(i + 12)     \
  CASES_1(i + 13)     \
  CASES_1(i + 14)     \
  CASES_1(i + 15)
#define CASES_256     \
  CASES_16(0)         \
  CASES_16(16)        \
  CASES_16(32)        \
  CASES_16(48)        \
  CASES_16(64)        \
  CASES_16(80)        \
  CASES_16(96)        \
  CASES_16(112)       \
  CASES_16(128)       \
  CASES_16(144)       \
  CASES_16(160)       \
  CASES_16(176)       \
  CASES_16(192)       \
  CASES_16(208)       \
  CASES_16(224)       \
  CASES_16(240)
    CASES_256
#undef CASES_1
#undef CASES_16
#undef CASES_256
  default:
    break;
  }
  return acc;
}
#undef OP_CHUNK_3
#undef CASES_BODY_3

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_ICacheMiss, iters) {
  folly::BenchmarkSuspender susp;
  using Serializer = CompactSerializer;
  using Struct = LargeSetInt;
  auto strct = create<Struct>();
  folly::IOBufQueue q;
  Serializer::serialize(strct, &q);
  auto buf = q.move();
  buf->coalesce();

  constexpr size_t chase_size = 1024;
  std::vector<void*> chase_buffer(chase_size);
  for (size_t i = 0; i < chase_size; ++i) {
    chase_buffer[i] = &chase_buffer[(i * 61 + 17) % chase_size];
  }
  void* p = chase_buffer[0];

  susp.dismiss();

  int64_t fake_dep = 0;
  for (size_t i = 0; i < iters; ++i) {
    auto obj = protocol::parseObject<GetReader<Serializer>>(*buf);
    folly::doNotOptimizeAway(obj);

    for (int j = 0; j < 500; ++j) {
      p = *static_cast<void**>(p);
      int64_t payload = reinterpret_cast<intptr_t>(p);
      switch (j % 3) {
        case 0:
          fake_dep = heavy_alu_1(payload + fake_dep);
          break;
        case 1:
          fake_dep = heavy_alu_2(payload + fake_dep);
          break;
        case 2:
          fake_dep = heavy_alu_3(payload + fake_dep);
          break;
      }
    }
  }
  folly::doNotOptimizeAway(fake_dep);
  susp.rehire();
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
