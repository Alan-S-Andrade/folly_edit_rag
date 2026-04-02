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
#include <numeric>
#include <random>

using namespace apache::thrift;
using namespace thrift::benchmark;

namespace {

// To target L1i MPKI of ~28, we need a large code footprint.
// This is achieved by creating 3 large, no-inlined functions with 256-case
// switch statements. The benchmark rotates through these functions.
// Each case has a number of ALU operations with unique constants to prevent
// compiler optimizations and increase code size.
// 7 chunks * 4 ops/chunk = 28 ops per case.

__attribute__((noinline)) void worker_func_0(int val, volatile int& acc) {
#define ALU_CHUNK(v)                             \
  acc = (acc * (v + 13)) + (v + 29);             \
  acc = (acc * (v + 31)) ^ (v + 47);
#define ALU_CHUNK_7(v)                           \
  ALU_CHUNK(v + 0) ALU_CHUNK(v + 64) ALU_CHUNK(v + 128)   \
  ALU_CHUNK(v + 192) ALU_CHUNK(v + 256) ALU_CHUNK(v + 320) \
  ALU_CHUNK(v + 384)
#define MAKE_CASE(i, base)                       \
  case i:                                        \
    ALU_CHUNK_7((base) + (i)*512);               \
    break;

  switch (val & 0xFF) {
#define CASES_16(i, base)                                                      \
    MAKE_CASE(i + 0, base) MAKE_CASE(i + 1, base) MAKE_CASE(i + 2, base)        \
    MAKE_CASE(i + 3, base) MAKE_CASE(i + 4, base) MAKE_CASE(i + 5, base)        \
    MAKE_CASE(i + 6, base) MAKE_CASE(i + 7, base) MAKE_CASE(i + 8, base)        \
    MAKE_CASE(i + 9, base) MAKE_CASE(i + 10, base) MAKE_CASE(i + 11, base)      \
    MAKE_CASE(i + 12, base) MAKE_CASE(i + 13, base) MAKE_CASE(i + 14, base)     \
    MAKE_CASE(i + 15, base)

    CASES_16(0, 10000) CASES_16(16, 10000) CASES_16(32, 10000)
    CASES_16(48, 10000) CASES_16(64, 10000) CASES_16(80, 10000)
    CASES_16(96, 10000) CASES_16(112, 10000) CASES_16(128, 10000)
    CASES_16(144, 10000) CASES_16(160, 10000) CASES_16(176, 10000)
    CASES_16(192, 10000) CASES_16(208, 10000) CASES_16(224, 10000)
    CASES_16(240, 10000)
#undef CASES_16
  }
#undef MAKE_CASE
#undef ALU_CHUNK_7
#undef ALU_CHUNK
}

__attribute__((noinline)) void worker_func_1(int val, volatile int& acc) {
#define ALU_CHUNK(v)                             \
  acc = (acc * (v + 13)) + (v + 29);             \
  acc = (acc * (v + 31)) ^ (v + 47);
#define ALU_CHUNK_7(v)                           \
  ALU_CHUNK(v + 0) ALU_CHUNK(v + 64) ALU_CHUNK(v + 128)   \
  ALU_CHUNK(v + 192) ALU_CHUNK(v + 256) ALU_CHUNK(v + 320) \
  ALU_CHUNK(v + 384)
#define MAKE_CASE(i, base)                       \
  case i:                                        \
    ALU_CHUNK_7((base) + (i)*512);               \
    break;

  switch (val & 0xFF) {
#define CASES_16(i, base)                                                      \
    MAKE_CASE(i + 0, base) MAKE_CASE(i + 1, base) MAKE_CASE(i + 2, base)        \
    MAKE_CASE(i + 3, base) MAKE_CASE(i + 4, base) MAKE_CASE(i + 5, base)        \
    MAKE_CASE(i + 6, base) MAKE_CASE(i + 7, base) MAKE_CASE(i + 8, base)        \
    MAKE_CASE(i + 9, base) MAKE_CASE(i + 10, base) MAKE_CASE(i + 11, base)      \
    MAKE_CASE(i + 12, base) MAKE_CASE(i + 13, base) MAKE_CASE(i + 14, base)     \
    MAKE_CASE(i + 15, base)

    CASES_16(0, 2000000) CASES_16(16, 2000000) CASES_16(32, 2000000)
    CASES_16(48, 2000000) CASES_16(64, 2000000) CASES_16(80, 2000000)
    CASES_16(96, 2000000) CASES_16(112, 2000000) CASES_16(128, 2000000)
    CASES_16(144, 2000000) CASES_16(160, 2000000) CASES_16(176, 2000000)
    CASES_16(192, 2000000) CASES_16(208, 2000000) CASES_16(224, 2000000)
    CASES_16(240, 2000000)
#undef CASES_16
  }
#undef MAKE_CASE
#undef ALU_CHUNK_7
#undef ALU_CHUNK
}

__attribute__((noinline)) void worker_func_2(int val, volatile int& acc) {
#define ALU_CHUNK(v)                             \
  acc = (acc * (v + 13)) + (v + 29);             \
  acc = (acc * (v + 31)) ^ (v + 47);
#define ALU_CHUNK_7(v)                           \
  ALU_CHUNK(v + 0) ALU_CHUNK(v + 64) ALU_CHUNK(v + 128)   \
  ALU_CHUNK(v + 192) ALU_CHUNK(v + 256) ALU_CHUNK(v + 320) \
  ALU_CHUNK(v + 384)
#define MAKE_CASE(i, base)                       \
  case i:                                        \
    ALU_CHUNK_7((base) + (i)*512);               \
    break;

  switch (val & 0xFF) {
#define CASES_16(i, base)                                                      \
    MAKE_CASE(i + 0, base) MAKE_CASE(i + 1, base) MAKE_CASE(i + 2, base)        \
    MAKE_CASE(i + 3, base) MAKE_CASE(i + 4, base) MAKE_CASE(i + 5, base)        \
    MAKE_CASE(i + 6, base) MAKE_CASE(i + 7, base) MAKE_CASE(i + 8, base)        \
    MAKE_CASE(i + 9, base) MAKE_CASE(i + 10, base) MAKE_CASE(i + 11, base)      \
    MAKE_CASE(i + 12, base) MAKE_CASE(i + 13, base) MAKE_CASE(i + 14, base)     \
    MAKE_CASE(i + 15, base)

    CASES_16(0, 4000000) CASES_16(16, 4000000) CASES_16(32, 4000000)
    CASES_16(48, 4000000) CASES_16(64, 4000000) CASES_16(80, 4000000)
    CASES_16(96, 4000000) CASES_16(112, 4000000) CASES_16(128, 4000000)
    CASES_16(144, 4000000) CASES_16(160, 4000000) CASES_16(176, 4000000)
    CASES_16(192, 4000000) CASES_16(208, 4000000) CASES_16(224, 4000000)
    CASES_16(240, 4000000)
#undef CASES_16
  }
#undef MAKE_CASE
#undef ALU_CHUNK_7
#undef ALU_CHUNK
}

} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_icache_v1, iters) {
  folly::BenchmarkSuspender susp;

  constexpr size_t kChaseSize = 4096;
  struct ChaseNode {
    int next_idx;
    int payload;
  };
  std::vector<ChaseNode> chase_buffer(kChaseSize);
  std::vector<int> indices(kChaseSize);
  std::iota(indices.begin(), indices.end(), 0);

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  for (size_t i = 0; i < kChaseSize; ++i) {
    chase_buffer[indices[i]].next_idx = indices[(i + 1) % kChaseSize];
    chase_buffer[indices[i]].payload = g();
  }

  volatile int acc = 1;
  size_t chase_idx = 0;

  susp.dismiss();

  for (size_t i = 0; i < iters; ++i) {
    chase_idx = chase_buffer[chase_idx].next_idx;
    int payload = chase_buffer[chase_idx].payload;

    switch (i % 3) {
      case 0:
        worker_func_0(payload, acc);
        break;
      case 1:
        worker_func_1(payload, acc);
        break;
      case 2:
        worker_func_2(payload, acc);
        break;
    }
  }

  folly::doNotOptimizeAway(acc);
  susp.rehire();
}

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

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
