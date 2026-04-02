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

#define OPS_112(acc, d, c) \
    acc = (acc * (c + 1)) + d[1];  acc = (acc ^ (c + 2)) - d[2];  acc = (acc + (c + 3)) * d[3];  acc = (acc - (c + 4)) ^ d[4]; \
    acc = (acc * (c + 5)) + d[5];  acc = (acc ^ (c + 6)) - d[6];  acc = (acc + (c + 7)) * d[7];  acc = (acc - (c + 8)) ^ d[0]; \
    acc = (acc * (c + 9)) + d[1];  acc = (acc ^ (c + 10)) - d[2]; acc = (acc + (c + 11)) * d[3]; acc = (acc - (c + 12)) ^ d[4]; \
    acc = (acc * (c + 13)) + d[5]; acc = (acc ^ (c + 14)) - d[6]; acc = (acc + (c + 15)) * d[7]; acc = (acc - (c + 16)) ^ d[0]; \
    acc = (acc * (c + 17)) + d[1]; acc = (acc ^ (c + 18)) - d[2]; acc = (acc + (c + 19)) * d[3]; acc = (acc - (c + 20)) ^ d[4]; \
    acc = (acc * (c + 21)) + d[5]; acc = (acc ^ (c + 22)) - d[6]; acc = (acc + (c + 23)) * d[7]; acc = (acc - (c + 24)) ^ d[0]; \
    acc = (acc * (c + 25)) + d[1]; acc = (acc ^ (c + 26)) - d[2]; acc = (acc + (c + 27)) * d[3]; acc = (acc - (c + 28)) ^ d[4]; \
    acc = (acc * (c + 29)) + d[1];  acc = (acc ^ (c + 30)) - d[2];  acc = (acc + (c + 31)) * d[3];  acc = (acc - (c + 32)) ^ d[4]; \
    acc = (acc * (c + 33)) + d[5];  acc = (acc ^ (c + 34)) - d[6];  acc = (acc + (c + 35)) * d[7];  acc = (acc - (c + 36)) ^ d[0]; \
    acc = (acc * (c + 37)) + d[1];  acc = (acc ^ (c + 38)) - d[2]; acc = (acc + (c + 39)) * d[3]; acc = (acc - (c + 40)) ^ d[4]; \
    acc = (acc * (c + 41)) + d[5]; acc = (acc ^ (c + 42)) - d[6]; acc = (acc + (c + 43)) * d[7]; acc = (acc - (c + 44)) ^ d[0]; \
    acc = (acc * (c + 45)) + d[1]; acc = (acc ^ (c + 46)) - d[2]; acc = (acc + (c + 47)) * d[3]; acc = (acc - (c + 48)) ^ d[4]; \
    acc = (acc * (c + 49)) + d[5]; acc = (acc ^ (c + 50)) - d[6]; acc = (acc + (c + 51)) * d[7]; acc = (acc - (c + 52)) ^ d[0]; \
    acc = (acc * (c + 53)) + d[1]; acc = (acc ^ (c + 54)) - d[2]; acc = (acc + (c + 55)) * d[3]; acc = (acc - (c + 56)) ^ d[4]; \
    acc = (acc * (c + 57)) + d[1];  acc = (acc ^ (c + 58)) - d[2];  acc = (acc + (c + 59)) * d[3];  acc = (acc - (c + 60)) ^ d[4]; \
    acc = (acc * (c + 61)) + d[5];  acc = (acc ^ (c + 62)) - d[6];  acc = (acc + (c + 63)) * d[7];  acc = (acc - (c + 64)) ^ d[0]; \
    acc = (acc * (c + 65)) + d[1];  acc = (acc ^ (c + 66)) - d[2]; acc = (acc + (c + 67)) * d[3]; acc = (acc - (c + 68)) ^ d[4]; \
    acc = (acc * (c + 69)) + d[5]; acc = (acc ^ (c + 70)) - d[6]; acc = (acc + (c + 71)) * d[7]; acc = (acc - (c + 72)) ^ d[0]; \
    acc = (acc * (c + 73)) + d[1]; acc = (acc ^ (c + 74)) - d[2]; acc = (acc + (c + 75)) * d[3]; acc = (acc - (c + 76)) ^ d[4]; \
    acc = (acc * (c + 77)) + d[5]; acc = (acc ^ (c + 78)) - d[6]; acc = (acc + (c + 79)) * d[7]; acc = (acc - (c + 80)) ^ d[0]; \
    acc = (acc * (c + 81)) + d[1]; acc = (acc ^ (c + 82)) - d[2]; acc = (acc + (c + 83)) * d[3]; acc = (acc - (c + 84)) ^ d[4]; \
    acc = (acc * (c + 85)) + d[1];  acc = (acc ^ (c + 86)) - d[2];  acc = (acc + (c + 87)) * d[3];  acc = (acc - (c + 88)) ^ d[4]; \
    acc = (acc * (c + 89)) + d[5];  acc = (acc ^ (c + 90)) - d[6];  acc = (acc + (c + 91)) * d[7];  acc = (acc - (c + 92)) ^ d[0]; \
    acc = (acc * (c + 93)) + d[1];  acc = (acc ^ (c + 94)) - d[2]; acc = (acc + (c + 95)) * d[3]; acc = (acc - (c + 96)) ^ d[4]; \
    acc = (acc * (c + 97)) + d[5]; acc = (acc ^ (c + 98)) - d[6]; acc = (acc + (c + 99)) * d[7]; acc = (acc - (c + 100)) ^ d[0]; \
    acc = (acc * (c + 101)) + d[1]; acc = (acc ^ (c + 102)) - d[2]; acc = (acc + (c + 103)) * d[3]; acc = (acc - (c + 104)) ^ d[4]; \
    acc = (acc * (c + 105)) + d[5]; acc = (acc ^ (c + 106)) - d[6]; acc = (acc + (c + 107)) * d[7]; acc = (acc - (c + 108)) ^ d[0]; \
    acc = (acc * (c + 109)) + d[1]; acc = (acc ^ (c + 110)) - d[2]; acc = (acc + (c + 111)) * d[3]; acc = (acc - (c + 112)) ^ d[4];

#define CASE(i, C) \
    case i: { OPS_112(acc, data, C * 256 + i); break; }

#define CASES_16(i, C) \
    CASE(i, C) CASE(i+1, C) CASE(i+2, C) CASE(i+3, C) \
    CASE(i+4, C) CASE(i+5, C) CASE(i+6, C) CASE(i+7, C) \
    CASE(i+8, C) CASE(i+9, C) CASE(i+10, C) CASE(i+11, C) \
    CASE(i+12, C) CASE(i+13, C) CASE(i+14, C) CASE(i+15, C)

#define CASES_256(C) \
    CASES_16(0, C) CASES_16(16, C) CASES_16(32, C) CASES_16(48, C) \
    CASES_16(64, C) CASES_16(80, C) CASES_16(96, C) CASES_16(112, C) \
    CASES_16(128, C) CASES_16(144, C) CASES_16(160, C) CASES_16(176, C) \
    CASES_16(192, C) CASES_16(208, C) CASES_16(224, C) CASES_16(240, C)

template <int C>
__attribute__((noinline)) int process_data(int acc, const uint8_t* data) {
    switch(data[0]) {
        CASES_256(C)
        default:
            break;
    }
    return acc;
}
} // namespace


BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v17, iters) {
    folly::BenchmarkSuspender susp;

    auto strct = create<LargeSetInt>();
    folly::IOBufQueue q;
    CompactSerializer::serialize(strct, &q);
    auto buf = q.move();
    buf->coalesce();
    const uint8_t* data = buf->data();
    size_t data_len = buf->length();

    // Guard: need at least 8 bytes for OPS_28 to index d[0]..d[7] safely
    CHECK_GE(data_len, 8u);

    size_t current_offset = 0;
    int acc = 0;

    susp.dismiss();

    for (size_t i = 0; i < iters; ++i) {
        const uint8_t* p = data + current_offset;

        switch (i % 3) {
            case 0: acc = process_data<1>(acc, p); break;
            case 1: acc = process_data<2>(acc, p); break;
            case 2: acc = process_data<3>(acc, p); break;
        }

        current_offset += (1 + (p[1] % 32));
        // Ensure we always have at least 8 bytes available after current_offset
        if (current_offset + 8 >= data_len) {
            current_offset = 0;
        }
    }

    susp.rehire();
    folly::doNotOptimizeAway(acc);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
