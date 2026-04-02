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

namespace {

// Use a volatile to prevent compiler optimizing away the ALU ops.
volatile int64_t icache_saturation_sink = 0;

#define ALU_OPS_28_0(N)                                                        \
  acc += 0x12345678 + N; acc ^= 0xABCDEF01 + N; acc *= 3 + N;                  \
  acc += 0x23456789 + N; acc ^= 0xBCDEF012 + N; acc *= 5 + N;                  \
  acc += 0x3456789A + N; acc ^= 0xCDEF0123 + N; acc *= 7 + N;                  \
  acc += 0x456789AB + N; acc ^= 0xDEF01234 + N; acc *= 11 + N;                 \
  acc += 0x56789ABC + N; acc ^= 0xEF012345 + N; acc *= 13 + N;                 \
  acc += 0x6789ABCD + N; acc ^= 0xF0123456 + N; acc *= 17 + N;                 \
  acc += 0x789ABCDE + N; acc ^= 0x01234567 + N; acc *= 19 + N;                 \
  acc += 0x89ABCDEF + N; acc ^= 0x12345678 + N; acc *= 23 + N;                 \
  acc += 0x9ABCDEF0 + N; acc ^= 0x23456789 + N; acc *= 29 + N;                 \
  acc += 0xABCDEF01 + N; acc ^= 0x3456789A + N;

#define ALU_OPS_28_1(N)                                                        \
  acc += 0x23456781 + N; acc ^= 0xBCDEF01A + N; acc *= 3 + N;                  \
  acc += 0x34567892 + N; acc ^= 0xCDEF012B + N; acc *= 5 + N;                  \
  acc += 0x456789A3 + N; acc ^= 0xDEF0123C + N; acc *= 7 + N;                  \
  acc += 0x56789AB4 + N; acc ^= 0xEF01234D + N; acc *= 11 + N;                 \
  acc += 0x6789ABC5 + N; acc ^= 0xF012345E + N; acc *= 13 + N;                 \
  acc += 0x789ABCD6 + N; acc ^= 0x0123456F + N; acc *= 17 + N;                 \
  acc += 0x89ABCDE7 + N; acc ^= 0x12345670 + N; acc *= 19 + N;                 \
  acc += 0x9ABCDEF8 + N; acc ^= 0x23456781 + N; acc *= 23 + N;                 \
  acc += 0xABCDEF09 + N; acc ^= 0x34567892 + N; acc *= 29 + N;                 \
  acc += 0xBCDEF01A + N; acc ^= 0x456789A3 + N;

#define ALU_OPS_28_2(N)                                                        \
  acc += 0xFEDCBA98 + N; acc ^= 0x01234567 + N; acc *= 3 + N;                  \
  acc += 0xEDCBA987 + N; acc ^= 0x12345678 + N; acc *= 5 + N;                  \
  acc += 0xDCBA9876 + N; acc ^= 0x23456789 + N; acc *= 7 + N;                  \
  acc += 0xCBA98765 + N; acc ^= 0x3456789A + N; acc *= 11 + N;                 \
  acc += 0xBA987654 + N; acc ^= 0x456789AB + N; acc *= 13 + N;                 \
  acc += 0xA9876543 + N; acc ^= 0x56789ABC + N; acc *= 17 + N;                 \
  acc += 0x98765432 + N; acc ^= 0x6789ABCD + N; acc *= 19 + N;                 \
  acc += 0x87654321 + N; acc ^= 0x789ABCDE + N; acc *= 23 + N;                 \
  acc += 0x76543210 + N; acc ^= 0x89ABCDEF + N; acc *= 29 + N;                 \
  acc += 0x6543210F + N; acc ^= 0x9ABCDEF0 + N;

#define CASE(i, N, M) \
  case i:             \
    M(N);             \
    break;
#define CASES_16(i, N, M)                                                      \
  CASE(i + 0, N + 0, M)                                                        \
  CASE(i + 1, N + 1, M)                                                        \
  CASE(i + 2, N + 2, M)                                                        \
  CASE(i + 3, N + 3, M)                                                        \
  CASE(i + 4, N + 4, M)                                                        \
  CASE(i + 5, N + 5, M)                                                        \
  CASE(i + 6, N + 6, M)                                                        \
  CASE(i + 7, N + 7, M)                                                        \
  CASE(i + 8, N + 8, M)                                                        \
  CASE(i + 9, N + 9, M)                                                        \
  CASE(i + 10, N + 10, M)                                                      \
  CASE(i + 11, N + 11, M)                                                      \
  CASE(i + 12, N + 12, M)                                                      \
  CASE(i + 13, N + 13, M)                                                      \
  CASE(i + 14, N + 14, M)                                                      \
  CASE(i + 15, N + 15, M)
#define CASES_256(N, M)                                                        \
  CASES_16(0, N + 0, M)                                                        \
  CASES_16(16, N + 16, M)                                                      \
  CASES_16(32, N + 32, M)                                                      \
  CASES_16(48, N + 48, M)                                                      \
  CASES_16(64, N + 64, M)                                                      \
  CASES_16(80, N + 80, M)                                                      \
  CASES_16(96, N + 96, M)                                                      \
  CASES_16(112, N + 112, M)                                                    \
  CASES_16(128, N + 128, M)                                                    \
  CASES_16(144, N + 144, M)                                                    \
  CASES_16(160, N + 160, M)                                                    \
  CASES_16(176, N + 176, M)                                                    \
  CASES_16(192, N + 192, M)                                                    \
  CASES_16(208, N + 208, M)                                                    \
  CASES_16(224, N + 224, M)                                                    \
  CASES_16(240, N + 240, M)

__attribute__((noinline)) int64_t saturate_icache_0(int64_t acc, uint8_t val) {
  switch (val) {
    CASES_256(0, ALU_OPS_28_0)
  }
  return acc;
}

__attribute__((noinline)) int64_t saturate_icache_1(int64_t acc, uint8_t val) {
  switch (val) {
    CASES_256(256, ALU_OPS_28_1)
  }
  return acc;
}

__attribute__((noinline)) int64_t saturate_icache_2(int64_t acc, uint8_t val) {
  switch (val) {
    CASES_256(512, ALU_OPS_28_2)
  }
  return acc;
}
} // namespace

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
  susp.dismiss();

  folly::IOBufQueue q;
  int64_t acc = 0;
  while (iters--) {
    if constexpr (kSerializerMethod == SerializerMethod::Object) {
      protocol::serializeObject<GetWriter<Serializer>>(obj, q);
      folly::doNotOptimizeAway(q);
    } else {
      Serializer::serialize(strct, &q);
      folly::doNotOptimizeAway(q);
    }

    // I-cache saturation loop
    auto buf = q.front();
    if (buf && buf->length() > 0) {
      auto coalesced = buf->clone();
      coalesced->coalesce();
      const size_t len = coalesced->length();
      const uint8_t* data = coalesced->data();
      for (size_t i = 0; i < 512; ++i) { // fixed number of iterations
        uint8_t val = data[i % len];
        switch (i % 3) {
          case 0:
            acc = saturate_icache_0(acc, val);
            break;
          case 1:
            acc = saturate_icache_1(acc, val);
            break;
          case 2:
            acc = saturate_icache_2(acc, val);
            break;
        }
      }
    }

    // Reuse the queue across iterations to avoid allocating a new buffer for
    // each struct (which would dominate the measurement), but keep only the
    // tail to avoid unbounded growth.
    if (auto head = q.front(); head && head->isChained()) {
      q.append(q.move()->prev()->unlink());
    }
  }
  icache_saturation_sink = acc;
  susp.rehire();
}

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

BENCHMARK(
    OpEncodeBinaryProtocol_write_SortedVecSetInt_IcacheSaturate,
    iters) {
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
