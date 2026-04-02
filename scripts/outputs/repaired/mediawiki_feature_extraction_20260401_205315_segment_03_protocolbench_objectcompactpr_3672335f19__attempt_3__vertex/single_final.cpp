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

namespace {

#define OPS_PER_CASE(i, acc, salt) \
  acc += (i * 3 + salt) * (acc | (0x123 + salt)); \
  acc ^= (i * 5 + salt) * (acc | (0x456 + salt)); \
  acc += (i * 7 + salt) * (acc | (0x789 + salt)); \
  acc ^= (i * 11 + salt) * (acc | (0xABC + salt)); \
  acc += (i * 13 + salt) * (acc | (0xDEF + salt)); \
  acc ^= (i * 17 + salt) * (acc | (0x135 + salt)); \
  acc += (i * 19 + salt) * (acc | (0x246 + salt)); \
  acc ^= (i * 23 + salt) * (acc | (0x79B + salt)); \
  acc += (i * 29 + salt) * (acc | (0x8AD + salt)); \
  acc ^= (i * 31 + salt) * (acc | (0xCEF + salt)); \
  acc += (i * 37 + salt) * (acc | (0x1234 + salt)); \
  acc ^= (i * 41 + salt) * (acc | (0x5678 + salt)); \
  acc += (i * 43 + salt) * (acc | (0x9ABC + salt)); \
  acc ^= (i * 47 + salt) * (acc | (0xDEF0 + salt)); \
  acc += (i * 53 + salt) * (acc | (0x1357 + salt)); \
  acc ^= (i * 59 + salt) * (acc | (0x2468 + salt)); \
  acc += (i * 61 + salt) * (acc | (0x9BDF + salt)); \
  acc ^= (i * 67 + salt) * (acc | (0xACE0 + salt)); \
  acc += (i * 71 + salt) * (acc | (0x1122 + salt)); \
  acc ^= (i * 73 + salt) * (acc | (0x3344 + salt));

#define CASE_STMT(i, salt) \
    case i: \
        OPS_PER_CASE(i, acc, salt) \
        break;

#define R1(i, salt) CASE_STMT(i, salt)
#define R4(i, salt) R1(i, salt) R1(i+1, salt) R1(i+2, salt) R1(i+3, salt)
#define R16(i, salt) R4(i, salt) R4(i+4, salt) R4(i+8, salt) R4(i+12, salt)
#define R64(i, salt) R16(i, salt) R16(i+16, salt) R16(i+32, salt) R16(i+48, salt)
#define R256(i, salt) R64(i, salt) R64(i+64, salt) R64(i+128, salt) R64(i+192, salt)

__attribute__((noinline))
int64_t large_switch_func_1(int64_t acc, int val) {
    switch (val) {
        R256(0, 1)
    }
    return acc;
}

__attribute__((noinline))
int64_t large_switch_func_2(int64_t acc, int val) {
    switch (val) {
        R256(0, 2)
    }
    return acc;
}

__attribute__((noinline))
int64_t large_switch_func_3(int64_t acc, int val) {
    switch (val) {
        R256(0, 3)
    }
    return acc;
}

#undef R256
#undef R64
#undef R16
#undef R4
#undef R1
#undef CASE_STMT
#undef OPS_PER_CASE

} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_icache_v3, iters) {
    folly::BenchmarkSuspender susp;
    using Serializer = CompactSerializer;
    using Struct = LargeSetInt;
    using Reader = GetReader<Serializer>;
    auto strct = create<Struct>();
    folly::IOBufQueue q;
    Serializer::serialize(strct, &q);
    auto buf = q.move();
    buf->coalesce();

    constexpr size_t kChaseArraySize = 256;
    std::vector<uint8_t> chase_array(kChaseArraySize);
    for (size_t i = 0; i < kChaseArraySize; ++i) {
        chase_array[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(chase_array.begin(), chase_array.end(), g);

    susp.dismiss();

    uint8_t chase_idx = 0;
    int64_t acc = 0;

    for (size_t i = 0; i < iters; ++i) {
        auto obj = protocol::parseObject<Reader>(*buf);
        folly::doNotOptimizeAway(obj);

        chase_idx = chase_array[chase_idx];
        uint8_t switch_val = chase_idx;

        switch (i % 3) {
            case 0:
                acc = large_switch_func_1(acc, switch_val);
                break;
            case 1:
                acc = large_switch_func_2(acc, switch_val);
                break;
            case 2:
                acc = large_switch_func_3(acc, switch_val);
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
