/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may in a copy of the License at
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

namespace {
__attribute__((noinline))
int64_t hot_code_path_0(int input) {
  int64_t acc = input;
  switch (input & 0xFF) {
#define GEN_CASE(i, base) \
    case i: { acc += base+i*0x100+0; acc ^= base+i*0x100+1; acc *= base+i*0x100+2; acc -= base+i*0x100+3; acc += base+i*0x100+4; acc ^= base+i*0x100+5; acc *= base+i*0x100+6; acc -= base+i*0x100+7; acc += base+i*0x100+8; acc ^= base+i*0x100+9; acc *= base+i*0x100+10; acc -= base+i*0x100+11; acc += base+i*0x100+12; acc ^= base+i*0x100+13; acc *= base+i*0x100+14; acc -= base+i*0x100+15; acc += base+i*0x100+16; acc ^= base+i*0x100+17; acc *= base+i*0x100+18; acc -= base+i*0x100+19; acc += base+i*0x100+20; acc ^= base+i*0x100+21; acc *= base+i*0x100+22; acc -= base+i*0x100+23; acc += base+i*0x100+24; acc ^= base+i*0x100+25; acc *= base+i*0x100+26; acc -= base+i*0x100+27; break; }
#define GEN_16_CASES(i, base) \
    GEN_CASE(i*16+0, base) GEN_CASE(i*16+1, base) GEN_CASE(i*16+2, base) GEN_CASE(i*16+3, base) \
    GEN_CASE(i*16+4, base) GEN_CASE(i*16+5, base) GEN_CASE(i*16+6, base) GEN_CASE(i*16+7, base) \
    GEN_CASE(i*16+8, base) GEN_CASE(i*16+9, base) GEN_CASE(i*16+10, base) GEN_CASE(i*16+11, base) \
    GEN_CASE(i*16+12, base) GEN_CASE(i*16+13, base) GEN_CASE(i*16+14, base) GEN_CASE(i*16+15, base)
GEN_16_CASES(0, 0x1000) GEN_16_CASES(1, 0x1000) GEN_16_CASES(2, 0x1000) GEN_16_CASES(3, 0x1000)
GEN_16_CASES(4, 0x1000) GEN_16_CASES(5, 0x1000) GEN_16_CASES(6, 0x1000) GEN_16_CASES(7, 0x1000)
GEN_16_CASES(8, 0x1000) GEN_16_CASES(9, 0x1000) GEN_16_CASES(10, 0x1000) GEN_16_CASES(11, 0x1000)
GEN_16_CASES(12, 0x1000) GEN_16_CASES(13, 0x1000) GEN_16_CASES(14, 0x1000) GEN_16_CASES(15, 0x1000)
#undef GEN_16_CASES
#undef GEN_CASE
  }
  return acc;
}
__attribute__((noinline))
int64_t hot_code_path_1(int input) {
  int64_t acc = input;
  switch (input & 0xFF) {
#define GEN_CASE(i, base) \
    case i: { acc += base+i*0x100+0; acc ^= base+i*0x100+1; acc *= base+i*0x100+2; acc -= base+i*0x100+3; acc += base+i*0x100+4; acc ^= base+i*0x100+5; acc *= base+i*0x100+6; acc -= base+i*0x100+7; acc += base+i*0x100+8; acc ^= base+i*0x100+9; acc *= base+i*0x100+10; acc -= base+i*0x100+11; acc += base+i*0x100+12; acc ^= base+i*0x100+13; acc *= base+i*0x100+14; acc -= base+i*0x100+15; acc += base+i*0x100+16; acc ^= base+i*0x100+17; acc *= base+i*0x100+18; acc -= base+i*0x100+19; acc += base+i*0x100+20; acc ^= base+i*0x100+21; acc *= base+i*0x100+22; acc -= base+i*0x100+23; acc += base+i*0x100+24; acc ^= base+i*0x100+25; acc *= base+i*0x100+26; acc -= base+i*0x100+27; break; }
#define GEN_16_CASES(i, base) \
    GEN_CASE(i*16+0, base) GEN_CASE(i*16+1, base) GEN_CASE(i*16+2, base) GEN_CASE(i*16+3, base) \
    GEN_CASE(i*16+4, base) GEN_CASE(i*16+5, base) GEN_CASE(i*16+6, base) GEN_CASE(i*16+7, base) \
    GEN_CASE(i*16+8, base) GEN_CASE(i*16+9, base) GEN_CASE(i*16+10, base) GEN_CASE(i*16+11, base) \
    GEN_CASE(i*16+12, base) GEN_CASE(i*16+13, base) GEN_CASE(i*16+14, base) GEN_CASE(i*16+15, base)
GEN_16_CASES(0, 0x100000) GEN_16_CASES(1, 0x100000) GEN_16_CASES(2, 0x100000) GEN_16_CASES(3, 0x100000)
GEN_16_CASES(4, 0x100000) GEN_16_CASES(5, 0x100000) GEN_16_CASES(6, 0x100000) GEN_16_CASES(7, 0x100000)
GEN_16_CASES(8, 0x100000) GEN_16_CASES(9, 0x100000) GEN_16_CASES(10, 0x100000) GEN_16_CASES(11, 0x100000)
GEN_16_CASES(12, 0x100000) GEN_16_CASES(13, 0x100000) GEN_16_CASES(14, 0x100000) GEN_16_CASES(15, 0x100000)
#undef GEN_16_CASES
#undef GEN_CASE
  }
  return acc;
}
__attribute__((noinline))
int64_t hot_code_path_2(int input) {
  int64_t acc = input;
  switch (input & 0xFF) {
#define GEN_CASE(i, base) \
    case i: { acc += base+i*0x100+0; acc ^= base+i*0x100+1; acc *= base+i*0x100+2; acc -= base+i*0x100+3; acc += base+i*0x100+4; acc ^= base+i*0x100+5; acc *= base+i*0x100+6; acc -= base+i*0x100+7; acc += base+i*0x100+8; acc ^= base+i*0x100+9; acc *= base+i*0x100+10; acc -= base+i*0x100+11; acc += base+i*0x100+12; acc ^= base+i*0x100+13; acc *= base+i*0x100+14; acc -= base+i*0x100+15; acc += base+i*0x100+16; acc ^= base+i*0x100+17; acc *= base+i*0x100+18; acc -= base+i*0x100+19; acc += base+i*0x100+20; acc ^= base+i*0x100+21; acc *= base+i*0x100+22; acc -= base+i*0x100+23; acc += base+i*0x100+24; acc ^= base+i*0x100+25; acc *= base+i*0x100+26; acc -= base+i*0x100+27; break; }
#define GEN_16_CASES(i, base) \
    GEN_CASE(i*16+0, base) GEN_CASE(i*16+1, base) GEN_CASE(i*16+2, base) GEN_CASE(i*16+3, base) \
    GEN_CASE(i*16+4, base) GEN_CASE(i*16+5, base) GEN_CASE(i*16+6, base) GEN_CASE(i*16+7, base) \
    GEN_CASE(i*16+8, base) GEN_CASE(i*16+9, base) GEN_CASE(i*16+10, base) GEN_CASE(i*16+11, base) \
    GEN_CASE(i*16+12, base) GEN_CASE(i*16+13, base) GEN_CASE(i*16+14, base) GEN_CASE(i*16+15, base)
GEN_16_CASES(0, 0x200000) GEN_16_CASES(1, 0x200000) GEN_16_CASES(2, 0x200000) GEN_16_CASES(3, 0x200000)
GEN_16_CASES(4, 0x200000) GEN_16_CASES(5, 0x200000) GEN_16_CASES(6, 0x200000) GEN_16_CASES(7, 0x200000)
GEN_16_CASES(8, 0x200000) GEN_16_CASES(9, 0x200000) GEN_16_CASES(10, 0x200000) GEN_16_CASES(11, 0x200000)
GEN_16_CASES(12, 0x200000) GEN_16_CASES(13, 0x200000) GEN_16_CASES(14, 0x200000) GEN_16_CASES(15, 0x200000)
#undef GEN_16_CASES
#undef GEN_CASE
  }
  return acc;
}
} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_ipc_v1, iters) {
    folly::BenchmarkSuspender susp;

    constexpr size_t kDataSize = 4096;
    struct Node {
        Node* next;
        int payload;
    };
    auto node_storage = std::make_unique<std::vector<Node>>(kDataSize);
    std::vector<Node>& data = *node_storage;

    for (size_t i = 0; i < kDataSize; ++i) {
        data[i].next = &data[(i + 37) % kDataSize]; // small prime stride
        data[i].payload = i * 13;
    }
    Node* current = &data[0];

    int64_t (*funcs[])(int) = {hot_code_path_0, hot_code_path_1, hot_code_path_2};
    int64_t total = 0;

    susp.dismiss();

    for (size_t i = 0; i < iters; ++i) {
        total += funcs[i % 3](current->payload);
        current = current->next;
    }

    folly::doNotOptimizeAway(total);
    susp.rehire();
}


int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
