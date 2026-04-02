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

#include <memory>

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

namespace {

template <int I, int F>
inline int64_t alu_heavy_op(int64_t acc) {
  // 28 ops with unique constants based on I (case) and F (function)
  constexpr int64_t base_offset = F * 256 * 56;
  constexpr int64_t case_offset = I * 56;
  acc += (base_offset + case_offset + 1);
  acc ^= (base_offset + case_offset + 2);
  acc += (base_offset + case_offset + 3);
  acc ^= (base_offset + case_offset + 4);
  acc += (base_offset + case_offset + 5);
  acc ^= (base_offset + case_offset + 6);
  acc += (base_offset + case_offset + 7);
  acc ^= (base_offset + case_offset + 8);
  acc += (base_offset + case_offset + 9);
  acc ^= (base_offset + case_offset + 10);
  acc += (base_offset + case_offset + 11);
  acc ^= (base_offset + case_offset + 12);
  acc += (base_offset + case_offset + 13);
  acc ^= (base_offset + case_offset + 14);
  acc += (base_offset + case_offset + 15);
  acc ^= (base_offset + case_offset + 16);
  acc += (base_offset + case_offset + 17);
  acc ^= (base_offset + case_offset + 18);
  acc += (base_offset + case_offset + 19);
  acc ^= (base_offset + case_offset + 20);
  acc += (base_offset + case_offset + 21);
  acc ^= (base_offset + case_offset + 22);
  acc += (base_offset + case_offset + 23);
  acc ^= (base_offset + case_offset + 24);
  acc += (base_offset + case_offset + 25);
  acc ^= (base_offset + case_offset + 26);
  acc += (base_offset + case_offset + 27);
  acc ^= (base_offset + case_offset + 28);
  return acc;
}

#define CASE(i) \
  case i:     \
    return alu_heavy_op<i, F>(acc)
#define C8(i) \
  CASE(i);    \
  CASE(i + 1);CASE(i + 2);CASE(i + 3);CASE(i + 4);CASE(i + 5);CASE(i + 6);CASE(i + 7)
#define C64(i) \
  C8(i);       \
  C8(i + 8);   \
  C8(i + 16);  \
  C8(i + 24);  \
  C8(i + 32);  \
  C8(i + 40);  \
  C8(i + 48);  \
  C8(i + 56)
#define C256() \
  C64(0);      \
  C64(64);     \
  C64(128);    \
  C64(192)

template <int F>
__attribute__((noinline)) int64_t generated_func(int64_t acc, int selector) {
  switch (selector) {
    C256();
    default:
      return acc;
  }
}

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v5, iters) {
  folly::BenchmarkSuspender susp;
  constexpr size_t kPointerChaseSize = 4096;
  auto chase_buffer = std::make_unique<uintptr_t[]>(kPointerChaseSize);
  for (size_t i = 0; i < kPointerChaseSize; ++i) {
    chase_buffer[i] =
        reinterpret_cast<uintptr_t>(&chase_buffer[(i + 13) % kPointerChaseSize]);
  }
  uintptr_t p = reinterpret_cast<uintptr_t>(&chase_buffer[0]);
  int64_t acc = 0;

  susp.dismiss();

  for (uint64_t i = 0; i < iters; ++i) {
    p = *reinterpret_cast<uintptr_t*>(p);
    int selector = p & 0xff;
    switch (i % 3) {
      case 0:
        acc = generated_func<0>(acc, selector);
        break;
      case 1:
        acc = generated_func<1>(acc, selector);
        break;
      case 2:
        acc = generated_func<2>(acc, selector);
        break;
    }
  }
  folly::doNotOptimizeAway(acc);
  folly::doNotOptimizeAway(p);
}

#undef C256
#undef C64
#undef C8
#undef CASE
} // namespace

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
