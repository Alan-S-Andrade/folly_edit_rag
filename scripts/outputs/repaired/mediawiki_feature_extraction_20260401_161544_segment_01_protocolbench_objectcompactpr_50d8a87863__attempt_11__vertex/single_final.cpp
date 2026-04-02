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
#include <cstdint>
#include <memory>
#include <numeric>
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
#define CASE_CHUNK(acc, i, c1, c2, c3, c4, c5, c6)                              \
  acc += i + c1;                                                               \
  acc *= c2;                                                                   \
  acc ^= c3;                                                                   \
  acc -= i + c4;                                                               \
  acc |= c5;                                                                   \
  acc &= c6;

#define CASE_BODY(i, s1, s2, s3, s4)                                           \
  case i: {                                                                    \
    CASE_CHUNK(acc, i, (s1 + i), (s2 + i), (s3 + i), (s4 + i), (s1* i), (s2* i));  \
    CASE_CHUNK(acc, i, (s1 - i), (s2 - i), (s3 - i), (s4 - i), (s3* i), (s4* i));  \
    CASE_CHUNK(acc, i, (s1 ^ i), (s2 ^ i), (s3 ^ i), (s4 ^ i), (s1& i), (s2& i));  \
    CASE_CHUNK(acc, i, (s1 | i), (s2 | i), (s3 | i), (s4 | i), (s3& i), (s4& i));  \
    break;                                                                     \
  }

#define C0(j, s1, s2, s3, s4) CASE_BODY(j + 0, s1, s2, s3, s4)
#define C1(j, s1, s2, s3, s4) C0(j, s1, s2, s3, s4) C0(j + 1, s1, s2, s3, s4)
#define C2(j, s1, s2, s3, s4) C1(j, s1, s2, s3, s4) C1(j + 2, s1, s2, s3, s4)
#define C3(j, s1, s2, s3, s4) C2(j, s1, s2, s3, s4) C2(j + 4, s1, s2, s3, s4)
#define C4(j, s1, s2, s3, s4) C3(j, s1, s2, s3, s4) C3(j + 8, s1, s2, s3, s4)
#define C5(j, s1, s2, s3, s4) C4(j, s1, s2, s3, s4) C4(j + 16, s1, s2, s3, s4)
#define C6(j, s1, s2, s3, s4) C5(j, s1, s2, s3, s4) C5(j + 32, s1, s2, s3, s4)
#define C7(j, s1, s2, s3, s4) C6(j, s1, s2, s3, s4) C6(j + 64, s1, s2, s3, s4)
#define ALL_CASES_256(s1, s2, s3, s4)                                          \
  C7(0, s1, s2, s3, s4) C7(128, s1, s2, s3, s4)

template <int N>
__attribute__((noinline)) int64_t hotspot_func(uint8_t val) {
  int64_t acc = val;
  switch (val) {
    ALL_CASES_256(N, N + 1, N + 2, N + 3)
    default:
      break;
  }
  return acc;
}

#undef CASE_CHUNK
#undef CASE_BODY
#undef C0
#undef C1
#undef C2
#undef C3
#undef C4
#undef C5
#undef C6
#undef C7
#undef ALL_CASES_256

auto hotspot_func0 = hotspot_func<0>;
auto hotspot_func1 = hotspot_func<1000>;
auto hotspot_func2 = hotspot_func<2000>;

} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v11, iters) {
  folly::BenchmarkSuspender susp;

  constexpr size_t kNumNodes = 16 * 1024;
  struct ChaseNode {
    uint64_t payload;
    ChaseNode* next;
  };

  auto nodes = std::make_unique<ChaseNode[]>(kNumNodes);
  std::vector<size_t> indices(kNumNodes);
  std::iota(indices.begin(), indices.end(), 0);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  for (size_t i = 0; i < kNumNodes; ++i) {
    nodes[indices[i]].next = &nodes[indices[(i + 1) % kNumNodes]];
    nodes[indices[i]].payload = g();
  }

  ChaseNode* current = &nodes[0];
  int64_t sink = 0;

  susp.dismiss();

  for (size_t j = 0; j < iters; ++j) {
    uint8_t switch_val = current->payload & 0xFF;
    switch (j % 3) {
      case 0:
        sink += hotspot_func0(switch_val);
        break;
      case 1:
        sink += hotspot_func1(switch_val);
        break;
      case 2:
        sink += hotspot_func2(switch_val);
        break;
    }
    current = current->next;
  }

  folly::doNotOptimizeAway(sink);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
