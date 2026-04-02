/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may
 * obtain a copy of the License at
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

#include <numeric>
#include <random>

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

namespace {

struct ChaseNode {
  ChaseNode* next;
  uint64_t payload;
};

#define ALU_OPS_28(v, i, c1, c2, c3, c4)      \
  v += c1 * (i + 1);                          \
  v ^= c2 * (i + 1);                          \
  v *= ((c3 * (i + 1)) | 1);                  \
  v += c4 * (i + 1);                          \
  v += (c1 + 1) * (i + 1);                    \
  v ^= (c2 + 1) * (i + 1);                    \
  v *= (((c3 + 1) * (i + 1)) | 1);            \
  v += (c4 + 1) * (i + 1);                    \
  v += (c1 + 2) * (i + 1);                    \
  v ^= (c2 + 2) * (i + 1);                    \
  v *= (((c3 + 2) * (i + 1)) | 1);            \
  v += (c4 + 2) * (i + 1);                    \
  v += (c1 + 3) * (i + 1);                    \
  v ^= (c2 + 3) * (i + 1);                    \
  v *= (((c3 + 3) * (i + 1)) | 1);            \
  v += (c4 + 3) * (i + 1);                    \
  v += (c1 + 4) * (i + 1);                    \
  v ^= (c2 + 4) * (i + 1);                    \
  v *= (((c3 + 4) * (i + 1)) | 1);            \
  v += (c4 + 4) * (i + 1);                    \
  v += (c1 + 5) * (i + 1);                    \
  v ^= (c2 + 5) * (i + 1);                    \
  v *= (((c3 + 5) * (i + 1)) | 1);            \
  v += (c4 + 5) * (i + 1);                    \
  v += (c1 + 6) * (i + 1);                    \
  v ^= (c2 + 6) * (i + 1);                    \
  v *= (((c3 + 6) * (i + 1)) | 1);            \
  v += (c4 + 6) * (i + 1);

#define GenCase(i, v, c1, c2, c3, c4) \
  case i:                             \
    ALU_OPS_28(v, i, c1, c2, c3, c4)  \
    break;

#define GenCases16(base, v, c1, c2, c3, c4) \
  GenCase(base + 0, v, c1, c2, c3, c4)     \
  GenCase(base + 1, v, c1, c2, c3, c4)     \
  GenCase(base + 2, v, c1, c2, c3, c4)     \
  GenCase(base + 3, v, c1, c2, c3, c4)     \
  GenCase(base + 4, v, c1, c2, c3, c4)     \
  GenCase(base + 5, v, c1, c2, c3, c4)     \
  GenCase(base + 6, v, c1, c2, c3, c4)     \
  GenCase(base + 7, v, c1, c2, c3, c4)     \
  GenCase(base + 8, v, c1, c2, c3, c4)     \
  GenCase(base + 9, v, c1, c2, c3, c4)     \
  GenCase(base + 10, v, c1, c2, c3, c4)    \
  GenCase(base + 11, v, c1, c2, c3, c4)    \
  GenCase(base + 12, v, c1, c2, c3, c4)    \
  GenCase(base + 13, v, c1, c2, c3, c4)    \
  GenCase(base + 14, v, c1, c2, c3, c4)    \
  GenCase(base + 15, v, c1, c2, c3, c4)

#define GenCases256(v, c1, c2, c3, c4)  \
  GenCases16(0, v, c1, c2, c3, c4)      \
  GenCases16(16, v, c1, c2, c3, c4)     \
  GenCases16(32, v, c1, c2, c3, c4)     \
  GenCases16(48, v, c1, c2, c3, c4)     \
  GenCases16(64, v, c1, c2, c3, c4)     \
  GenCases16(80, v, c1, c2, c3, c4)     \
  GenCases16(96, v, c1, c2, c3, c4)     \
  GenCases16(112, v, c1, c2, c3, c4)    \
  GenCases16(128, v, c1, c2, c3, c4)    \
  GenCases16(144, v, c1, c2, c3, c4)    \
  GenCases16(160, v, c1, c2, c3, c4)    \
  GenCases16(176, v, c1, c2, c3, c4)    \
  GenCases16(192, v, c1, c2, c3, c4)    \
  GenCases16(208, v, c1, c2, c3, c4)    \
  GenCases16(224, v, c1, c2, c3, c4)    \
  GenCases16(240, v, c1, c2, c3, c4)

__attribute__((noinline)) uint64_t
icache_func0(uint64_t val, uint8_t switch_val) {
  switch (switch_val) {
    GenCases256(val, 0x10, 0x20, 0x30, 0x40)
  }
  return val;
}

__attribute__((noinline)) uint64_t
icache_func1(uint64_t val, uint8_t switch_val) {
  switch (switch_val) {
    GenCases256(val, 0x50, 0x60, 0x70, 0x80)
  }
  return val;
}

__attribute__((noinline)) uint64_t
icache_func2(uint64_t val, uint8_t switch_val) {
  switch (switch_val) {
    GenCases256(val, 0x90, 0xA0, 0xB0, 0xC0)
  }
  return val;
}
} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_icache_v7, iters) {
  folly::BenchmarkSuspender susp;

  constexpr size_t kNumNodes = 65536;
  std::vector<ChaseNode> nodes(kNumNodes);
  std::vector<int> indices(kNumNodes);
  std::iota(indices.begin(), indices.end(), 0);

  std::mt19937 g(12345);
  std::shuffle(indices.begin(), indices.end(), g);

  for (size_t i = 0; i < kNumNodes; ++i) {
    nodes[indices[i]].next = &nodes[indices[(i + 1) % kNumNodes]];
    nodes[indices[i]].payload = g();
  }

  ChaseNode* current = &nodes[0];
  volatile uint64_t accum = 0;

  susp.dismiss();

  for (size_t j = 0; j < iters; ++j) {
    uint8_t switch_idx = current->payload & 0xFF;
    switch (j % 3) {
      case 0:
        accum = icache_func0(accum, switch_idx);
        break;
      case 1:
        accum = icache_func1(accum, switch_idx);
        break;
      case 2:
        accum = icache_func2(accum, switch_idx);
        break;
    }
    current = current->next;
  }

  folly::doNotOptimizeAway(accum);
}


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
