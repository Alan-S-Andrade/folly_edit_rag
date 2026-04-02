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

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

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

#define ALU_CASE(val, base)                                                      \
  case val: {                                                                    \
    auto& a = *acc;                                                              \
    a += (base) + 1;                                                             \
    a ^= (base) + 2;                                                             \
    a += (base) + 3;                                                             \
    a ^= (base) + 4;                                                             \
    a += (base) + 5;                                                             \
    a ^= (base) + 6;                                                             \
    a += (base) + 7;                                                             \
    a *= (base) + 8;                                                             \
    a += (base) + 9;                                                             \
    a ^= (base) + 10;                                                            \
    a += (base) + 11;                                                            \
    a ^= (base) + 12;                                                            \
    a += (base) + 13;                                                            \
    a ^= (base) + 14;                                                            \
    a *= (base) + 15;                                                            \
    a += (base) + 16;                                                            \
    a += (base) + 17;                                                            \
    a ^= (base) + 18;                                                            \
    a += (base) + 19;                                                            \
    a ^= (base) + 20;                                                            \
    a += (base) + 21;                                                            \
    a ^= (base) + 22;                                                            \
    a *= (base) + 23;                                                            \
    a += (base) + 24;                                                            \
    a += (base) + 25;                                                            \
    a ^= (base) + 26;                                                            \
    a += (base) + 27;                                                            \
    a ^= (base) + 28;                                                            \
    break;                                                                       \
  }

#define GEN_SWITCH_CASES_16(val_base, const_base)                                \
  ALU_CASE(val_base + 0, const_base + (val_base + 0) * 28)                       \
  ALU_CASE(val_base + 1, const_base + (val_base + 1) * 28)                       \
  ALU_CASE(val_base + 2, const_base + (val_base + 2) * 28)                       \
  ALU_CASE(val_base + 3, const_base + (val_base + 3) * 28)                       \
  ALU_CASE(val_base + 4, const_base + (val_base + 4) * 28)                       \
  ALU_CASE(val_base + 5, const_base + (val_base + 5) * 28)                       \
  ALU_CASE(val_base + 6, const_base + (val_base + 6) * 28)                       \
  ALU_CASE(val_base + 7, const_base + (val_base + 7) * 28)                       \
  ALU_CASE(val_base + 8, const_base + (val_base + 8) * 28)                       \
  ALU_CASE(val_base + 9, const_base + (val_base + 9) * 28)                       \
  ALU_CASE(val_base + 10, const_base + (val_base + 10) * 28)                     \
  ALU_CASE(val_base + 11, const_base + (val_base + 11) * 28)                     \
  ALU_CASE(val_base + 12, const_base + (val_base + 12) * 28)                     \
  ALU_CASE(val_base + 13, const_base + (val_base + 13) * 28)                     \
  ALU_CASE(val_base + 14, const_base + (val_base + 14) * 28)                     \
  ALU_CASE(val_base + 15, const_base + (val_base + 15) * 28)

#define GEN_SWITCH_CASES_256(const_base)                                         \
  GEN_SWITCH_CASES_16(0, const_base)                                             \
  GEN_SWITCH_CASES_16(16, const_base)                                            \
  GEN_SWITCH_CASES_16(32, const_base)                                            \
  GEN_SWITCH_CASES_16(48, const_base)                                            \
  GEN_SWITCH_CASES_16(64, const_base)                                            \
  GEN_SWITCH_CASES_16(80, const_base)                                            \
  GEN_SWITCH_CASES_16(96, const_base)                                            \
  GEN_SWITCH_CASES_16(112, const_base)                                           \
  GEN_SWITCH_CASES_16(128, const_base)                                           \
  GEN_SWITCH_CASES_16(144, const_base)                                           \
  GEN_SWITCH_CASES_16(160, const_base)                                           \
  GEN_SWITCH_CASES_16(176, const_base)                                           \
  GEN_SWITCH_CASES_16(192, const_base)                                           \
  GEN_SWITCH_CASES_16(208, const_base)                                           \
  GEN_SWITCH_CASES_16(224, const_base)                                           \
  GEN_SWITCH_CASES_16(240, const_base)

namespace {
__attribute__((noinline)) void
generated_switch_func_0(uint8_t val, volatile int* acc) {
  switch (val) {
    GEN_SWITCH_CASES_256(10000)
  }
}
__attribute__((noinline)) void
generated_switch_func_1(uint8_t val, volatile int* acc) {
  switch (val) {
    GEN_SWITCH_CASES_256(20000)
  }
}
__attribute__((noinline)) void
generated_switch_func_2(uint8_t val, volatile int* acc) {
  switch (val) {
    GEN_SWITCH_CASES_256(30000)
  }
}

using switch_func_t = void (*)(uint8_t, volatile int*);
switch_func_t funcs[] = {
    &generated_switch_func_0,
    &generated_switch_func_1,
    &generated_switch_func_2};

struct alignas(64) Node {
  Node* next;
  uint8_t payload;
};

constexpr size_t kNumNodes = 65536;
std::vector<Node> nodes;
Node* start_node = nullptr;

void setup_pointer_chase() {
  if (start_node) {
    return;
  }
  nodes.resize(kNumNodes);
  std::vector<size_t> indices(kNumNodes);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 g(1); // deterministic for benchmark
  std::shuffle(indices.begin(), indices.end(), g);

  for (size_t i = 0; i < kNumNodes; ++i) {
    nodes[indices[i]].next = &nodes[indices[(i + 1) % kNumNodes]];
    nodes[indices[i]].payload = (indices[i] * 37) & 0xFF;
  }
  start_node = &nodes[indices[0]];
}
} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_icache_v1, iters) {
  folly::BenchmarkSuspender susp;
  setup_pointer_chase();
  Node* p = start_node;
  volatile int accumulator = 0;

  susp.dismiss();

  for (size_t i = 0; i < iters; ++i) {
    p = p->next;
    uint8_t payload = p->payload;
    funcs[i % 3](payload, &accumulator);
  }

  susp.rehire();
  folly::doNotOptimizeAway(p);
  folly::doNotOptimizeAway(accumulator);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
