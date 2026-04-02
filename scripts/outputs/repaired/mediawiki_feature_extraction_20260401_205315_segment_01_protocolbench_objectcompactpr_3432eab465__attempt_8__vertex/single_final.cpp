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

#include <glog/logging.h>
#include <folly/Benchmark.h>
#include <folly/BenchmarkUtil.h>
#include <folly/Optional.h>
#include <folly/init/Init.h>
#include <folly/portability/GFlags.h>

using namespace apache::thrift;
using namespace thrift::benchmark;

#define ALU_OPS_28(acc, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14) \
  acc += C1; acc ^= C2; acc *= C3; acc += C4; acc ^= C5; acc *= C6; acc += C7; \
  acc ^= C8; acc *= C9; acc += C10; acc ^= C11; acc *= C12; acc += C13; acc ^= C14; \
  acc += C14; acc ^= C13; acc *= C12; acc += C11; acc ^= C10; acc *= C9; acc += C8; \
  acc ^= C7; acc *= C6; acc += C5; acc ^= C4; acc *= C3; acc += C2; acc ^= C1;

#define SWITCH_CASE_IMPL(i, acc, offset) \
  case i: \
    ALU_OPS_28(acc, (i + offset) * 14 + 1, (i + offset) * 14 + 2, (i + offset) * 14 + 3, (i + offset) * 14 + 4, (i + offset) * 14 + 5, (i + offset) * 14 + 6, (i + offset) * 14 + 7, (i + offset) * 14 + 8, (i + offset) * 14 + 9, (i + offset) * 14 + 10, (i + offset) * 14 + 11, (i + offset) * 14 + 12, (i + offset) * 14 + 13, (i + offset) * 14 + 14); \
    break;

#define SWITCH_CHUNK_4(i, acc, offset) \
    SWITCH_CASE_IMPL(i, acc, offset) \
    SWITCH_CASE_IMPL(i+1, acc, offset) \
    SWITCH_CASE_IMPL(i+2, acc, offset) \
    SWITCH_CASE_IMPL(i+3, acc, offset)

#define SWITCH_CHUNK_16(i, acc, offset) \
    SWITCH_CHUNK_4(i, acc, offset) \
    SWITCH_CHUNK_4(i+4, acc, offset) \
    SWITCH_CHUNK_4(i+8, acc, offset) \
    SWITCH_CHUNK_4(i+12, acc, offset)

#define SWITCH_CHUNK_64(i, acc, offset) \
    SWITCH_CHUNK_16(i, acc, offset) \
    SWITCH_CHUNK_16(i+16, acc, offset) \
    SWITCH_CHUNK_16(i+32, acc, offset) \
    SWITCH_CHUNK_16(i+48, acc, offset)

#define SWITCH_CHUNK_256(i, acc, offset) \
    SWITCH_CHUNK_64(i, acc, offset) \
    SWITCH_CHUNK_64(i+64, acc, offset) \
    SWITCH_CHUNK_64(i+128, acc, offset) \
    SWITCH_CHUNK_64(i+192, acc, offset)

__attribute__((noinline))
uint64_t large_switch_func_0(uint64_t acc, uint8_t selector) {
  switch (selector) {
    SWITCH_CHUNK_256(0, acc, 0)
  }
  return acc;
}

__attribute__((noinline))
uint64_t large_switch_func_1(uint64_t acc, uint8_t selector) {
  switch (selector) {
    SWITCH_CHUNK_256(0, acc, 256)
  }
  return acc;
}

__attribute__((noinline))
uint64_t large_switch_func_2(uint64_t acc, uint8_t selector) {
  switch (selector) {
    SWITCH_CHUNK_256(0, acc, 512)
  }
  return acc;
}

#undef ALU_OPS_28
#undef SWITCH_CASE_IMPL
#undef SWITCH_CHUNK_4
#undef SWITCH_CHUNK_16
#undef SWITCH_CHUNK_64
#undef SWITCH_CHUNK_256

struct Node {
  Node* next;
  uint64_t payload;
};

const size_t kNodeCount_icache_v8 = 1 << 16;
static std::vector<Node> chase_nodes_icache_v8;
static Node* chase_head_icache_v8 = nullptr;

void setup_chase_array_icache_v8() {
  if (chase_head_icache_v8) return;

  chase_nodes_icache_v8.resize(kNodeCount_icache_v8);
  std::vector<size_t> indices(kNodeCount_icache_v8);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 g(folly::randomNumberSeed());
  std::shuffle(indices.begin(), indices.end(), g);

  for (size_t i = 0; i < kNodeCount_icache_v8; ++i) {
    chase_nodes_icache_v8[indices[i]].next = &chase_nodes_icache_v8[indices[(i + 1) % kNodeCount_icache_v8]];
    chase_nodes_icache_v8[indices[i]].payload = folly::Random::rand64(g);
  }
  chase_head_icache_v8 = &chase_nodes_icache_v8[0];
}


BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_icache_v8, iters) {
  folly::BenchmarkSuspender susp;
  setup_chase_array_icache_v8();
  auto* p = chase_head_icache_v8;
  uint64_t acc = 0;
  susp.dismiss();

  for (size_t i = 0; i < iters; ++i) {
    p = p->next;
    uint8_t selector = p->payload & 0xFF;
    switch (i % 3) {
      case 0:
        acc = large_switch_func_0(acc, selector);
        break;
      case 1:
        acc = large_switch_func_1(acc, selector);
        break;
      case 2:
        acc = large_switch_func_2(acc, selector);
        break;
    }
  }

  folly::doNotOptimizeAway(acc);
  folly::doNotOptimizeAway(p);
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
