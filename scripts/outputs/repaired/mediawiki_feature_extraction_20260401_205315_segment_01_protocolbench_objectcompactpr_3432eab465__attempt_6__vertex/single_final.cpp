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
// For a new benchmark to match L1-icache-load-misses_MPKI
volatile uint64_t L1i_sink;

struct ChaseNode {
  ChaseNode* next;
  uint8_t payload;
};

std::vector<ChaseNode> chase_nodes;
ChaseNode* chase_head = nullptr;

void setup_chase_chain() {
  constexpr size_t num_nodes = 4096;
  if (chase_nodes.empty()) {
    chase_nodes.resize(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
      chase_nodes[i].payload = static_cast<uint8_t>(i);
    }
    std::mt19937 g(12345); // deterministic shuffle
    std::shuffle(chase_nodes.begin(), chase_nodes.end(), g);
    for (size_t i = 0; i < num_nodes - 1; ++i) {
      chase_nodes[i].next = &chase_nodes[i + 1];
    }
    chase_nodes[num_nodes - 1].next = &chase_nodes[0];
    chase_head = &chase_nodes[0];
  }
}

// Macro to generate a case block with N ALU ops
#define ALU_OPS_28(i, off)                                          \
  acc += (i + off + 0) * 0x123456789ABCDEF0ULL;                      \
  acc ^= (i + off + 1) * 0xFEDCBA9876543210ULL;                      \
  acc *= (i + off + 2) * 0x13579BDF02468ACEULL;                      \
  acc += (i + off + 3) * 0xACE86420FDB97531ULL;                      \
  acc ^= (i + off + 4) * 0x5555555555555555ULL;                      \
  acc *= (i + off + 5) * 0xAAAAAAAAAAAAAAAAULL;                      \
  acc += (i + off + 6) * 0x1111111111111111ULL;                      \
  acc ^= (i + off + 7) * 0x2222222222222222ULL;                      \
  acc *= (i + off + 8) * 0x3333333333333333ULL;                      \
  acc += (i + off + 9) * 0x4444444444444444ULL;                      \
  acc ^= (i + off + 10) * 0x5555555555555555ULL;                     \
  acc *= (i + off + 11) * 0x6666666666666666ULL;                     \
  acc += (i + off + 12) * 0x7777777777777777ULL;                     \
  acc ^= (i + off + 13) * 0x8888888888888888ULL;                     \
  acc *= (i + off + 14) * 0x9999999999999999ULL;                     \
  acc += (i + off + 15) * 0x0101010101010101ULL;                     \
  acc ^= (i + off + 16) * 0x2323232323232323ULL;                     \
  acc *= (i + off + 17) * 0x4545454545454545ULL;                     \
  acc += (i + off + 18) * 0x6767676767676767ULL;                     \
  acc ^= (i + off + 19) * 0x8989898989898989ULL;                     \
  acc *= (i + off + 20) * 0xababababababababULL;                     \
  acc += (i + off + 21) * 0xcdcdcdcdcdcdcdcdULL;                     \
  acc ^= (i + off + 22) * 0xefefefefefefefefULL;                     \
  acc *= (i + off + 23) * 0x1212121212121212ULL;                     \
  acc += (i + off + 24) * 0x3434343434343434ULL;                     \
  acc ^= (i + off + 25) * 0x5656565656565656ULL;                     \
  acc *= (i + off + 26) * 0x7878787878787878ULL;                     \
  acc += (i + off + 27) * 0x9a9a9a9a9a9a9a9aULL

#define CASE_BLOCK(i, off) \
  case i:                  \
    ALU_OPS_28(i, off);    \
    break;

#define REPEAT_2(m, i, off) m(i, off) m(i + 1, off)
#define REPEAT_4(m, i, off) \
  REPEAT_2(m, i, off) REPEAT_2(m, i + 2, off)
#define REPEAT_8(m, i, off) \
  REPEAT_4(m, i, off) REPEAT_4(m, i + 4, off)
#define REPEAT_16(m, i, off) \
  REPEAT_8(m, i, off) REPEAT_8(m, i + 8, off)
#define REPEAT_32(m, i, off) \
  REPEAT_16(m, i, off) REPEAT_16(m, i + 16, off)
#define REPEAT_64(m, i, off) \
  REPEAT_32(m, i, off) REPEAT_32(m, i + 32, off)
#define REPEAT_128(m, i, off) \
  REPEAT_64(m, i, off) REPEAT_64(m, i + 64, off)
#define REPEAT_256(m, i, off) \
  REPEAT_128(m, i, off) REPEAT_128(m, i + 128, off)

__attribute__((noinline)) void switch_func_0(uint8_t sel, uint64_t& acc) {
  switch (sel) {
    REPEAT_256(CASE_BLOCK, 0, 0x10)
  }
}

__attribute__((noinline)) void switch_func_1(uint8_t sel, uint64_t& acc) {
  switch (sel) {
    REPEAT_256(CASE_BLOCK, 0, 0x30)
  }
}

__attribute__((noinline)) void switch_func_2(uint8_t sel, uint64_t& acc) {
  switch (sel) {
    REPEAT_256(CASE_BLOCK, 0, 0x50)
  }
}

template <typename Serializer>
void L1iHeavyReadBench(size_t iters) {
  folly::BenchmarkSuspender susp;

  auto strct = create<LargeSetInt>();
  folly::IOBufQueue q;
  Serializer::serialize(strct, &q);
  auto buf = q.move();
  buf->coalesce();

  setup_chase_chain();
  auto* current_node = chase_head;
  uint64_t acc = 0;

  susp.dismiss();

  for (size_t i = 0; i < iters; ++i) {
    // Original work
    auto obj = protocol::parseObject<GetReader<Serializer>>(*buf);
    folly::doNotOptimizeAway(obj);

    // Added work for L1i misses
    for (size_t j = 0; j < 5; ++j) {
      uint8_t sel = current_node->payload;
      current_node = current_node->next;
      switch (j % 3) {
        case 0:
          switch_func_0(sel, acc);
          break;
        case 1:
          switch_func_1(sel, acc);
          break;
        case 2:
          switch_func_2(sel, acc);
          break;
      }
    }
  }

  susp.rehire();
  L1i_sink = acc;
}
} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v6, iters) {
  L1iHeavyReadBench<CompactSerializer>(iters);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
