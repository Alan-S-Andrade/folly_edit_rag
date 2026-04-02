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

#include <folly/Random.h>
#include <vector>

using namespace apache::thrift;
using namespace thrift::benchmark;

namespace {
struct Node {
  Node* next;
  int data;
};

// These functions are designed to have a large instruction footprint to
// stress the instruction cache. They consist of a 256-way switch statement,
// and each case has a sequence of arithmetic operations. Three similar but
// distinct functions are used and rotated through to prevent the instruction
// cache from getting too hot.
// The number of ALU ops is tuned to match the L1i-MPKI target.
// With ~5 ops per pair of lines, and 6 pairs, we get ~30 ops per case.
#define CASE_BLOCK(i, base_op1, base_op2)                                      \
  case i:                                                                      \
    result = (result + (base_op1 + i)) * (base_op2 + i);                       \
    result ^= (result + (base_op1 + i + 1)) * (base_op2 + i + 1);              \
    result = (result + (base_op1 + i + 2)) * (base_op2 + i + 2);               \
    result ^= (result + (base_op1 + i + 3)) * (base_op2 + i + 3);              \
    result = (result + (base_op1 + i + 4)) * (base_op2 + i + 4);               \
    result ^= (result + (base_op1 + i + 5)) * (base_op2 + i + 5);              \
    result = (result + (base_op1 + i + 6)) * (base_op2 + i + 6);               \
    result ^= (result + (base_op1 + i + 7)) * (base_op2 + i + 7);              \
    result = (result + (base_op1 + i + 8)) * (base_op2 + i + 8);               \
    result ^= (result + (base_op1 + i + 9)) * (base_op2 + i + 9);              \
    result = (result + (base_op1 + i + 10)) * (base_op2 + i + 10);             \
    result ^= (result + (base_op1 + i + 11)) * (base_op2 + i + 11);            \
    break;

#define GEN_CASES_4(i, base_op1, base_op2)                                     \
  CASE_BLOCK(i, base_op1, base_op2)                                            \
  CASE_BLOCK(i + 1, base_op1, base_op2)                                        \
  CASE_BLOCK(i + 2, base_op1, base_op2)                                        \
  CASE_BLOCK(i + 3, base_op1, base_op2)
#define GEN_CASES_16(i, base_op1, base_op2)                                    \
  GEN_CASES_4(i, base_op1, base_op2)                                           \
  GEN_CASES_4(i + 4, base_op1, base_op2)                                       \
  GEN_CASES_4(i + 8, base_op1, base_op2)                                       \
  GEN_CASES_4(i + 12, base_op1, base_op2)
#define GEN_CASES_64(i, base_op1, base_op2)                                    \
  GEN_CASES_16(i, base_op1, base_op2)                                          \
  GEN_CASES_16(i + 16, base_op1, base_op2)                                     \
  GEN_CASES_16(i + 32, base_op1, base_op2)                                     \
  GEN_CASES_16(i + 48, base_op1, base_op2)
#define GEN_CASES_256(base_op1, base_op2)                                      \
  GEN_CASES_64(0, base_op1, base_op2)                                          \
  GEN_CASES_64(64, base_op1, base_op2)                                         \
  GEN_CASES_64(128, base_op1, base_op2)                                        \
  GEN_CASES_64(192, base_op1, base_op2)

__attribute__((noinline)) int process_data_0(int x, int y) {
  int result = x;
  switch (y) {
    GEN_CASES_256(100, 200)
  }
  return result;
}

__attribute__((noinline)) int process_data_1(int x, int y) {
  int result = x;
  switch (y) {
    GEN_CASES_256(300, 400)
  }
  return result;
}

__attribute__((noinline)) int process_data_2(int x, int y) {
  int result = x;
  switch (y) {
    GEN_CASES_256(500, 600)
  }
  return result;
}

#undef CASE_BLOCK
#undef GEN_CASES_4
#undef GEN_CASES_16
#undef GEN_CASES_64
#undef GEN_CASES_256

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

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_Icache, iters) {
  folly::BenchmarkSuspender susp;

  constexpr size_t num_nodes = 4096;
  std::vector<Node> nodes(num_nodes);
  for (size_t i = 0; i < num_nodes; ++i) {
    nodes[i].next = &nodes[(i + 1) % num_nodes];
    nodes[i].data = folly::Random::rand32();
  }
  Node* current = &nodes[0];
  int accumulator = 0;

  using ProcessFunc = int (*)(int, int);
  ProcessFunc funcs[] = {process_data_0, process_data_1, process_data_2};

  susp.dismiss();

  for (size_t i = 0; i < iters; ++i) {
    current = current->next;
    int switch_val = current->data & 0xFF;
    accumulator = funcs[i % 3](accumulator, switch_val);
  }

  folly::doNotOptimizeAway(accumulator);
  susp.rehire();
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
