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
#include <folly/Random.h>
#include <folly/init/Init.h>
#include <folly/portability/GFlags.h>

#include <cstdint>
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

struct Node {
  Node* next;
  int payload;
};

#define ALU_OPS_IMPL(accu, i, j, salt)                                         \
  accu += salt + 0x1000 * (j) + 0x100 * (i) + 0;                                \
  accu ^= salt + 0x2000 * (j) + 0x100 * (i) + 1;                                \
  accu *= salt + 0x3000 * (j) + 0x100 * (i) + 2;

#define OPS_36(accu, i, salt)                                                  \
  ALU_OPS_IMPL(accu, i, 0, salt)                                               \
  ALU_OPS_IMPL(accu, i, 1, salt)                                               \
  ALU_OPS_IMPL(accu, i, 2, salt)                                               \
  ALU_OPS_IMPL(accu, i, 3, salt)                                               \
  ALU_OPS_IMPL(accu, i, 4, salt)                                               \
  ALU_OPS_IMPL(accu, i, 5, salt)                                               \
  ALU_OPS_IMPL(accu, i, 6, salt)                                               \
  ALU_OPS_IMPL(accu, i, 7, salt)                                               \
  ALU_OPS_IMPL(accu, i, 8, salt)                                               \
  ALU_OPS_IMPL(accu, i, 9, salt)                                               \
  ALU_OPS_IMPL(accu, i, 10, salt)                                              \
  ALU_OPS_IMPL(accu, i, 11, salt)

#define CASE_IMPL(accu, i, salt)                                               \
  case i:                                                                      \
    OPS_36(accu, i, salt);                                                     \
    break;

#define CASES_16(accu, i, salt)                                                \
  CASE_IMPL(accu, i + 0, salt)                                                 \
  CASE_IMPL(accu, i + 1, salt)                                                 \
  CASE_IMPL(accu, i + 2, salt)                                                 \
  CASE_IMPL(accu, i + 3, salt)                                                 \
  CASE_IMPL(accu, i + 4, salt)                                                 \
  CASE_IMPL(accu, i + 5, salt)                                                 \
  CASE_IMPL(accu, i + 6, salt)                                                 \
  CASE_IMPL(accu, i + 7, salt)                                                 \
  CASE_IMPL(accu, i + 8, salt)                                                 \
  CASE_IMPL(accu, i + 9, salt)                                                 \
  CASE_IMPL(accu, i + 10, salt)                                                \
  CASE_IMPL(accu, i + 11, salt)                                                \
  CASE_IMPL(accu, i + 12, salt)                                                \
  CASE_IMPL(accu, i + 13, salt)                                                \
  CASE_IMPL(accu, i + 14, salt)                                                \
  CASE_IMPL(accu, i + 15, salt)

#define CASES_256(accu, salt)                                                  \
  CASES_16(accu, 0, salt)                                                      \
  CASES_16(accu, 16, salt)                                                     \
  CASES_16(accu, 32, salt)                                                     \
  CASES_16(accu, 48, salt)                                                     \
  CASES_16(accu, 64, salt)                                                     \
  CASES_16(accu, 80, salt)                                                     \
  CASES_16(accu, 96, salt)                                                     \
  CASES_16(accu, 112, salt)                                                    \
  CASES_16(accu, 128, salt)                                                    \
  CASES_16(accu, 144, salt)                                                    \
  CASES_16(accu, 160, salt)                                                    \
  CASES_16(accu, 176, salt)                                                    \
  CASES_16(accu, 192, salt)                                                    \
  CASES_16(accu, 208, salt)                                                    \
  CASES_16(accu, 224, salt)                                                    \
  CASES_16(accu, 240, salt)

__attribute__((noinline)) void switch_func_0(
    unsigned long long& accu,
    int payload) {
  switch (payload & 0xFF) {
    CASES_256(accu, 0)
    default:
      break;
  }
}

__attribute__((noinline)) void switch_func_1(
    unsigned long long& accu,
    int payload) {
  switch (payload & 0xFF) {
    CASES_256(accu, 0x100000)
    default:
      break;
  }
}

__attribute__((noinline)) void switch_func_2(
    unsigned long long& accu,
    int payload) {
  switch (payload & 0xFF) {
    CASES_256(accu, 0x200000)
    default:
      break;
  }
}

} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v1, iters) {
  folly::BenchmarkSuspender susp;

  constexpr size_t kNumNodes = 65536;
  std::vector<Node> nodes(kNumNodes);
  for (size_t i = 0; i < kNumNodes; ++i) {
    // Semi-random permutation to defeat prefetchers
    nodes[i].next = &nodes[((uint64_t)i * 3935559000370003845ULL +
                            2691343689449562083ULL) %
                           kNumNodes];
    nodes[i].payload = folly::Random::rand32();
  }

  Node* current = &nodes[0];
  unsigned long long accu = 0;

  void (*funcs[])(unsigned long long&, int) = {
      switch_func_0, switch_func_1, switch_func_2};

  susp.dismiss();

  for (size_t j = 0; j < iters; ++j) {
    funcs[j % 3](accu, current->payload);
    current = current->next;
  }

  folly::doNotOptimizeAway(accu);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
