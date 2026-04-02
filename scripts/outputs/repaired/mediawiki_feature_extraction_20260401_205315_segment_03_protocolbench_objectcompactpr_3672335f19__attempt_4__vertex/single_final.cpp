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

namespace {

// I-cache-heavy workload to increase L1-icache-load-misses_MPKI
#define OPS_V4(acc, c)                                     \
  acc += c + 1;                                            \
  acc ^= c + 2;                                            \
  acc *= c + 3;                                            \
  acc += c + 4;                                            \
  acc ^= c + 5;                                            \
  acc += c + 6;                                            \
  acc ^= c + 7;                                            \
  acc *= c + 8;                                            \
  acc += c + 9;                                            \
  acc ^= c + 10;                                           \
  acc += c + 11;                                           \
  acc ^= c + 12;                                           \
  acc *= c + 13;                                           \
  acc += c + 14;                                           \
  acc ^= c + 15;                                           \
  acc += c + 16;                                           \
  acc ^= c + 17;                                           \
  acc *= c + 18;                                           \
  acc += c + 19;                                           \
  acc ^= c + 20;                                           \
  acc += c + 21;                                           \
  acc ^= c + 22;                                           \
  acc *= c + 23;                                           \
  acc += c + 24;                                           \
  acc ^= c + 25;                                           \
  acc += c + 26;                                           \
  acc ^= c + 27;                                           \
  acc *= c + 28;

#define CASES_16_V4(i, func_offset)                                 \
  case i + 0:                                                       \
    OPS_V4(acc, (i + 0) * 28 + func_offset);                        \
    break;                                                          \
  case i + 1:                                                       \
    OPS_V4(acc, (i + 1) * 28 + func_offset);                        \
    break;                                                          \
  case i + 2:                                                       \
    OPS_V4(acc, (i + 2) * 28 + func_offset);                        \
    break;                                                          \
  case i + 3:                                                       \
    OPS_V4(acc, (i + 3) * 28 + func_offset);                        \
    break;                                                          \
  case i + 4:                                                       \
    OPS_V4(acc, (i + 4) * 28 + func_offset);                        \
    break;                                                          \
  case i + 5:                                                       \
    OPS_V4(acc, (i + 5) * 28 + func_offset);                        \
    break;                                                          \
  case i + 6:                                                       \
    OPS_V4(acc, (i + 6) * 28 + func_offset);                        \
    break;                                                          \
  case i + 7:                                                       \
    OPS_V4(acc, (i + 7) * 28 + func_offset);                        \
    break;                                                          \
  case i + 8:                                                       \
    OPS_V4(acc, (i + 8) * 28 + func_offset);                        \
    break;                                                          \
  case i + 9:                                                       \
    OPS_V4(acc, (i + 9) * 28 + func_offset);                        \
    break;                                                          \
  case i + 10:                                                      \
    OPS_V4(acc, (i + 10) * 28 + func_offset);                       \
    break;                                                          \
  case i + 11:                                                      \
    OPS_V4(acc, (i + 11) * 28 + func_offset);                       \
    break;                                                          \
  case i + 12:                                                      \
    OPS_V4(acc, (i + 12) * 28 + func_offset);                       \
    break;                                                          \
  case i + 13:                                                      \
    OPS_V4(acc, (i + 13) * 28 + func_offset);                       \
    break;                                                          \
  case i + 14:                                                      \
    OPS_V4(acc, (i + 14) * 28 + func_offset);                       \
    break;                                                          \
  case i + 15:                                                      \
    OPS_V4(acc, (i + 15) * 28 + func_offset);                       \
    break;

#define ALL_CASES_V4(func_offset) \
  CASES_16_V4(0, func_offset)     \
  CASES_16_V4(16, func_offset)    \
  CASES_16_V4(32, func_offset)    \
  CASES_16_V4(48, func_offset)    \
  CASES_16_V4(64, func_offset)    \
  CASES_16_V4(80, func_offset)    \
  CASES_16_V4(96, func_offset)    \
  CASES_16_V4(112, func_offset)   \
  CASES_16_V4(128, func_offset)   \
  CASES_16_V4(144, func_offset)   \
  CASES_16_V4(160, func_offset)   \
  CASES_16_V4(176, func_offset)   \
  CASES_16_V4(192, func_offset)   \
  CASES_16_V4(208, func_offset)   \
  CASES_16_V4(224, func_offset)   \
  CASES_16_V4(240, func_offset)

__attribute__((noinline)) int do_work_v4_0(int x, int y) {
  int acc = y;
  switch (x) {
    ALL_CASES_V4(0)
  }
  return acc;
}
__attribute__((noinline)) int do_work_v4_1(int x, int y) {
  int acc = y;
  switch (x) {
    ALL_CASES_V4(1000000)
  }
  return acc;
}
__attribute__((noinline)) int do_work_v4_2(int x, int y) {
  int acc = y;
  switch (x) {
    ALL_CASES_V4(2000000)
  }
  return acc;
}

#undef OPS_V4
#undef CASES_16_V4
#undef ALL_CASES_V4

struct Node_v4 {
  Node_v4* next;
  unsigned char payload;
};

template <
    SerializerMethod kSerializerMethod,
    typename Serializer,
    typename Struct>
void readBench_LargeSetInt_v4(size_t iters) {
  folly::BenchmarkSuspender susp;
  auto strct = create<Struct>();
  folly::IOBufQueue q;
  Serializer::serialize(strct, &q);
  auto buf = q.move();
  buf->coalesce();

  constexpr size_t kNumNodes = 2048;
  std::vector<Node_v4> nodes(kNumNodes);
  std::vector<int> indices(kNumNodes);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 g(1); // deterministic seed
  std::shuffle(indices.begin(), indices.end(), g);

  for (size_t i = 0; i < kNumNodes; ++i) {
    nodes[i].next = &nodes[indices[i]];
    nodes[i].payload = indices[i] & 0xff;
  }
  Node_v4* current_node = &nodes[0];
  int acc = 0;

  susp.dismiss();

  size_t j = 0;
  for (size_t i = 0; i < iters; ++i) {
    if constexpr (kSerializerMethod == SerializerMethod::Object) {
      auto obj = protocol::parseObject<GetReader<Serializer>>(*buf);
      folly::doNotOptimizeAway(obj);
    } else {
      Struct data;
      Serializer::deserialize(buf.get(), data);
      folly::doNotOptimizeAway(data);
    }

    // Spin to increase I-cache misses
    for (int k = 0; k < 15; ++k) {
      current_node = current_node->next;
      switch (j % 3) {
        case 0:
          acc = do_work_v4_0(current_node->payload, acc);
          break;
        case 1:
          acc = do_work_v4_1(current_node->payload, acc);
          break;
        case 2:
          acc = do_work_v4_2(current_node->payload, acc);
          break;
      }
      j++;
    }
  }
  folly::doNotOptimizeAway(acc);
  susp.rehire();
}

} // namespace
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

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v4, iters) {
  readBench_LargeSetInt_v4<
      SerializerMethod::Object,
      CompactSerializer,
      LargeSetInt>(iters);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
