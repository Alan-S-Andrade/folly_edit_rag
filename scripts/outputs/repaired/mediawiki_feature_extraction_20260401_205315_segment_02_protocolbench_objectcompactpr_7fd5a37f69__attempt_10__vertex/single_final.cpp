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

#include <cstdint>
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
  uint64_t payload;
};

#define OP_3(acc, c) \
  acc += c;          \
  acc ^= (c + 1);    \
  acc *= (c + 2);
#define OPS_9(acc, c) \
  OP_3(acc, c);       \
  OP_3(acc, c + 3);   \
  OP_3(acc, c + 6);
#define OPS_27(acc, c) \
  OPS_9(acc, c);       \
  OPS_9(acc, c + 9);   \
  OPS_9(acc, c + 18);

#define CASE(i, acc, C)      \
  case i:                    \
    OPS_27(acc, C + i * 30); \
    break;

#define CASES_16(i, acc, C)  \
  CASE(i + 0, acc, C)        \
  CASE(i + 1, acc, C)        \
  CASE(i + 2, acc, C)        \
  CASE(i + 3, acc, C)        \
  CASE(i + 4, acc, C)        \
  CASE(i + 5, acc, C)        \
  CASE(i + 6, acc, C)        \
  CASE(i + 7, acc, C)        \
  CASE(i + 8, acc, C)        \
  CASE(i + 9, acc, C)        \
  CASE(i + 10, acc, C)       \
  CASE(i + 11, acc, C)       \
  CASE(i + 12, acc, C)       \
  CASE(i + 13, acc, C)       \
  CASE(i + 14, acc, C)       \
  CASE(i + 15, acc, C)

#define CASES_256(acc, C)     \
  CASES_16(0, acc, C)         \
  CASES_16(16, acc, C)        \
  CASES_16(32, acc, C)        \
  CASES_16(48, acc, C)        \
  CASES_16(64, acc, C)        \
  CASES_16(80, acc, C)        \
  CASES_16(96, acc, C)        \
  CASES_16(112, acc, C)       \
  CASES_16(128, acc, C)       \
  CASES_16(144, acc, C)       \
  CASES_16(160, acc, C)       \
  CASES_16(176, acc, C)       \
  CASES_16(192, acc, C)       \
  CASES_16(208, acc, C)       \
  CASES_16(224, acc, C)       \
  CASES_16(240, acc, C)

__attribute__((noinline)) uint64_t do_work_0(uint64_t acc, uint8_t val) {
  switch (val) {
    CASES_256(acc, 1000)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) uint64_t do_work_1(uint64_t acc, uint8_t val) {
  switch (val) {
    CASES_256(acc, 2000000)
    default:
      break;
  }
  return acc;
}

__attribute__((noinline)) uint64_t do_work_2(uint64_t acc, uint8_t val) {
  switch (val) {
    CASES_256(acc, 4000000)
    default:
      break;
  }
  return acc;
}

} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_ipc, iters) {
  folly::BenchmarkSuspender susp;

  struct State {
    std::vector<Node> nodes;
    Node* current;
    State() : nodes(1 << 16) {
      const size_t num_nodes = nodes.size();
      std::vector<size_t> indices(num_nodes);
      std::iota(indices.begin(), indices.end(), 0);
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(indices.begin(), indices.end(), g);

      for (size_t i = 0; i < num_nodes; ++i) {
        nodes[indices[i]].next = &nodes[indices[(i + 1) % num_nodes]];
        nodes[indices[i]].payload = (i * 1234567891ULL) ^ 987654321ULL;
      }
      current = &nodes[indices[0]];
    }
  };
  static State state;

  uint64_t acc = 0;
  susp.dismiss();

  while (iters--) {
    uint64_t local_acc = 0;
    for (int j = 0; j < 300; ++j) {
      uint8_t switch_val = state.current->payload & 0xFF;
      switch (j % 3) {
        case 0:
          local_acc = do_work_0(local_acc, switch_val);
          break;
        case 1:
          local_acc = do_work_1(local_acc, switch_val);
          break;
        case 2:
          local_acc = do_work_2(local_acc, switch_val);
          break;
      }
      state.current = state.current->next;
    }
    acc += local_acc;
  }

  susp.rehire();
  folly::doNotOptimizeAway(acc);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
