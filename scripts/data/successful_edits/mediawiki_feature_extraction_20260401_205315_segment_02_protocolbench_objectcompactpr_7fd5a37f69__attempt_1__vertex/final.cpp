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
struct PtrChaseNode {
  PtrChaseNode* next;
  uint64_t payload;
};

constexpr size_t kPtrChaseSize = 64;
std::vector<PtrChaseNode> ptr_chase_nodes(kPtrChaseSize);

void setup_ptr_chase_data() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  for (size_t i = 0; i < kPtrChaseSize; ++i) {
    ptr_chase_nodes[i].next = &ptr_chase_nodes[(i + 1) % kPtrChaseSize];
    ptr_chase_nodes[i].payload =
        (i * 0x9ddfea08eb382d69ULL) ^ 0x2203199374238761ULL;
  }
  initialized = true;
}
} // namespace

template <int N>
__attribute__((noinline)) uint64_t big_switch_func(uint64_t val) {
  uint64_t v = val;
  switch (val & 0xff) {
#define V(op, i, j) (v op((i << 8) + j + N))
#define OP_PAIRS(i)                                                  \
  v = V(+, i, 1);  v = V(^, i, 2);  v = V(*, i, 3);  v = V(+, i, 4);  \
  v = V(^, i, 5);  v = V(*, i, 6);  v = V(+, i, 7);  v = V(^, i, 8);  \
  v = V(*, i, 9);  v = V(+, i, 10); v = V(^, i, 11); v = V(*, i, 12); \
  v = V(+, i, 13); v = V(^, i, 14); v = V(*, i, 15); v = V(+, i, 16); \
  v = V(^, i, 17); v = V(*, i, 18); v = V(+, i, 19); v = V(^, i, 20); \
  v = V(*, i, 21); v = V(+, i, 22); v = V(^, i, 23); v = V(*, i, 24); \
  v = V(+, i, 25); v = V(^, i, 26);

#define R1(I)      \
  case I:          \
    OP_PAIRS(I); \
    break;
#define R4(I) R1(I) R1(I + 1) R1(I + 2) R1(I + 3)
#define R16(I) R4(I) R4(I + 4) R4(I + 8) R4(I + 12)
#define R64(I) R16(I) R16(I + 16) R16(I + 32) R16(I + 48)
#define R256(I) R64(I) R64(I + 64) R64(I + 128) R64(I + 192)
    R256(0)
    default:
      break;
#undef V
#undef OP_PAIRS
#undef R1
#undef R4
#undef R16
#undef R64
#undef R256
  }
  return v;
}

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

template <typename Serializer, typename Struct>
void readBench_LargeSetInt_v1(size_t iters) {
  folly::BenchmarkSuspender susp;
  auto strct = create<Struct>();
  folly::IOBufQueue q;
  Serializer::serialize(strct, &q);
  auto buf = q.move();
  // coalesce the IOBuf chain to test fast path
  buf->coalesce();

  setup_ptr_chase_data();
  PtrChaseNode* current_node = &ptr_chase_nodes[0];
  uint64_t accum = 0;
  int j = 0;

  susp.dismiss();

  while (iters--) {
    auto obj = protocol::parseObject<GetReader<Serializer>>(*buf);
    folly::doNotOptimizeAway(obj);

    for (int k = 0; k < 1000; ++k) {
      current_node = current_node->next;
      uint64_t payload = current_node->payload;
      switch (j % 3) {
        case 0:
          accum = big_switch_func<1>(payload + accum);
          break;
        case 1:
          accum = big_switch_func<2>(payload + accum);
          break;
        case 2:
          accum = big_switch_func<3>(payload + accum);
          break;
      }
      j++;
    }
  }
  folly::doNotOptimizeAway(accum);
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

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v1, iters) {
  readBench_LargeSetInt_v1<CompactSerializer, LargeSetInt>(iters);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
