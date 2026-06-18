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

#define ALU_OPS_28(v, i, offset) \
    v = (v * 3) ^ (i * 17 + offset + 0); \
    v = (v * 5) + (i * 19 + offset + 1); \
    v = (v * 7) ^ (i * 23 + offset + 2); \
    v = (v * 11) + (i * 29 + offset + 3); \
    v = (v * 13) ^ (i * 31 + offset + 4); \
    v = (v * 3) + (i * 37 + offset + 5); \
    v = (v * 5) ^ (i * 41 + offset + 6); \
    v = (v * 7) + (i * 43 + offset + 7); \
    v = (v * 11) ^ (i * 47 + offset + 8); \
    v = (v * 13) + (i * 53 + offset + 9); \
    v = (v * 3) ^ (i * 59 + offset + 10); \
    v = (v * 5) + (i * 61 + offset + 11); \
    v = (v * 7) ^ (i * 67 + offset + 12); \
    v = (v * 11) + (i * 71 + offset + 13); \
    v = (v * 13) ^ (i * 73 + offset + 14); \
    v = (v * 3) + (i * 79 + offset + 15); \
    v = (v * 5) ^ (i * 83 + offset + 16); \
    v = (v * 7) + (i * 89 + offset + 17); \
    v = (v * 11) ^ (i * 97 + offset + 18); \
    v = (v * 13) + (i * 101 + offset + 19); \
    v = (v * 3) ^ (i * 103 + offset + 20); \
    v = (v * 5) + (i * 107 + offset + 21); \
    v = (v * 7) ^ (i * 109 + offset + 22); \
    v = (v * 11) + (i * 113 + offset + 23); \
    v = (v * 13) ^ (i * 127 + offset + 24); \
    v = (v * 3) + (i * 131 + offset + 25); \
    v = (v * 5) ^ (i * 137 + offset + 26); \
    v = (v * 7) + (i * 139 + offset + 27);

#define CASE_BLOCK(i, offset) \
    case i: \
        ALU_OPS_28(val, i, offset); \
        break;

#define U(p,n) p##n
#define G_INNER(n, offset) \
  CASE_BLOCK(U(0x, n##0), offset) \
  CASE_BLOCK(U(0x, n##1), offset) \
  CASE_BLOCK(U(0x, n##2), offset) \
  CASE_BLOCK(U(0x, n##3), offset) \
  CASE_BLOCK(U(0x, n##4), offset) \
  CASE_BLOCK(U(0x, n##5), offset) \
  CASE_BLOCK(U(0x, n##6), offset) \
  CASE_BLOCK(U(0x, n##7), offset) \
  CASE_BLOCK(U(0x, n##8), offset) \
  CASE_BLOCK(U(0x, n##9), offset) \
  CASE_BLOCK(U(0x, n##a), offset) \
  CASE_BLOCK(U(0x, n##b), offset) \
  CASE_BLOCK(U(0x, n##c), offset) \
  CASE_BLOCK(U(0x, n##d), offset) \
  CASE_BLOCK(U(0x, n##e), offset) \
  CASE_BLOCK(U(0x, n##f), offset)

#define G(offset) \
    G_INNER(0, offset) G_INNER(1, offset) G_INNER(2, offset) G_INNER(3, offset) \
    G_INNER(4, offset) G_INNER(5, offset) G_INNER(6, offset) G_INNER(7, offset) \
    G_INNER(8, offset) G_INNER(9, offset) G_INNER(a, offset) G_INNER(b, offset) \
    G_INNER(c, offset) G_INNER(d, offset) G_INNER(e, offset) G_INNER(f, offset)

__attribute__((noinline))
int L1i_switch_func_0(int x, int y) {
    int val = y;
    switch (x & 0xFF) {
    G(0)
    default:
        break;
    }
    return val;
}

__attribute__((noinline))
int L1i_switch_func_1(int x, int y) {
    int val = y;
    switch (x & 0xFF) {
    G(256 * 28)
    default:
        break;
    }
    return val;
}

__attribute__((noinline))
int L1i_switch_func_2(int x, int y) {
    int val = y;
    switch (x & 0xFF) {
    G(256 * 28 * 2)
    default:
        break;
    }
    return val;
}

#undef G
#undef G_INNER
#undef U
#undef CASE_BLOCK
#undef ALU_OPS_28

} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_L1i_v1, iters) {
    folly::BenchmarkSuspender susp;

    struct Node {
        int payload;
        int next_idx;
    };
    constexpr size_t array_size = 1024 * 16;
    std::vector<Node> node_array(array_size);

    std::vector<int> indices(array_size);
    for (size_t i = 0; i < array_size; ++i) {
        indices[i] = i;
    }
    // simple shuffle
    for (size_t i = 0; i < array_size - 1; ++i) {
        size_t j = i + folly::Random::rand32() % (array_size - i);
        std::swap(indices[i], indices[j]);
    }

    for (size_t i = 0; i < array_size; ++i) {
        node_array[indices[i]].next_idx = indices[(i + 1) % array_size];
        node_array[indices[i]].payload = folly::Random::rand32();
    }
    
    int current_idx = 0;
    int dep = 0;
    
    susp.dismiss();

    for (size_t i = 0; i < iters; ++i) {
        const auto& node = node_array[current_idx];
        int val = node.payload;
        
        switch (i % 3) {
            case 0: dep = L1i_switch_func_0(val, dep); break;
            case 1: dep = L1i_switch_func_1(val, dep); break;
            case 2: dep = L1i_switch_func_2(val, dep); break;
        }
        current_idx = node.next_idx;
    }
    
    folly::doNotOptimizeAway(dep);
    susp.rehire();
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
