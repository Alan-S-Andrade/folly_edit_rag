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

#define CASE_BLOCK(i, salt) \
    case i: \
        acc += i * 3 + salt; acc ^= 0x12345678 + salt; acc *= (i+1); \
        acc += i * 5 + salt; acc ^= 0x87654321 + salt; acc *= (i+2); \
        acc += i * 7 + salt; acc ^= 0xabcdef01 + salt; acc *= (i+3); \
        acc += i * 9 + salt; acc ^= 0xfedcba98 + salt; acc *= (i+4); \
        acc += i * 11 + salt; acc ^= 0xdeadbeef + salt; acc *= (i+5); \
        acc += i * 13 + salt; acc ^= 0xc0ffee + salt; acc *= (i+6); \
        acc += i * 15 + salt; acc ^= 0xbadf00d + salt; acc *= (i+7); \
        acc += i * 17 + salt; acc ^= 0x1337 + salt; acc *= (i+8); \
        acc += i * 19 + salt; acc ^= 0xbadcafe + salt; acc *= (i+9); \
        break;

#define CASES_16(m, base, salt) \
    m(base + 0, salt) m(base + 1, salt) m(base + 2, salt) m(base + 3, salt) \
    m(base + 4, salt) m(base + 5, salt) m(base + 6, salt) m(base + 7, salt) \
    m(base + 8, salt) m(base + 9, salt) m(base + 10, salt) m(base + 11, salt) \
    m(base + 12, salt) m(base + 13, salt) m(base + 14, salt) m(base + 15, salt)

#define ALL_CASES(m, salt) \
    CASES_16(m, 0, salt) CASES_16(m, 16, salt) CASES_16(m, 32, salt) CASES_16(m, 48, salt) \
    CASES_16(m, 64, salt) CASES_16(m, 80, salt) CASES_16(m, 96, salt) CASES_16(m, 112, salt) \
    CASES_16(m, 128, salt) CASES_16(m, 144, salt) CASES_16(m, 160, salt) CASES_16(m, 176, salt) \
    CASES_16(m, 192, salt) CASES_16(m, 208, salt) CASES_16(m, 224, salt) CASES_16(m, 240, salt)

__attribute__((noinline))
long op_func_0(long acc, int val) {
    switch (val) {
        ALL_CASES(CASE_BLOCK, 0)
    }
    return acc;
}
__attribute__((noinline))
long op_func_1(long acc, int val) {
    switch (val) {
        ALL_CASES(CASE_BLOCK, 1)
    }
    return acc;
}
__attribute__((noinline))
long op_func_2(long acc, int val) {
    switch (val) {
        ALL_CASES(CASE_BLOCK, 2)
    }
    return acc;
}

struct PointerChaseNode {
    PointerChaseNode* next;
    uint8_t payload;
};

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v12, iters) {
  folly::BenchmarkSuspender susp;
  constexpr size_t num_nodes = 1024 * 16;
  std::vector<PointerChaseNode> nodes(num_nodes);
  std::vector<size_t> indices(num_nodes);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 g(12345);
  std::shuffle(indices.begin(), indices.end(), g);

  for(size_t i = 0; i < num_nodes; ++i) {
      nodes[indices[i]].next = &nodes[indices[(i + 1) % num_nodes]];
      nodes[indices[i]].payload = g() % 256;
  }

  susp.dismiss();

  long acc = 0;
  auto* p = &nodes[0];
  for (size_t j = 0; j < iters; ++j) {
      uint8_t val = p->payload;
      switch(j % 3) {
          case 0: acc = op_func_0(acc, val); break;
          case 1: acc = op_func_1(acc, val); break;
          case 2: acc = op_func_2(acc, val); break;
      }
      p = p->next;
  }
  folly::doNotOptimizeAway(acc);
  susp.rehire();
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
// clang-format on
