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

#define ALU_OPS_0(i, acc) \
    acc += (i * 101 + 1); acc ^= (i * 103 + 2); acc *= (i * 107 + 3); acc += (i * 109 + 4); \
    acc ^= (i * 113 + 5); acc *= (i * 127 + 6); acc += (i * 131 + 7); acc ^= (i * 137 + 8); \
    acc *= (i * 139 + 9); acc += (i * 149 + 10); acc ^= (i * 151 + 11); acc *= (i * 157 + 12); \
    acc += (i * 163 + 13); acc ^= (i * 167 + 14); acc *= (i * 173 + 15); acc += (i * 179 + 16); \
    acc ^= (i * 181 + 17); acc *= (i * 191 + 18); acc += (i * 193 + 19); acc ^= (i * 197 + 20); \
    acc *= (i * 199 + 21); acc += (i * 211 + 22); acc ^= (i * 223 + 23); acc *= (i * 227 + 24); \
    acc += (i * 229 + 25); acc ^= (i * 233 + 26); acc *= (i * 239 + 27); acc += (i * 241 + 28);

#define ALU_OPS_1(i, acc) \
    acc += (i * 251 + 29); acc ^= (i * 257 + 30); acc *= (i * 263 + 31); acc += (i * 269 + 32); \
    acc ^= (i * 271 + 33); acc *= (i * 277 + 34); acc += (i * 281 + 35); acc ^= (i * 283 + 36); \
    acc *= (i * 293 + 37); acc += (i * 307 + 38); acc ^= (i * 311 + 39); acc *= (i * 313 + 40); \
    acc += (i * 317 + 41); acc ^= (i * 331 + 42); acc *= (i * 337 + 43); acc += (i * 347 + 44); \
    acc ^= (i * 349 + 45); acc *= (i * 353 + 46); acc += (i * 359 + 47); acc ^= (i * 367 + 48); \
    acc *= (i * 373 + 49); acc += (i * 379 + 50); acc ^= (i * 383 + 51); acc *= (i * 389 + 52); \
    acc += (i * 397 + 53); acc ^= (i * 401 + 54); acc *= (i * 409 + 55); acc += (i * 419 + 56);

#define ALU_OPS_2(i, acc) \
    acc += (i * 421 + 57); acc ^= (i * 431 + 58); acc *= (i * 433 + 59); acc += (i * 439 + 60); \
    acc ^= (i * 443 + 61); acc *= (i * 449 + 62); acc += (i * 457 + 63); acc ^= (i * 461 + 64); \
    acc *= (i * 463 + 65); acc += (i * 467 + 66); acc ^= (i * 479 + 67); acc *= (i * 487 + 68); \
    acc += (i * 491 + 69); acc ^= (i * 499 + 70); acc *= (i * 503 + 71); acc += (i * 509 + 72); \
    acc ^= (i * 521 + 73); acc *= (i * 523 + 74); acc += (i * 541 + 75); acc ^= (i * 547 + 76); \
    acc *= (i * 557 + 77); acc += (i * 563 + 78); acc ^= (i * 569 + 79); acc *= (i * 571 + 80); \
    acc += (i * 577 + 81); acc ^= (i * 587 + 82); acc *= (i * 593 + 83); acc += (i * 599 + 84);

#define EXPAND_16(m, base) \
    m(base+0) m(base+1) m(base+2) m(base+3) m(base+4) m(base+5) m(base+6) m(base+7) \
    m(base+8) m(base+9) m(base+10) m(base+11) m(base+12) m(base+13) m(base+14) m(base+15)

#define EXPAND_256(m) \
    EXPAND_16(m, 0) EXPAND_16(m, 16) EXPAND_16(m, 32) EXPAND_16(m, 48) \
    EXPAND_16(m, 64) EXPAND_16(m, 80) EXPAND_16(m, 96) EXPAND_16(m, 112) \
    EXPAND_16(m, 128) EXPAND_16(m, 144) EXPAND_16(m, 160) EXPAND_16(m, 176) \
    EXPAND_16(m, 192) EXPAND_16(m, 208) EXPAND_16(m, 224) EXPAND_16(m, 240)

__attribute__((noinline))
long long worker_func_0(char val, long long acc) {
    switch (static_cast<unsigned char>(val)) {
#define CASE_STMT_0(i) case i: ALU_OPS_0(i, acc); break;
    EXPAND_256(CASE_STMT_0)
#undef CASE_STMT_0
    }
    return acc;
}

__attribute__((noinline))
long long worker_func_1(char val, long long acc) {
    switch (static_cast<unsigned char>(val)) {
#define CASE_STMT_1(i) case i: ALU_OPS_1(i, acc); break;
    EXPAND_256(CASE_STMT_1)
#undef CASE_STMT_1
    }
    return acc;
}

__attribute__((noinline))
long long worker_func_2(char val, long long acc) {
    switch (static_cast<unsigned char>(val)) {
#define CASE_STMT_2(i) case i: ALU_OPS_2(i, acc); break;
    EXPAND_256(CASE_STMT_2)
#undef CASE_STMT_2
    }
    return acc;
}

struct Node {
    Node* next;
    char payload;
};

} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_v5, iters) {
    folly::BenchmarkSuspender susp;

    constexpr size_t num_nodes = 16 * 1024;
    std::vector<Node> nodes(num_nodes);
    std::vector<int> indices(num_nodes);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 g(42); // deterministic shuffle
    std::shuffle(indices.begin(), indices.end(), g);

    for (size_t i = 0; i < num_nodes; ++i) {
        nodes[indices[i]].next = &nodes[indices[(i + 1) % num_nodes]];
        nodes[indices[i]].payload = static_cast<char>(indices[i] & 0xFF);
    }

    Node* current = &nodes[0];
    long long acc = 0;

    susp.dismiss();

    for (size_t i = 0; i < iters; ++i) {
        char payload = current->payload;
        switch (i % 3) {
            case 0:
                acc = worker_func_0(payload, acc);
                break;
            case 1:
                acc = worker_func_1(payload, acc);
                break;
            case 2:
                acc = worker_func_2(payload, acc);
                break;
        }
        current = current->next;
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
