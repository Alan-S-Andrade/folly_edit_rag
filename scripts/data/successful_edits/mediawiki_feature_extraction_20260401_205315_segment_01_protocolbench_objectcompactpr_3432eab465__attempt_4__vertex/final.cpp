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

#define GEN_CASES_0_127(F) \
    F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8) F(9) F(10) F(11) F(12) F(13) F(14) F(15) \
    F(16) F(17) F(18) F(19) F(20) F(21) F(22) F(23) F(24) F(25) F(26) F(27) F(28) F(29) F(30) F(31) \
    F(32) F(33) F(34) F(35) F(36) F(37) F(38) F(39) F(40) F(41) F(42) F(43) F(44) F(45) F(46) F(47) \
    F(48) F(49) F(50) F(51) F(52) F(53) F(54) F(55) F(56) F(57) F(58) F(59) F(60) F(61) F(62) F(63) \
    F(64) F(65) F(66) F(67) F(68) F(69) F(70) F(71) F(72) F(73) F(74) F(75) F(76) F(77) F(78) F(79) \
    F(80) F(81) F(82) F(83) F(84) F(85) F(86) F(87) F(88) F(89) F(90) F(91) F(92) F(93) F(94) F(95) \
    F(96) F(97) F(98) F(99) F(100) F(101) F(102) F(103) F(104) F(105) F(106) F(107) F(108) F(109) F(110) F(111) \
    F(112) F(113) F(114) F(115) F(116) F(117) F(118) F(119) F(120) F(121) F(122) F(123) F(124) F(125) F(126) F(127)

#define GEN_CASES_128_255(F) \
    F(128) F(129) F(130) F(131) F(132) F(133) F(134) F(135) F(136) F(137) F(138) F(139) F(140) F(141) F(142) F(143) \
    F(144) F(145) F(146) F(147) F(148) F(149) F(150) F(151) F(152) F(153) F(154) F(155) F(156) F(157) F(158) F(159) \
    F(160) F(161) F(162) F(163) F(164) F(165) F(166) F(167) F(168) F(169) F(170) F(171) F(172) F(173) F(174) F(175) \
    F(176) F(177) F(178) F(179) F(180) F(181) F(182) F(183) F(184) F(185) F(186) F(187) F(188) F(189) F(190) F(191) \
    F(192) F(193) F(194) F(195) F(196) F(197) F(198) F(199) F(200) F(201) F(202) F(203) F(204) F(205) F(206) F(207) \
    F(208) F(209) F(210) F(211) F(212) F(213) F(214) F(215) F(216) F(217) F(218) F(219) F(220) F(221) F(222) F(223) \
    F(224) F(225) F(226) F(227) F(228) F(229) F(230) F(231) F(232) F(233) F(234) F(235) F(236) F(237) F(238) F(239) \
    F(240) F(241) F(242) F(243) F(244) F(245) F(246) F(247) F(248) F(249) F(250) F(251) F(252) F(253) F(254) F(255)

#define OPS_ipc_v2_1(i) \
    acc = acc * (i * 37 + 1) + p[8 + (i % 16)]; \
    acc = acc ^ (i * 41 + 2) + p[8 + ((i + 1) % 16)]; \
    acc = acc * (i * 43 + 3) - p[8 + ((i + 2) % 16)]; \
    acc = acc ^ (i * 47 + 4) + p[8 + ((i + 3) % 16)]; \
    acc = acc * (i * 53 + 5) + p[8 + ((i + 4) % 16)]; \
    acc = acc ^ (i * 59 + 6) + p[8 + ((i + 5) % 16)]; \
    acc = acc * (i * 61 + 7) + p[8 + ((i + 6) % 16)]; \
    acc = acc ^ (i * 67 + 8) + p[8 + ((i + 7) % 16)]; \
    acc = acc * (i * 71 + 9) - p[8 + ((i + 8) % 16)]; \
    acc = acc ^ (i * 73 + 10) + p[8 + ((i + 9) % 16)]; \
    acc = acc * (i * 79 + 11) + p[8 + ((i + 10) % 16)]; \
    acc = acc ^ (i * 83 + 12) + p[8 + ((i + 11) % 16)]; \
    acc = acc * (i * 89 + 13) + p[8 + ((i + 12) % 16)];

#define CASE_ipc_v2_1(i) case i: { OPS_ipc_v2_1(i) } break;

__attribute__((noinline))
long worker_func_1_ipc_v2(const char* p) {
    long acc = p[8] + p[9];
    switch (p[8] & 0xFF) {
        GEN_CASES_0_127(CASE_ipc_v2_1)
        GEN_CASES_128_255(CASE_ipc_v2_1)
    }
    return acc;
}

#define OPS_ipc_v2_2(i) \
    acc = acc * (i * 97 + 1) + p[8 + (i % 16)]; \
    acc = acc ^ (i * 101 + 2) + p[8 + ((i + 1) % 16)]; \
    acc = acc * (i * 103 + 3) - p[8 + ((i + 2) % 16)]; \
    acc = acc ^ (i * 107 + 4) + p[8 + ((i + 3) % 16)]; \
    acc = acc * (i * 109 + 5) + p[8 + ((i + 4) % 16)]; \
    acc = acc ^ (i * 113 + 6) + p[8 + ((i + 5) % 16)]; \
    acc = acc * (i * 127 + 7) + p[8 + ((i + 6) % 16)]; \
    acc = acc ^ (i * 131 + 8) + p[8 + ((i + 7) % 16)]; \
    acc = acc * (i * 137 + 9) - p[8 + ((i + 8) % 16)]; \
    acc = acc ^ (i * 139 + 10) + p[8 + ((i + 9) % 16)]; \
    acc = acc * (i * 149 + 11) + p[8 + ((i + 10) % 16)]; \
    acc = acc ^ (i * 151 + 12) + p[8 + ((i + 11) % 16)]; \
    acc = acc * (i * 157 + 13) + p[8 + ((i + 12) % 16)];

#define CASE_ipc_v2_2(i) case i: { OPS_ipc_v2_2(i) } break;

__attribute__((noinline))
long worker_func_2_ipc_v2(const char* p) {
    long acc = p[8] + p[9];
    switch (p[8] & 0xFF) {
        GEN_CASES_0_127(CASE_ipc_v2_2)
        GEN_CASES_128_255(CASE_ipc_v2_2)
    }
    return acc;
}

#define OPS_ipc_v2_3(i) \
    acc = acc * (i * 163 + 1) + p[8 + (i % 16)]; \
    acc = acc ^ (i * 167 + 2) + p[8 + ((i + 1) % 16)]; \
    acc = acc * (i * 173 + 3) - p[8 + ((i + 2) % 16)]; \
    acc = acc ^ (i * 179 + 4) + p[8 + ((i + 3) % 16)]; \
    acc = acc * (i * 181 + 5) + p[8 + ((i + 4) % 16)]; \
    acc = acc ^ (i * 191 + 6) + p[8 + ((i + 5) % 16)]; \
    acc = acc * (i * 193 + 7) + p[8 + ((i + 6) % 16)]; \
    acc = acc ^ (i * 197 + 8) + p[8 + ((i + 7) % 16)]; \
    acc = acc * (i * 199 + 9) - p[8 + ((i + 8) % 16)]; \
    acc = acc ^ (i * 211 + 10) + p[8 + ((i + 9) % 16)]; \
    acc = acc * (i * 223 + 11) + p[8 + ((i + 10) % 16)]; \
    acc = acc ^ (i * 227 + 12) + p[8 + ((i + 11) % 16)]; \
    acc = acc * (i * 229 + 13) + p[8 + ((i + 12) % 16)];

#define CASE_ipc_v2_3(i) case i: { OPS_ipc_v2_3(i) } break;

__attribute__((noinline))
long worker_func_3_ipc_v2(const char* p) {
    long acc = p[8] + p[9];
    switch (p[8] & 0xFF) {
        GEN_CASES_0_127(CASE_ipc_v2_3)
        GEN_CASES_128_255(CASE_ipc_v2_3)
    }
    return acc;
}
} // namespace

BENCHMARK(ObjectCompactProtocol_read_LargeSetInt_ipc_v2, iters) {
    folly::BenchmarkSuspender susp;

    constexpr size_t kNumNodes = 16 * 1024;
    constexpr size_t kNodeSize = 64;
    struct Node {
        Node* next;
        char payload[kNodeSize - sizeof(Node*)];
    };

    std::vector<Node> nodes(kNumNodes);
    for (size_t i = 0; i < kNumNodes; ++i) {
        nodes[i].next = &nodes[(i + 1) % kNumNodes];
        for (size_t j = 0; j < sizeof(nodes[i].payload); ++j) {
            nodes[i].payload[j] = (i + j) & 0xFF;
        }
    }

    const char* p = reinterpret_cast<const char*>(&nodes[0]);
    long acc = 0;

    susp.dismiss();

    for (size_t i = 0; i < iters; ++i) {
        p = *reinterpret_cast<const char* const*>(p);
        switch (i % 3) {
            case 0:
                acc += worker_func_1_ipc_v2(p);
                break;
            case 1:
                acc += worker_func_2_ipc_v2(p);
                break;
            case 2:
                acc += worker_func_3_ipc_v2(p);
                break;
        }
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
