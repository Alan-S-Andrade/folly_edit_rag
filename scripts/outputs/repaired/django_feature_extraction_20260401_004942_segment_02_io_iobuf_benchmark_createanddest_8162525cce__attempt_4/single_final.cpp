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

#include <folly/Benchmark.h>
#include <folly/io/IOBuf.h>

using folly::IOBuf;

BENCHMARK(createAndDestroy, iters) {
  while (iters--) {
    IOBuf buf(IOBuf::CREATE, 10);
    folly::doNotOptimizeAway(buf.capacity());
  }
}

BENCHMARK(cloneOneBenchmark, iters) {
  IOBuf buf(IOBuf::CREATE, 10);
  while (iters--) {
    auto copy = buf.cloneOne();
    folly::doNotOptimizeAway(copy->capacity());
  }
}

BENCHMARK(cloneOneIntoBenchmark, iters) {
  IOBuf buf(IOBuf::CREATE, 10);
  IOBuf copy;
  while (iters--) {
    buf.cloneOneInto(copy);
    folly::doNotOptimizeAway(copy.capacity());
  }
}

BENCHMARK(cloneBenchmark, iters) {
  IOBuf buf(IOBuf::CREATE, 10);
  while (iters--) {
    auto copy = buf.clone();
    folly::doNotOptimizeAway(copy->capacity());
  }
}

BENCHMARK(cloneIntoBenchmark, iters) {
  IOBuf buf(IOBuf::CREATE, 10);
  IOBuf copy;
  while (iters--) {
    buf.cloneInto(copy);
    folly::doNotOptimizeAway(copy.capacity());
  }
}

BENCHMARK(moveBenchmark, iters) {
  IOBuf buf(IOBuf::CREATE, 10);
  while (iters--) {
    auto tmp = std::move(buf);
    folly::doNotOptimizeAway(tmp.capacity());
    buf = std::move(tmp);
  }
}

BENCHMARK(copyBenchmark, iters) {
  IOBuf buf(IOBuf::CREATE, 10);
  while (iters--) {
    auto copy = buf;
    folly::doNotOptimizeAway(copy.capacity());
  }
}

BENCHMARK(copyBufferFromStringBenchmark, iters) {
  std::string s("Hello World");
  while (iters--) {
    auto copy = IOBuf::copyBuffer(s);
    folly::doNotOptimizeAway(copy->capacity());
  }
}

BENCHMARK(copyBufferFromStringPieceBenchmark, iters) {
  folly::StringPiece s("Hello World");
  while (iters--) {
    auto copy = IOBuf::copyBuffer(s);
    folly::doNotOptimizeAway(copy->capacity());
  }
}

BENCHMARK(cloneCoalescedBaseline, iters) {
  std::unique_ptr<IOBuf> buf = IOBuf::createChain(100, 10);
  while (iters--) {
    auto clone = buf->cloneAsValue();
    clone.coalesce();
    folly::doNotOptimizeAway(clone.capacity());
  }
}

BENCHMARK_RELATIVE(cloneCoalescedBenchmark, iters) {
  std::unique_ptr<IOBuf> buf = IOBuf::createChain(100, 10);
  while (iters--) {
    auto copy = buf->cloneCoalescedAsValue();
    folly::doNotOptimizeAway(copy.capacity());
  }
}

BENCHMARK(takeOwnershipBenchmark, iters) {
  size_t data = 0;
  while (iters--) {
    std::unique_ptr<IOBuf> buf(
        IOBuf::takeOwnership(
            &data,
            sizeof(data),
            [](void* /*unused*/, void* /*unused*/) {},
            nullptr));
  }
}

namespace {
// Helper for createAndDestroyMulti_IndirectBranch
[[maybe_unused]] int switch3(const std::vector<uint64_t>& vec, int idx, uint64_t& current) {
  switch (idx % 3) {
    case 0:
      return vec[current & 0xFF] & 0xFF;
    case 1:
      return vec[(current >> 8) & 0xFF] & 0xFF;
    case 2:
      return vec[(current >> 16) & 0xFF] & 0xFF;
  }
  return 0; // Unreachable
}

[[maybe_unused]] int switch2(const std::vector<uint64_t>& vec, int idx, uint64_t& current) {
  switch (idx % 2) {
    case 0:
      return vec[current & 0xFF] & 0xFF;
    case 1:
      return vec[(current >> 8) & 0xFF] & 0xFF;
  }
  return 0; // Unreachable
}

// PROVEN PATTERN: Create 3 __attribute__((noinline)) functions each containing a 256-case switch
// with N ALU ops per case (add/xor/mul on a local accumulator). Rotate among them with j%3
// in the hot loop. The switch index should be payload & 0xFF from a pointer-chase load.
// L1i MPKI scales approximately linearly with ALU ops per switch case: 8 ops/case ~ 11 MPKI,
// 10 ~ 10 MPKI, 12 ~ 12 MPKI, 14 ~ 15 MPKI. Adjust ops-per-case to reach target L1i MPKI.
// Each switch case must use unique literal constants to prevent compiler deduplication.
// The 3-function rotation forces icache thrashing across a massive code footprint.

// Start with 10 ALU ops per case. This corresponds to roughly 10 L1i MPKI.
// We need ~31 L1i MPKI, so we'll need to increase ops-per-case or iterations.
// Each switch case must use unique literal constants to prevent compiler deduplication.

#define ALU_OPS_PER_CASE 10
#define CASES 256

[[maybe_unused]] __attribute__((noinline)) void switchFuncA(uint64_t& acc, const std::vector<uint64_t>& indirect_data, uint64_t payload) {
  int idx = payload & 0xFF;
  switch (idx) {
    case 0: acc ^= indirect_data[0] * 1 + 0x1234567890ABCDEFULL; break; case 1: acc ^= indirect_data[1] * 2 + 0xFEDCBA0987654321ULL; break; case 2: acc ^= indirect_data[2] * 3 + 0x1122334455667788ULL; break; case 3: acc ^= indirect_data[3] * 4 + 0x8877665544332211ULL; break; case 4: acc ^= indirect_data[4] * 5 + 0xAABBCCDDEEFF0011ULL; break; case 5: acc ^= indirect_data[5] * 6 + 0x1100FFEECCDDABULL; break; case 6: acc ^= indirect_data[6] * 7 + 0x33445566778899AALL; break; case 7: acc ^= indirect_data[7] * 8 + 0xAA99887766554433ULL; break; case 8: acc ^= indirect_data[8] * 9 + 0x0011223344556677ULL; break; case 9: acc ^= indirect_data[9] * 10 + 0x7766554433221100ULL; break;
    case 10: acc ^= indirect_data[10] * 11 + 0x13579BDF2468ACE0ULL; break; case 11: acc ^= indirect_data[11] * 12 + 0xE0AC8642FD9B7531ULL; break; case 12: acc ^= indirect_data[12] * 13 + 0x2468ACE013579BDFULL; break; case 13: acc ^= indirect_data[13] * 14 + 0xFD9B7531E0AC8642ULL; break; case 14: acc ^= indirect_data[14] * 15 + 0x1133557799BBCCDDULL; break; case 15: acc ^= indirect_data[15] * 16 + 0xDDCCBBAA77553311ULL; break; case 16: acc ^= indirect_data[16] * 17 + 0x02468ACE13579BDFULL; break; case 17: acc ^= indirect_data[17] * 18 + 0xFD9B7531E0AC8642ULL; break; case 18: acc ^= indirect_data[18] * 19 + 0x3579BDF2468ACE0ULL; break; case 19: acc ^= indirect_data[19] * 20 + 0xE0AC8642FD9B7531ULL; break;
    case 20: acc ^= indirect_data[20] * 21 + 0x9876543210FEDCBAULL; break; case 21: acc ^= indirect_data[21] * 22 + 0x0123456789ABCDEFULL; break; case 22: acc ^= indirect_data[22] * 23 + 0xFEDCBA0987654321ULL; break; case 23: acc ^= indirect_data[23] * 24 + 0x1234567890ABCDEFULL; break; case 24: acc ^= indirect_data[24] * 25 + 0x33445566778899AALL; break; case 25: acc ^= indirect_data[25] * 26 + 0xAA99887766554433ULL; break; case 26: acc ^= indirect_data[26] * 27 + 0x0011223344556677ULL; break; case 27: acc ^= indirect_data[27] * 28 + 0x7766554433221100ULL; break; case 28: acc ^= indirect_data[28] * 29 + 0x1122334455667788ULL; break; case 29: acc ^= indirect_data[29] * 30 + 0x8877665544332211ULL; break;
    case 30: acc ^= indirect_data[30] * 31 + 0xABCDEF0123456789ULL; break; case 31: acc ^= indirect_data[31] * 32 + 0x9876543210FEDCBAULL; break; case 32: acc ^= indirect_data[32] * 33 + 0x1234567890ABCDEFULL; break; case 33: acc ^= indirect_data[33] * 34 + 0xFEDCBA0987654321ULL; break; case 34: acc ^= indirect_data[34] * 35 + 0x1133557799BBCCDDULL; break; case 35: acc ^= indirect_data[35] * 36 + 0xDDCCBBAA77553311ULL; break; case 36: acc ^= indirect_data[36] * 37 + 0x02468ACE13579BDFULL; break; case 37: acc ^= indirect_data[37] * 38 + 0xFD9B7531E0AC8642ULL; break; case 38: acc ^= indirect_data[38] * 39 + 0x3579BDF2468ACE0ULL; break; case 39: acc ^= indirect_data[39] * 40 + 0xE0AC8642FD9B7531ULL; break;
    case 40: acc ^= indirect_data[40] * 41 + 0x5566778899AABBCCULL; break; case 41: acc ^= indirect_data[41] * 42 + 0xCCBBAA9988776655ULL; break; case 42: acc ^= indirect_data[42] * 43 + 0x1122334455667788ULL; break; case 43: acc ^= indirect_data[43] * 44 + 0x8877665544332211ULL; break; case 44: acc ^= indirect_data[44] * 45 + 0xAABBCCDDEEFF0011ULL; break; case 45: acc ^= indirect_data[45] * 46 + 0x1100FFEECCDDABULL; break; case 46: acc ^= indirect_data[46] * 47 + 0x33445566778899AALL; break; case 47: acc ^= indirect_data[47] * 48 + 0xAA99887766554433ULL; break; case 48: acc ^= indirect_data[48] * 49 + 0x0011223344556677ULL; break; case 49: acc ^= indirect_data[49] * 50 + 0x7766554433221100ULL; break;
    case 50: acc ^= indirect_data[50] * 51 + 0x13579BDF2468ACE0ULL; break; case 51: acc ^= indirect_data[51] * 52 + 0xE0AC8642FD9B7531ULL; break; case 52: acc ^= indirect_data[52] * 53 + 0x2468ACE013579BDFULL; break; case 53: acc ^= indirect_data[53] * 54 + 0xFD9B7531E0AC8642ULL; break; case 54: acc ^= indirect_data[54] * 55 + 0x1133557799BBCCDDULL; break; case 55: acc ^= indirect_data[55] * 56 + 0xDDCCBBAA77553311ULL; break; case 56: acc ^= indirect_data[56] * 57 + 0x02468ACE13579BDFULL; break; case 57: acc ^= indirect_data[57] * 58 + 0xFD9B7531E0AC8642ULL; break; case 58: acc ^= indirect_data[58] * 59 + 0x3579BDF2468ACE0ULL; break; case 59: acc ^= indirect_data[59] * 60 + 0xE0AC8642FD9B7531ULL; break;
    case 60: acc ^= indirect_data[60] * 61 + 0xFEDCBA0987654321ULL; break; case 61: acc ^= indirect_data[61] * 62 + 0x1234567890ABCDEFULL; break; case 62: acc ^= indirect_data[62] * 63 + 0x33445566778899AALL; break; case 63: acc ^= indirect_data[63] * 64 + 0xAA99887766554433ULL; break; case 64: acc ^= indirect_data[64] * 65 + 0x0011223344556677ULL; break; case 65: acc ^= indirect_data[65] * 66 + 0x7766554433221100ULL; break; case 66: acc ^= indirect_data[66] * 67 + 0x1122334455667788ULL; break; case 67: acc ^= indirect_data[67] * 68 + 0x8877665544332211ULL; break; case 68: acc ^= indirect_data[68] * 69 + 0xAABBCCDDEEFF0011ULL; break; case 69: acc ^= indirect_data[69] * 70 + 0x1100FFEECCDDABULL; break;
    case 70: acc ^= indirect_data[70] * 71 + 0x33445566778899AALL; break; case 71: acc ^= indirect_data[71] * 72 + 0xAA99887766554433ULL; break; case 72: acc ^= indirect_data[72] * 73 + 0x0011223344556677ULL; break; case 73: acc ^= indirect_data[73] * 74 + 0x7766554433221100ULL; break; case 74: acc ^= indirect_data[74] * 75 + 0x13579BDF2468ACE0ULL; break; case 75: acc ^= indirect_data[75] * 76 + 0xE0AC8642FD9B7531ULL; break; case 76: acc ^= indirect_data[76] * 77 + 0x2468ACE013579BDFULL; break; case 77: acc ^= indirect_data[77] * 78 + 0xFD9B7531E0AC8642ULL; break; case 78: acc ^= indirect_data[78] * 79 + 0x1133557799BBCCDDULL; break; case 79: acc ^= indirect_data[79] * 80 + 0xDDCCBBAA77553311ULL; break;
    case 80: acc ^= indirect_data[80] * 81 + 0x02468ACE13579BDFULL; break; case 81: acc ^= indirect_data[81] * 82 + 0xFD9B7531E0AC8642ULL; break; case 82: acc ^= indirect_data[82] * 83 + 0x3579BDF2468ACE0ULL; break; case 83: acc ^= indirect_data[83] * 84 + 0xE0AC8642FD9B7531ULL; break; case 84: acc ^= indirect_data[84] * 85 + 0x9876543210FEDCBAULL; break; case 85: acc ^= indirect_data[85] * 86 + 0x0123456789ABCDEFULL; break; case 86: acc ^= indirect_data[86] * 87 + 0xFEDCBA0987654321ULL; break; case 87: acc ^= indirect_data[87] * 88 + 0x1234567890ABCDEFULL; break; case 88: acc ^= indirect_data[88] * 89 + 0x33445566778899AALL; break; case 89: acc ^= indirect_data[89] * 90 + 0xAA99887766554433ULL; break;
    case 90: acc ^= indirect_data[90] * 91 + 0x0011223344556677ULL; break; case 91: acc ^= indirect_data[91] * 92 + 0x7766554433221100ULL; break; case 92: acc ^= indirect_data[92] * 93 + 0x1122334455667788ULL; break; case 93: acc ^= indirect_data[93] * 94 + 0x8877665544332211ULL; break; case 94: acc ^= indirect_data[94] * 95 + 0xAABBCCDDEEFF0011ULL; break; case 95: acc ^= indirect_data[95] * 96 + 0x1100FFEECCDDABULL; break; case 96: acc ^= indirect_data[96] * 97 + 0x33445566778899AALL; break; case 97: acc ^= indirect_data[97] * 98 + 0xAA99887766554433ULL; break; case 98: acc ^= indirect_data[98] * 99 + 0x0011223344556677ULL; break; case 99: acc ^= indirect_data[99] * 100 + 0x7766554433221100ULL; break;
    case 100: acc ^= indirect_data[100] * 101 + 0x13579BDF2468ACE0ULL; break; case 101: acc ^= indirect_data[101] * 102 + 0xE0AC8642FD9B7531ULL; break; case 102: acc ^= indirect_data[102] * 103 + 0x2468ACE013579BDFULL; break; case 103: acc ^= indirect_data[103] * 104 + 0xFD9B7531E0AC8642ULL; break; case 104: acc ^= indirect_data[104] * 105 + 0x1133557799BBCCDDULL; break; case 105: acc ^= indirect_data[105] * 106 + 0xDDCCBBAA77553311ULL; break; case 106: acc ^= indirect_data[106] * 107 + 0x02468ACE13579BDFULL; break; case 107: acc ^= indirect_data[107] * 108 + 0xFD9B7531E0AC8642ULL; break; case 108: acc ^= indirect_data[108] * 109 + 0x3579BDF2468ACE0ULL; break; case 109: acc ^= indirect_data[109] * 110 + 0xE0AC8642FD9B7531ULL; break;
    case 110: acc ^= indirect_data[110] * 111 + 0xFEDCBA0987654321ULL; break; case 111: acc ^= indirect_data[111] * 112 + 0x1234567890ABCDEFULL; break; case 112: acc ^= indirect_data[112] * 113 + 0x33445566778899AALL; break; case 113: acc ^= indirect_data[113] * 114 + 0xAA99887766554433ULL; break; case 114: acc ^= indirect_data[114] * 115 + 0x0011223344556677ULL; break; case 115: acc ^= indirect_data[115] * 116 + 0x7766554433221100ULL; break; case 116: acc ^= indirect_data[116] * 117 + 0x1122334455667788ULL; break; case 117: acc ^= indirect_data[117] * 118 + 0x8877665544332211ULL; break; case 118: acc ^= indirect_data[118] * 119 + 0xAABBCCDDEEFF0011ULL; break; case 119: acc ^= indirect_data[119] * 120 + 0x1100FFEECCDDABULL; break;
    case 120: acc ^= indirect_data[120] * 121 + 0x33445566778899AALL; break; case 121: acc ^= indirect_data[121] * 122 + 0xAA99887766554433ULL; break; case 122: acc ^= indirect_data[122] * 123 + 0x0011223344556677ULL; break; case 123: acc ^= indirect_data[123] * 124 + 0x7766554433221100ULL; break; case 124: acc ^= indirect_data[124] * 125 + 0x13579BDF2468ACE0ULL; break; case 125: acc ^= indirect_data[125] * 126 + 0xE0AC8642FD9B7531ULL; break; case 126: acc ^= indirect_data[126] * 127 + 0x2468ACE013579BDFULL; break; case 127: acc ^= indirect_data[127] * 128 + 0xFD9B7531E0AC8642ULL; break; case 128: acc ^= indirect_data[128] * 129 + 0x1133557799BBCCDDULL; break; case 129: acc ^= indirect_data[129] * 130 + 0xDDCCBBAA77553311ULL; break;
    case 130: acc ^= indirect_data[130] * 131 + 0x02468ACE13579BDFULL; break; case 131: acc ^= indirect_data[131] * 132 + 0xFD9B7531E0AC8642ULL; break; case 132: acc ^= indirect_data[132] * 133 + 0x3579BDF2468ACE0ULL; break; case 133: acc ^= indirect_data[133] * 134 + 0xE0AC8642FD9B7531ULL; break; case 134: acc ^= indirect_data[134] * 135 + 0x9876543210FEDCBAULL; break; case 135: acc ^= indirect_data[135] * 136 + 0x0123456789ABCDEFULL; break; case 136: acc ^= indirect_data[136] * 137 + 0xFEDCBA0987654321ULL; break; case 137: acc ^= indirect_data[137] * 138 + 0x1234567890ABCDEFULL; break; case 138: acc ^= indirect_data[138] * 139 + 0x33445566778899AALL; break; case 139: acc ^= indirect_data[139] * 140 + 0xAA99887766554433ULL; break;
    case 140: acc ^= indirect_data[140] * 141 + 0x0011223344556677ULL; break; case 141: acc ^= indirect_data[141] * 142 + 0x7766554433221100ULL; break; case 142: acc ^= indirect_data[142] * 143 + 0x1122334455667788ULL; break; case 143: acc ^= indirect_data[143] * 144 + 0x8877665544332211ULL; break; case 144: acc ^= indirect_data[144] * 145 + 0xAABBCCDDEEFF0011ULL; break; case 145: acc ^= indirect_data[145] * 146 + 0x1100FFEECCDDABULL; break; case 146: acc ^= indirect_data[146] * 147 + 0x33445566778899AALL; break; case 147: acc ^= indirect_data[147] * 148 + 0xAA99887766554433ULL; break; case 148: acc ^= indirect_data[148] * 149 + 0x0011223344556677ULL; break; case 149: acc ^= indirect_data[149] * 150 + 0x7766554433221100ULL; break;
    case 150: acc ^= indirect_data[150] * 151 + 0x13579BDF2468ACE0ULL; break; case 151: acc ^= indirect_data[151] * 152 + 0xE0AC8642FD9B7531ULL; break; case 152: acc ^= indirect_data[152] * 153 + 0x2468ACE013579BDFULL; break; case 153: acc ^= indirect_data[153] * 154 + 0xFD9B7531E0AC8642ULL; break; case 154: acc ^= indirect_data[154] * 155 + 0x1133557799BBCCDDULL; break; case 155: acc ^= indirect_data[155] * 156 + 0xDDCCBBAA77553311ULL; break; case 156: acc ^= indirect_data[156] * 157 + 0x02468ACE13579BDFULL; break; case 157: acc ^= indirect_data[157] * 158 + 0xFD9B7531E0AC8642ULL; break; case 158: acc ^= indirect_data[158] * 159 + 0x3579BDF2468ACE0ULL; break; case 159: acc ^= indirect_data[159] * 160 + 0xE0AC8642FD9B7531ULL; break;
    case 160: acc ^= indirect_data[160] * 161 + 0xFEDCBA0987654321ULL; break; case 161: acc ^= indirect_data[161] * 162 + 0x1234567890ABCDEFULL; break; case 162: acc ^= indirect_data[162] * 163 + 0x33445566778899AALL; break; case 163: acc ^= indirect_data[163] * 164 + 0xAA99887766554433ULL; break; case 164: acc ^= indirect_data[164] * 165 + 0x0011223344556677ULL; break; case 165: acc ^= indirect_data[165] * 166 + 0x7766554433221100ULL; break; case 166: acc ^= indirect_data[166] * 167 + 0x1122334455667788ULL; break; case 167: acc ^= indirect_data[167] * 168 + 0x8877665544332211ULL; break; case 168: acc ^= indirect_data[168] * 169 + 0xAABBCCDDEEFF0011ULL; break; case 169: acc ^= indirect_data[169] * 170 + 0x1100FFEECCDDABULL; break;
    case 170: acc ^= indirect_data[170] * 171 + 0x33445566778899AALL; break; case 171: acc ^= indirect_data[171] * 172 + 0xAA99887766554433ULL; break; case 172: acc ^= indirect_data[172] * 173 + 0x0011223344556677ULL; break; case 173: acc ^= indirect_data[173] * 174 + 0x7766554433221100ULL; break; case 174: acc ^= indirect_data[174] * 175 + 0x13579BDF2468ACE0ULL; break; case 175: acc ^= indirect_data[175] * 176 + 0xE0AC8642FD9B7531ULL; break; case 176: acc ^= indirect_data[176] * 177 + 0x2468ACE013579BDFULL; break; case 177: acc ^= indirect_data[177] * 178 + 0xFD9B7531E0AC8642ULL; break; case 178: acc ^= indirect_data[178] * 179 + 0x1133557799BBCCDDULL; break; case 179: acc ^= indirect_data[179] * 180 + 0xDDCCBBAA77553311ULL; break;
    case 180: acc ^= indirect_data[180] * 181 + 0x02468ACE13579BDFULL; break; case 181: acc ^= indirect_data[181] * 182 + 0xFD9B7531E0AC8642ULL; break; case 182: acc ^= indirect_data[182] * 183 + 0x3579BDF2468ACE0ULL; break; case 183: acc ^= indirect_data[183] * 184 + 0xE0AC8642FD9B7531ULL; break; case 184: acc ^= indirect_data[184] * 185 + 0x9876543210FEDCBAULL; break; case 185: acc ^= indirect_data[185] * 186 + 0x0123456789ABCDEFULL; break; case 186: acc ^= indirect_data[186] * 187 + 0xFEDCBA0987654321ULL; break; case 187: acc ^= indirect_data[187] * 188 + 0x1234567890ABCDEFULL; break; case 188: acc ^= indirect_data[188] * 189 + 0x33445566778899AALL; break; case 189: acc ^= indirect_data[189] * 190 + 0xAA99887766554433ULL; break;
    case 190: acc ^= indirect_data[190] * 191 + 0x0011223344556677ULL; break; case 191: acc ^= indirect_data[191] * 192 + 0x7766554433221100ULL; break; case 192: acc ^= indirect_data[192] * 193 + 0x1122334455667788ULL; break; case 193: acc ^= indirect_data[193] * 194 + 0x8877665544332211ULL; break; case 194: acc ^= indirect_data[194] * 195 + 0xAABBCCDDEEFF0011ULL; break; case 195: acc ^= indirect_data[195] * 196 + 0x1100FFEECCDDABULL; break; case 196: acc ^= indirect_data[196] * 197 + 0x33445566778899AALL; break; case 197: acc ^= indirect_data[197] * 198 + 0xAA99887766554433ULL; break; case 198: acc ^= indirect_data[198] * 199 + 0x0011223344556677ULL; break; case 199: acc ^= indirect_data[199] * 200 + 0x7766554433221100ULL; break;
    case 200: acc ^= indirect_data[200] * 201 + 0x13579BDF2468ACE0ULL; break; case 201: acc ^= indirect_data[201] * 202 + 0xE0AC8642FD9B7531ULL; break; case 202: acc ^= indirect_data[202] * 203 + 0x2468ACE013579BDFULL; break; case 203: acc ^= indirect_data[203] * 204 + 0xFD9B7531E0AC8642ULL; break; case 204: acc ^= indirect_data[204] * 205 + 0x1133557799BBCCDDULL; break; case 205: acc ^= indirect_data[205] * 206 + 0xDDCCBBAA77553311ULL; break; case 206: acc ^= indirect_data[206] * 207 + 0x02468ACE13579BDFULL; break; case 207: acc ^= indirect_data[207] * 208 + 0xFD9B7531E0AC8642ULL; break; case 208: acc ^= indirect_data[208] * 209 + 0x3579BDF2468ACE0ULL; break; case 209: acc ^= indirect_data[209] * 210 + 0xE0AC8642FD9B7531ULL; break;
    case 210: acc ^= indirect_data[210] * 211 + 0xFEDCBA0987654321ULL; break; case 211: acc ^= indirect_data[211] * 212 + 0x1234567890ABCDEFULL; break; case 212: acc ^= indirect_data[212] * 213 + 0x33445566778899AALL; break; case 213: acc ^= indirect_data[213] * 214 + 0xAA99887766554433ULL; break; case 214: acc ^= indirect_data[214] * 215 + 0x0011223344556677ULL; break; case 215: acc ^= indirect_data[215] * 216 + 0x7766554433221100ULL; break; case 216: acc ^= indirect_data[216] * 217 + 0x1122334455667788ULL; break; case 217: acc ^= indirect_data[217] * 218 + 0x8877665544332211ULL; break; case 218: acc ^= indirect_data[218] * 219 + 0xAABBCCDDEEFF0011ULL; break; case 219: acc ^= indirect_data[219] * 220 + 0x1100FFEECCDDABULL; break;
    case 220: acc ^= indirect_data[220] * 221 + 0x33445566778899AALL; break; case 221: acc ^= indirect_data[221] * 222 + 0xAA99887766554433ULL; break; case 222: acc ^= indirect_data[222] * 223 + 0x0011223344556677ULL; break; case 223: acc ^= indirect_data[223] * 224 + 0x7766554433221100ULL; break; case 224: acc ^= indirect_data[224] * 225 + 0x13579BDF2468ACE0ULL; break; case 225: acc ^= indirect_data[225] * 226 + 0xE0AC8642FD9B7531ULL; break; case 226: acc ^= indirect_data[226] * 227 + 0x2468ACE013579BDFULL; break; case 227: acc ^= indirect_data[227] * 228 + 0xFD9B7531E0AC8642ULL; break; case 228: acc ^= indirect_data[228] * 229 + 0x1133557799BBCCDDULL; break; case 229: acc ^= indirect_data[229] * 230 + 0xDDCCBBAA77553311ULL; break;
    case 230: acc ^= indirect_data[230] * 231 + 0x02468ACE13579BDFULL; break; case 231: acc ^= indirect_data[231] * 232 + 0xFD9B7531E0AC8642ULL; break; case 232: acc ^= indirect_data[232] * 233 + 0x3579BDF2468ACE0ULL; break; case 233: acc ^= indirect_data[233] * 234 + 0xE0AC8642FD9B7531ULL; break; case 234: acc ^= indirect_data[234] * 235 + 0x9876543210FEDCBAULL; break; case 235: acc ^= indirect_data[235] * 236 + 0x0123456789ABCDEFULL; break; case 236: acc ^= indirect_data[236] * 237 + 0xFEDCBA0987654321ULL; break; case 237: acc ^= indirect_data[237] * 238 + 0x1234567890ABCDEFULL; break; case 238: acc ^= indirect_data[238] * 239 + 0x33445566778899AALL; break; case 239: acc ^= indirect_data[239] * 240 + 0xAA99887766554433ULL; break;
    case 240: acc ^= indirect_data[240] * 241 + 0x0011223344556677ULL; break; case 241: acc ^= indirect_data[241] * 242 + 0x7766554433221100ULL; break; case 242: acc ^= indirect_data[242] * 243 + 0x1122334455667788ULL; break; case 243: acc ^= indirect_data[243] * 244 + 0x8877665544332211ULL; break; case 244: acc ^= indirect_data[244] * 245 + 0xAABBCCDDEEFF0011ULL; break; case 245: acc ^= indirect_data[245] * 246 + 0x1100FFEECCDDABULL; break; case 246: acc ^= indirect_data[246] * 247 + 0x33445566778899AALL; break; case 247: acc ^= indirect_data[247] * 248 + 0xAA99887766554433ULL; break; case 248: acc ^= indirect_data[248] * 249 + 0x0011223344556677ULL; break; case 249: acc ^= indirect_data[249] * 250 + 0x7766554433221100ULL; break;
    case 250: acc ^= indirect_data[250] * 251 + 0x13579BDF2468ACE0ULL; break; case 251: acc ^= indirect_data[251] * 252 + 0xE0AC8642FD9B7531ULL; break; case 252: acc ^= indirect_data[252] * 253 + 0x2468ACE013579BDFULL; break; case 253: acc ^= indirect_data[253] * 254 + 0xFD9B7531E0AC8642ULL; break; case 254: acc ^= indirect_data[254] * 255 + 0x1133557799BBCCDDULL; break; case 255: acc ^= indirect_data[255] * 256 + 0xDDCCBBAA77553311ULL; break;
  }
}

[[maybe_unused]] __attribute__((noinline)) void switchFuncB(uint64_t& acc, const std::vector<uint64_t>& indirect_data, uint64_t payload) {
  int idx = (payload >> 8) & 0xFF;
  switch (idx) {
    case 0: acc ^= indirect_data[0] * 1 + 0x1234567890ABCDEFULL; break; case 1: acc ^= indirect_data[1] * 2 + 0xFEDCBA0987654321ULL; break; case 2: acc ^= indirect_data[2] * 3 + 0x1122334455667788ULL; break; case 3: acc ^= indirect_data[3] * 4 + 0x8877665544332211ULL; break; case 4: acc ^= indirect_data[4] * 5 + 0xAABBCCDDEEFF0011ULL; break; case 5: acc ^= indirect_data[5] * 6 + 0x1100FFEECCDDABULL; break; case 6: acc ^= indirect_data[6] * 7 + 0x33445566778899AALL; break; case 7: acc ^= indirect_data[7] * 8 + 0xAA99887766554433ULL; break; case 8: acc ^= indirect_data[8] * 9 + 0x0011223344556677ULL; break; case 9: acc ^= indirect_data[9] * 10 + 0x7766554433221100ULL; break;
    case 10: acc ^= indirect_data[10] * 11 + 0x13579BDF2468ACE0ULL; break; case 11: acc ^= indirect_data[11] * 12 + 0xE0AC8642FD9B7531ULL; break; case 12: acc ^= indirect_data[12] * 13 + 0x2468ACE013579BDFULL; break; case 13: acc ^= indirect_data[13] * 14 + 0xFD9B7531E0AC8642ULL; break; case 14: acc ^= indirect_data[14] * 15 + 0x1133557799BBCCDDULL; break; case 15: acc ^= indirect_data[15] * 16 + 0xDDCCBBAA77553311ULL; break; case 16: acc ^= indirect_data[16] * 17 + 0x02468ACE13579BDFULL; break; case 17: acc ^= indirect_data[17] * 18 + 0xFD9B7531E0AC8642ULL; break; case 18: acc ^= indirect_data[18] * 19 + 0x3579BDF2468ACE0ULL; break; case 19: acc ^= indirect_data[19] * 20 + 0xE0AC8642FD9B7531ULL; break;
    case 20: acc ^= indirect_data[20] * 21 + 0x9876543210FEDCBAULL; break; case 21: acc ^= indirect_data[21] * 22 + 0x0123456789ABCDEFULL; break; case 22: acc ^= indirect_data[22] * 23 + 0xFEDCBA0987654321ULL; break; case 23: acc ^= indirect_data[23] * 24 + 0x1234567890ABCDEFULL; break; case 24: acc ^= indirect_data[24] * 25 + 0x33445566778899AALL; break; case 25: acc ^= indirect_data[25] * 26 + 0xAA99887766554433ULL; break; case 26: acc ^= indirect_data[26] * 27 + 0x0011223344556677ULL; break; case 27: acc ^= indirect_data[27] * 28 + 0x7766554433221100ULL; break; case 28: acc ^= indirect_data[28] * 29 + 0x1122334455667788ULL; break; case 29: acc ^= indirect_data[29] * 30 + 0x8877665544332211ULL; break;
    case 30: acc ^= indirect_data[30] * 31 + 0xABCDEF0123456789ULL; break; case 31: acc ^= indirect_data[31] * 32 + 0x9876543210FEDCBAULL; break; case 32: acc ^= indirect_data[32] * 33 + 0x1234567890ABCDEFULL; break; case 33: acc ^= indirect_data[33] * 34 + 0xFEDCBA0987654321ULL; break; case 34: acc ^= indirect_data[34] * 35 + 0x1133557799BBCCDDULL; break; case 35: acc ^= indirect_data[35] * 36 + 0xDDCCBBAA77553311ULL; break; case 36: acc ^= indirect_data[36] * 37 + 0x02468ACE13579BDFULL; break; case 37: acc ^= indirect_data[37] * 38 + 0xFD9B7531E0AC8642ULL; break; case 38: acc ^= indirect_data[38] * 39 + 0x3579BDF2468ACE0ULL; break; case 39: acc ^= indirect_data[39] * 40 + 0xE0AC8642FD9B7531ULL; break;
    case 40: acc ^= indirect_data[40] * 41 + 0x5566778899AABBCCULL; break; case 41: acc ^= indirect_data[41] * 42 + 0xCCBBAA9988776655ULL; break; case 42: acc ^= indirect_data[42] * 43 + 0x1122334455667788ULL; break; case 43: acc ^= indirect_data[43] * 44 + 0x8877665544332211ULL; break; case 44: acc ^= indirect_data[44] * 45 + 0xAABBCCDDEEFF0011ULL; break; case 45: acc ^= indirect_data[45] * 46 + 0x1100FFEECCDDABULL; break; case 46: acc ^= indirect_data[46] * 47 + 0x33445566778899AALL; break; case 47: acc ^= indirect_data[47] * 48 + 0xAA99887766554433ULL; break; case 48: acc ^= indirect_data[48] * 49 + 0x0011223344556677ULL; break; case 49: acc ^= indirect_data[49] * 50 + 0x7766554433221100ULL; break;
    case 50: acc ^= indirect_data[50] * 51 + 0x13579BDF2468ACE0ULL; break; case 51: acc ^= indirect_data[51] * 52 + 0xE0AC8642FD9B7531ULL; break; case 52: acc ^= indirect_data[52] * 53 + 0x2468ACE013579BDFULL; break; case 53: acc ^= indirect_data[53] * 54 + 0xFD9B7531E0AC8642ULL; break; case 54: acc ^= indirect_data[54] * 55 + 0x1133557799BBCCDDULL; break; case 55: acc ^= indirect_data[55] * 56 + 0xDDCCBBAA77553311ULL; break; case 56: acc ^= indirect_data[56] * 57 + 0x02468ACE13579BDFULL; break; case 57: acc ^= indirect_data[57] * 58 + 0xFD9B7531E0AC8642ULL; break; case 58: acc ^= indirect_data[58] * 59 + 0x3579BDF2468ACE0ULL; break; case 59: acc ^= indirect_data[59] * 60 + 0xE0AC8642FD9B7531ULL; break;
    case 60: acc ^= indirect_data[60] * 61 + 0xFEDCBA0987654321ULL; break; case 61: acc ^= indirect_data[61] * 62 + 0x1234567890ABCDEFULL; break; case 62: acc ^= indirect_data[62] * 63 + 0x33445566778899AALL; break; case 63: acc ^= indirect_data[63] * 64 + 0xAA99887766554433ULL; break; case 64: acc ^= indirect_data[64] * 65 + 0x0011223344556677ULL; break; case 65: acc ^= indirect_data[65] * 66 + 0x7766554433221100ULL; break; case 66: acc ^= indirect_data[66] * 67 + 0x1122334455667788ULL; break; case 67: acc ^= indirect_data[67] * 68 + 0x8877665544332211ULL; break; case 68: acc ^= indirect_data[68] * 69 + 0xAABBCCDDEEFF0011ULL; break; case 69: acc ^= indirect_data[69] * 70 + 0x1100FFEECCDDABULL; break;
    case 70: acc ^= indirect_data[70] * 71 + 0x33445566778899AALL; break; case 71: acc ^= indirect_data[71] * 72 + 0xAA99887766554433ULL; break; case 72: acc ^= indirect_data[72] * 73 + 0x0011223344556677ULL; break; case 73: acc ^= indirect_data[73] * 74 + 0x7766554433221100ULL; break; case 74: acc ^= indirect_data[74] * 75 + 0x13579BDF2468ACE0ULL; break; case 75: acc ^= indirect_data[75] * 76 + 0xE0AC8642FD9B7531ULL; break; case 76: acc ^= indirect_data[76] * 77 + 0x2468ACE013579BDFULL; break; case 77: acc ^= indirect_data[77] * 78 + 0xFD9B7531E0AC8642ULL; break; case 78: acc ^= indirect_data[78] * 79 + 0x1133557799BBCCDDULL; break; case 79: acc ^= indirect_data[79] * 80 + 0xDDCCBBAA77553311ULL; break;
    case 80: acc ^= indirect_data[80] * 81 + 0x02468ACE13579BDFULL; break; case 81: acc ^= indirect_data[81] * 82 + 0xFD9B7531E0AC8642ULL; break; case 82: acc ^= indirect_data[82] * 83 + 0x3579BDF2468ACE0ULL; break; case 83: acc ^= indirect_data[83] * 84 + 0xE0AC8642FD9B7531ULL; break; case 84: acc ^= indirect_data[84] * 85 + 0x9876543210FEDCBAULL; break; case 85: acc ^= indirect_data[85] * 86 + 0x0123456789ABCDEFULL; break; case 86: acc ^= indirect_data[86] * 87 + 0xFEDCBA0987654321ULL; break; case 87: acc ^= indirect_data[87] * 88 + 0x1234567890ABCDEFULL; break; case 88: acc ^= indirect_data[88] * 89 + 0x33445566778899AALL; break; case 89: acc ^= indirect_data[89] * 90 + 0xAA99887766554433ULL; break;
    case 90: acc ^= indirect_data[90] * 91 + 0x0011223344556677ULL; break; case 91: acc ^= indirect_data[91] * 92 + 0x7766554433221100ULL; break; case 92: acc ^= indirect_data[92] * 93 + 0x1122334455667788ULL; break; case 93: acc ^= indirect_data[93] * 94 + 0x8877665544332211ULL; break; case 94: acc ^= indirect_data[94] * 95 + 0xAABBCCDDEEFF0011ULL; break; case 95: acc ^= indirect_data[95] * 96 + 0x1100FFEECCDDABULL; break; case 96: acc ^= indirect_data[96] * 97 + 0x33445566778899AALL; break; case 97: acc ^= indirect_data[97] * 98 + 0xAA99887766554433ULL; break; case 98: acc ^= indirect_data[98] * 99 + 0x0011223344556677ULL; break; case 99: acc ^= indirect_data[99] * 100 + 0x7766554433221100ULL; break;
    case 100: acc ^= indirect_data[100] * 101 + 0x13579BDF2468ACE0ULL; break; case 101: acc ^= indirect_data[101] * 102 + 0xE0AC8642FD9B7531ULL; break; case 102: acc ^= indirect_data[102] * 103 + 0x2468ACE013579BDFULL; break; case 103: acc ^= indirect_data[103] * 104 + 0xFD9B7531E0AC8642ULL; break; case 104: acc ^= indirect_data[104] * 105 + 0x1133557799BBCCDDULL; break; case 105: acc ^= indirect_data[105] * 106 + 0xDDCCBBAA77553311ULL; break; case 106: acc ^= indirect_data[106] * 107 + 0x02468ACE13579BDFULL; break; case 107: acc ^= indirect_data[107] * 108 + 0xFD9B7531E0AC8642ULL; break; case 108: acc ^= indirect_data[108] * 109 + 0x3579BDF2468ACE0ULL; break; case 109: acc ^= indirect_data[109] * 110 + 0xE0AC8642FD9B7531ULL; break;
    case 110: acc ^= indirect_data[110] * 111 + 0xFEDCBA0987654321ULL; break; case 111: acc ^= indirect_data[111] * 112 + 0x1234567890ABCDEFULL; break; case 112: acc ^= indirect_data[112] * 113 + 0x33445566778899AALL; break; case 113: acc ^= indirect_data[113] * 114 + 0xAA99887766554433ULL; break; case 114: acc ^= indirect_data[114] * 115 + 0x0011223344556677ULL; break; case 115: acc ^= indirect_data[115] * 116 + 0x7766554433221100ULL; break; case 116: acc ^= indirect_data[116] * 117 + 0x1122334455667788ULL; break; case 117: acc ^= indirect_data[117] * 118 + 0x8877665544332211ULL; break; case 118: acc ^= indirect_data[118] * 119 + 0xAABBCCDDEEFF0011ULL; break; case 119: acc ^= indirect_data[119] * 120 + 0x1100FFEECCDDABULL; break;
    case 120: acc ^= indirect_data[120] * 121 + 0x33445566778899AALL; break; case 121: acc ^= indirect_data[121] * 122 + 0xAA99887766554433ULL; break; case 122: acc ^= indirect_data[122] * 123 + 0x0011223344556677ULL; break; case 123: acc ^= indirect_data[123] * 124 + 0x7766554433221100ULL; break; case 124: acc ^= indirect_data[124] * 125 + 0x13579BDF2468ACE0ULL; break; case 125: acc ^= indirect_data[125] * 126 + 0xE0AC8642FD9B7531ULL; break; case 126: acc ^= indirect_data[126] * 127 + 0x2468ACE013579BDFULL; break; case 127: acc ^= indirect_data[127] * 128 + 0xFD9B7531E0AC8642ULL; break; case 128: acc ^= indirect_data[128] * 129 + 0x1133557799BBCCDDULL; break; case 129: acc ^= indirect_data[129] * 130 + 0xDDCCBBAA77553311ULL; break;
    case 130: acc ^= indirect_data[130] * 131 + 0x02468ACE13579BDFULL; break; case 131: acc ^= indirect_data[131] * 132 + 0xFD9B7531E0AC8642ULL; break; case 132: acc ^= indirect_data[132] * 133 + 0x3579BDF2468ACE0ULL; break; case 133: acc ^= indirect_data[133] * 134 + 0xE0AC8642FD9B7531ULL; break; case 134: acc ^= indirect_data[134] * 135 + 0x9876543210FEDCBAULL; break; case 135: acc ^= indirect_data[135] * 136 + 0x0123456789ABCDEFULL; break; case 136: acc ^= indirect_data[136] * 137 + 0xFEDCBA0987654321ULL; break; case 137: acc ^= indirect_data[137] * 138 + 0x1234567890ABCDEFULL; break; case 138: acc ^= indirect_data[138] * 139 + 0x33445566778899AALL; break; case 139: acc ^= indirect_data[139] * 140 + 0xAA99887766554433ULL; break;
    case 140: acc ^= indirect_data[140] * 141 + 0x0011223344556677ULL; break; case 141: acc ^= indirect_data[141] * 142 + 0x7766554433221100ULL; break; case 142: acc ^= indirect_data[142] * 143 + 0x1122334455667788ULL; break; case 143: acc ^= indirect_data[143] * 144 + 0x8877665544332211ULL; break; case 144: acc ^= indirect_data[144] * 145 + 0xAABBCCDDEEFF0011ULL; break; case 145: acc ^= indirect_data[145] * 146 + 0x1100FFEECCDDABULL; break; case 146: acc ^= indirect_data[146] * 147 + 0x33445566778899AALL; break; case 147: acc ^= indirect_data[147] * 148 + 0xAA99887766554433ULL; break; case 148: acc ^= indirect_data[148] * 149 + 0x0011223344556677ULL; break; case 149: acc ^= indirect_data[149] * 150 + 0x7766554433221100ULL; break;
    case 150: acc ^= indirect_data[150] * 151 + 0x13579BDF2468ACE0ULL; break; case 151: acc ^= indirect_data[151] * 152 + 0xE0AC8642FD9B7531ULL; break; case 152: acc ^= indirect_data[152] * 153 + 0x2468ACE013579BDFULL; break; case 153: acc ^= indirect_data[153] * 154 + 0xFD9B7531E0AC8642ULL; break; case 154: acc ^= indirect_data[154] * 155 + 0x1133557799BBCCDDULL; break; case 155: acc ^= indirect_data[155] * 156 + 0xDDCCBBAA77553311ULL; break; case 156: acc ^= indirect_data[156] * 157 + 0x02468ACE13579BDFULL; break; case 157: acc ^= indirect_data[157] * 158 + 0xFD9B7531E0AC8642ULL; break; case 158: acc ^= indirect_data[158] * 159 + 0x3579BDF2468ACE0ULL; break; case 159: acc ^= indirect_data[159] * 160 + 0xE0AC8642FD9B7531ULL; break;
    case 160: acc ^= indirect_data[160] * 161 + 0xFEDCBA0987654321ULL; break; case 161: acc ^= indirect_data[161] * 162 + 0x1234567890ABCDEFULL; break; case 162: acc ^= indirect_data[162] * 163 + 0x33445566778899AALL; break; case 163: acc ^= indirect_data[163] * 164 + 0xAA99887766554433ULL; break; case 164: acc ^= indirect_data[164] * 165 + 0x0011223344556677ULL; break; case 165: acc ^= indirect_data[165] * 166 + 0x7766554433221100ULL; break; case 166: acc ^= indirect_data[166] * 167 + 0x1122334455667788ULL; break; case 167: acc ^= indirect_data[167] * 168 + 0x8877665544332211ULL; break; case 168: acc ^= indirect_data[168] * 169 + 0xAABBCCDDEEFF0011ULL; break; case 169: acc ^= indirect_data[169] * 170 + 0x1100FFEECCDDABULL; break;
    case 170: acc ^= indirect_data[170] * 171 + 0x33445566778899AALL; break; case 171: acc ^= indirect_data[171] * 172 + 0xAA99887766554433ULL; break; case 172: acc ^= indirect_data[172] * 173 + 0x0011223344556677ULL; break; case 173: acc ^= indirect_data[173] * 174 + 0x7766554433221100ULL; break; case 174: acc ^= indirect_data[174] * 175 + 0x13579BDF2468ACE0ULL; break; case 175: acc ^= indirect_data[175] * 176 + 0xE0AC8642FD9B7531ULL; break; case 176: acc ^= indirect_data[176] * 177 + 0x2468ACE013579BDFULL; break; case 177: acc ^= indirect_data[177] * 178 + 0xFD9B7531E0AC8642ULL; break; case 178: acc ^= indirect_data[178] * 179 + 0x1133557799BBCCDDULL; break; case 179: acc ^= indirect_data[179] * 180 + 0xDDCCBBAA77553311ULL; break;
    case 180: acc ^= indirect_data[180] * 181 + 0x02468ACE13579BDFULL; break; case 181: acc ^= indirect_data[181] * 182 + 0xFD9B7531E0AC8642ULL; break; case 182: acc ^= indirect_data[182] * 183 + 0x3579BDF2468ACE0ULL; break; case 183: acc ^= indirect_data[183] * 184 + 0xE0AC8642FD9B7531ULL; break; case 184: acc ^= indirect_data[184] * 185 + 0x9876543210FEDCBAULL; break; case 185: acc ^= indirect_data[185] * 186 + 0x0123456789ABCDEFULL; break; case 186: acc ^= indirect_data[186] * 187 + 0xFEDCBA0987654321ULL; break; case 187: acc ^= indirect_data[187] * 188 + 0x1234567890ABCDEFULL; break; case 188: acc ^= indirect_data[188] * 189 + 0x33445566778899AALL; break; case 189: acc ^= indirect_data[189] * 190 + 0xAA99887766554433ULL; break;
    case 190: acc ^= indirect_data[190] * 191 + 0x0011223344556677ULL; break; case 191: acc ^= indirect_data[191] * 192 + 0x7766554433221100ULL; break; case 192: acc ^= indirect_data[192] * 193 + 0x1122334455667788ULL; break; case 193: acc ^= indirect_data[193] * 194 + 0x8877665544332211ULL; break; case 194: acc ^= indirect_data[194] * 195 + 0xAABBCCDDEEFF0011ULL; break; case 195: acc ^= indirect_data[195] * 196 + 0x1100FFEECCDDABULL; break; case 196: acc ^= indirect_data[196] * 197 + 0x33445566778899AALL; break; case 197: acc ^= indirect_data[197] * 198 + 0xAA99887766554433ULL; break; case 198: acc ^= indirect_data[198] * 199 + 0x0011223344556677ULL; break; case 199: acc ^= indirect_data[199] * 200 + 0x7766554433221100ULL; break;
    case 200: acc ^= indirect_data[200] * 201 + 0x13579BDF2468ACE0ULL; break; case 201: acc ^= indirect_data[201] * 202 + 0xE0AC8642FD9B7531ULL; break; case 202: acc ^= indirect_data[202] * 203 + 0x2468ACE013579BDFULL; break; case 203: acc ^= indirect_data[203] * 204 + 0xFD9B7531E0AC8642ULL; break; case 204: acc ^= indirect_data[204] * 205 + 0x1133557799BBCCDDULL; break; case 205: acc ^= indirect_data[205] * 206 + 0xDDCCBBAA77553311ULL; break; case 206: acc ^= indirect_data[206] * 207 + 0x02468ACE13579BDFULL; break; case 207: acc ^= indirect_data[207] * 208 + 0xFD9B7531E0AC8642ULL; break; case 208: acc ^= indirect_data[208] * 209 + 0x3579BDF2468ACE0ULL; break; case 209: acc ^= indirect_data[209] * 210 + 0xE0AC8642FD9B7531ULL; break;
    case 210: acc ^= indirect_data[210] * 211 + 0xFEDCBA0987654321ULL; break; case 211: acc ^= indirect_data[211] * 212 + 0x1234567890ABCDEFULL; break; case 212: acc ^= indirect_data[212] * 213 + 0x33445566778899AALL; break; case 213: acc ^= indirect_data[213] * 214 + 0xAA99887766554433ULL; break; case 214: acc ^= indirect_data[214] * 215 + 0x0011223344556677ULL; break; case 215: acc ^= indirect_data[215] * 216 + 0x7766554433221100ULL; break; case 216: acc ^= indirect_data[216] * 217 + 0x1122334455667788ULL; break; case 217: acc ^= indirect_data[217] * 218 + 0x8877665544332211ULL; break; case 218: acc ^= indirect_data[218] * 219 + 0xAABBCCDDEEFF0011ULL; break; case 219: acc ^= indirect_data[219] * 220 + 0x1100FFEECCDDABULL; break;
    case 220: acc ^= indirect_data[220] * 221 + 0x33445566778899AALL; break; case 221: acc ^= indirect_data[221] * 222 + 0xAA99887766554433ULL; break; case 222: acc ^= indirect_data[222] * 223 + 0x0011223344556677ULL; break; case 223: acc ^= indirect_data[223] * 224 + 0x7766554433221100ULL; break; case 224: acc ^= indirect_data[224] * 225 + 0x13579BDF2468ACE0ULL; break; case 225: acc ^= indirect_data[225] * 226 + 0xE0AC8642FD9B7531ULL; break; case 226: acc ^= indirect_data[226] * 227 + 0x2468ACE013579BDFULL; break; case 227: acc ^= indirect_data[227] * 228 + 0xFD9B7531E0AC8642ULL; break; case 228: acc ^= indirect_data[228] * 229 + 0x1133557799BBCCDDULL; break; case 229: acc ^= indirect_data[229] * 230 + 0xDDCCBBAA77553311ULL; break;
    case 230: acc ^= indirect_data[230] * 231 + 0x02468ACE13579BDFULL; break; case 231: acc ^= indirect_data[231] * 232 + 0xFD9B7531E0AC8642ULL; break; case 232: acc ^= indirect_data[232] * 233 + 0x3579BDF2468ACE0ULL; break; case 233: acc ^= indirect_data[233] * 234 + 0xE0AC8642FD9B7531ULL; break; case 234: acc ^= indirect_data[234] * 235 + 0x9876543210FEDCBAULL; break; case 235: acc ^= indirect_data[235] * 236 + 0x0123456789ABCDEFULL; break; case 236: acc ^= indirect_data[236] * 237 + 0xFEDCBA0987654321ULL; break; case 237: acc ^= indirect_data[237] * 238 + 0x1234567890ABCDEFULL; break; case 238: acc ^= indirect_data[238] * 239 + 0x33445566778899AALL; break; case 239: acc ^= indirect_data[239] * 240 + 0xAA99887766554433ULL; break;
    case 240: acc ^= indirect_data[240] * 241 + 0x0011223344556677ULL; break; case 241: acc ^= indirect_data[241] * 242 + 0x7766554433221100ULL; break; case 242: acc ^= indirect_data[242] * 243 + 0x1122334455667788ULL; break; case 243: acc ^= indirect_data[243] * 244 + 0x8877665544332211ULL; break; case 244: acc ^= indirect_data[244] * 245 + 0xAABBCCDDEEFF0011ULL; break; case 245: acc ^= indirect_data[245] * 246 + 0x1100FFEECCDDABULL; break; case 246: acc ^= indirect_data[246] * 247 + 0x33445566778899AALL; break; case 247: acc ^= indirect_data[247] * 248 + 0xAA99887766554433ULL; break; case 248: acc ^= indirect_data[248] * 249 + 0x0011223344556677ULL; break; case 249: acc ^= indirect_data[249] * 250 + 0x7766554433221100ULL; break;
    case 250: acc ^= indirect_data[250] * 251 + 0x13579BDF2468ACE0ULL; break; case 251: acc ^= indirect_data[251] * 252 + 0xE0AC8642FD9B7531ULL; break; case 252: acc ^= indirect_data[252] * 253 + 0x2468ACE013579BDFULL; break; case 253: acc ^= indirect_data[253] * 254 + 0xFD9B7531E0AC8642ULL; break; case 254: acc ^= indirect_data[254] * 255 + 0x1133557799BBCCDDULL; break; case 255: acc ^= indirect_data[255] * 256 + 0xDDCCBBAA77553311ULL; break;
  }
}

[[maybe_unused]] __attribute__((noinline)) void switchFuncC(uint64_t& acc, const std::vector<uint64_t>& indirect_data, uint64_t payload) {
  int idx = (payload >> 16) & 0xFF;
  switch (idx) {
    case 0: acc ^= indirect_data[0] * 1 + 0x1234567890ABCDEFULL; break; case 1: acc ^= indirect_data[1] * 2 + 0xFEDCBA0987654321ULL; break; case 2: acc ^= indirect_data[2] * 3 + 0x1122334455667788ULL; break; case 3: acc ^= indirect_data[3] * 4 + 0x8877665544332211ULL; break; case 4: acc ^= indirect_data[4] * 5 + 0xAABBCCDDEEFF0011ULL; break; case 5: acc ^= indirect_data[5] * 6 + 0x1100FFEECCDDABULL; break; case 6: acc ^= indirect_data[6] * 7 + 0x33445566778899AALL; break; case 7: acc ^= indirect_data[7] * 8 + 0xAA99887766554433ULL; break; case 8: acc ^= indirect_data[8] * 9 + 0x0011223344556677ULL; break; case 9: acc ^= indirect_data[9] * 10 + 0x7766554433221100ULL; break;
    case 10: acc ^= indirect_data[10] * 11 + 0x13579BDF2468ACE0ULL; break; case 11: acc ^= indirect_data[11] * 12 + 0xE0AC8642FD9B7531ULL; break; case 12: acc ^= indirect_data[12] * 13 + 0x2468ACE013579BDFULL; break; case 13: acc ^= indirect_data[13] * 14 + 0xFD9B7531E0AC8642ULL; break; case 14: acc ^= indirect_data[14] * 15 + 0x1133557799BBCCDDULL; break; case 15: acc ^= indirect_data[15] * 16 + 0xDDCCBBAA77553311ULL; break; case 16: acc ^= indirect_data[16] * 17 + 0x02468ACE13579BDFULL; break; case 17: acc ^= indirect_data[17] * 18 + 0xFD9B7531E0AC8642ULL; break; case 18: acc ^= indirect_data[18] * 19 + 0x3579BDF2468ACE0ULL; break; case 19: acc ^= indirect_data[19] * 20 + 0xE0AC8642FD9B7531ULL; break;
    case 20: acc ^= indirect_data[20] * 21 + 0x9876543210FEDCBAULL; break; case 21: acc ^= indirect_data[21] * 22 + 0x0123456789ABCDEFULL; break; case 22: acc ^= indirect_data[22] * 23 + 0xFEDCBA0987654321ULL; break; case 23: acc ^= indirect_data[23] * 24 + 0x1234567890ABCDEFULL; break; case 24: acc ^= indirect_data[24] * 25 + 0x33445566778899AALL; break; case 25: acc ^= indirect_data[25] * 26 + 0xAA99887766554433ULL; break; case 26: acc ^= indirect_data[26] * 27 + 0x0011223344556677ULL; break; case 27: acc ^= indirect_data[27] * 28 + 0x7766554433221100ULL; break; case 28: acc ^= indirect_data[28] * 29 + 0x1122334455667788ULL; break; case 29: acc ^= indirect_data[29] * 30 + 0x8877665544332211ULL; break;
    case 30: acc ^= indirect_data[30] * 31 + 0xABCDEF0123456789ULL; break; case 31: acc ^= indirect_data[31] * 32 + 0x9876543210FEDCBAULL; break; case 32: acc ^= indirect_data[32] * 33 + 0x1234567890ABCDEFULL; break; case 33: acc ^= indirect_data[33] * 34 + 0xFEDCBA0987654321ULL; break; case 34: acc ^= indirect_data[34] * 35 + 0x1133557799BBCCDDULL; break; case 35: acc ^= indirect_data[35] * 36 + 0xDDCCBBAA77553311ULL; break; case 36: acc ^= indirect_data[36] * 37 + 0x02468ACE13579BDFULL; break; case 37: acc ^= indirect_data[37] * 38 + 0xFD9B7531E0AC8642ULL; break; case 38: acc ^= indirect_data[38] * 39 + 0x3579BDF2468ACE0ULL; break; case 39: acc ^= indirect_data[39] * 40 + 0xE0AC8642FD9B7531ULL; break;
    case 40: acc ^= indirect_data[40] * 41 + 0x5566778899AABBCCULL; break; case 41: acc ^= indirect_data[41] * 42 + 0xCCBBAA9988776655ULL; break; case 42: acc ^= indirect_data[42] * 43 + 0x1122334455667788ULL; break; case 43: acc ^= indirect_data[43] * 44 + 0x8877665544332211ULL; break; case 44: acc ^= indirect_data[44] * 45 + 0xAABBCCDDEEFF0011ULL; break; case 45: acc ^= indirect_data[45] * 46 + 0x1100FFEECCDDABULL; break; case 46: acc ^= indirect_data[46] * 47 + 0x33445566778899AALL; break; case 47: acc ^= indirect_data[47] * 48 + 0xAA99887766554433ULL; break; case 48: acc ^= indirect_data[48] * 49 + 0x0011223344556677ULL; break; case 49: acc ^= indirect_data[49] * 50 + 0x7766554433221100ULL; break;
    case 50: acc ^= indirect_data[50] * 51 + 0x13579BDF2468ACE0ULL; break; case 51: acc ^= indirect_data[51] * 52 + 0xE0AC8642FD9B7531ULL; break; case 52: acc ^= indirect_data[52] * 53 + 0x2468ACE013579BDFULL; break; case 53: acc ^= indirect_data[53] * 54 + 0xFD9B7531E0AC8642ULL; break; case 54: acc ^= indirect_data[54] * 55 + 0x1133557799BBCCDDULL; break; case 55: acc ^= indirect_data[55] * 56 + 0xDDCCBBAA77553311ULL; break; case 56: acc ^= indirect_data[56] * 57 + 0x02468ACE13579BDFULL; break; case 57: acc ^= indirect_data[57] * 58 + 0xFD9B7531E0AC8642ULL; break; case 58: acc ^= indirect_data[58] * 59 + 0x3579BDF2468ACE0ULL; break; case 59: acc ^= indirect_data[59] * 60 + 0xE0AC8642FD9B7531ULL; break;
    case 60: acc ^= indirect_data[60] * 61 + 0xFEDCBA0987654321ULL; break; case 61: acc ^= indirect_data[61] * 62 + 0x1234567890ABCDEFULL; break; case 62: acc ^= indirect_data[62] * 63 + 0x33445566778899AALL; break; case 63: acc ^= indirect_data[63] * 64 + 0xAA99887766554433ULL; break; case 64: acc ^= indirect_data[64] * 65 + 0x0011223344556677ULL; break; case 65: acc ^= indirect_data[65] * 66 + 0x7766554433221100ULL; break; case 66: acc ^= indirect_data[66] * 67 + 0x1122334455667788ULL; break; case 67: acc ^= indirect_data[67] * 68 + 0x8877665544332211ULL; break; case 68: acc ^= indirect_data[68] * 69 + 0xAABBCCDDEEFF0011ULL; break; case 69: acc ^= indirect_data[69] * 70 + 0x1100FFEECCDDABULL; break;
    case 70: acc ^= indirect_data[70] * 71 + 0x33445566778899AALL; break; case 71: acc ^= indirect_data[71] * 72 + 0xAA99887766554433ULL; break; case 72: acc ^= indirect_data[72] * 73 + 0x0011223344556677ULL; break; case 73: acc ^= indirect_data[73] * 74 + 0x7766554433221100ULL; break; case 74: acc ^= indirect_data[74] * 75 + 0x13579BDF2468ACE0ULL; break; case 75: acc ^= indirect_data[75] * 76 + 0xE0AC8642FD9B7531ULL; break; case 76: acc ^= indirect_data[76] * 77 + 0x2468ACE013579BDFULL; break; case 77: acc ^= indirect_data[77] * 78 + 0xFD9B7531E0AC8642ULL; break; case 78: acc ^= indirect_data[78] * 79 + 0x1133557799BBCCDDULL; break; case 79: acc ^= indirect_data[79] * 80 + 0xDDCCBBAA77553311ULL; break;
    case 80: acc ^= indirect_data[80] * 81 + 0x02468ACE13579BDFULL; break; case 81: acc ^= indirect_data[81] * 82 + 0xFD9B7531E0AC8642ULL; break; case 82: acc ^= indirect_data[82] * 83 + 0x3579BDF2468ACE0ULL; break; case 83: acc ^= indirect_data[83] * 84 + 0xE0AC8642FD9B7531ULL; break; case 84: acc ^= indirect_data[84] * 85 + 0x9876543210FEDCBAULL; break; case 85: acc ^= indirect_data[85] * 86 + 0x0123456789ABCDEFULL; break; case 86: acc ^= indirect_data[86] * 87 + 0xFEDCBA0987654321ULL; break; case 87: acc ^= indirect_data[87] * 88 + 0x1234567890ABCDEFULL; break; case 88: acc ^= indirect_data[88] * 89 + 0x33445566778899AALL; break; case 89: acc ^= indirect_data[89] * 90 + 0xAA99887766554433ULL; break;
    case 90: acc ^= indirect_data[90] * 91 + 0x0011223344556677ULL; break; case 91: acc ^= indirect_data[91] * 92 + 0x7766554433221100ULL; break; case 92: acc ^= indirect_data[92] * 93 + 0x1122334455667788ULL; break; case 93: acc ^= indirect_data[93] * 94 + 0x8877665544332211ULL; break; case 94: acc ^= indirect_data[94] * 95 + 0xAABBCCDDEEFF0011ULL; break; case 95: acc ^= indirect_data[95] * 96 + 0x1100FFEECCDDABULL; break; case 96: acc ^= indirect_data[96] * 97 + 0x33445566778899AALL; break; case 97: acc ^= indirect_data[97] * 98 + 0xAA99887766554433ULL; break; case 98: acc ^= indirect_data[98] * 99 + 0x0011223344556677ULL; break; case 99: acc ^= indirect_data[99] * 100 + 0x7766554433221100ULL; break;
    case 100: acc ^= indirect_data[100] * 101 + 0x13579BDF2468ACE0ULL; break; case 101: acc ^= indirect_data[101] * 102 + 0xE0AC8642FD9B7531ULL; break; case 102: acc ^= indirect_data[102] * 103 + 0x2468ACE013579BDFULL; break; case 103: acc ^= indirect_data[103] * 104 + 0xFD9B7531E0AC8642ULL; break; case 104: acc ^= indirect_data[104] * 105 + 0x1133557799BBCCDDULL; break; case 105: acc ^= indirect_data[105] * 106 + 0xDDCCBBAA77553311ULL; break; case 106: acc ^= indirect_data[106] * 107 + 0x02468ACE13579BDFULL; break; case 107: acc ^= indirect_data[107] * 108 + 0xFD9B7531E0AC8642ULL; break; case 108: acc ^= indirect_data[108] * 109 + 0x3579BDF2468ACE0ULL; break; case 109: acc ^= indirect_data[109] * 110 + 0xE0AC8642FD9B7531ULL; break;
    case 110: acc ^= indirect_data[110] * 111 + 0xFEDCBA0987654321ULL; break; case 111: acc ^= indirect_data[111] * 112 + 0x1234567890ABCDEFULL; break; case 112: acc ^= indirect_data[112] * 113 + 0x33445566778899AALL; break; case 113: acc ^= indirect_data[113] * 114 + 0xAA99887766554433ULL; break; case 114: acc ^= indirect_data[114] * 115 + 0x0011223344556677ULL; break; case 115: acc ^= indirect_data[115] * 116 + 0x7766554433221100ULL; break; case 116: acc ^= indirect_data[116] * 117 + 0x1122334455667788ULL; break; case 117: acc ^= indirect_data[117] * 118 + 0x8877665544332211ULL; break; case 118: acc ^= indirect_data[118] * 119 + 0xAABBCCDDEEFF0011ULL; break; case 119: acc ^= indirect_data[119] * 120 + 0x1100FFEECCDDABULL; break;
    case 120: acc ^= indirect_data[120] * 121 + 0x33445566778899AALL; break; case 121: acc ^= indirect_data[121] * 122 + 0xAA99887766554433ULL; break; case 122: acc ^= indirect_data[122] * 123 + 0x0011223344556677ULL; break; case 123: acc ^= indirect_data[123] * 124 + 0x7766554433221100ULL; break; case 124: acc ^= indirect_data[124] * 125 + 0x13579BDF2468ACE0ULL; break; case 125: acc ^= indirect_data[125] * 126 + 0xE0AC8642FD9B7531ULL; break; case 126: acc ^= indirect_data[126] * 127 + 0x2468ACE013579BDFULL; break; case 127: acc ^= indirect_data[127] * 128 + 0xFD9B7531E0AC8642ULL; break; case 128: acc ^= indirect_data[128] * 129 + 0x1133557799BBCCDDULL; break; case 129: acc ^= indirect_data[129] * 130 + 0xDDCCBBAA77553311ULL; break;
    case 130: acc ^= indirect_data[130] * 131 + 0x02468ACE13579BDFULL; break; case 131: acc ^= indirect_data[131] * 132 + 0xFD9B7531E0AC8642ULL; break; case 132: acc ^= indirect_data[132] * 133 + 0x3579BDF2468ACE0ULL; break; case 133: acc ^= indirect_data[133] * 134 + 0xE0AC8642FD9B7531ULL; break; case 134: acc ^= indirect_data[134] * 135 + 0x9876543210FEDCBAULL; break; case 135: acc ^= indirect_data[135] * 136 + 0x0123456789ABCDEFULL; break; case 136: acc ^= indirect_data[136] * 137 + 0xFEDCBA0987654321ULL; break; case 137: acc ^= indirect_data[137] * 138 + 0x1234567890ABCDEFULL; break; case 138: acc ^= indirect_data[138] * 139 + 0x33445566778899AALL; break; case 139: acc ^= indirect_data[139] * 140 + 0xAA99887766554433ULL; break;
    case 140: acc ^= indirect_data[140] * 141 + 0x0011223344556677ULL; break; case 141: acc ^= indirect_data[141] * 142 + 0x7766554433221100ULL; break; case 142: acc ^= indirect_data[142] * 143 + 0x1122334455667788ULL; break; case 143: acc ^= indirect_data[143] * 144 + 0x8877665544332211ULL; break; case 144: acc ^= indirect_data[144] * 145 + 0xAABBCCDDEEFF0011ULL; break; case 145: acc ^= indirect_data[145] * 146 + 0x1100FFEECCDDABULL; break; case 146: acc ^= indirect_data[146] * 147 + 0x33445566778899AALL; break; case 147: acc ^= indirect_data[147] * 148 + 0xAA99887766554433ULL; break; case 148: acc ^= indirect_data[148] * 149 + 0x0011223344556677ULL; break; case 149: acc ^= indirect_data[149] * 150 + 0x7766554433221100ULL; break;
    case 150: acc ^= indirect_data[150] * 151 + 0x13579BDF2468ACE0ULL; break; case 151: acc ^= indirect_data[151] * 152 + 0xE0AC8642FD9B7531ULL; break; case 152: acc ^= indirect_data[152] * 153 + 0x2468ACE013579BDFULL; break; case 153: acc ^= indirect_data[153] * 154 + 0xFD9B7531E0AC8642ULL; break; case 154: acc ^= indirect_data[154] * 155 + 0x1133557799BBCCDDULL; break; case 155: acc ^= indirect_data[155] * 156 + 0xDDCCBBAA77553311ULL; break; case 156: acc ^= indirect_data[156] * 157 + 0x02468ACE13579BDFULL; break; case 157: acc ^= indirect_data[157] * 158 + 0xFD9B7531E0AC8642ULL; break; case 158: acc ^= indirect_data[158] * 159 + 0x3579BDF2468ACE0ULL; break; case 159: acc ^= indirect_data[159] * 160 + 0xE0AC8642FD9B7531ULL; break;
    case 160: acc ^= indirect_data[160] * 161 + 0xFEDCBA0987654321ULL; break; case 161: acc ^= indirect_data[161] * 162 + 0x1234567890ABCDEFULL; break; case 162: acc ^= indirect_data[162] * 163 + 0x33445566778899AALL; break; case 163: acc ^= indirect_data[163] * 164 + 0xAA99887766554433ULL; break; case 164: acc ^= indirect_data[164] * 165 + 0x0011223344556677ULL; break; case 165: acc ^= indirect_data[165] * 166 + 0x7766554433221100ULL; break; case 166: acc ^= indirect_data[166] * 167 + 0x1122334455667788ULL; break; case 167: acc ^= indirect_data[167] * 168 + 0x8877665544332211ULL; break; case 168: acc ^= indirect_data[168] * 169 + 0xAABBCCDDEEFF0011ULL; break; case 169: acc ^= indirect_data[169] * 170 + 0x1100FFEECCDDABULL; break;
    case 170: acc ^= indirect_data[170] * 171 + 0x33445566778899AALL; break; case 171: acc ^= indirect_data[171] * 172 + 0xAA99887766554433ULL; break; case 172: acc ^= indirect_data[172] * 173 + 0x0011223344556677ULL; break; case 173: acc ^= indirect_data[173] * 174 + 0x7766554433221100ULL; break; case 174: acc ^= indirect_data[174] * 175 + 0x13579BDF2468ACE0ULL; break; case 175: acc ^= indirect_data[175] * 176 + 0xE0AC8642FD9B7531ULL; break; case 176: acc ^= indirect_data[176] * 177 + 0x2468ACE013579BDFULL; break; case 177: acc ^= indirect_data[177] * 178 + 0xFD9B7531E0AC8642ULL; break; case 178: acc ^= indirect_data[178] * 179 + 0x1133557799BBCCDDULL; break; case 179: acc ^= indirect_data[179] * 180 + 0xDDCCBBAA77553311ULL; break;
    case 180: acc ^= indirect_data[180] * 181 + 0x02468ACE13579BDFULL; break; case 181: acc ^= indirect_data[181] * 182 + 0xFD9B7531E0AC8642ULL; break; case 182: acc ^= indirect_data[182] * 183 + 0x3579BDF2468ACE0ULL; break; case 183: acc ^= indirect_data[183] * 184 + 0xE0AC8642FD9B7531ULL; break; case 184: acc ^= indirect_data[184] * 185 + 0x9876543210FEDCBAULL; break; case 185: acc ^= indirect_data[185] * 186 + 0x0123456789ABCDEFULL; break; case 186: acc ^= indirect_data[186] * 187 + 0xFEDCBA0987654321ULL; break; case 187: acc ^= indirect_data[187] * 188 + 0x1234567890ABCDEFULL; break; case 188: acc ^= indirect_data[188] * 189 + 0x33445566778899AALL; break; case 189: acc ^= indirect_data[189] * 190 + 0xAA99887766554433ULL; break;
    case 190: acc ^= indirect_data[190] * 191 + 0x0011223344556677ULL; break; case 191: acc ^= indirect_data[191] * 192 + 0x7766554433221100ULL; break; case 192: acc ^= indirect_data[192] * 193 + 0x1122334455667788ULL; break; case 193: acc ^= indirect_data[193] * 194 + 0x8877665544332211ULL; break; case 194: acc ^= indirect_data[194] * 195 + 0xAABBCCDDEEFF0011ULL; break; case 195: acc ^= indirect_data[195] * 196 + 0x1100FFEECCDDABULL; break; case 196: acc ^= indirect_data[196] * 197 + 0x33445566778899AALL; break; case 197: acc ^= indirect_data[197] * 198 + 0xAA99887766554433ULL; break; case 198: acc ^= indirect_data[198] * 199 + 0x0011223344556677ULL; break; case 199: acc ^= indirect_data[199] * 200 + 0x7766554433221100ULL; break;
    case 200: acc ^= indirect_data[200] * 201 + 0x13579BDF2468ACE0ULL; break; case 201: acc ^= indirect_data[201] * 202 + 0xE0AC8642FD9B7531ULL; break; case 202: acc ^= indirect_data[202] * 203 + 0x2468ACE013579BDFULL; break; case 203: acc ^= indirect_data[203] * 204 + 0xFD9B7531E0AC8642ULL; break; case 204: acc ^= indirect_data[204] * 205 + 0x1133557799BBCCDDULL; break; case 205: acc ^= indirect_data[205] * 206 + 0xDDCCBBAA77553311ULL; break; case 206: acc ^= indirect_data[206] * 207 + 0x02468ACE13579BDFULL; break; case 207: acc ^= indirect_data[207] * 208 + 0xFD9B7531E0AC8642ULL; break; case 208: acc ^= indirect_data[208] * 209 + 0x3579BDF2468ACE0ULL; break; case 209: acc ^= indirect_data[209] * 210 + 0xE0AC8642FD9B7531ULL; break;
    case 210: acc ^= indirect_data[210] * 211 + 0xFEDCBA0987654321ULL; break; case 211: acc ^= indirect_data[211] * 212 + 0x1234567890ABCDEFULL; break; case 212: acc ^= indirect_data[212] * 213 + 0x33445566778899AALL; break; case 213: acc ^= indirect_data[213] * 214 + 0xAA99887766554433ULL; break; case 214: acc ^= indirect_data[214] * 215 + 0x0011223344556677ULL; break; case 215: acc ^= indirect_data[215] * 216 + 0x7766554433221100ULL; break; case 216: acc ^= indirect_data[216] * 217 + 0x1122334455667788ULL; break; case 217: acc ^= indirect_data[217] * 218 + 0x8877665544332211ULL; break; case 218: acc ^= indirect_data[218] * 219 + 0xAABBCCDDEEFF0011ULL; break; case 219: acc ^= indirect_data[219] * 220 + 0x1100FFEECCDDABULL; break;
    case 220: acc ^= indirect_data[220] * 221 + 0x33445566778899AALL; break; case 221: acc ^= indirect_data[221] * 222 + 0xAA99887766554433ULL; break; case 222: acc ^= indirect_data[222] * 223 + 0x0011223344556677ULL; break; case 223: acc ^= indirect_data[223] * 224 + 0x7766554433221100ULL; break; case 224: acc ^= indirect_data[224] * 225 + 0x13579BDF2468ACE0ULL; break; case 225: acc ^= indirect_data[225] * 226 + 0xE0AC8642FD9B7531ULL; break; case 226: acc ^= indirect_data[226] * 227 + 0x2468ACE013579BDFULL; break; case 227: acc ^= indirect_data[227] * 228 + 0xFD9B7531E0AC8642ULL; break; case 228: acc ^= indirect_data[228] * 229 + 0x1133557799BBCCDDULL; break; case 229: acc ^= indirect_data[229] * 230 + 0xDDCCBBAA77553311ULL; break;
    case 230: acc ^= indirect_data[230] * 231 + 0x02468ACE13579BDFULL; break; case 231: acc ^= indirect_data[231] * 232 + 0xFD9B7531E0AC8642ULL; break; case 232: acc ^= indirect_data[232] * 233 + 0x3579BDF2468ACE0ULL; break; case 233: acc ^= indirect_data[233] * 234 + 0xE0AC8642FD9B7531ULL; break; case 234: acc ^= indirect_data[234] * 235 + 0x9876543210FEDCBAULL; break; case 235: acc ^= indirect_data[235] * 236 + 0x0123456789ABCDEFULL; break; case 236: acc ^= indirect_data[236] * 237 + 0xFEDCBA0987654321ULL; break; case 237: acc ^= indirect_data[237] * 238 + 0x1234567890ABCDEFULL; break; case 238: acc ^= indirect_data[238] * 239 + 0x33445566778899AALL; break; case 239: acc ^= indirect_data[239] * 240 + 0xAA99887766554433ULL; break;
    case 240: acc ^= indirect_data[240] * 241 + 0x0011223344556677ULL; break; case 241: acc ^= indirect_data[241] * 242 + 0x7766554433221100ULL; break; case 242: acc ^= indirect_data[242] * 243 + 0x1122334455667788ULL; break; case 243: acc ^= indirect_data[243] * 244 + 0x8877665544332211ULL; break; case 244: acc ^= indirect_data[244] * 245 + 0xAABBCCDDEEFF0011ULL; break; case 245: acc ^= indirect_data[245] * 246 + 0x1100FFEECCDDABULL; break; case 246: acc ^= indirect_data[246] * 247 + 0x33445566778899AALL; break; case 247: acc ^= indirect_data[247] * 248 + 0xAA99887766554433ULL; break; case 248: acc ^= indirect_data[248] * 249 + 0x0011223344556677ULL; break; case 249: acc ^= indirect_data[249] * 250 + 0x7766554433221100ULL; break;
    case 250: acc ^= indirect_data[250] * 251 + 0x13579BDF2468ACE0ULL; break; case 251: acc ^= indirect_data[251] * 252 + 0xE0AC8642FD9B7531ULL; break; case 252: acc ^= indirect_data[252] * 253 + 0x2468ACE013579BDFULL; break; case 253: acc ^= indirect_data[253] * 254 + 0xFD9B7531E0AC8642ULL; break; case 254: acc ^= indirect_data[254] * 255 + 0x1133557799BBCCDDULL; break; case 255: acc ^= indirect_data[255] * 256 + 0xDDCCBBAA77553311ULL; break;
  }
}

} // namespace

static void createAndDestroyMulti(size_t iters, size_t size) {
  static constexpr auto kSize = 1024;
  std::array<std::unique_ptr<IOBuf>, kSize> buffers;

  while (iters--) {
    for (auto i = 0; i < kSize; ++i) {
      buffers[i] = IOBuf::create(size);
    }
  }
}

BENCHMARK_NAMED_PARAM(createAndDestroyMulti, 64, 64)
BENCHMARK_NAMED_PARAM(createAndDestroyMulti, 256, 256)
BENCHMARK_NAMED_PARAM(createAndDestroyMulti, 1024, 1024)
BENCHMARK_NAMED_PARAM(createAndDestroyMulti, 4096, 4096)
BENCHMARK_NAMED_PARAM(createAndDestroyMulti, 5000, 5000)
BENCHMARK_NAMED_PARAM(createAndDestroyMulti, 5120, 5120)
BENCHMARK_NAMED_PARAM(createAndDestroyMulti, 8192, 8192)
BENCHMARK_NAMED_PARAM(createAndDestroyMulti, 10000, 10000)
BENCHMARK_NAMED_PARAM(createAndDestroyMulti, 10240, 10240)
BENCHMARK_NAMED_PARAM(createAndDestroyMulti, 16384, 16384)
BENCHMARK_NAMED_PARAM(createAndDestroyMulti, 17000, 17000)

static void createAndDestroyMulti_IndirectBranch(size_t iters) {
  static constexpr auto kSize = 1024;
  std::vector<uint64_t> indirect_data(256);
  for (size_t i = 0; i < indirect_data.size(); ++i) {
    indirect_data[i] = i * 3; // Some arbitrary calculation
  }

  std::array<std::unique_ptr<IOBuf>, kSize> buffers;
  uint64_t current = 0; // Payload

  while (iters--) {
    for (auto i = 0; i < kSize; ++i) {
      buffers[i] = IOBuf::create(128); // Small size to keep IOBuf creation fast
    }
    // Perform a series of operations that involve indirect branches
    // and pointer chasing to defeat branch prediction.
    for (int j = 0; j < 1024; ++j) {
      current ^= indirect_data[switch3(indirect_data, j, current)];
      current ^= indirect_data[switch2(indirect_data, j, current)];
    }
    folly::doNotOptimizeAway(current);
  }
}
BENCHMARK_NAMED_PARAM(createAndDestroyMulti_IndirectBranch, 1024)

BENCHMARK(createAndDestroyMulti_LargeAlloc, iters) {
  static constexpr auto kSize = 1024;
  std::array<std::unique_ptr<IOBuf>, kSize> buffers;
  size_t size = 1024 * 1024; // 1MB allocation

  std::vector<uint64_t> indirect_data(256);
  for (size_t i = 0; i < indirect_data.size(); ++i) {
    indirect_data[i] = i * 3 + 1; // Some arbitrary calculation
  }

  uint64_t current = 1; // Payload for the switch functions

  while (iters--) {
    for (auto i = 0; i < kSize; ++i) {
      buffers[i] = IOBuf::create(size);
    }
    // Inject the PROVEN PATTERN for L1i MPKI
    for (int j = 0; j < 1024; ++j) {
      switch (j % 3) {
        case 0:
          switchFuncA(current, indirect_data, current);
          break;
        case 1:
          switchFuncB(current, indirect_data, current);
          break;
        case 2:
          switchFuncC(current, indirect_data, current);
          break;
      }
    }
    folly::doNotOptimizeAway(current);
  }
}

static void createAndDestroyMulti_1024_Variant(size_t iters) {
  static constexpr auto kSize = 1024;
  std::array<std::unique_ptr<IOBuf>, kSize> buffers;

  while (iters--) {
    for (auto i = 0; i < kSize; ++i) {
      buffers[i] = IOBuf::create(1024);
    }
  }
}
BENCHMARK_NAMED_PARAM(createAndDestroyMulti_1024_Variant, 1024)


BENCHMARK_DRAW_LINE();

/**
 * folly/io/test:iobuf_benchmark -- --bm_min_iters 100000
 *  ============================================================================
 *  folly/io/test/IOBufBenchmark.cpp                relative  time/iter  iters/s
 *  ============================================================================
 *  createAndDestroy                                            17.42ns   57.41M
 *  cloneOneBenchmark                                           23.73ns   42.14M
 *  cloneOneIntoBenchmark                                       19.08ns   52.40M
 *  cloneBenchmark                                              24.92ns   40.13M
 *  cloneIntoBenchmark                                          21.74ns   45.99M
 *  moveBenchmark                                                8.61ns  116.17M
 *  copyBenchmark                                               21.23ns   47.11M
 *  cloneCoalescedBaseline                                     201.31ns    4.97M
 *  cloneCoalescedBenchmark                          555.93%    36.21ns   27.62M
 *  takeOwnershipBenchmark                                      36.01ns   27.77M
 *  ----------------------------------------------------------------------------
 *  createAndDestroyMulti(64)                                   32.74us   30.54K
 *  createAndDestroyMulti(256)                                  34.08us   29.34K
 *  createAndDestroyMulti(1024)                                 36.09us   27.71K
 *  createAndDestroyMulti(4096)                                 70.16us   14.25K
 *  createAndDestroyMulti(5000)                                 69.27us   14.44K
 *  createAndDestroyMulti(5120)                                 79.56us   12.57K
 *  createAndDestroyMulti(8192)                                 83.61us   11.96K
 *  createAndDestroyMulti(10000)                                84.54us   11.83K
 *  createAndDestroyMulti(10240)                                83.83us   11.93K
 *  createAndDestroyMulti(16384)                                93.03us   10.75K
 *  createAndDestroyMulti(17000)                                93.85us   10.66K
 *  ----------------------------------------------------------------------------
 *  createAndDestroyMulti_IndirectBranch(1024)                  88.62us   11.28K
 *  ----------------------------------------------------------------------------
 *  createAndDestroyMulti_LargeAlloc                           ??? ns    ??? K
 *  ============================================================================
 */

int main(int argc, char** argv) {
  folly::gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::runBenchmarks();
  return 0;
}
