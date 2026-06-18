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

namespace {
__attribute__((noinline)) void compute_heavy_part1(uint64_t& acc, const uint64_t* data) {
  for (int i = 0; i < 256; ++i) {
    switch (i) {
      case 0: acc ^= data[0] * 1; break;
      case 1: acc ^= data[1] * 2; break;
      case 2: acc ^= data[2] * 3; break;
      case 3: acc ^= data[3] * 4; break;
      case 4: acc ^= data[4] * 5; break;
      case 5: acc ^= data[5] * 6; break;
      case 6: acc ^= data[6] * 7; break;
      case 7: acc ^= data[7] * 8; break;
      case 8: acc ^= data[8] * 9; break;
      case 9: acc ^= data[9] * 10; break;
      case 10: acc ^= data[10] * 11; break;
      case 11: acc ^= data[11] * 12; break;
      case 12: acc ^= data[12] * 13; break;
      case 13: acc ^= data[13] * 14; break;
      case 14: acc ^= data[14] * 15; break;
      case 15: acc ^= data[15] * 16; break;
      case 16: acc ^= data[16] * 17; break;
      case 17: acc ^= data[17] * 18; break;
      case 18: acc ^= data[18] * 19; break;
      case 19: acc ^= data[19] * 20; break;
      case 20: acc ^= data[20] * 21; break;
      case 21: acc ^= data[21] * 22; break;
      case 22: acc ^= data[22] * 23; break;
      case 23: acc ^= data[23] * 24; break;
      case 24: acc ^= data[24] * 25; break;
      case 25: acc ^= data[25] * 26; break;
      case 26: acc ^= data[26] * 27; break;
      case 27: acc ^= data[27] * 28; break;
      case 28: acc ^= data[28] * 29; break;
      case 29: acc ^= data[29] * 30; break;
      case 30: acc ^= data[30] * 31; break;
      case 31: acc ^= data[31] * 32; break;
      case 32: acc ^= data[32] * 33; break;
      case 33: acc ^= data[33] * 34; break;
      case 34: acc ^= data[34] * 35; break;
      case 35: acc ^= data[35] * 36; break;
      case 36: acc ^= data[36] * 37; break;
      case 37: acc ^= data[37] * 38; break;
      case 38: acc ^= data[38] * 39; break;
      case 39: acc ^= data[39] * 40; break;
      case 40: acc ^= data[40] * 41; break;
      case 41: acc ^= data[41] * 42; break;
      case 42: acc ^= data[42] * 43; break;
      case 43: acc ^= data[43] * 44; break;
      case 44: acc ^= data[44] * 45; break;
      case 45: acc ^= data[45] * 46; break;
      case 46: acc ^= data[46] * 47; break;
      case 47: acc ^= data[47] * 48; break;
      case 48: acc ^= data[48] * 49; break;
      case 49: acc ^= data[49] * 50; break;
      case 50: acc ^= data[50] * 51; break;
      case 51: acc ^= data[51] * 52; break;
      case 52: acc ^= data[52] * 53; break;
      case 53: acc ^= data[53] * 54; break;
      case 54: acc ^= data[54] * 55; break;
      case 55: acc ^= data[55] * 56; break;
      case 56: acc ^= data[56] * 57; break;
      case 57: acc ^= data[57] * 58; break;
      case 58: acc ^= data[58] * 59; break;
      case 59: acc ^= data[59] * 60; break;
      case 60: acc ^= data[60] * 61; break;
      case 61: acc ^= data[61] * 62; break;
      case 62: acc ^= data[62] * 63; break;
      case 63: acc ^= data[63] * 64; break;
      case 64: acc ^= data[64] * 65; break;
      case 65: acc ^= data[65] * 66; break;
      case 66: acc ^= data[66] * 67; break;
      case 67: acc ^= data[67] * 68; break;
      case 68: acc ^= data[68] * 69; break;
      case 69: acc ^= data[69] * 70; break;
      case 70: acc ^= data[70] * 71; break;
      case 71: acc ^= data[71] * 72; break;
      case 72: acc ^= data[72] * 73; break;
      case 73: acc ^= data[73] * 74; break;
      case 74: acc ^= data[74] * 75; break;
      case 75: acc ^= data[75] * 76; break;
      case 76: acc ^= data[76] * 77; break;
      case 77: acc ^= data[77] * 78; break;
      case 78: acc ^= data[78] * 79; break;
      case 79: acc ^= data[79] * 80; break;
      case 80: acc ^= data[80] * 81; break;
      case 81: acc ^= data[81] * 82; break;
      case 82: acc ^= data[82] * 83; break;
      case 83: acc ^= data[83] * 84; break;
      case 84: acc ^= data[84] * 85; break;
      case 85: acc ^= data[85] * 86; break;
      case 86: acc ^= data[86] * 87; break;
      case 87: acc ^= data[87] * 88; break;
      case 88: acc ^= data[88] * 89; break;
      case 89: acc ^= data[89] * 90; break;
      case 90: acc ^= data[90] * 91; break;
      case 91: acc ^= data[91] * 92; break;
      case 92: acc ^= data[92] * 93; break;
      case 93: acc ^= data[93] * 94; break;
      case 94: acc ^= data[94] * 95; break;
      case 95: acc ^= data[95] * 96; break;
      case 96: acc ^= data[96] * 97; break;
      case 97: acc ^= data[97] * 98; break;
      case 98: acc ^= data[98] * 99; break;
      case 99: acc ^= data[99] * 100; break;
      case 100: acc ^= data[100] * 101; break;
      case 101: acc ^= data[101] * 102; break;
      case 102: acc ^= data[102] * 103; break;
      case 103: acc ^= data[103] * 104; break;
      case 104: acc ^= data[104] * 105; break;
      case 105: acc ^= data[105] * 106; break;
      case 106: acc ^= data[106] * 107; break;
      case 107: acc ^= data[107] * 108; break;
      case 108: acc ^= data[108] * 109; break;
      case 109: acc ^= data[109] * 110; break;
      case 110: acc ^= data[110] * 111; break;
      case 111: acc ^= data[111] * 112; break;
      case 112: acc ^= data[112] * 113; break;
      case 113: acc ^= data[113] * 114; break;
      case 114: acc ^= data[114] * 115; break;
      case 115: acc ^= data[115] * 116; break;
      case 116: acc ^= data[116] * 117; break;
      case 117: acc ^= data[117] * 118; break;
      case 118: acc ^= data[118] * 119; break;
      case 119: acc ^= data[119] * 120; break;
      case 120: acc ^= data[120] * 121; break;
      case 121: acc ^= data[121] * 122; break;
      case 122: acc ^= data[122] * 123; break;
      case 123: acc ^= data[123] * 124; break;
      case 124: acc ^= data[124] * 125; break;
      case 125: acc ^= data[125] * 126; break;
      case 126: acc ^= data[126] * 127; break;
      case 127: acc ^= data[127] * 128; break;
      case 128: acc ^= data[128] * 129; break;
      case 129: acc ^= data[129] * 130; break;
      case 130: acc ^= data[130] * 131; break;
      case 131: acc ^= data[131] * 132; break;
      case 132: acc ^= data[132] * 133; break;
      case 133: acc ^= data[133] * 134; break;
      case 134: acc ^= data[134] * 135; break;
      case 135: acc ^= data[135] * 136; break;
      case 136: acc ^= data[136] * 137; break;
      case 137: acc ^= data[137] * 138; break;
      case 138: acc ^= data[138] * 139; break;
      case 139: acc ^= data[139] * 140; break;
      case 140: acc ^= data[140] * 141; break;
      case 141: acc ^= data[141] * 142; break;
      case 142: acc ^= data[142] * 143; break;
      case 143: acc ^= data[143] * 144; break;
      case 144: acc ^= data[144] * 145; break;
      case 145: acc ^= data[145] * 146; break;
      case 146: acc ^= data[146] * 147; break;
      case 147: acc ^= data[147] * 148; break;
      case 148: acc ^= data[148] * 149; break;
      case 149: acc ^= data[149] * 150; break;
      case 150: acc ^= data[150] * 151; break;
      case 151: acc ^= data[151] * 152; break;
      case 152: acc ^= data[152] * 153; break;
      case 153: acc ^= data[153] * 154; break;
      case 154: acc ^= data[154] * 155; break;
      case 155: acc ^= data[155] * 156; break;
      case 156: acc ^= data[156] * 157; break;
      case 157: acc ^= data[157] * 158; break;
      case 158: acc ^= data[158] * 159; break;
      case 159: acc ^= data[159] * 160; break;
      case 160: acc ^= data[160] * 161; break;
      case 161: acc ^= data[161] * 162; break;
      case 162: acc ^= data[162] * 163; break;
      case 163: acc ^= data[163] * 164; break;
      case 164: acc ^= data[164] * 165; break;
      case 165: acc ^= data[165] * 166; break;
      case 166: acc ^= data[166] * 167; break;
      case 167: acc ^= data[167] * 168; break;
      case 168: acc ^= data[168] * 169; break;
      case 169: acc ^= data[169] * 170; break;
      case 170: acc ^= data[170] * 171; break;
      case 171: acc ^= data[171] * 172; break;
      case 172: acc ^= data[172] * 173; break;
      case 173: acc ^= data[173] * 174; break;
      case 174: acc ^= data[174] * 175; break;
      case 175: acc ^= data[175] * 176; break;
      case 176: acc ^= data[176] * 177; break;
      case 177: acc ^= data[177] * 178; break;
      case 178: acc ^= data[178] * 179; break;
      case 179: acc ^= data[179] * 180; break;
      case 180: acc ^= data[180] * 181; break;
      case 181: acc ^= data[181] * 182; break;
      case 182: acc ^= data[182] * 183; break;
      case 183: acc ^= data[183] * 184; break;
      case 184: acc ^= data[184] * 185; break;
      case 185: acc ^= data[185] * 186; break;
      case 186: acc ^= data[186] * 187; break;
      case 187: acc ^= data[187] * 188; break;
      case 188: acc ^= data[188] * 189; break;
      case 189: acc ^= data[189] * 190; break;
      case 190: acc ^= data[190] * 191; break;
      case 191: acc ^= data[191] * 192; break;
      case 192: acc ^= data[192] * 193; break;
      case 193: acc ^= data[193] * 194; break;
      case 194: acc ^= data[194] * 195; break;
      case 195: acc ^= data[195] * 196; break;
      case 196: acc ^= data[196] * 197; break;
      case 197: acc ^= data[197] * 198; break;
      case 198: acc ^= data[198] * 199; break;
      case 199: acc ^= data[199] * 200; break;
      case 200: acc ^= data[200] * 201; break;
      case 201: acc ^= data[201] * 202; break;
      case 202: acc ^= data[202] * 203; break;
      case 203: acc ^= data[203] * 204; break;
      case 204: acc ^= data[204] * 205; break;
      case 205: acc ^= data[205] * 206; break;
      case 206: acc ^= data[206] * 207; break;
      case 207: acc ^= data[207] * 208; break;
      case 208: acc ^= data[208] * 209; break;
      case 209: acc ^= data[209] * 210; break;
      case 210: acc ^= data[210] * 211; break;
      case 211: acc ^= data[211] * 212; break;
      case 212: acc ^= data[212] * 213; break;
      case 213: acc ^= data[213] * 214; break;
      case 214: acc ^= data[214] * 215; break;
      case 215: acc ^= data[215] * 216; break;
      case 216: acc ^= data[216] * 217; break;
      case 217: acc ^= data[217] * 218; break;
      case 218: acc ^= data[218] * 219; break;
      case 219: acc ^= data[219] * 220; break;
      case 220: acc ^= data[220] * 221; break;
      case 221: acc ^= data[221] * 222; break;
      case 222: acc ^= data[222] * 223; break;
      case 223: acc ^= data[223] * 224; break;
      case 224: acc ^= data[224] * 225; break;
      case 225: acc ^= data[225] * 226; break;
      case 226: acc ^= data[226] * 227; break;
      case 227: acc ^= data[227] * 228; break;
      case 228: acc ^= data[228] * 229; break;
      case 229: acc ^= data[229] * 230; break;
      case 230: acc ^= data[230] * 231; break;
      case 231: acc ^= data[231] * 232; break;
      case 232: acc ^= data[232] * 233; break;
      case 233: acc ^= data[233] * 234; break;
      case 234: acc ^= data[234] * 235; break;
      case 235: acc ^= data[235] * 236; break;
      case 236: acc ^= data[236] * 237; break;
      case 237: acc ^= data[237] * 238; break;
      case 238: acc ^= data[238] * 239; break;
      case 239: acc ^= data[239] * 240; break;
      case 240: acc ^= data[240] * 241; break;
      case 241: acc ^= data[241] * 242; break;
      case 242: acc ^= data[242] * 243; break;
      case 243: acc ^= data[243] * 244; break;
      case 244: acc ^= data[244] * 245; break;
      case 245: acc ^= data[245] * 246; break;
      case 246: acc ^= data[246] * 247; break;
      case 247: acc ^= data[247] * 248; break;
      case 248: acc ^= data[248] * 249; break;
      case 249: acc ^= data[249] * 250; break;
      case 250: acc ^= data[250] * 251; break;
      case 251: acc ^= data[251] * 252; break;
      case 252: acc ^= data[252] * 253; break;
      case 253: acc ^= data[253] * 254; break;
      case 254: acc ^= data[254] * 255; break;
      case 255: acc ^= data[255] * 256; break;
    }
  }
}

__attribute__((noinline)) void compute_heavy_part2(uint64_t& acc, const uint64_t* data) {
  for (int i = 0; i < 256; ++i) {
    switch (i) {
      case 0: acc += data[0] + 1; break;
      case 1: acc += data[1] + 2; break;
      case 2: acc += data[2] + 3; break;
      case 3: acc += data[3] + 4; break;
      case 4: acc += data[4] + 5; break;
      case 5: acc += data[5] + 6; break;
      case 6: acc += data[6] + 7; break;
      case 7: acc += data[7] + 8; break;
      case 8: acc += data[8] + 9; break;
      case 9: acc += data[9] + 10; break;
      case 10: acc += data[10] + 11; break;
      case 11: acc += data[11] + 12; break;
      case 12: acc += data[12] + 13; break;
      case 13: acc += data[13] + 14; break;
      case 14: acc += data[14] + 15; break;
      case 15: acc += data[15] + 16; break;
      case 16: acc += data[16] + 17; break;
      case 17: acc += data[17] + 18; break;
      case 18: acc += data[18] + 19; break;
      case 19: acc += data[19] + 20; break;
      case 20: acc += data[20] + 21; break;
      case 21: acc += data[21] + 22; break;
      case 22: acc += data[22] + 23; break;
      case 23: acc += data[23] + 24; break;
      case 24: acc += data[24] + 25; break;
      case 25: acc += data[25] + 26; break;
      case 26: acc += data[26] + 27; break;
      case 27: acc += data[27] + 28; break;
      case 28: acc += data[28] + 29; break;
      case 29: acc += data[29] + 30; break;
      case 30: acc += data[30] + 31; break;
      case 31: acc += data[31] + 32; break;
      case 32: acc += data[32] + 33; break;
      case 33: acc += data[33] + 34; break;
      case 34: acc += data[34] + 35; break;
      case 35: acc += data[35] + 36; break;
      case 36: acc += data[36] + 37; break;
      case 37: acc += data[37] + 38; break;
      case 38: acc += data[38] + 39; break;
      case 39: acc += data[39] + 40; break;
      case 40: acc += data[40] + 41; break;
      case 41: acc += data[41] + 42; break;
      case 42: acc += data[42] + 43; break;
      case 43: acc += data[43] + 44; break;
      case 44: acc += data[44] + 45; break;
      case 45: acc += data[45] + 46; break;
      case 46: acc += data[46] + 47; break;
      case 47: acc += data[47] + 48; break;
      case 48: acc += data[48] + 49; break;
      case 49: acc += data[49] + 50; break;
      case 50: acc += data[50] + 51; break;
      case 51: acc += data[51] + 52; break;
      case 52: acc += data[52] + 53; break;
      case 53: acc += data[53] + 54; break;
      case 54: acc += data[54] + 55; break;
      case 55: acc += data[55] + 56; break;
      case 56: acc += data[56] + 57; break;
      case 57: acc += data[57] + 58; break;
      case 58: acc += data[58] + 59; break;
      case 59: acc += data[59] + 60; break;
      case 60: acc += data[60] + 61; break;
      case 61: acc += data[61] + 62; break;
      case 62: acc += data[62] + 63; break;
      case 63: acc += data[63] + 64; break;
      case 64: acc += data[64] + 65; break;
      case 65: acc += data[65] + 66; break;
      case 66: acc += data[66] + 67; break;
      case 67: acc += data[67] + 68; break;
      case 68: acc += data[68] + 69; break;
      case 69: acc += data[69] + 70; break;
      case 70: acc += data[70] + 71; break;
      case 71: acc += data[71] + 72; break;
      case 72: acc += data[72] + 73; break;
      case 73: acc += data[73] + 74; break;
      case 74: acc += data[74] + 75; break;
      case 75: acc += data[75] + 76; break;
      case 76: acc += data[76] + 77; break;
      case 77: acc += data[77] + 78; break;
      case 78: acc += data[78] + 79; break;
      case 79: acc += data[79] + 80; break;
      case 80: acc += data[80] + 81; break;
      case 81: acc += data[81] + 82; break;
      case 82: acc += data[82] + 83; break;
      case 83: acc += data[83] + 84; break;
      case 84: acc += data[84] + 85; break;
      case 85: acc += data[85] + 86; break;
      case 86: acc += data[86] + 87; break;
      case 87: acc += data[87] + 88; break;
      case 88: acc += data[88] + 89; break;
      case 89: acc += data[89] + 90; break;
      case 90: acc += data[90] + 91; break;
      case 91: acc += data[91] + 92; break;
      case 92: acc += data[92] + 93; break;
      case 93: acc += data[93] + 94; break;
      case 94: acc += data[94] + 95; break;
      case 95: acc += data[95] + 96; break;
      case 96: acc += data[96] + 97; break;
      case 97: acc += data[97] + 98; break;
      case 98: acc += data[98] + 99; break;
      case 99: acc += data[99] + 100; break;
      case 100: acc += data[100] + 101; break;
      case 101: acc += data[101] + 102; break;
      case 102: acc += data[102] + 103; break;
      case 103: acc += data[103] + 104; break;
      case 104: acc += data[104] + 105; break;
      case 105: acc += data[105] + 106; break;
      case 106: acc += data[106] + 107; break;
      case 107: acc += data[107] + 108; break;
      case 108: acc += data[108] + 109; break;
      case 109: acc += data[109] + 110; break;
      case 110: acc += data[110] + 111; break;
      case 111: acc += data[111] + 112; break;
      case 112: acc += data[112] + 113; break;
      case 113: acc += data[113] + 114; break;
      case 114: acc += data[114] + 115; break;
      case 115: acc += data[115] + 116; break;
      case 116: acc += data[116] + 117; break;
      case 117: acc += data[117] + 118; break;
      case 118: acc += data[118] + 119; break;
      case 119: acc += data[119] + 120; break;
      case 120: acc += data[120] + 121; break;
      case 121: acc += data[121] + 122; break;
      case 122: acc += data[122] + 123; break;
      case 123: acc += data[123] + 124; break;
      case 124: acc += data[124] + 125; break;
      case 125: acc += data[125] + 126; break;
      case 126: acc += data[126] + 127; break;
      case 127: acc += data[127] + 128; break;
      case 128: acc += data[128] + 129; break;
      case 129: acc += data[129] + 130; break;
      case 130: acc += data[130] + 131; break;
      case 131: acc += data[131] + 132; break;
      case 132: acc += data[132] + 133; break;
      case 133: acc += data[133] + 134; break;
      case 134: acc += data[134] + 135; break;
      case 135: acc += data[135] + 136; break;
      case 136: acc += data[136] + 137; break;
      case 137: acc += data[137] + 138; break;
      case 138: acc += data[138] + 139; break;
      case 139: acc += data[139] + 140; break;
      case 140: acc += data[140] + 141; break;
      case 141: acc += data[141] + 142; break;
      case 142: acc += data[142] + 143; break;
      case 143: acc += data[143] + 144; break;
      case 144: acc += data[144] + 145; break;
      case 145: acc += data[145] + 146; break;
      case 146: acc += data[146] + 147; break;
      case 147: acc += data[147] + 148; break;
      case 148: acc += data[148] + 149; break;
      case 149: acc += data[149] + 150; break;
      case 150: acc += data[150] + 151; break;
      case 151: acc += data[151] + 152; break;
      case 152: acc += data[152] + 153; break;
      case 153: acc += data[153] + 154; break;
      case 154: acc += data[154] + 155; break;
      case 155: acc += data[155] + 156; break;
      case 156: acc += data[156] + 157; break;
      case 157: acc += data[157] + 158; break;
      case 158: acc += data[158] + 159; break;
      case 159: acc += data[159] + 160; break;
      case 160: acc += data[160] + 161; break;
      case 161: acc += data[161] + 162; break;
      case 162: acc += data[162] + 163; break;
      case 163: acc += data[163] + 164; break;
      case 164: acc += data[164] + 165; break;
      case 165: acc += data[165] + 166; break;
      case 166: acc += data[166] + 167; break;
      case 167: acc += data[167] + 168; break;
      case 168: acc += data[168] + 169; break;
      case 169: acc += data[169] + 170; break;
      case 170: acc += data[170] + 171; break;
      case 171: acc += data[171] + 172; break;
      case 172: acc += data[172] + 173; break;
      case 173: acc += data[173] + 174; break;
      case 174: acc += data[174] + 175; break;
      case 175: acc += data[175] + 176; break;
      case 176: acc += data[176] + 177; break;
      case 177: acc += data[177] + 178; break;
      case 178: acc += data[178] + 179; break;
      case 179: acc += data[179] + 180; break;
      case 180: acc += data[180] + 181; break;
      case 181: acc += data[181] + 182; break;
      case 182: acc += data[182] + 183; break;
      case 183: acc += data[183] + 184; break;
      case 184: acc += data[184] + 185; break;
      case 185: acc += data[185] + 186; break;
      case 186: acc += data[186] + 187; break;
      case 187: acc += data[187] + 188; break;
      case 188: acc += data[188] + 189; break;
      case 189: acc += data[189] + 190; break;
      case 190: acc += data[190] + 191; break;
      case 191: acc += data[191] + 192; break;
      case 192: acc += data[192] + 193; break;
      case 193: acc += data[193] + 194; break;
      case 194: acc += data[194] + 195; break;
      case 195: acc += data[195] + 196; break;
      case 196: acc += data[196] + 197; break;
      case 197: acc += data[197] + 198; break;
      case 198: acc += data[198] + 199; break;
      case 199: acc += data[199] + 200; break;
      case 200: acc += data[200] + 201; break;
      case 201: acc += data[201] + 202; break;
      case 202: acc += data[202] + 203; break;
      case 203: acc += data[203] + 204; break;
      case 204: acc += data[204] + 205; break;
      case 205: acc += data[205] + 206; break;
      case 206: acc += data[206] + 207; break;
      case 207: acc += data[207] + 208; break;
      case 208: acc += data[208] + 209; break;
      case 209: acc += data[209] + 210; break;
      case 210: acc += data[210] + 211; break;
      case 211: acc += data[211] + 212; break;
      case 212: acc += data[212] + 213; break;
      case 213: acc += data[213] + 214; break;
      case 214: acc += data[214] + 215; break;
      case 215: acc += data[215] + 216; break;
      case 216: acc += data[216] + 217; break;
      case 217: acc += data[217] + 218; break;
      case 218: acc += data[218] + 219; break;
      case 219: acc += data[219] + 220; break;
      case 220: acc += data[220] + 221; break;
      case 221: acc += data[221] + 222; break;
      case 222: acc += data[222] + 223; break;
      case 223: acc += data[223] + 224; break;
      case 224: acc += data[224] + 225; break;
      case 225: acc += data[225] + 226; break;
      case 226: acc += data[226] + 227; break;
      case 227: acc += data[227] + 228; break;
      case 228: acc += data[228] + 229; break;
      case 229: acc += data[229] + 230; break;
      case 230: acc += data[230] + 231; break;
      case 231: acc += data[231] + 232; break;
      case 232: acc += data[232] + 233; break;
      case 233: acc += data[233] + 234; break;
      case 234: acc += data[234] + 235; break;
      case 235: acc += data[235] + 236; break;
      case 236: acc += data[236] + 237; break;
      case 237: acc += data[237] + 238; break;
      case 238: acc += data[238] + 239; break;
      case 239: acc += data[239] + 240; break;
      case 240: acc += data[240] + 241; break;
      case 241: acc += data[241] + 242; break;
      case 242: acc += data[242] + 243; break;
      case 243: acc += data[243] + 244; break;
      case 244: acc += data[244] + 245; break;
      case 245: acc += data[245] + 246; break;
      case 246: acc += data[246] + 247; break;
      case 247: acc += data[247] + 248; break;
      case 248: acc += data[248] + 249; break;
      case 249: acc += data[249] + 250; break;
      case 250: acc += data[250] + 251; break;
      case 251: acc += data[251] + 252; break;
      case 252: acc += data[252] + 253; break;
      case 253: acc += data[253] + 254; break;
      case 254: acc += data[254] + 255; break;
      case 255: acc += data[255] + 256; break;
    }
  }
}

__attribute__((noinline)) void compute_heavy_part3(uint64_t& acc, const uint64_t* data) {
  for (int i = 0; i < 256; ++i) {
    switch (i) {
      case 0: acc ^= data[0] ^ 1; break;
      case 1: acc ^= data[1] ^ 2; break;
      case 2: acc ^= data[2] ^ 3; break;
      case 3: acc ^= data[3] ^ 4; break;
      case 4: acc ^= data[4] ^ 5; break;
      case 5: acc ^= data[5] ^ 6; break;
      case 6: acc ^= data[6] ^ 7; break;
      case 7: acc ^= data[7] ^ 8; break;
      case 8: acc ^= data[8] ^ 9; break;
      case 9: acc ^= data[9] ^ 10; break;
      case 10: acc ^= data[10] ^ 11; break;
      case 11: acc ^= data[11] ^ 12; break;
      case 12: acc ^= data[12] ^ 13; break;
      case 13: acc ^= data[13] ^ 14; break;
      case 14: acc ^= data[14] ^ 15; break;
      case 15: acc ^= data[15] ^ 16; break;
      case 16: acc ^= data[16] ^ 17; break;
      case 17: acc ^= data[17] ^ 18; break;
      case 18: acc ^= data[18] ^ 19; break;
      case 19: acc ^= data[19] ^ 20; break;
      case 20: acc ^= data[20] ^ 21; break;
      case 21: acc ^= data[21] ^ 22; break;
      case 22: acc ^= data[22] ^ 23; break;
      case 23: acc ^= data[23] ^ 24; break;
      case 24: acc ^= data[24] ^ 25; break;
      case 25: acc ^= data[25] ^ 26; break;
      case 26: acc ^= data[26] ^ 27; break;
      case 27: acc ^= data[27] ^ 28; break;
      case 28: acc ^= data[28] ^ 29; break;
      case 29: acc ^= data[29] ^ 30; break;
      case 30: acc ^= data[30] ^ 31; break;
      case 31: acc ^= data[31] ^ 32; break;
      case 32: acc ^= data[32] ^ 33; break;
      case 33: acc ^= data[33] ^ 34; break;
      case 34: acc ^= data[34] ^ 35; break;
      case 35: acc ^= data[35] ^ 36; break;
      case 36: acc ^= data[36] ^ 37; break;
      case 37: acc ^= data[37] ^ 38; break;
      case 38: acc ^= data[38] ^ 39; break;
      case 39: acc ^= data[39] ^ 40; break;
      case 40: acc ^= data[40] ^ 41; break;
      case 41: acc ^= data[41] ^ 42; break;
      case 42: acc ^= data[42] ^ 43; break;
      case 43: acc ^= data[43] ^ 44; break;
      case 44: acc ^= data[44] ^ 45; break;
      case 45: acc ^= data[45] ^ 46; break;
      case 46: acc ^= data[46] ^ 47; break;
      case 47: acc ^= data[47] ^ 48; break;
      case 48: acc ^= data[48] ^ 49; break;
      case 49: acc ^= data[49] ^ 50; break;
      case 50: acc ^= data[50] ^ 51; break;
      case 51: acc ^= data[51] ^ 52; break;
      case 52: acc ^= data[52] ^ 53; break;
      case 53: acc ^= data[53] ^ 54; break;
      case 54: acc ^= data[54] ^ 55; break;
      case 55: acc ^= data[55] ^ 56; break;
      case 56: acc ^= data[56] ^ 57; break;
      case 57: acc ^= data[57] ^ 58; break;
      case 58: acc ^= data[58] ^ 59; break;
      case 59: acc ^= data[59] ^ 60; break;
      case 60: acc ^= data[60] ^ 61; break;
      case 61: acc ^= data[61] ^ 62; break;
      case 62: acc ^= data[62] ^ 63; break;
      case 63: acc ^= data[63] ^ 64; break;
      case 64: acc ^= data[64] ^ 65; break;
      case 65: acc ^= data[65] ^ 66; break;
      case 66: acc ^= data[66] ^ 67; break;
      case 67: acc ^= data[67] ^ 68; break;
      case 68: acc ^= data[68] ^ 69; break;
      case 69: acc ^= data[69] ^ 70; break;
      case 70: acc ^= data[70] ^ 71; break;
      case 71: acc ^= data[71] ^ 72; break;
      case 72: acc ^= data[72] ^ 73; break;
      case 73: acc ^= data[73] ^ 74; break;
      case 74: acc ^= data[74] ^ 75; break;
      case 75: acc ^= data[75] ^ 76; break;
      case 76: acc ^= data[76] ^ 77; break;
      case 77: acc ^= data[77] ^ 78; break;
      case 78: acc ^= data[78] ^ 79; break;
      case 79: acc ^= data[79] ^ 80; break;
      case 80: acc ^= data[80] ^ 81; break;
      case 81: acc ^= data[81] ^ 82; break;
      case 82: acc ^= data[82] ^ 83; break;
      case 83: acc ^= data[83] ^ 84; break;
      case 84: acc ^= data[84] ^ 85; break;
      case 85: acc ^= data[85] ^ 86; break;
      case 86: acc ^= data[86] ^ 87; break;
      case 87: acc ^= data[87] ^ 88; break;
      case 88: acc ^= data[88] ^ 89; break;
      case 89: acc ^= data[89] ^ 90; break;
      case 90: acc ^= data[90] ^ 91; break;
      case 91: acc ^= data[91] ^ 92; break;
      case 92: acc ^= data[92] ^ 93; break;
      case 93: acc ^= data[93] ^ 94; break;
      case 94: acc ^= data[94] ^ 95; break;
      case 95: acc ^= data[95] ^ 96; break;
      case 96: acc ^= data[96] ^ 97; break;
      case 97: acc ^= data[97] ^ 98; break;
      case 98: acc ^= data[98] ^ 99; break;
      case 99: acc ^= data[99] ^ 100; break;
      case 100: acc ^= data[100] ^ 101; break;
      case 101: acc ^= data[101] ^ 102; break;
      case 102: acc ^= data[102] ^ 103; break;
      case 103: acc ^= data[103] ^ 104; break;
      case 104: acc ^= data[104] ^ 105; break;
      case 105: acc ^= data[105] ^ 106; break;
      case 106: acc ^= data[106] ^ 107; break;
      case 107: acc ^= data[107] ^ 108; break;
      case 108: acc ^= data[108] ^ 109; break;
      case 109: acc ^= data[109] ^ 110; break;
      case 110: acc ^= data[110] ^ 111; break;
      case 111: acc ^= data[111] ^ 112; break;
      case 112: acc ^= data[112] ^ 113; break;
      case 113: acc ^= data[113] ^ 114; break;
      case 114: acc ^= data[114] ^ 115; break;
      case 115: acc ^= data[115] ^ 116; break;
      case 116: acc ^= data[116] ^ 117; break;
      case 117: acc ^= data[117] ^ 118; break;
      case 118: acc ^= data[118] ^ 119; break;
      case 119: acc ^= data[119] ^ 120; break;
      case 120: acc ^= data[120] ^ 121; break;
      case 121: acc ^= data[121] ^ 122; break;
      case 122: acc ^= data[122] ^ 123; break;
      case 123: acc ^= data[123] ^ 124; break;
      case 124: acc ^= data[124] ^ 125; break;
      case 125: acc ^= data[125] ^ 126; break;
      case 126: acc ^= data[126] ^ 127; break;
      case 127: acc ^= data[127] ^ 128; break;
      case 128: acc ^= data[128] ^ 129; break;
      case 129: acc ^= data[129] ^ 130; break;
      case 130: acc ^= data[130] ^ 131; break;
      case 131: acc ^= data[131] ^ 132; break;
      case 132: acc ^= data[132] ^ 133; break;
      case 133: acc ^= data[133] ^ 134; break;
      case 134: acc ^= data[134] ^ 135; break;
      case 135: acc ^= data[135] ^ 136; break;
      case 136: acc ^= data[136] ^ 137; break;
      case 137: acc ^= data[137] ^ 138; break;
      case 138: acc ^= data[138] ^ 139; break;
      case 139: acc ^= data[139] ^ 140; break;
      case 140: acc ^= data[140] ^ 141; break;
      case 141: acc ^= data[141] ^ 142; break;
      case 142: acc ^= data[142] ^ 143; break;
      case 143: acc ^= data[143] ^ 144; break;
      case 144: acc ^= data[144] ^ 145; break;
      case 145: acc ^= data[145] ^ 146; break;
      case 146: acc ^= data[146] ^ 147; break;
      case 147: acc ^= data[147] ^ 148; break;
      case 148: acc ^= data[148] ^ 149; break;
      case 149: acc ^= data[149] ^ 150; break;
      case 150: acc ^= data[150] ^ 151; break;
      case 151: acc ^= data[151] ^ 152; break;
      case 152: acc ^= data[152] ^ 153; break;
      case 153: acc ^= data[153] ^ 154; break;
      case 154: acc ^= data[154] ^ 155; break;
      case 155: acc ^= data[155] ^ 156; break;
      case 156: acc ^= data[156] ^ 157; break;
      case 157: acc ^= data[157] ^ 158; break;
      case 158: acc ^= data[158] ^ 159; break;
      case 159: acc ^= data[159] ^ 160; break;
      case 160: acc ^= data[160] ^ 161; break;
      case 161: acc ^= data[161] ^ 162; break;
      case 162: acc ^= data[162] ^ 163; break;
      case 163: acc ^= data[163] ^ 164; break;
      case 164: acc ^= data[164] ^ 165; break;
      case 165: acc ^= data[165] ^ 166; break;
      case 166: acc ^= data[166] ^ 167; break;
      case 167: acc ^= data[167] ^ 168; break;
      case 168: acc ^= data[168] ^ 169; break;
      case 169: acc ^= data[169] ^ 170; break;
      case 170: acc ^= data[170] ^ 171; break;
      case 171: acc ^= data[171] ^ 172; break;
      case 172: acc ^= data[172] ^ 173; break;
      case 173: acc ^= data[173] ^ 174; break;
      case 174: acc ^= data[174] ^ 175; break;
      case 175: acc ^= data[175] ^ 176; break;
      case 176: acc ^= data[176] ^ 177; break;
      case 177: acc ^= data[177] ^ 178; break;
      case 178: acc ^= data[178] ^ 179; break;
      case 179: acc ^= data[179] ^ 180; break;
      case 180: acc ^= data[180] ^ 181; break;
      case 181: acc ^= data[181] ^ 182; break;
      case 182: acc ^= data[182] ^ 183; break;
      case 183: acc ^= data[183] ^ 184; break;
      case 184: acc ^= data[184] ^ 185; break;
      case 185: acc ^= data[185] ^ 186; break;
      case 186: acc ^= data[186] ^ 187; break;
      case 187: acc ^= data[187] ^ 188; break;
      case 188: acc ^= data[188] ^ 189; break;
      case 189: acc ^= data[189] ^ 190; break;
      case 190: acc ^= data[190] ^ 191; break;
      case 191: acc ^= data[191] ^ 192; break;
      case 192: acc ^= data[192] ^ 193; break;
      case 193: acc ^= data[193] ^ 194; break;
      case 194: acc ^= data[194] ^ 195; break;
      case 195: acc ^= data[195] ^ 196; break;
      case 196: acc ^= data[196] ^ 197; break;
      case 197: acc ^= data[197] ^ 198; break;
      case 198: acc ^= data[198] ^ 199; break;
      case 199: acc ^= data[199] ^ 200; break;
      case 200: acc ^= data[200] ^ 201; break;
      case 201: acc ^= data[201] ^ 202; break;
      case 202: acc ^= data[202] ^ 203; break;
      case 203: acc ^= data[203] ^ 204; break;
      case 204: acc ^= data[204] ^ 205; break;
      case 205: acc ^= data[205] ^ 206; break;
      case 206: acc ^= data[206] ^ 207; break;
      case 207: acc ^= data[207] ^ 208; break;
      case 208: acc ^= data[208] ^ 209; break;
      case 209: acc ^= data[209] ^ 210; break;
      case 210: acc ^= data[210] ^ 211; break;
      case 211: acc ^= data[211] ^ 212; break;
      case 212: acc ^= data[212] ^ 213; break;
      case 213: acc ^= data[213] ^ 214; break;
      case 214: acc ^= data[214] ^ 215; break;
      case 215: acc ^= data[215] ^ 216; break;
      case 216: acc ^= data[216] ^ 217; break;
      case 217: acc ^= data[217] ^ 218; break;
      case 218: acc ^= data[218] ^ 219; break;
      case 219: acc ^= data[219] ^ 220; break;
      case 220: acc ^= data[220] ^ 221; break;
      case 221: acc ^= data[221] ^ 222; break;
      case 222: acc ^= data[222] ^ 223; break;
      case 223: acc ^= data[223] ^ 224; break;
      case 224: acc ^= data[224] ^ 225; break;
      case 225: acc ^= data[225] ^ 226; break;
      case 226: acc ^= data[226] ^ 227; break;
      case 227: acc ^= data[227] ^ 228; break;
      case 228: acc ^= data[228] ^ 229; break;
      case 229: acc ^= data[229] ^ 230; break;
      case 230: acc ^= data[230] ^ 231; break;
      case 231: acc ^= data[231] ^ 232; break;
      case 232: acc ^= data[232] ^ 233; break;
      case 233: acc ^= data[233] ^ 234; break;
      case 234: acc ^= data[234] ^ 235; break;
      case 235: acc ^= data[235] ^ 236; break;
      case 236: acc ^= data[236] ^ 237; break;
      case 237: acc ^= data[237] ^ 238; break;
      case 238: acc ^= data[238] ^ 239; break;
      case 239: acc ^= data[239] ^ 240; break;
      case 240: acc ^= data[240] ^ 241; break;
      case 241: acc ^= data[241] ^ 242; break;
      case 242: acc ^= data[242] ^ 243; break;
      case 243: acc ^= data[243] ^ 244; break;
      case 244: acc ^= data[244] ^ 245; break;
      case 245: acc ^= data[245] ^ 246; break;
      case 246: acc ^= data[246] ^ 247; break;
      case 247: acc ^= data[247] ^ 248; break;
      case 248: acc ^= data[248] ^ 249; break;
      case 249: acc ^= data[249] ^ 250; break;
      case 250: acc ^= data[250] ^ 251; break;
      case 251: acc ^= data[251] ^ 252; break;
      case 252: acc ^= data[252] ^ 253; break;
      case 253: acc ^= data[253] ^ 254; break;
      case 254: acc ^= data[254] ^ 255; break;
      case 255: acc ^= data[255] ^ 256; break;
    }
  }
}
} // namespace

BENCHMARK(createAndDestroyMulti_LargeAlloc, iters) {
  static constexpr auto kSize = 1024;
  std::array<std::unique_ptr<IOBuf>, kSize> buffers;
  size_t size = 1024 * 1024; // 1MB allocation

  // Prepare data for the computation-heavy functions
  std::vector<uint64_t> computation_data(256);
  for (size_t i = 0; i < computation_data.size(); ++i) {
    computation_data[i] = i * 17; // Arbitrary calculation
  }

  uint64_t payload = 0; // Value to be used in the heavy computation

  while (iters--) {
    for (auto i = 0; i < kSize; ++i) {
      buffers[i] = IOBuf::create(size);
    }
    // Perform a series of operations that involve indirect branches
    // and pointer chasing to defeat branch prediction.
    for (int j = 0; j < 4; ++j) { // Reduced iterations to focus on computation
      switch (j % 3) {
        case 0: compute_heavy_part1(payload, computation_data.data()); break;
        case 1: compute_heavy_part2(payload, computation_data.data()); break;
        case 2: compute_heavy_part3(payload, computation_data.data()); break;
      }
    }
    folly::doNotOptimizeAway(payload);
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
