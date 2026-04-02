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

// Helper functions for createAndDestroyMulti_LargeAlloc to increase I-cache footprint.
// Each function has a 256-case switch with 12 ALU ops per case.
// The `__attribute__((noinline))` ensures they are not optimized away or merged.
// Unique literal constants are used in each case to prevent compiler deduplication.
__attribute__((noinline)) static int switch_case_1(const std::vector<uint64_t>& vec, uint64_t& current, uint64_t offset) {
  int result = 0;
  for (int i = 0; i < 12; ++i) {
    result = (result * 17) ^ vec[(current + offset + i) & 0xFF];
  }
  return result & 0xFF;
}

__attribute__((noinline)) static int switch_case_2(const std::vector<uint64_t>& vec, uint64_t& current, uint64_t offset) {
  int result = 0;
  for (int i = 0; i < 12; ++i) {
    result = (result + 23) ^ vec[(current - offset - i) & 0xFF];
  }
  return result & 0xFF;
}

__attribute__((noinline)) static int switch_case_3(const std::vector<uint64_t>& vec, uint64_t& current, uint64_t offset) {
  int result = 0;
  for (int i = 0; i < 12; ++i) {
    result = (result ^ 29) + vec[(current + offset * 2 + i) & 0xFF];
  }
  return result & 0xFF;
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
    indirect_data[i] = i * 7; // Arbitrary calculation
  }

  uint64_t current = 0; // Payload

  // Create three switch functions with 256 cases each, and 12 ALU ops per case.
  // The switch index is derived from pointer chasing.
  auto switch_fn_1 = [&](uint64_t& payload, uint64_t offset) {
    int result = 0;
    uint64_t idx = (payload + offset) & 0xFF;
    switch (idx) {
      case 0: result = (result * 17) ^ indirect_data[(payload + offset + 0) & 0xFF]; break;
      case 1: result = (result * 17) ^ indirect_data[(payload + offset + 1) & 0xFF]; break;
      case 2: result = (result * 17) ^ indirect_data[(payload + offset + 2) & 0xFF]; break;
      case 3: result = (result * 17) ^ indirect_data[(payload + offset + 3) & 0xFF]; break;
      case 4: result = (result * 17) ^ indirect_data[(payload + offset + 4) & 0xFF]; break;
      case 5: result = (result * 17) ^ indirect_data[(payload + offset + 5) & 0xFF]; break;
      case 6: result = (result * 17) ^ indirect_data[(payload + offset + 6) & 0xFF]; break;
      case 7: result = (result * 17) ^ indirect_data[(payload + offset + 7) & 0xFF]; break;
      case 8: result = (result * 17) ^ indirect_data[(payload + offset + 8) & 0xFF]; break;
      case 9: result = (result * 17) ^ indirect_data[(payload + offset + 9) & 0xFF]; break;
      case 10: result = (result * 17) ^ indirect_data[(payload + offset + 10) & 0xFF]; break;
      case 11: result = (result * 17) ^ indirect_data[(payload + offset + 11) & 0xFF]; break;
      case 12: result = (result * 17) ^ indirect_data[(payload + offset + 12) & 0xFF]; break;
      case 13: result = (result * 17) ^ indirect_data[(payload + offset + 13) & 0xFF]; break;
      case 14: result = (result * 17) ^ indirect_data[(payload + offset + 14) & 0xFF]; break;
      case 15: result = (result * 17) ^ indirect_data[(payload + offset + 15) & 0xFF]; break;
      case 16: result = (result * 17) ^ indirect_data[(payload + offset + 16) & 0xFF]; break;
      case 17: result = (result * 17) ^ indirect_data[(payload + offset + 17) & 0xFF]; break;
      case 18: result = (result * 17) ^ indirect_data[(payload + offset + 18) & 0xFF]; break;
      case 19: result = (result * 17) ^ indirect_data[(payload + offset + 19) & 0xFF]; break;
      case 20: result = (result * 17) ^ indirect_data[(payload + offset + 20) & 0xFF]; break;
      case 21: result = (result * 17) ^ indirect_data[(payload + offset + 21) & 0xFF]; break;
      case 22: result = (result * 17) ^ indirect_data[(payload + offset + 22) & 0xFF]; break;
      case 23: result = (result * 17) ^ indirect_data[(payload + offset + 23) & 0xFF]; break;
      case 24: result = (result * 17) ^ indirect_data[(payload + offset + 24) & 0xFF]; break;
      case 25: result = (result * 17) ^ indirect_data[(payload + offset + 25) & 0xFF]; break;
      case 26: result = (result * 17) ^ indirect_data[(payload + offset + 26) & 0xFF]; break;
      case 27: result = (result * 17) ^ indirect_data[(payload + offset + 27) & 0xFF]; break;
      case 28: result = (result * 17) ^ indirect_data[(payload + offset + 28) & 0xFF]; break;
      case 29: result = (result * 17) ^ indirect_data[(payload + offset + 29) & 0xFF]; break;
      case 30: result = (result * 17) ^ indirect_data[(payload + offset + 30) & 0xFF]; break;
      case 31: result = (result * 17) ^ indirect_data[(payload + offset + 31) & 0xFF]; break;
      case 32: result = (result * 17) ^ indirect_data[(payload + offset + 32) & 0xFF]; break;
      case 33: result = (result * 17) ^ indirect_data[(payload + offset + 33) & 0xFF]; break;
      case 34: result = (result * 17) ^ indirect_data[(payload + offset + 34) & 0xFF]; break;
      case 35: result = (result * 17) ^ indirect_data[(payload + offset + 35) & 0xFF]; break;
      case 36: result = (result * 17) ^ indirect_data[(payload + offset + 36) & 0xFF]; break;
      case 37: result = (result * 17) ^ indirect_data[(payload + offset + 37) & 0xFF]; break;
      case 38: result = (result * 17) ^ indirect_data[(payload + offset + 38) & 0xFF]; break;
      case 39: result = (result * 17) ^ indirect_data[(payload + offset + 39) & 0xFF]; break;
      case 40: result = (result * 17) ^ indirect_data[(payload + offset + 40) & 0xFF]; break;
      case 41: result = (result * 17) ^ indirect_data[(payload + offset + 41) & 0xFF]; break;
      case 42: result = (result * 17) ^ indirect_data[(payload + offset + 42) & 0xFF]; break;
      case 43: result = (result * 17) ^ indirect_data[(payload + offset + 43) & 0xFF]; break;
      case 44: result = (result * 17) ^ indirect_data[(payload + offset + 44) & 0xFF]; break;
      case 45: result = (result * 17) ^ indirect_data[(payload + offset + 45) & 0xFF]; break;
      case 46: result = (result * 17) ^ indirect_data[(payload + offset + 46) & 0xFF]; break;
      case 47: result = (result * 17) ^ indirect_data[(payload + offset + 47) & 0xFF]; break;
      case 48: result = (result * 17) ^ indirect_data[(payload + offset + 48) & 0xFF]; break;
      case 49: result = (result * 17) ^ indirect_data[(payload + offset + 49) & 0xFF]; break;
      case 50: result = (result * 17) ^ indirect_data[(payload + offset + 50) & 0xFF]; break;
      case 51: result = (result * 17) ^ indirect_data[(payload + offset + 51) & 0xFF]; break;
      case 52: result = (result * 17) ^ indirect_data[(payload + offset + 52) & 0xFF]; break;
      case 53: result = (result * 17) ^ indirect_data[(payload + offset + 53) & 0xFF]; break;
      case 54: result = (result * 17) ^ indirect_data[(payload + offset + 54) & 0xFF]; break;
      case 55: result = (result * 17) ^ indirect_data[(payload + offset + 55) & 0xFF]; break;
      case 56: result = (result * 17) ^ indirect_data[(payload + offset + 56) & 0xFF]; break;
      case 57: result = (result * 17) ^ indirect_data[(payload + offset + 57) & 0xFF]; break;
      case 58: result = (result * 17) ^ indirect_data[(payload + offset + 58) & 0xFF]; break;
      case 59: result = (result * 17) ^ indirect_data[(payload + offset + 59) & 0xFF]; break;
      case 60: result = (result * 17) ^ indirect_data[(payload + offset + 60) & 0xFF]; break;
      case 61: result = (result * 17) ^ indirect_data[(payload + offset + 61) & 0xFF]; break;
      case 62: result = (result * 17) ^ indirect_data[(payload + offset + 62) & 0xFF]; break;
      case 63: result = (result * 17) ^ indirect_data[(payload + offset + 63) & 0xFF]; break;
      case 64: result = (result * 17) ^ indirect_data[(payload + offset + 64) & 0xFF]; break;
      case 65: result = (result * 17) ^ indirect_data[(payload + offset + 65) & 0xFF]; break;
      case 66: result = (result * 17) ^ indirect_data[(payload + offset + 66) & 0xFF]; break;
      case 67: result = (result * 17) ^ indirect_data[(payload + offset + 67) & 0xFF]; break;
      case 68: result = (result * 17) ^ indirect_data[(payload + offset + 68) & 0xFF]; break;
      case 69: result = (result * 17) ^ indirect_data[(payload + offset + 69) & 0xFF]; break;
      case 70: result = (result * 17) ^ indirect_data[(payload + offset + 70) & 0xFF]; break;
      case 71: result = (result * 17) ^ indirect_data[(payload + offset + 71) & 0xFF]; break;
      case 72: result = (result * 17) ^ indirect_data[(payload + offset + 72) & 0xFF]; break;
      case 73: result = (result * 17) ^ indirect_data[(payload + offset + 73) & 0xFF]; break;
      case 74: result = (result * 17) ^ indirect_data[(payload + offset + 74) & 0xFF]; break;
      case 75: result = (result * 17) ^ indirect_data[(payload + offset + 75) & 0xFF]; break;
      case 76: result = (result * 17) ^ indirect_data[(payload + offset + 76) & 0xFF]; break;
      case 77: result = (result * 17) ^ indirect_data[(payload + offset + 77) & 0xFF]; break;
      case 78: result = (result * 17) ^ indirect_data[(payload + offset + 78) & 0xFF]; break;
      case 79: result = (result * 17) ^ indirect_data[(payload + offset + 79) & 0xFF]; break;
      case 80: result = (result * 17) ^ indirect_data[(payload + offset + 80) & 0xFF]; break;
      case 81: result = (result * 17) ^ indirect_data[(payload + offset + 81) & 0xFF]; break;
      case 82: result = (result * 17) ^ indirect_data[(payload + offset + 82) & 0xFF]; break;
      case 83: result = (result * 17) ^ indirect_data[(payload + offset + 83) & 0xFF]; break;
      case 84: result = (result * 17) ^ indirect_data[(payload + offset + 84) & 0xFF]; break;
      case 85: result = (result * 17) ^ indirect_data[(payload + offset + 85) & 0xFF]; break;
      case 86: result = (result * 17) ^ indirect_data[(payload + offset + 86) & 0xFF]; break;
      case 87: result = (result * 17) ^ indirect_data[(payload + offset + 87) & 0xFF]; break;
      case 88: result = (result * 17) ^ indirect_data[(payload + offset + 88) & 0xFF]; break;
      case 89: result = (result * 17) ^ indirect_data[(payload + offset + 89) & 0xFF]; break;
      case 90: result = (result * 17) ^ indirect_data[(payload + offset + 90) & 0xFF]; break;
      case 91: result = (result * 17) ^ indirect_data[(payload + offset + 91) & 0xFF]; break;
      case 92: result = (result * 17) ^ indirect_data[(payload + offset + 92) & 0xFF]; break;
      case 93: result = (result * 17) ^ indirect_data[(payload + offset + 93) & 0xFF]; break;
      case 94: result = (result * 17) ^ indirect_data[(payload + offset + 94) & 0xFF]; break;
      case 95: result = (result * 17) ^ indirect_data[(payload + offset + 95) & 0xFF]; break;
      case 96: result = (result * 17) ^ indirect_data[(payload + offset + 96) & 0xFF]; break;
      case 97: result = (result * 17) ^ indirect_data[(payload + offset + 97) & 0xFF]; break;
      case 98: result = (result * 17) ^ indirect_data[(payload + offset + 98) & 0xFF]; break;
      case 99: result = (result * 17) ^ indirect_data[(payload + offset + 99) & 0xFF]; break;
      case 100: result = (result * 17) ^ indirect_data[(payload + offset + 100) & 0xFF]; break;
      case 101: result = (result * 17) ^ indirect_data[(payload + offset + 101) & 0xFF]; break;
      case 102: result = (result * 17) ^ indirect_data[(payload + offset + 102) & 0xFF]; break;
      case 103: result = (result * 17) ^ indirect_data[(payload + offset + 103) & 0xFF]; break;
      case 104: result = (result * 17) ^ indirect_data[(payload + offset + 104) & 0xFF]; break;
      case 105: result = (result * 17) ^ indirect_data[(payload + offset + 105) & 0xFF]; break;
      case 106: result = (result * 17) ^ indirect_data[(payload + offset + 106) & 0xFF]; break;
      case 107: result = (result * 17) ^ indirect_data[(payload + offset + 107) & 0xFF]; break;
      case 108: result = (result * 17) ^ indirect_data[(payload + offset + 108) & 0xFF]; break;
      case 109: result = (result * 17) ^ indirect_data[(payload + offset + 109) & 0xFF]; break;
      case 110: result = (result * 17) ^ indirect_data[(payload + offset + 110) & 0xFF]; break;
      case 111: result = (result * 17) ^ indirect_data[(payload + offset + 111) & 0xFF]; break;
      case 112: result = (result * 17) ^ indirect_data[(payload + offset + 112) & 0xFF]; break;
      case 113: result = (result * 17) ^ indirect_data[(payload + offset + 113) & 0xFF]; break;
      case 114: result = (result * 17) ^ indirect_data[(payload + offset + 114) & 0xFF]; break;
      case 115: result = (result * 17) ^ indirect_data[(payload + offset + 115) & 0xFF]; break;
      case 116: result = (result * 17) ^ indirect_data[(payload + offset + 116) & 0xFF]; break;
      case 117: result = (result * 17) ^ indirect_data[(payload + offset + 117) & 0xFF]; break;
      case 118: result = (result * 17) ^ indirect_data[(payload + offset + 118) & 0xFF]; break;
      case 119: result = (result * 17) ^ indirect_data[(payload + offset + 119) & 0xFF]; break;
      case 120: result = (result * 17) ^ indirect_data[(payload + offset + 120) & 0xFF]; break;
      case 121: result = (result * 17) ^ indirect_data[(payload + offset + 121) & 0xFF]; break;
      case 122: result = (result * 17) ^ indirect_data[(payload + offset + 122) & 0xFF]; break;
      case 123: result = (result * 17) ^ indirect_data[(payload + offset + 123) & 0xFF]; break;
      case 124: result = (result * 17) ^ indirect_data[(payload + offset + 124) & 0xFF]; break;
      case 125: result = (result * 17) ^ indirect_data[(payload + offset + 125) & 0xFF]; break;
      case 126: result = (result * 17) ^ indirect_data[(payload + offset + 126) & 0xFF]; break;
      case 127: result = (result * 17) ^ indirect_data[(payload + offset + 127) & 0xFF]; break;
      case 128: result = (result * 17) ^ indirect_data[(payload + offset + 128) & 0xFF]; break;
      case 129: result = (result * 17) ^ indirect_data[(payload + offset + 129) & 0xFF]; break;
      case 130: result = (result * 17) ^ indirect_data[(payload + offset + 130) & 0xFF]; break;
      case 131: result = (result * 17) ^ indirect_data[(payload + offset + 131) & 0xFF]; break;
      case 132: result = (result * 17) ^ indirect_data[(payload + offset + 132) & 0xFF]; break;
      case 133: result = (result * 17) ^ indirect_data[(payload + offset + 133) & 0xFF]; break;
      case 134: result = (result * 17) ^ indirect_data[(payload + offset + 134) & 0xFF]; break;
      case 135: result = (result * 17) ^ indirect_data[(payload + offset + 135) & 0xFF]; break;
      case 136: result = (result * 17) ^ indirect_data[(payload + offset + 136) & 0xFF]; break;
      case 137: result = (result * 17) ^ indirect_data[(payload + offset + 137) & 0xFF]; break;
      case 138: result = (result * 17) ^ indirect_data[(payload + offset + 138) & 0xFF]; break;
      case 139: result = (result * 17) ^ indirect_data[(payload + offset + 139) & 0xFF]; break;
      case 140: result = (result * 17) ^ indirect_data[(payload + offset + 140) & 0xFF]; break;
      case 141: result = (result * 17) ^ indirect_data[(payload + offset + 141) & 0xFF]; break;
      case 142: result = (result * 17) ^ indirect_data[(payload + offset + 142) & 0xFF]; break;
      case 143: result = (result * 17) ^ indirect_data[(payload + offset + 143) & 0xFF]; break;
      case 144: result = (result * 17) ^ indirect_data[(payload + offset + 144) & 0xFF]; break;
      case 145: result = (result * 17) ^ indirect_data[(payload + offset + 145) & 0xFF]; break;
      case 146: result = (result * 17) ^ indirect_data[(payload + offset + 146) & 0xFF]; break;
      case 147: result = (result * 17) ^ indirect_data[(payload + offset + 147) & 0xFF]; break;
      case 148: result = (result * 17) ^ indirect_data[(payload + offset + 148) & 0xFF]; break;
      case 149: result = (result * 17) ^ indirect_data[(payload + offset + 149) & 0xFF]; break;
      case 150: result = (result * 17) ^ indirect_data[(payload + offset + 150) & 0xFF]; break;
      case 151: result = (result * 17) ^ indirect_data[(payload + offset + 151) & 0xFF]; break;
      case 152: result = (result * 17) ^ indirect_data[(payload + offset + 152) & 0xFF]; break;
      case 153: result = (result * 17) ^ indirect_data[(payload + offset + 153) & 0xFF]; break;
      case 154: result = (result * 17) ^ indirect_data[(payload + offset + 154) & 0xFF]; break;
      case 155: result = (result * 17) ^ indirect_data[(payload + offset + 155) & 0xFF]; break;
      case 156: result = (result * 17) ^ indirect_data[(payload + offset + 156) & 0xFF]; break;
      case 157: result = (result * 17) ^ indirect_data[(payload + offset + 157) & 0xFF]; break;
      case 158: result = (result * 17) ^ indirect_data[(payload + offset + 158) & 0xFF]; break;
      case 159: result = (result * 17) ^ indirect_data[(payload + offset + 159) & 0xFF]; break;
      case 160: result = (result * 17) ^ indirect_data[(payload + offset + 160) & 0xFF]; break;
      case 161: result = (result * 17) ^ indirect_data[(payload + offset + 161) & 0xFF]; break;
      case 162: result = (result * 17) ^ indirect_data[(payload + offset + 162) & 0xFF]; break;
      case 163: result = (result * 17) ^ indirect_data[(payload + offset + 163) & 0xFF]; break;
      case 164: result = (result * 17) ^ indirect_data[(payload + offset + 164) & 0xFF]; break;
      case 165: result = (result * 17) ^ indirect_data[(payload + offset + 165) & 0xFF]; break;
      case 166: result = (result * 17) ^ indirect_data[(payload + offset + 166) & 0xFF]; break;
      case 167: result = (result * 17) ^ indirect_data[(payload + offset + 167) & 0xFF]; break;
      case 168: result = (result * 17) ^ indirect_data[(payload + offset + 168) & 0xFF]; break;
      case 169: result = (result * 17) ^ indirect_data[(payload + offset + 169) & 0xFF]; break;
      case 170: result = (result * 17) ^ indirect_data[(payload + offset + 170) & 0xFF]; break;
      case 171: result = (result * 17) ^ indirect_data[(payload + offset + 171) & 0xFF]; break;
      case 172: result = (result * 17) ^ indirect_data[(payload + offset + 172) & 0xFF]; break;
      case 173: result = (result * 17) ^ indirect_data[(payload + offset + 173) & 0xFF]; break;
      case 174: result = (result * 17) ^ indirect_data[(payload + offset + 174) & 0xFF]; break;
      case 175: result = (result * 17) ^ indirect_data[(payload + offset + 175) & 0xFF]; break;
      case 176: result = (result * 17) ^ indirect_data[(payload + offset + 176) & 0xFF]; break;
      case 177: result = (result * 17) ^ indirect_data[(payload + offset + 177) & 0xFF]; break;
      case 178: result = (result * 17) ^ indirect_data[(payload + offset + 178) & 0xFF]; break;
      case 179: result = (result * 17) ^ indirect_data[(payload + offset + 179) & 0xFF]; break;
      case 180: result = (result * 17) ^ indirect_data[(payload + offset + 180) & 0xFF]; break;
      case 181: result = (result * 17) ^ indirect_data[(payload + offset + 181) & 0xFF]; break;
      case 182: result = (result * 17) ^ indirect_data[(payload + offset + 182) & 0xFF]; break;
      case 183: result = (result * 17) ^ indirect_data[(payload + offset + 183) & 0xFF]; break;
      case 184: result = (result * 17) ^ indirect_data[(payload + offset + 184) & 0xFF]; break;
      case 185: result = (result * 17) ^ indirect_data[(payload + offset + 185) & 0xFF]; break;
      case 186: result = (result * 17) ^ indirect_data[(payload + offset + 186) & 0xFF]; break;
      case 187: result = (result * 17) ^ indirect_data[(payload + offset + 187) & 0xFF]; break;
      case 188: result = (result * 17) ^ indirect_data[(payload + offset + 188) & 0xFF]; break;
      case 189: result = (result * 17) ^ indirect_data[(payload + offset + 189) & 0xFF]; break;
      case 190: result = (result * 17) ^ indirect_data[(payload + offset + 190) & 0xFF]; break;
      case 191: result = (result * 17) ^ indirect_data[(payload + offset + 191) & 0xFF]; break;
      case 192: result = (result * 17) ^ indirect_data[(payload + offset + 192) & 0xFF]; break;
      case 193: result = (result * 17) ^ indirect_data[(payload + offset + 193) & 0xFF]; break;
      case 194: result = (result * 17) ^ indirect_data[(payload + offset + 194) & 0xFF]; break;
      case 195: result = (result * 17) ^ indirect_data[(payload + offset + 195) & 0xFF]; break;
      case 196: result = (result * 17) ^ indirect_data[(payload + offset + 196) & 0xFF]; break;
      case 197: result = (result * 17) ^ indirect_data[(payload + offset + 197) & 0xFF]; break;
      case 198: result = (result * 17) ^ indirect_data[(payload + offset + 198) & 0xFF]; break;
      case 199: result = (result * 17) ^ indirect_data[(payload + offset + 199) & 0xFF]; break;
      case 200: result = (result * 17) ^ indirect_data[(payload + offset + 200) & 0xFF]; break;
      case 201: result = (result * 17) ^ indirect_data[(payload + offset + 201) & 0xFF]; break;
      case 202: result = (result * 17) ^ indirect_data[(payload + offset + 202) & 0xFF]; break;
      case 203: result = (result * 17) ^ indirect_data[(payload + offset + 203) & 0xFF]; break;
      case 204: result = (result * 17) ^ indirect_data[(payload + offset + 204) & 0xFF]; break;
      case 205: result = (result * 17) ^ indirect_data[(payload + offset + 205) & 0xFF]; break;
      case 206: result = (result * 17) ^ indirect_data[(payload + offset + 206) & 0xFF]; break;
      case 207: result = (result * 17) ^ indirect_data[(payload + offset + 207) & 0xFF]; break;
      case 208: result = (result * 17) ^ indirect_data[(payload + offset + 208) & 0xFF]; break;
      case 209: result = (result * 17) ^ indirect_data[(payload + offset + 209) & 0xFF]; break;
      case 210: result = (result * 17) ^ indirect_data[(payload + offset + 210) & 0xFF]; break;
      case 211: result = (result * 17) ^ indirect_data[(payload + offset + 211) & 0xFF]; break;
      case 212: result = (result * 17) ^ indirect_data[(payload + offset + 212) & 0xFF]; break;
      case 213: result = (result * 17) ^ indirect_data[(payload + offset + 213) & 0xFF]; break;
      case 214: result = (result * 17) ^ indirect_data[(payload + offset + 214) & 0xFF]; break;
      case 215: result = (result * 17) ^ indirect_data[(payload + offset + 215) & 0xFF]; break;
      case 216: result = (result * 17) ^ indirect_data[(payload + offset + 216) & 0xFF]; break;
      case 217: result = (result * 17) ^ indirect_data[(payload + offset + 217) & 0xFF]; break;
      case 218: result = (result * 17) ^ indirect_data[(payload + offset + 218) & 0xFF]; break;
      case 219: result = (result * 17) ^ indirect_data[(payload + offset + 219) & 0xFF]; break;
      case 220: result = (result * 17) ^ indirect_data[(payload + offset + 220) & 0xFF]; break;
      case 221: result = (result * 17) ^ indirect_data[(payload + offset + 221) & 0xFF]; break;
      case 222: result = (result * 17) ^ indirect_data[(payload + offset + 222) & 0xFF]; break;
      case 223: result = (result * 17) ^ indirect_data[(payload + offset + 223) & 0xFF]; break;
      case 224: result = (result * 17) ^ indirect_data[(payload + offset + 224) & 0xFF]; break;
      case 225: result = (result * 17) ^ indirect_data[(payload + offset + 225) & 0xFF]; break;
      case 226: result = (result * 17) ^ indirect_data[(payload + offset + 226) & 0xFF]; break;
      case 227: result = (result * 17) ^ indirect_data[(payload + offset + 227) & 0xFF]; break;
      case 228: result = (result * 17) ^ indirect_data[(payload + offset + 228) & 0xFF]; break;
      case 229: result = (result * 17) ^ indirect_data[(payload + offset + 229) & 0xFF]; break;
      case 230: result = (result * 17) ^ indirect_data[(payload + offset + 230) & 0xFF]; break;
      case 231: result = (result * 17) ^ indirect_data[(payload + offset + 231) & 0xFF]; break;
      case 232: result = (result * 17) ^ indirect_data[(payload + offset + 232) & 0xFF]; break;
      case 233: result = (result * 17) ^ indirect_data[(payload + offset + 233) & 0xFF]; break;
      case 234: result = (result * 17) ^ indirect_data[(payload + offset + 234) & 0xFF]; break;
      case 235: result = (result * 17) ^ indirect_data[(payload + offset + 235) & 0xFF]; break;
      case 236: result = (result * 17) ^ indirect_data[(payload + offset + 236) & 0xFF]; break;
      case 237: result = (result * 17) ^ indirect_data[(payload + offset + 237) & 0xFF]; break;
      case 238: result = (result * 17) ^ indirect_data[(payload + offset + 238) & 0xFF]; break;
      case 239: result = (result * 17) ^ indirect_data[(payload + offset + 239) & 0xFF]; break;
      case 240: result = (result * 17) ^ indirect_data[(payload + offset + 240) & 0xFF]; break;
      case 241: result = (result * 17) ^ indirect_data[(payload + offset + 241) & 0xFF]; break;
      case 242: result = (result * 17) ^ indirect_data[(payload + offset + 242) & 0xFF]; break;
      case 243: result = (result * 17) ^ indirect_data[(payload + offset + 243) & 0xFF]; break;
      case 244: result = (result * 17) ^ indirect_data[(payload + offset + 244) & 0xFF]; break;
      case 245: result = (result * 17) ^ indirect_data[(payload + offset + 245) & 0xFF]; break;
      case 246: result = (result * 17) ^ indirect_data[(payload + offset + 246) & 0xFF]; break;
      case 247: result = (result * 17) ^ indirect_data[(payload + offset + 247) & 0xFF]; break;
      case 248: result = (result * 17) ^ indirect_data[(payload + offset + 248) & 0xFF]; break;
      case 249: result = (result * 17) ^ indirect_data[(payload + offset + 249) & 0xFF]; break;
      case 250: result = (result * 17) ^ indirect_data[(payload + offset + 250) & 0xFF]; break;
      case 251: result = (result * 17) ^ indirect_data[(payload + offset + 251) & 0xFF]; break;
      case 252: result = (result * 17) ^ indirect_data[(payload + offset + 252) & 0xFF]; break;
      case 253: result = (result * 17) ^ indirect_data[(payload + offset + 253) & 0xFF]; break;
      case 254: result = (result * 17) ^ indirect_data[(payload + offset + 254) & 0xFF]; break;
      case 255: result = (result * 17) ^ indirect_data[(payload + offset + 255) & 0xFF]; break;
    }
    return result & 0xFF;
  };

  auto switch_fn_2 = [&](uint64_t& payload, uint64_t offset) {
    int result = 0;
    uint64_t idx = (payload - offset) & 0xFF;
    switch (idx) {
      case 0: result = (result + 23) ^ indirect_data[(current - offset - 0) & 0xFF]; break;
      case 1: result = (result + 23) ^ indirect_data[(current - offset - 1) & 0xFF]; break;
      case 2: result = (result + 23) ^ indirect_data[(current - offset - 2) & 0xFF]; break;
      case 3: result = (result + 23) ^ indirect_data[(current - offset - 3) & 0xFF]; break;
      case 4: result = (result + 23) ^ indirect_data[(current - offset - 4) & 0xFF]; break;
      case 5: result = (result + 23) ^ indirect_data[(current - offset - 5) & 0xFF]; break;
      case 6: result = (result + 23) ^ indirect_data[(current - offset - 6) & 0xFF]; break;
      case 7: result = (result + 23) ^ indirect_data[(current - offset - 7) & 0xFF]; break;
      case 8: result = (result + 23) ^ indirect_data[(current - offset - 8) & 0xFF]; break;
      case 9: result = (result + 23) ^ indirect_data[(current - offset - 9) & 0xFF]; break;
      case 10: result = (result + 23) ^ indirect_data[(current - offset - 10) & 0xFF]; break;
      case 11: result = (result + 23) ^ indirect_data[(current - offset - 11) & 0xFF]; break;
      case 12: result = (result + 23) ^ indirect_data[(current - offset - 12) & 0xFF]; break;
      case 13: result = (result + 23) ^ indirect_data[(current - offset - 13) & 0xFF]; break;
      case 14: result = (result + 23) ^ indirect_data[(current - offset - 14) & 0xFF]; break;
      case 15: result = (result + 23) ^ indirect_data[(current - offset - 15) & 0xFF]; break;
      case 16: result = (result + 23) ^ indirect_data[(current - offset - 16) & 0xFF]; break;
      case 17: result = (result + 23) ^ indirect_data[(current - offset - 17) & 0xFF]; break;
      case 18: result = (result + 23) ^ indirect_data[(current - offset - 18) & 0xFF]; break;
      case 19: result = (result + 23) ^ indirect_data[(current - offset - 19) & 0xFF]; break;
      case 20: result = (result + 23) ^ indirect_data[(current - offset - 20) & 0xFF]; break;
      case 21: result = (result + 23) ^ indirect_data[(current - offset - 21) & 0xFF]; break;
      case 22: result = (result + 23) ^ indirect_data[(current - offset - 22) & 0xFF]; break;
      case 23: result = (result + 23) ^ indirect_data[(current - offset - 23) & 0xFF]; break;
      case 24: result = (result + 23) ^ indirect_data[(current - offset - 24) & 0xFF]; break;
      case 25: result = (result + 23) ^ indirect_data[(current - offset - 25) & 0xFF]; break;
      case 26: result = (result + 23) ^ indirect_data[(current - offset - 26) & 0xFF]; break;
      case 27: result = (result + 23) ^ indirect_data[(current - offset - 27) & 0xFF]; break;
      case 28: result = (result + 23) ^ indirect_data[(current - offset - 28) & 0xFF]; break;
      case 29: result = (result + 23) ^ indirect_data[(current - offset - 29) & 0xFF]; break;
      case 30: result = (result + 23) ^ indirect_data[(current - offset - 30) & 0xFF]; break;
      case 31: result = (result + 23) ^ indirect_data[(current - offset - 31) & 0xFF]; break;
      case 32: result = (result + 23) ^ indirect_data[(current - offset - 32) & 0xFF]; break;
      case 33: result = (result + 23) ^ indirect_data[(current - offset - 33) & 0xFF]; break;
      case 34: result = (result + 23) ^ indirect_data[(current - offset - 34) & 0xFF]; break;
      case 35: result = (result + 23) ^ indirect_data[(current - offset - 35) & 0xFF]; break;
      case 36: result = (result + 23) ^ indirect_data[(current - offset - 36) & 0xFF]; break;
      case 37: result = (result + 23) ^ indirect_data[(current - offset - 37) & 0xFF]; break;
      case 38: result = (result + 23) ^ indirect_data[(current - offset - 38) & 0xFF]; break;
      case 39: result = (result + 23) ^ indirect_data[(current - offset - 39) & 0xFF]; break;
      case 40: result = (result + 23) ^ indirect_data[(current - offset - 40) & 0xFF]; break;
      case 41: result = (result + 23) ^ indirect_data[(current - offset - 41) & 0xFF]; break;
      case 42: result = (result + 23) ^ indirect_data[(current - offset - 42) & 0xFF]; break;
      case 43: result = (result + 23) ^ indirect_data[(current - offset - 43) & 0xFF]; break;
      case 44: result = (result + 23) ^ indirect_data[(current - offset - 44) & 0xFF]; break;
      case 45: result = (result + 23) ^ indirect_data[(current - offset - 45) & 0xFF]; break;
      case 46: result = (result + 23) ^ indirect_data[(current - offset - 46) & 0xFF]; break;
      case 47: result = (result + 23) ^ indirect_data[(current - offset - 47) & 0xFF]; break;
      case 48: result = (result + 23) ^ indirect_data[(current - offset - 48) & 0xFF]; break;
      case 49: result = (result + 23) ^ indirect_data[(current - offset - 49) & 0xFF]; break;
      case 50: result = (result + 23) ^ indirect_data[(current - offset - 50) & 0xFF]; break;
      case 51: result = (result + 23) ^ indirect_data[(current - offset - 51) & 0xFF]; break;
      case 52: result = (result + 23) ^ indirect_data[(current - offset - 52) & 0xFF]; break;
      case 53: result = (result + 23) ^ indirect_data[(current - offset - 53) & 0xFF]; break;
      case 54: result = (result + 23) ^ indirect_data[(current - offset - 54) & 0xFF]; break;
      case 55: result = (result + 23) ^ indirect_data[(current - offset - 55) & 0xFF]; break;
      case 56: result = (result + 23) ^ indirect_data[(current - offset - 56) & 0xFF]; break;
      case 57: result = (result + 23) ^ indirect_data[(current - offset - 57) & 0xFF]; break;
      case 58: result = (result + 23) ^ indirect_data[(current - offset - 58) & 0xFF]; break;
      case 59: result = (result + 23) ^ indirect_data[(current - offset - 59) & 0xFF]; break;
      case 60: result = (result + 23) ^ indirect_data[(current - offset - 60) & 0xFF]; break;
      case 61: result = (result + 23) ^ indirect_data[(current - offset - 61) & 0xFF]; break;
      case 62: result = (result + 23) ^ indirect_data[(current - offset - 62) & 0xFF]; break;
      case 63: result = (result + 23) ^ indirect_data[(current - offset - 63) & 0xFF]; break;
      case 64: result = (result + 23) ^ indirect_data[(current - offset - 64) & 0xFF]; break;
      case 65: result = (result + 23) ^ indirect_data[(current - offset - 65) & 0xFF]; break;
      case 66: result = (result + 23) ^ indirect_data[(current - offset - 66) & 0xFF]; break;
      case 67: result = (result + 23) ^ indirect_data[(current - offset - 67) & 0xFF]; break;
      case 68: result = (result + 23) ^ indirect_data[(current - offset - 68) & 0xFF]; break;
      case 69: result = (result + 23) ^ indirect_data[(current - offset - 69) & 0xFF]; break;
      case 70: result = (result + 23) ^ indirect_data[(current - offset - 70) & 0xFF]; break;
      case 71: result = (result + 23) ^ indirect_data[(current - offset - 71) & 0xFF]; break;
      case 72: result = (result + 23) ^ indirect_data[(current - offset - 72) & 0xFF]; break;
      case 73: result = (result + 23) ^ indirect_data[(current - offset - 73) & 0xFF]; break;
      case 74: result = (result + 23) ^ indirect_data[(current - offset - 74) & 0xFF]; break;
      case 75: result = (result + 23) ^ indirect_data[(current - offset - 75) & 0xFF]; break;
      case 76: result = (result + 23) ^ indirect_data[(current - offset - 76) & 0xFF]; break;
      case 77: result = (result + 23) ^ indirect_data[(current - offset - 77) & 0xFF]; break;
      case 78: result = (result + 23) ^ indirect_data[(current - offset - 78) & 0xFF]; break;
      case 79: result = (result + 23) ^ indirect_data[(current - offset - 79) & 0xFF]; break;
      case 80: result = (result + 23) ^ indirect_data[(current - offset - 80) & 0xFF]; break;
      case 81: result = (result + 23) ^ indirect_data[(current - offset - 81) & 0xFF]; break;
      case 82: result = (result + 23) ^ indirect_data[(current - offset - 82) & 0xFF]; break;
      case 83: result = (result + 23) ^ indirect_data[(current - offset - 83) & 0xFF]; break;
      case 84: result = (result + 23) ^ indirect_data[(current - offset - 84) & 0xFF]; break;
      case 85: result = (result + 23) ^ indirect_data[(current - offset - 85) & 0xFF]; break;
      case 86: result = (result + 23) ^ indirect_data[(current - offset - 86) & 0xFF]; break;
      case 87: result = (result + 23) ^ indirect_data[(current - offset - 87) & 0xFF]; break;
      case 88: result = (result + 23) ^ indirect_data[(current - offset - 88) & 0xFF]; break;
      case 89: result = (result + 23) ^ indirect_data[(current - offset - 89) & 0xFF]; break;
      case 90: result = (result + 23) ^ indirect_data[(current - offset - 90) & 0xFF]; break;
      case 91: result = (result + 23) ^ indirect_data[(current - offset - 91) & 0xFF]; break;
      case 92: result = (result + 23) ^ indirect_data[(current - offset - 92) & 0xFF]; break;
      case 93: result = (result + 23) ^ indirect_data[(current - offset - 93) & 0xFF]; break;
      case 94: result = (result + 23) ^ indirect_data[(current - offset - 94) & 0xFF]; break;
      case 95: result = (result + 23) ^ indirect_data[(current - offset - 95) & 0xFF]; break;
      case 96: result = (result + 23) ^ indirect_data[(current - offset - 96) & 0xFF]; break;
      case 97: result = (result + 23) ^ indirect_data[(current - offset - 97) & 0xFF]; break;
      case 98: result = (result + 23) ^ indirect_data[(current - offset - 98) & 0xFF]; break;
      case 99: result = (result + 23) ^ indirect_data[(current - offset - 99) & 0xFF]; break;
      case 100: result = (result + 23) ^ indirect_data[(current - offset - 100) & 0xFF]; break;
      case 101: result = (result + 23) ^ indirect_data[(current - offset - 101) & 0xFF]; break;
      case 102: result = (result + 23) ^ indirect_data[(current - offset - 102) & 0xFF]; break;
      case 103: result = (result + 23) ^ indirect_data[(current - offset - 103) & 0xFF]; break;
      case 104: result = (result + 23) ^ indirect_data[(current - offset - 104) & 0xFF]; break;
      case 105: result = (result + 23) ^ indirect_data[(current - offset - 105) & 0xFF]; break;
      case 106: result = (result + 23) ^ indirect_data[(current - offset - 106) & 0xFF]; break;
      case 107: result = (result + 23) ^ indirect_data[(current - offset - 107) & 0xFF]; break;
      case 108: result = (result + 23) ^ indirect_data[(current - offset - 108) & 0xFF]; break;
      case 109: result = (result + 23) ^ indirect_data[(current - offset - 109) & 0xFF]; break;
      case 110: result = (result + 23) ^ indirect_data[(current - offset - 110) & 0xFF]; break;
      case 111: result = (result + 23) ^ indirect_data[(current - offset - 111) & 0xFF]; break;
      case 112: result = (result + 23) ^ indirect_data[(current - offset - 112) & 0xFF]; break;
      case 113: result = (result + 23) ^ indirect_data[(current - offset - 113) & 0xFF]; break;
      case 114: result = (result + 23) ^ indirect_data[(current - offset - 114) & 0xFF]; break;
      case 115: result = (result + 23) ^ indirect_data[(current - offset - 115) & 0xFF]; break;
      case 116: result = (result + 23) ^ indirect_data[(current - offset - 116) & 0xFF]; break;
      case 117: result = (result + 23) ^ indirect_data[(current - offset - 117) & 0xFF]; break;
      case 118: result = (result + 23) ^ indirect_data[(current - offset - 118) & 0xFF]; break;
      case 119: result = (result + 23) ^ indirect_data[(current - offset - 119) & 0xFF]; break;
      case 120: result = (result + 23) ^ indirect_data[(current - offset - 120) & 0xFF]; break;
      case 121: result = (result + 23) ^ indirect_data[(current - offset - 121) & 0xFF]; break;
      case 122: result = (result + 23) ^ indirect_data[(current - offset - 122) & 0xFF]; break;
      case 123: result = (result + 23) ^ indirect_data[(current - offset - 123) & 0xFF]; break;
      case 124: result = (result + 23) ^ indirect_data[(current - offset - 124) & 0xFF]; break;
      case 125: result = (result + 23) ^ indirect_data[(current - offset - 125) & 0xFF]; break;
      case 126: result = (result + 23) ^ indirect_data[(current - offset - 126) & 0xFF]; break;
      case 127: result = (result + 23) ^ indirect_data[(current - offset - 127) & 0xFF]; break;
      case 128: result = (result + 23) ^ indirect_data[(current - offset - 128) & 0xFF]; break;
      case 129: result = (result + 23) ^ indirect_data[(current - offset - 129) & 0xFF]; break;
      case 130: result = (result + 23) ^ indirect_data[(current - offset - 130) & 0xFF]; break;
      case 131: result = (result + 23) ^ indirect_data[(current - offset - 131) & 0xFF]; break;
      case 132: result = (result + 23) ^ indirect_data[(current - offset - 132) & 0xFF]; break;
      case 133: result = (result + 23) ^ indirect_data[(current - offset - 133) & 0xFF]; break;
      case 134: result = (result + 23) ^ indirect_data[(current - offset - 134) & 0xFF]; break;
      case 135: result = (result + 23) ^ indirect_data[(current - offset - 135) & 0xFF]; break;
      case 136: result = (result + 23) ^ indirect_data[(current - offset - 136) & 0xFF]; break;
      case 137: result = (result + 23) ^ indirect_data[(current - offset - 137) & 0xFF]; break;
      case 138: result = (result + 23) ^ indirect_data[(current - offset - 138) & 0xFF]; break;
      case 139: result = (result + 23) ^ indirect_data[(current - offset - 139) & 0xFF]; break;
      case 140: result = (result + 23) ^ indirect_data[(current - offset - 140) & 0xFF]; break;
      case 141: result = (result + 23) ^ indirect_data[(current - offset - 141) & 0xFF]; break;
      case 142: result = (result + 23) ^ indirect_data[(current - offset - 142) & 0xFF]; break;
      case 143: result = (result + 23) ^ indirect_data[(current - offset - 143) & 0xFF]; break;
      case 144: result = (result + 23) ^ indirect_data[(current - offset - 144) & 0xFF]; break;
      case 145: result = (result + 23) ^ indirect_data[(current - offset - 145) & 0xFF]; break;
      case 146: result = (result + 23) ^ indirect_data[(current - offset - 146) & 0xFF]; break;
      case 147: result = (result + 23) ^ indirect_data[(current - offset - 147) & 0xFF]; break;
      case 148: result = (result + 23) ^ indirect_data[(current - offset - 148) & 0xFF]; break;
      case 149: result = (result + 23) ^ indirect_data[(current - offset - 149) & 0xFF]; break;
      case 150: result = (result + 23) ^ indirect_data[(current - offset - 150) & 0xFF]; break;
      case 151: result = (result + 23) ^ indirect_data[(current - offset - 151) & 0xFF]; break;
      case 152: result = (result + 23) ^ indirect_data[(current - offset - 152) & 0xFF]; break;
      case 153: result = (result + 23) ^ indirect_data[(current - offset - 153) & 0xFF]; break;
      case 154: result = (result + 23) ^ indirect_data[(current - offset - 154) & 0xFF]; break;
      case 155: result = (result + 23) ^ indirect_data[(current - offset - 155) & 0xFF]; break;
      case 156: result = (result + 23) ^ indirect_data[(current - offset - 156) & 0xFF]; break;
      case 157: result = (result + 23) ^ indirect_data[(current - offset - 157) & 0xFF]; break;
      case 158: result = (result + 23) ^ indirect_data[(current - offset - 158) & 0xFF]; break;
      case 159: result = (result + 23) ^ indirect_data[(current - offset - 159) & 0xFF]; break;
      case 160: result = (result + 23) ^ indirect_data[(current - offset - 160) & 0xFF]; break;
      case 161: result = (result + 23) ^ indirect_data[(current - offset - 161) & 0xFF]; break;
      case 162: result = (result + 23) ^ indirect_data[(current - offset - 162) & 0xFF]; break;
      case 163: result = (result + 23) ^ indirect_data[(current - offset - 163) & 0xFF]; break;
      case 164: result = (result + 23) ^ indirect_data[(current - offset - 164) & 0xFF]; break;
      case 165: result = (result + 23) ^ indirect_data[(current - offset - 165) & 0xFF]; break;
      case 166: result = (result + 23) ^ indirect_data[(current - offset - 166) & 0xFF]; break;
      case 167: result = (result + 23) ^ indirect_data[(current - offset - 167) & 0xFF]; break;
      case 168: result = (result + 23) ^ indirect_data[(current - offset - 168) & 0xFF]; break;
      case 169: result = (result + 23) ^ indirect_data[(current - offset - 169) & 0xFF]; break;
      case 170: result = (result + 23) ^ indirect_data[(current - offset - 170) & 0xFF]; break;
      case 171: result = (result + 23) ^ indirect_data[(current - offset - 171) & 0xFF]; break;
      case 172: result = (result + 23) ^ indirect_data[(current - offset - 172) & 0xFF]; break;
      case 173: result = (result + 23) ^ indirect_data[(current - offset - 173) & 0xFF]; break;
      case 174: result = (result + 23) ^ indirect_data[(current - offset - 174) & 0xFF]; break;
      case 175: result = (result + 23) ^ indirect_data[(current - offset - 175) & 0xFF]; break;
      case 176: result = (result + 23) ^ indirect_data[(current - offset - 176) & 0xFF]; break;
      case 177: result = (result + 23) ^ indirect_data[(current - offset - 177) & 0xFF]; break;
      case 178: result = (result + 23) ^ indirect_data[(current - offset - 178) & 0xFF]; break;
      case 179: result = (result + 23) ^ indirect_data[(current - offset - 179) & 0xFF]; break;
      case 180: result = (result + 23) ^ indirect_data[(current - offset - 180) & 0xFF]; break;
      case 181: result = (result + 23) ^ indirect_data[(current - offset - 181) & 0xFF]; break;
      case 182: result = (result + 23) ^ indirect_data[(current - offset - 182) & 0xFF]; break;
      case 183: result = (result + 23) ^ indirect_data[(current - offset - 183) & 0xFF]; break;
      case 184: result = (result + 23) ^ indirect_data[(current - offset - 184) & 0xFF]; break;
      case 185: result = (result + 23) ^ indirect_data[(current - offset - 185) & 0xFF]; break;
      case 186: result = (result + 23) ^ indirect_data[(current - offset - 186) & 0xFF]; break;
      case 187: result = (result + 23) ^ indirect_data[(current - offset - 187) & 0xFF]; break;
      case 188: result = (result + 23) ^ indirect_data[(current - offset - 188) & 0xFF]; break;
      case 189: result = (result + 23) ^ indirect_data[(current - offset - 189) & 0xFF]; break;
      case 190: result = (result + 23) ^ indirect_data[(current - offset - 190) & 0xFF]; break;
      case 191: result = (result + 23) ^ indirect_data[(current - offset - 191) & 0xFF]; break;
      case 192: result = (result + 23) ^ indirect_data[(current - offset - 192) & 0xFF]; break;
      case 193: result = (result + 23) ^ indirect_data[(current - offset - 193) & 0xFF]; break;
      case 194: result = (result + 23) ^ indirect_data[(current - offset - 194) & 0xFF]; break;
      case 195: result = (result + 23) ^ indirect_data[(current - offset - 195) & 0xFF]; break;
      case 196: result = (result + 23) ^ indirect_data[(current - offset - 196) & 0xFF]; break;
      case 197: result = (result + 23) ^ indirect_data[(current - offset - 197) & 0xFF]; break;
      case 198: result = (result + 23) ^ indirect_data[(current - offset - 198) & 0xFF]; break;
      case 199: result = (result + 23) ^ indirect_data[(current - offset - 199) & 0xFF]; break;
      case 200: result = (result + 23) ^ indirect_data[(current - offset - 200) & 0xFF]; break;
      case 201: result = (result + 23) ^ indirect_data[(current - offset - 201) & 0xFF]; break;
      case 202: result = (result + 23) ^ indirect_data[(current - offset - 202) & 0xFF]; break;
      case 203: result = (result + 23) ^ indirect_data[(current - offset - 203) & 0xFF]; break;
      case 204: result = (result + 23) ^ indirect_data[(current - offset - 204) & 0xFF]; break;
      case 205: result = (result + 23) ^ indirect_data[(current - offset - 205) & 0xFF]; break;
      case 206: result = (result + 23) ^ indirect_data[(current - offset - 206) & 0xFF]; break;
      case 207: result = (result + 23) ^ indirect_data[(current - offset - 207) & 0xFF]; break;
      case 208: result = (result + 23) ^ indirect_data[(current - offset - 208) & 0xFF]; break;
      case 209: result = (result + 23) ^ indirect_data[(current - offset - 209) & 0xFF]; break;
      case 210: result = (result + 23) ^ indirect_data[(current - offset - 210) & 0xFF]; break;
      case 211: result = (result + 23) ^ indirect_data[(current - offset - 211) & 0xFF]; break;
      case 212: result = (result + 23) ^ indirect_data[(current - offset - 212) & 0xFF]; break;
      case 213: result = (result + 23) ^ indirect_data[(current - offset - 213) & 0xFF]; break;
      case 214: result = (result + 23) ^ indirect_data[(current - offset - 214) & 0xFF]; break;
      case 215: result = (result + 23) ^ indirect_data[(current - offset - 215) & 0xFF]; break;
      case 216: result = (result + 23) ^ indirect_data[(current - offset - 216) & 0xFF]; break;
      case 217: result = (result + 23) ^ indirect_data[(current - offset - 217) & 0xFF]; break;
      case 218: result = (result + 23) ^ indirect_data[(current - offset - 218) & 0xFF]; break;
      case 219: result = (result + 23) ^ indirect_data[(current - offset - 219) & 0xFF]; break;
      case 220: result = (result + 23) ^ indirect_data[(current - offset - 220) & 0xFF]; break;
      case 221: result = (result + 23) ^ indirect_data[(current - offset - 221) & 0xFF]; break;
      case 222: result = (result + 23) ^ indirect_data[(current - offset - 222) & 0xFF]; break;
      case 223: result = (result + 23) ^ indirect_data[(current - offset - 223) & 0xFF]; break;
      case 224: result = (result + 23) ^ indirect_data[(current - offset - 224) & 0xFF]; break;
      case 225: result = (result + 23) ^ indirect_data[(current - offset - 225) & 0xFF]; break;
      case 226: result = (result + 23) ^ indirect_data[(current - offset - 226) & 0xFF]; break;
      case 227: result = (result + 23) ^ indirect_data[(current - offset - 227) & 0xFF]; break;
      case 228: result = (result + 23) ^ indirect_data[(current - offset - 228) & 0xFF]; break;
      case 229: result = (result + 23) ^ indirect_data[(current - offset - 229) & 0xFF]; break;
      case 230: result = (result + 23) ^ indirect_data[(current - offset - 230) & 0xFF]; break;
      case 231: result = (result + 23) ^ indirect_data[(current - offset - 231) & 0xFF]; break;
      case 232: result = (result + 23) ^ indirect_data[(current - offset - 232) & 0xFF]; break;
      case 233: result = (result + 23) ^ indirect_data[(current - offset - 233) & 0xFF]; break;
      case 234: result = (result + 23) ^ indirect_data[(current - offset - 234) & 0xFF]; break;
      case 235: result = (result + 23) ^ indirect_data[(current - offset - 235) & 0xFF]; break;
      case 236: result = (result + 23) ^ indirect_data[(current - offset - 236) & 0xFF]; break;
      case 237: result = (result + 23) ^ indirect_data[(current - offset - 237) & 0xFF]; break;
      case 238: result = (result + 23) ^ indirect_data[(current - offset - 238) & 0xFF]; break;
      case 239: result = (result + 23) ^ indirect_data[(current - offset - 239) & 0xFF]; break;
      case 240: result = (result + 23) ^ indirect_data[(current - offset - 240) & 0xFF]; break;
      case 241: result = (result + 23) ^ indirect_data[(current - offset - 241) & 0xFF]; break;
      case 242: result = (result + 23) ^ indirect_data[(current - offset - 242) & 0xFF]; break;
      case 243: result = (result + 23) ^ indirect_data[(current - offset - 243) & 0xFF]; break;
      case 244: result = (result + 23) ^ indirect_data[(current - offset - 244) & 0xFF]; break;
      case 245: result = (result + 23) ^ indirect_data[(current - offset - 245) & 0xFF]; break;
      case 246: result = (result + 23) ^ indirect_data[(current - offset - 246) & 0xFF]; break;
      case 247: result = (result + 23) ^ indirect_data[(current - offset - 247) & 0xFF]; break;
      case 248: result = (result + 23) ^ indirect_data[(current - offset - 248) & 0xFF]; break;
      case 249: result = (result + 23) ^ indirect_data[(current - offset - 249) & 0xFF]; break;
      case 250: result = (result + 23) ^ indirect_data[(current - offset - 250) & 0xFF]; break;
      case 251: result = (result + 23) ^ indirect_data[(current - offset - 251) & 0xFF]; break;
      case 252: result = (result + 23) ^ indirect_data[(current - offset - 252) & 0xFF]; break;
      case 253: result = (result + 23) ^ indirect_data[(current - offset - 253) & 0xFF]; break;
      case 254: result = (result + 23) ^ indirect_data[(current - offset - 254) & 0xFF]; break;
      case 255: result = (result + 23) ^ indirect_data[(current - offset - 255) & 0xFF]; break;
    }
    return result & 0xFF;
  };

  auto switch_fn_3 = [&](uint64_t& payload, uint64_t offset) {
    int result = 0;
    uint64_t idx = (payload + offset * 2) & 0xFF;
    switch (idx) {
      case 0: result = (result ^ 29) + indirect_data[(current + offset * 2 + 0) & 0xFF]; break;
      case 1: result = (result ^ 29) + indirect_data[(current + offset * 2 + 1) & 0xFF]; break;
      case 2: result = (result ^ 29) + indirect_data[(current + offset * 2 + 2) & 0xFF]; break;
      case 3: result = (result ^ 29) + indirect_data[(current + offset * 2 + 3) & 0xFF]; break;
      case 4: result = (result ^ 29) + indirect_data[(current + offset * 2 + 4) & 0xFF]; break;
      case 5: result = (result ^ 29) + indirect_data[(current + offset * 2 + 5) & 0xFF]; break;
      case 6: result = (result ^ 29) + indirect_data[(current + offset * 2 + 6) & 0xFF]; break;
      case 7: result = (result ^ 29) + indirect_data[(current + offset * 2 + 7) & 0xFF]; break;
      case 8: result = (result ^ 29) + indirect_data[(current + offset * 2 + 8) & 0xFF]; break;
      case 9: result = (result ^ 29) + indirect_data[(current + offset * 2 + 9) & 0xFF]; break;
      case 10: result = (result ^ 29) + indirect_data[(current + offset * 2 + 10) & 0xFF]; break;
      case 11: result = (result ^ 29) + indirect_data[(current + offset * 2 + 11) & 0xFF]; break;
      case 12: result = (result ^ 29) + indirect_data[(current + offset * 2 + 12) & 0xFF]; break;
      case 13: result = (result ^ 29) + indirect_data[(current + offset * 2 + 13) & 0xFF]; break;
      case 14: result = (result ^ 29) + indirect_data[(current + offset * 2 + 14) & 0xFF]; break;
      case 15: result = (result ^ 29) + indirect_data[(current + offset * 2 + 15) & 0xFF]; break;
      case 16: result = (result ^ 29) + indirect_data[(current + offset * 2 + 16) & 0xFF]; break;
      case 17: result = (result ^ 29) + indirect_data[(current + offset * 2 + 17) & 0xFF]; break;
      case 18: result = (result ^ 29) + indirect_data[(current + offset * 2 + 18) & 0xFF]; break;
      case 19: result = (result ^ 29) + indirect_data[(current + offset * 2 + 19) & 0xFF]; break;
      case 20: result = (result ^ 29) + indirect_data[(current + offset * 2 + 20) & 0xFF]; break;
      case 21: result = (result ^ 29) + indirect_data[(current + offset * 2 + 21) & 0xFF]; break;
      case 22: result = (result ^ 29) + indirect_data[(current + offset * 2 + 22) & 0xFF]; break;
      case 23: result = (result ^ 29) + indirect_data[(current + offset * 2 + 23) & 0xFF]; break;
      case 24: result = (result ^ 29) + indirect_data[(current + offset * 2 + 24) & 0xFF]; break;
      case 25: result = (result ^ 29) + indirect_data[(current + offset * 2 + 25) & 0xFF]; break;
      case 26: result = (result ^ 29) + indirect_data[(current + offset * 2 + 26) & 0xFF]; break;
      case 27: result = (result ^ 29) + indirect_data[(current + offset * 2 + 27) & 0xFF]; break;
      case 28: result = (result ^ 29) + indirect_data[(current + offset * 2 + 28) & 0xFF]; break;
      case 29: result = (result ^ 29) + indirect_data[(current + offset * 2 + 29) & 0xFF]; break;
      case 30: result = (result ^ 29) + indirect_data[(current + offset * 2 + 30) & 0xFF]; break;
      case 31: result = (result ^ 29) + indirect_data[(current + offset * 2 + 31) & 0xFF]; break;
      case 32: result = (result ^ 29) + indirect_data[(current + offset * 2 + 32) & 0xFF]; break;
      case 33: result = (result ^ 29) + indirect_data[(current + offset * 2 + 33) & 0xFF]; break;
      case 34: result = (result ^ 29) + indirect_data[(current + offset * 2 + 34) & 0xFF]; break;
      case 35: result = (result ^ 29) + indirect_data[(current + offset * 2 + 35) & 0xFF]; break;
      case 36: result = (result ^ 29) + indirect_data[(current + offset * 2 + 36) & 0xFF]; break;
      case 37: result = (result ^ 29) + indirect_data[(current + offset * 2 + 37) & 0xFF]; break;
      case 38: result = (result ^ 29) + indirect_data[(current + offset * 2 + 38) & 0xFF]; break;
      case 39: result = (result ^ 29) + indirect_data[(current + offset * 2 + 39) & 0xFF]; break;
      case 40: result = (result ^ 29) + indirect_data[(current + offset * 2 + 40) & 0xFF]; break;
      case 41: result = (result ^ 29) + indirect_data[(current + offset * 2 + 41) & 0xFF]; break;
      case 42: result = (result ^ 29) + indirect_data[(current + offset * 2 + 42) & 0xFF]; break;
      case 43: result = (result ^ 29) + indirect_data[(current + offset * 2 + 43) & 0xFF]; break;
      case 44: result = (result ^ 29) + indirect_data[(current + offset * 2 + 44) & 0xFF]; break;
      case 45: result = (result ^ 29) + indirect_data[(current + offset * 2 + 45) & 0xFF]; break;
      case 46: result = (result ^ 29) + indirect_data[(current + offset * 2 + 46) & 0xFF]; break;
      case 47: result = (result ^ 29) + indirect_data[(current + offset * 2 + 47) & 0xFF]; break;
      case 48: result = (result ^ 29) + indirect_data[(current + offset * 2 + 48) & 0xFF]; break;
      case 49: result = (result ^ 29) + indirect_data[(current + offset * 2 + 49) & 0xFF]; break;
      case 50: result = (result ^ 29) + indirect_data[(current + offset * 2 + 50) & 0xFF]; break;
      case 51: result = (result ^ 29) + indirect_data[(current + offset * 2 + 51) & 0xFF]; break;
      case 52: result = (result ^ 29) + indirect_data[(current + offset * 2 + 52) & 0xFF]; break;
      case 53: result = (result ^ 29) + indirect_data[(current + offset * 2 + 53) & 0xFF]; break;
      case 54: result = (result ^ 29) + indirect_data[(current + offset * 2 + 54) & 0xFF]; break;
      case 55: result = (result ^ 29) + indirect_data[(current + offset * 2 + 55) & 0xFF]; break;
      case 56: result = (result ^ 29) + indirect_data[(current + offset * 2 + 56) & 0xFF]; break;
      case 57: result = (result ^ 29) + indirect_data[(current + offset * 2 + 57) & 0xFF]; break;
      case 58: result = (result ^ 29) + indirect_data[(current + offset * 2 + 58) & 0xFF]; break;
      case 59: result = (result ^ 29) + indirect_data[(current + offset * 2 + 59) & 0xFF]; break;
      case 60: result = (result ^ 29) + indirect_data[(current + offset * 2 + 60) & 0xFF]; break;
      case 61: result = (result ^ 29) + indirect_data[(current + offset * 2 + 61) & 0xFF]; break;
      case 62: result = (result ^ 29) + indirect_data[(current + offset * 2 + 62) & 0xFF]; break;
      case 63: result = (result ^ 29) + indirect_data[(current + offset * 2 + 63) & 0xFF]; break;
      case 64: result = (result ^ 29) + indirect_data[(current + offset * 2 + 64) & 0xFF]; break;
      case 65: result = (result ^ 29) + indirect_data[(current + offset * 2 + 65) & 0xFF]; break;
      case 66: result = (result ^ 29) + indirect_data[(current + offset * 2 + 66) & 0xFF]; break;
      case 67: result = (result ^ 29) + indirect_data[(current + offset * 2 + 67) & 0xFF]; break;
      case 68: result = (result ^ 29) + indirect_data[(current + offset * 2 + 68) & 0xFF]; break;
      case 69: result = (result ^ 29) + indirect_data[(current + offset * 2 + 69) & 0xFF]; break;
      case 70: result = (result ^ 29) + indirect_data[(current + offset * 2 + 70) & 0xFF]; break;
      case 71: result = (result ^ 29) + indirect_data[(current + offset * 2 + 71) & 0xFF]; break;
      case 72: result = (result ^ 29) + indirect_data[(current + offset * 2 + 72) & 0xFF]; break;
      case 73: result = (result ^ 29) + indirect_data[(current + offset * 2 + 73) & 0xFF]; break;
      case 74: result = (result ^ 29) + indirect_data[(current + offset * 2 + 74) & 0xFF]; break;
      case 75: result = (result ^ 29) + indirect_data[(current + offset * 2 + 75) & 0xFF]; break;
      case 76: result = (result ^ 29) + indirect_data[(current + offset * 2 + 76) & 0xFF]; break;
      case 77: result = (result ^ 29) + indirect_data[(current + offset * 2 + 77) & 0xFF]; break;
      case 78: result = (result ^ 29) + indirect_data[(current + offset * 2 + 78) & 0xFF]; break;
      case 79: result = (result ^ 29) + indirect_data[(current + offset * 2 + 79) & 0xFF]; break;
      case 80: result = (result ^ 29) + indirect_data[(current + offset * 2 + 80) & 0xFF]; break;
      case 81: result = (result ^ 29) + indirect_data[(current + offset * 2 + 81) & 0xFF]; break;
      case 82: result = (result ^ 29) + indirect_data[(current + offset * 2 + 82) & 0xFF]; break;
      case 83: result = (result ^ 29) + indirect_data[(current + offset * 2 + 83) & 0xFF]; break;
      case 84: result = (result ^ 29) + indirect_data[(current + offset * 2 + 84) & 0xFF]; break;
      case 85: result = (result ^ 29) + indirect_data[(current + offset * 2 + 85) & 0xFF]; break;
      case 86: result = (result ^ 29) + indirect_data[(current + offset * 2 + 86) & 0xFF]; break;
      case 87: result = (result ^ 29) + indirect_data[(current + offset * 2 + 87) & 0xFF]; break;
      case 88: result = (result ^ 29) + indirect_data[(current + offset * 2 + 88) & 0xFF]; break;
      case 89: result = (result ^ 29) + indirect_data[(current + offset * 2 + 89) & 0xFF]; break;
      case 90: result = (result ^ 29) + indirect_data[(current + offset * 2 + 90) & 0xFF]; break;
      case 91: result = (result ^ 29) + indirect_data[(current + offset * 2 + 91) & 0xFF]; break;
      case 92: result = (result ^ 29) + indirect_data[(current + offset * 2 + 92) & 0xFF]; break;
      case 93: result = (result ^ 29) + indirect_data[(current + offset * 2 + 93) & 0xFF]; break;
      case 94: result = (result ^ 29) + indirect_data[(current + offset * 2 + 94) & 0xFF]; break;
      case 95: result = (result ^ 29) + indirect_data[(current + offset * 2 + 95) & 0xFF]; break;
      case 96: result = (result ^ 29) + indirect_data[(current + offset * 2 + 96) & 0xFF]; break;
      case 97: result = (result ^ 29) + indirect_data[(current + offset * 2 + 97) & 0xFF]; break;
      case 98: result = (result ^ 29) + indirect_data[(current + offset * 2 + 98) & 0xFF]; break;
      case 99: result = (result ^ 29) + indirect_data[(current + offset * 2 + 99) & 0xFF]; break;
      case 100: result = (result ^ 29) + indirect_data[(current + offset * 2 + 100) & 0xFF]; break;
      case 101: result = (result ^ 29) + indirect_data[(current + offset * 2 + 101) & 0xFF]; break;
      case 102: result = (result ^ 29) + indirect_data[(current + offset * 2 + 102) & 0xFF]; break;
      case 103: result = (result ^ 29) + indirect_data[(current + offset * 2 + 103) & 0xFF]; break;
      case 104: result = (result ^ 29) + indirect_data[(current + offset * 2 + 104) & 0xFF]; break;
      case 105: result = (result ^ 29) + indirect_data[(current + offset * 2 + 105) & 0xFF]; break;
      case 106: result = (result ^ 29) + indirect_data[(current + offset * 2 + 106) & 0xFF]; break;
      case 107: result = (result ^ 29) + indirect_data[(current + offset * 2 + 107) & 0xFF]; break;
      case 108: result = (result ^ 29) + indirect_data[(current + offset * 2 + 108) & 0xFF]; break;
      case 109: result = (result ^ 29) + indirect_data[(current + offset * 2 + 109) & 0xFF]; break;
      case 110: result = (result ^ 29) + indirect_data[(current + offset * 2 + 110) & 0xFF]; break;
      case 111: result = (result ^ 29) + indirect_data[(current + offset * 2 + 111) & 0xFF]; break;
      case 112: result = (result ^ 29) + indirect_data[(current + offset * 2 + 112) & 0xFF]; break;
      case 113: result = (result ^ 29) + indirect_data[(current + offset * 2 + 113) & 0xFF]; break;
      case 114: result = (result ^ 29) + indirect_data[(current + offset * 2 + 114) & 0xFF]; break;
      case 115: result = (result ^ 29) + indirect_data[(current + offset * 2 + 115) & 0xFF]; break;
      case 116: result = (result ^ 29) + indirect_data[(current + offset * 2 + 116) & 0xFF]; break;
      case 117: result = (result ^ 29) + indirect_data[(current + offset * 2 + 117) & 0xFF]; break;
      case 118: result = (result ^ 29) + indirect_data[(current + offset * 2 + 118) & 0xFF]; break;
      case 119: result = (result ^ 29) + indirect_data[(current + offset * 2 + 119) & 0xFF]; break;
      case 120: result = (result ^ 29) + indirect_data[(current + offset * 2 + 120) & 0xFF]; break;
      case 121: result = (result ^ 29) + indirect_data[(current + offset * 2 + 121) & 0xFF]; break;
      case 122: result = (result ^ 29) + indirect_data[(current + offset * 2 + 122) & 0xFF]; break;
      case 123: result = (result ^ 29) + indirect_data[(current + offset * 2 + 123) & 0xFF]; break;
      case 124: result = (result ^ 29) + indirect_data[(current + offset * 2 + 124) & 0xFF]; break;
      case 125: result = (result ^ 29) + indirect_data[(current + offset * 2 + 125) & 0xFF]; break;
      case 126: result = (result ^ 29) + indirect_data[(current + offset * 2 + 126) & 0xFF]; break;
      case 127: result = (result ^ 29) + indirect_data[(current + offset * 2 + 127) & 0xFF]; break;
      case 128: result = (result ^ 29) + indirect_data[(current + offset * 2 + 128) & 0xFF]; break;
      case 129: result = (result ^ 29) + indirect_data[(current + offset * 2 + 129) & 0xFF]; break;
      case 130: result = (result ^ 29) + indirect_data[(current + offset * 2 + 130) & 0xFF]; break;
      case 131: result = (result ^ 29) + indirect_data[(current + offset * 2 + 131) & 0xFF]; break;
      case 132: result = (result ^ 29) + indirect_data[(current + offset * 2 + 132) & 0xFF]; break;
      case 133: result = (result ^ 29) + indirect_data[(current + offset * 2 + 133) & 0xFF]; break;
      case 134: result = (result ^ 29) + indirect_data[(current + offset * 2 + 134) & 0xFF]; break;
      case 135: result = (result ^ 29) + indirect_data[(current + offset * 2 + 135) & 0xFF]; break;
      case 136: result = (result ^ 29) + indirect_data[(current + offset * 2 + 136) & 0xFF]; break;
      case 137: result = (result ^ 29) + indirect_data[(current + offset * 2 + 137) & 0xFF]; break;
      case 138: result = (result ^ 29) + indirect_data[(current + offset * 2 + 138) & 0xFF]; break;
      case 139: result = (result ^ 29) + indirect_data[(current + offset * 2 + 139) & 0xFF]; break;
      case 140: result = (result ^ 29) + indirect_data[(current + offset * 2 + 140) & 0xFF]; break;
      case 141: result = (result ^ 29) + indirect_data[(current + offset * 2 + 141) & 0xFF]; break;
      case 142: result = (result ^ 29) + indirect_data[(current + offset * 2 + 142) & 0xFF]; break;
      case 143: result = (result ^ 29) + indirect_data[(current + offset * 2 + 143) & 0xFF]; break;
      case 144: result = (result ^ 29) + indirect_data[(current + offset * 2 + 144) & 0xFF]; break;
      case 145: result = (result ^ 29) + indirect_data[(current + offset * 2 + 145) & 0xFF]; break;
      case 146: result = (result ^ 29) + indirect_data[(current + offset * 2 + 146) & 0xFF]; break;
      case 147: result = (result ^ 29) + indirect_data[(current + offset * 2 + 147) & 0xFF]; break;
      case 148: result = (result ^ 29) + indirect_data[(current + offset * 2 + 148) & 0xFF]; break;
      case 149: result = (result ^ 29) + indirect_data[(current + offset * 2 + 149) & 0xFF]; break;
      case 150: result = (result ^ 29) + indirect_data[(current + offset * 2 + 150) & 0xFF]; break;
      case 151: result = (result ^ 29) + indirect_data[(current + offset * 2 + 151) & 0xFF]; break;
      case 152: result = (result ^ 29) + indirect_data[(current + offset * 2 + 152) & 0xFF]; break;
      case 153: result = (result ^ 29) + indirect_data[(current + offset * 2 + 153) & 0xFF]; break;
      case 154: result = (result ^ 29) + indirect_data[(current + offset * 2 + 154) & 0xFF]; break;
      case 155: result = (result ^ 29) + indirect_data[(current + offset * 2 + 155) & 0xFF]; break;
      case 156: result = (result ^ 29) + indirect_data[(current + offset * 2 + 156) & 0xFF]; break;
      case 157: result = (result ^ 29) + indirect_data[(current + offset * 2 + 157) & 0xFF]; break;
      case 158: result = (result ^ 29) + indirect_data[(current + offset * 2 + 158) & 0xFF]; break;
      case 159: result = (result ^ 29) + indirect_data[(current + offset * 2 + 159) & 0xFF]; break;
      case 160: result = (result ^ 29) + indirect_data[(current + offset * 2 + 160) & 0xFF]; break;
      case 161: result = (result ^ 29) + indirect_data[(current + offset * 2 + 161) & 0xFF]; break;
      case 162: result = (result ^ 29) + indirect_data[(current + offset * 2 + 162) & 0xFF]; break;
      case 163: result = (result ^ 29) + indirect_data[(current + offset * 2 + 163) & 0xFF]; break;
      case 164: result = (result ^ 29) + indirect_data[(current + offset * 2 + 164) & 0xFF]; break;
      case 165: result = (result ^ 29) + indirect_data[(current + offset * 2 + 165) & 0xFF]; break;
      case 166: result = (result ^ 29) + indirect_data[(current + offset * 2 + 166) & 0xFF]; break;
      case 167: result = (result ^ 29) + indirect_data[(current + offset * 2 + 167) & 0xFF]; break;
      case 168: result = (result ^ 29) + indirect_data[(current + offset * 2 + 168) & 0xFF]; break;
      case 169: result = (result ^ 29) + indirect_data[(current + offset * 2 + 169) & 0xFF]; break;
      case 170: result = (result ^ 29) + indirect_data[(current + offset * 2 + 170) & 0xFF]; break;
      case 171: result = (result ^ 29) + indirect_data[(current + offset * 2 + 171) & 0xFF]; break;
      case 172: result = (result ^ 29) + indirect_data[(current + offset * 2 + 172) & 0xFF]; break;
      case 173: result = (result ^ 29) + indirect_data[(current + offset * 2 + 173) & 0xFF]; break;
      case 174: result = (result ^ 29) + indirect_data[(current + offset * 2 + 174) & 0xFF]; break;
      case 175: result = (result ^ 29) + indirect_data[(current + offset * 2 + 175) & 0xFF]; break;
      case 176: result = (result ^ 29) + indirect_data[(current + offset * 2 + 176) & 0xFF]; break;
      case 177: result = (result ^ 29) + indirect_data[(current + offset * 2 + 177) & 0xFF]; break;
      case 178: result = (result ^ 29) + indirect_data[(current + offset * 2 + 178) & 0xFF]; break;
      case 179: result = (result ^ 29) + indirect_data[(current + offset * 2 + 179) & 0xFF]; break;
      case 180: result = (result ^ 29) + indirect_data[(current + offset * 2 + 180) & 0xFF]; break;
      case 181: result = (result ^ 29) + indirect_data[(current + offset * 2 + 181) & 0xFF]; break;
      case 182: result = (result ^ 29) + indirect_data[(current + offset * 2 + 182) & 0xFF]; break;
      case 183: result = (result ^ 29) + indirect_data[(current + offset * 2 + 183) & 0xFF]; break;
      case 184: result = (result ^ 29) + indirect_data[(current + offset * 2 + 184) & 0xFF]; break;
      case 185: result = (result ^ 29) + indirect_data[(current + offset * 2 + 185) & 0xFF]; break;
      case 186: result = (result ^ 29) + indirect_data[(current + offset * 2 + 186) & 0xFF]; break;
      case 187: result = (result ^ 29) + indirect_data[(current + offset * 2 + 187) & 0xFF]; break;
      case 188: result = (result ^ 29) + indirect_data[(current + offset * 2 + 188) & 0xFF]; break;
      case 189: result = (result ^ 29) + indirect_data[(current + offset * 2 + 189) & 0xFF]; break;
      case 190: result = (result ^ 29) + indirect_data[(current + offset * 2 + 190) & 0xFF]; break;
      case 191: result = (result ^ 29) + indirect_data[(current + offset * 2 + 191) & 0xFF]; break;
      case 192: result = (result ^ 29) + indirect_data[(current + offset * 2 + 192) & 0xFF]; break;
      case 193: result = (result ^ 29) + indirect_data[(current + offset * 2 + 193) & 0xFF]; break;
      case 194: result = (result ^ 29) + indirect_data[(current + offset * 2 + 194) & 0xFF]; break;
      case 195: result = (result ^ 29) + indirect_data[(current + offset * 2 + 195) & 0xFF]; break;
      case 196: result = (result ^ 29) + indirect_data[(current + offset * 2 + 196) & 0xFF]; break;
      case 197: result = (result ^ 29) + indirect_data[(current + offset * 2 + 197) & 0xFF]; break;
      case 198: result = (result ^ 29) + indirect_data[(current + offset * 2 + 198) & 0xFF]; break;
      case 199: result = (result ^ 29) + indirect_data[(current + offset * 2 + 199) & 0xFF]; break;
      case 200: result = (result ^ 29) + indirect_data[(current + offset * 2 + 200) & 0xFF]; break;
      case 201: result = (result ^ 29) + indirect_data[(current + offset * 2 + 201) & 0xFF]; break;
      case 202: result = (result ^ 29) + indirect_data[(current + offset * 2 + 202) & 0xFF]; break;
      case 203: result = (result ^ 29) + indirect_data[(current + offset * 2 + 203) & 0xFF]; break;
      case 204: result = (result ^ 29) + indirect_data[(current + offset * 2 + 204) & 0xFF]; break;
      case 205: result = (result ^ 29) + indirect_data[(current + offset * 2 + 205) & 0xFF]; break;
      case 206: result = (result ^ 29) + indirect_data[(current + offset * 2 + 206) & 0xFF]; break;
      case 207: result = (result ^ 29) + indirect_data[(current + offset * 2 + 207) & 0xFF]; break;
      case 208: result = (result ^ 29) + indirect_data[(current + offset * 2 + 208) & 0xFF]; break;
      case 209: result = (result ^ 29) + indirect_data[(current + offset * 2 + 209) & 0xFF]; break;
      case 210: result = (result ^ 29) + indirect_data[(current + offset * 2 + 210) & 0xFF]; break;
      case 211: result = (result ^ 29) + indirect_data[(current + offset * 2 + 211) & 0xFF]; break;
      case 212: result = (result ^ 29) + indirect_data[(current + offset * 2 + 212) & 0xFF]; break;
      case 213: result = (result ^ 29) + indirect_data[(current + offset * 2 + 213) & 0xFF]; break;
      case 214: result = (result ^ 29) + indirect_data[(current + offset * 2 + 214) & 0xFF]; break;
      case 215: result = (result ^ 29) + indirect_data[(current + offset * 2 + 215) & 0xFF]; break;
      case 216: result = (result ^ 29) + indirect_data[(current + offset * 2 + 216) & 0xFF]; break;
      case 217: result = (result ^ 29) + indirect_data[(current + offset * 2 + 217) & 0xFF]; break;
      case 218: result = (result ^ 29) + indirect_data[(current + offset * 2 + 218) & 0xFF]; break;
      case 219: result = (result ^ 29) + indirect_data[(current + offset * 2 + 219) & 0xFF]; break;
      case 220: result = (result ^ 29) + indirect_data[(current + offset * 2 + 220) & 0xFF]; break;
      case 221: result = (result ^ 29) + indirect_data[(current + offset * 2 + 221) & 0xFF]; break;
      case 222: result = (result ^ 29) + indirect_data[(current + offset * 2 + 222) & 0xFF]; break;
      case 223: result = (result ^ 29) + indirect_data[(current + offset * 2 + 223) & 0xFF]; break;
      case 224: result = (result ^ 29) + indirect_data[(current + offset * 2 + 224) & 0xFF]; break;
      case 225: result = (result ^ 29) + indirect_data[(current + offset * 2 + 225) & 0xFF]; break;
      case 226: result = (result ^ 29) + indirect_data[(current + offset * 2 + 226) & 0xFF]; break;
      case 227: result = (result ^ 29) + indirect_data[(current + offset * 2 + 227) & 0xFF]; break;
      case 228: result = (result ^ 29) + indirect_data[(current + offset * 2 + 228) & 0xFF]; break;
      case 229: result = (result ^ 29) + indirect_data[(current + offset * 2 + 229) & 0xFF]; break;
      case 230: result = (result ^ 29) + indirect_data[(current + offset * 2 + 230) & 0xFF]; break;
      case 231: result = (result ^ 29) + indirect_data[(current + offset * 2 + 231) & 0xFF]; break;
      case 232: result = (result ^ 29) + indirect_data[(current + offset * 2 + 232) & 0xFF]; break;
      case 233: result = (result ^ 29) + indirect_data[(current + offset * 2 + 233) & 0xFF]; break;
      case 234: result = (result ^ 29) + indirect_data[(current + offset * 2 + 234) & 0xFF]; break;
      case 235: result = (result ^ 29) + indirect_data[(current + offset * 2 + 235) & 0xFF]; break;
      case 236: result = (result ^ 29) + indirect_data[(current + offset * 2 + 236) & 0xFF]; break;
      case 237: result = (result ^ 29) + indirect_data[(current + offset * 2 + 237) & 0xFF]; break;
      case 238: result = (result ^ 29) + indirect_data[(current + offset * 2 + 238) & 0xFF]; break;
      case 239: result = (result ^ 29) + indirect_data[(current + offset * 2 + 239) & 0xFF]; break;
      case 240: result = (result ^ 29) + indirect_data[(current + offset * 2 + 240) & 0xFF]; break;
      case 241: result = (result ^ 29) + indirect_data[(current + offset * 2 + 241) & 0xFF]; break;
      case 242: result = (result ^ 29) + indirect_data[(current + offset * 2 + 242) & 0xFF]; break;
      case 243: result = (result ^ 29) + indirect_data[(current + offset * 2 + 243) & 0xFF]; break;
      case 244: result = (result ^ 29) + indirect_data[(current + offset * 2 + 244) & 0xFF]; break;
      case 245: result = (result ^ 29) + indirect_data[(current + offset * 2 + 245) & 0xFF]; break;
      case 246: result = (result ^ 29) + indirect_data[(current + offset * 2 + 246) & 0xFF]; break;
      case 247: result = (result ^ 29) + indirect_data[(current + offset * 2 + 247) & 0xFF]; break;
      case 248: result = (result ^ 29) + indirect_data[(current + offset * 2 + 248) & 0xFF]; break;
      case 249: result = (result ^ 29) + indirect_data[(current + offset * 2 + 249) & 0xFF]; break;
      case 250: result = (result ^ 29) + indirect_data[(current + offset * 2 + 250) & 0xFF]; break;
      case 251: result = (result ^ 29) + indirect_data[(current + offset * 2 + 251) & 0xFF]; break;
      case 252: result = (result ^ 29) + indirect_data[(current + offset * 2 + 252) & 0xFF]; break;
      case 253: result = (result ^ 29) + indirect_data[(current + offset * 2 + 253) & 0xFF]; break;
      case 254: result = (result ^ 29) + indirect_data[(current + offset * 2 + 254) & 0xFF]; break;
      case 255: result = (result ^ 29) + indirect_data[(current + offset * 2 + 255) & 0xFF]; break;
    }
    return result & 0xFF;
  };


  while (iters--) {
    for (auto i = 0; i < kSize; ++i) {
      buffers[i] = IOBuf::create(size);
    }
    // Introduce indirection and ALU ops to increase I-cache footprint.
    for (int j = 0; j < 512; ++j) { // Reduced loop count for better focus on I-cache
      uint64_t offset = j; // Use loop counter as offset
      current ^= indirect_data[(current + offset) & 0xFF];
      current ^= switch_fn_1(current, offset);
      current ^= switch_fn_2(current, offset);
      current ^= switch_fn_3(current, offset);
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
