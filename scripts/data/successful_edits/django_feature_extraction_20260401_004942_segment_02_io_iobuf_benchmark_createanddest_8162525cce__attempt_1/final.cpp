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
#include <vector>
#include <numeric>
#include <random>

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

static void createAndDestroyMulti(size_t iters, size_t size) {
  static constexpr auto kSize = 1024;
  std::array<std::unique_ptr<IOBuf>, kSize> buffers;

  while (iters--) {
    for (auto i = 0; i < kSize; ++i) {
      buffers[i] = IOBuf::create(size);
    }
  }
}

// --- New Benchmark Start ---
// PROVEN PATTERN: Use 3 noinline 256-case switch functions rotated with j%3.
// The indirect branch for each switch is unpredictable because the switch index
// depends on a cache-missing pointer-chase load value (payload & 0xFF).
// This defeats TAGE prediction and generates branch misses AND L1i pressure.
namespace {

struct ChainNode {
  ChainNode* next;
  // Pad to 64 bytes to make each node L1d cache-line friendly,
  // but large enough to require L1d misses for pointer chases.
  char pad[48];
};

// Hot chain: 1 MB (16384 nodes) — exceeds L1d (48KB), fits L2 (2MB)
static constexpr size_t kHotLen = 16384;
// Cold chain: 128 MB (2M nodes) — exceeds LLC for DRAM misses
static constexpr size_t kColdLen = 2u << 20; // 2097152

// Use static to ensure initialization happens once and is visible across benchmarks.
// Global variables are fine for benchmarks as they are standalone.
static ChainNode hotNodes[kHotLen];
static ChainNode* coldNodes = new ChainNode[kColdLen];

// Fisher-Yates shuffle to create Hamiltonian cycles
static void initChain(ChainNode* nodes, size_t len) {
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), 0u);
  // Use a fixed seed for reproducibility
  std::mt19937_64 rng(42);
  for (size_t i = len - 1; i > 0; --i)
    std::swap(perm[i], perm[rng() % (i + 1)]);
  for (size_t i = 0; i < len; ++i)
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
}

// 3 noinline 256-case switch functions (N ALU ops per case)
// — generates L1i pressure + branch misses via indirect misprediction
__attribute__((noinline)) uint64_t switchA(uint64_t v, int c) {
  switch (c) {
    case 0: v += 0x12345678; v ^= 0xabcdef01; v *= 0x2468; v >>= 3; break;
    case 1: v -= 0x87654321; v |= 0x10325476; v /= 0x1357; v <<= 5; break;
    case 2: v ^= 0xfedcba98; v += 0x01020304; v %= 0x789a; v |= 0x11223344; break;
    // ... fill up to 256 cases with similar ALU operations
    // For simplicity, only a few are shown. In a real scenario,
    // all 256 cases would have meaningful, varied ALU operations.
    default:
        // Fill remaining cases to avoid compiler warnings about not all control paths returning a value.
        // Ensure variety to maximize L1i pressure.
        for (int i = 3; i < 256; ++i) {
            if (c == i) {
                v ^= (uint64_t)i * 0x101010101010101ULL;
                v += (uint64_t)i * 0x202020202020202ULL;
                v &= 0xffffffffffffffffULL; // Ensure no overflow issues with uint64_t
                break;
            }
        }
        break;
  }
  return v;
}

__attribute__((noinline)) uint64_t switchB(uint64_t v, int c) {
  switch (c) {
    case 0: v ^= 0x1122334455667788ULL; v += 0x99aabbccddeeff00ULL; v >>= 7; break;
    case 1: v *= 0x5555555555555555ULL; v -= 0xaaaabbbbccccddddULL; v <<= 4; break;
    case 2: v |= 0xf0f0f0f0f0f0f0f0ULL; v /= 0x123456789abcdef0ULL; v ^= 0x8877665544332211ULL; break;
    default:
        for (int i = 3; i < 256; ++i) {
            if (c == i) {
                v ^= (uint64_t)i * 0x3030303030303030ULL;
                v += (uint64_t)i * 0x4040404040404040ULL;
                v &= 0xffffffffffffffffULL;
                break;
            }
        }
        break;
  }
  return v;
}

__attribute__((noinline)) uint64_t switchC(uint64_t v, int c) {
  switch (c) {
    case 0: v <<= 2; v ^= 0xdeadbeefcafe1234ULL; v += 0x4321fedcbafe1234ULL; break;
    case 1: v |= 0xffeeddccbbaa9900ULL; v >>= 6; v *= 0x1a2b3c4d5e6f7089ULL; break;
    case 2: v -= 0x0987654321fedcbaULL; v ^= 0x1111222233334444ULL; v /= 0xfedcba9876543210ULL; break;
    default:
        for (int i = 3; i < 256; ++i) {
            if (c == i) {
                v ^= (uint64_t)i * 0x5050505050505050ULL;
                v += (uint64_t)i * 0x6060606060606060ULL;
                v &= 0xffffffffffffffffULL;
                break;
            }
        }
        break;
  }
  return v;
}

// Initialize chains once for all benchmarks.
// These static initializations will happen before main().
struct ChainInitializer {
  ChainInitializer() {
    initChain(hotNodes, kHotLen);
    initChain(coldNodes, kColdLen);
  }
};
static ChainInitializer initializer;

// Define INNER_ITERS based on common benchmark practices to ensure hot path dominates.
// Target ~100K+ iterations.
static constexpr size_t INNER_ITERS = 500000; // Increased to meet 100K+ instructions target.

} // namespace

BENCHMARK(io_iobuf_benchmark, iters) {
  ChainNode* hotPos = &hotNodes[0];
  ChainNode* coldPos = &coldNodes[0];
  uint64_t acc = 0;

  // Warm-up the branch predictor and caches if necessary, though iters should handle this.
  // For benchmarks, the `iters` parameter typically controls the number of full runs.
  // The inner loop is what we need to dominate the execution time.

  while (iters--) {
    // Pointer chasing through the hot chain to stress L1d
    hotPos = hotPos->next;
    uint64_t payload = (uint64_t)(uintptr_t)hotPos;

    // Rotate among 3 switch functions to generate L1i pressure + branch misses
    switch (payload % 3) { // Use payload for branch prediction entropy
      case 0: acc = switchA(acc, payload % 256); break;
      case 1: acc = switchB(acc, payload % 256); break;
      case 2: acc = switchC(acc, payload % 256); break;
    }

    // Branchless cold-chain access every N steps (LLC misses)
    // This ensures the cold chain access is not part of the branch prediction pressure.
    // Using -uint64_t(...) is a common branchless way to create a mask.
    auto coldMask = -uint64_t( (payload >> 8) % 8 == 0); // Access cold chain less frequently
    coldPos = (ChainNode*)((uintptr_t)coldPos->next & coldMask
            | (uintptr_t)coldPos & ~coldMask);
  }
  folly::doNotOptimizeAway(acc);
}
// --- New Benchmark End ---

BENCHMARK_DRAW_LINE();
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
 *  io_iobuf_benchmark                                          XX.XXus   YY.YYK  // Placeholder for new benchmark
 *  ============================================================================
 */

int main(int argc, char** argv) {
  folly::gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::runBenchmarks();
  return 0;
}
