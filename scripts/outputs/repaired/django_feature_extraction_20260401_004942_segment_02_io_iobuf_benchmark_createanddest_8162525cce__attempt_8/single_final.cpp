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

// Helper functions for createAndDestroyMulti_LargeAlloc_Variant
__attribute__((noinline)) static void largeAllocSwitchFn1(uint64_t& payload) {
  uint64_t acc = payload;
  for (int i = 0; i < 8; ++i) { // 8 ALU ops per case
    acc = (acc * 0x13767C0000000000ULL) ^ (acc >> 7);
    acc = (acc + 0x7476797371706162ULL) ^ (acc << 3);
    acc = (acc * 0x13767C0000000000ULL) ^ (acc >> 7);
    acc = (acc + 0x7476797371706162ULL) ^ (acc << 3);
    acc = (acc * 0x13767C0000000000ULL) ^ (acc >> 7);
    acc = (acc + 0x7476797371706162ULL) ^ (acc << 3);
    acc = (acc * 0x13767C0000000000ULL) ^ (acc >> 7);
    acc = (acc + 0x7476797371706162ULL) ^ (acc << 3);
  }
  payload = acc;
}

__attribute__((noinline)) static void largeAllocSwitchFn2(uint64_t& payload) {
  uint64_t acc = payload;
  for (int i = 0; i < 8; ++i) { // 8 ALU ops per case
    acc = (acc ^ 0x13767C0000000000ULL) + (acc >> 7);
    acc = (acc - 0x7476797371706162ULL) ^ (acc >> 3);
    acc = (acc ^ 0x13767C0000000000ULL) + (acc >> 7);
    acc = (acc - 0x7476797371706162ULL) ^ (acc >> 3);
    acc = (acc ^ 0x13767C0000000000ULL) + (acc >> 7);
    acc = (acc - 0x7476797371706162ULL) ^ (acc >> 3);
    acc = (acc ^ 0x13767C0000000000ULL) + (acc >> 7);
    acc = (acc - 0x7476797371706162ULL) ^ (acc >> 3);
  }
  payload = acc;
}

__attribute__((noinline)) static void largeAllocSwitchFn3(uint64_t& payload) {
  uint64_t acc = payload;
  for (int i = 0; i < 8; ++i) { // 8 ALU ops per case
    acc = (acc << 7) ^ (acc * 0x13767C0000000000ULL);
    acc = (acc + 0x7476797371706162ULL) ^ (acc << 3);
    acc = (acc << 7) ^ (acc * 0x13767C0000000000ULL);
    acc = (acc + 0x7476797371706162ULL) ^ (acc << 3);
    acc = (acc << 7) ^ (acc * 0x13767C0000000000ULL);
    acc = (acc + 0x7476797371706162ULL) ^ (acc << 3);
    acc = (acc << 7) ^ (acc * 0x13767C0000000000ULL);
    acc = (acc + 0x7476797371706162ULL) ^ (acc << 3);
  }
  payload = acc;
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

  uint64_t payload = 0;
  while (iters--) {
    for (auto i = 0; i < kSize; ++i) {
      buffers[i] = IOBuf::create(size);
    }
    // Introduce indirect branches and pointer chasing
    for (int j = 0; j < 256; ++j) {
      switch (j % 3) {
        case 0:
          largeAllocSwitchFn1(payload);
          break;
        case 1:
          largeAllocSwitchFn2(payload);
          break;
        case 2:
          largeAllocSwitchFn3(payload);
          break;
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
