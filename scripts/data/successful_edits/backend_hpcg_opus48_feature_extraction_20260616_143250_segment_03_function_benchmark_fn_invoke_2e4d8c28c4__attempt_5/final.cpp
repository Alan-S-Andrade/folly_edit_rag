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

#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <glog/logging.h>

#include <folly/Benchmark.h>
#include <folly/ScopeGuard.h>
#include <folly/portability/GFlags.h>
#include <folly/test/function_benchmark/benchmark_impl.h>
#include <folly/test/function_benchmark/test_functions.h>

using folly::makeGuard;

// Directly invoking a function
BENCHMARK(fn_invoke, iters) {
  for (size_t n = 0; n < iters; ++n) {
    doNothing();
  }
}

namespace {

// Pointer-chase chain node, one cache line per node.
struct ChainNode {
  ChainNode* next;
  char pad[56];
};

// Hot chain: 1 MB (16384 x 64B) — exceeds L1d, drives L1d misses.
constexpr size_t kHotLen = 16384;
// Cold chain: 128 MB (2M x 64B) — exceeds LLC, drives LLC misses.
constexpr size_t kColdLen = 2u << 20;

// Build a single Hamiltonian cycle over a permutation of the nodes so that
// the dependent loads are unpredictable to the hardware prefetcher.
ChainNode* makeChain(size_t len) {
  auto* nodes = new ChainNode[len];
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), size_t{0});
  std::mt19937_64 rng(42);
  for (size_t i = len - 1; i > 0; --i) {
    std::swap(perm[i], perm[rng() % (i + 1)]);
  }
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
  return nodes;
}

// 64-case switches generated via macro to add L1i pressure and produce
// data-dependent indirect-branch mispredictions on a random payload.
#define FNB_OP(n, S)                                          \
  case (n):                                                   \
    v = (v + 0x9e3779b97f4a7c15ull * ((n) + 1) + (S)) ^       \
        (v >> (((n) & 7) ? 7 : 5));                           \
    break;
#define FNB_OP8(b, S)                                                   \
  FNB_OP((b) + 0, S) FNB_OP((b) + 1, S) FNB_OP((b) + 2, S)              \
  FNB_OP((b) + 3, S) FNB_OP((b) + 4, S) FNB_OP((b) + 5, S)              \
  FNB_OP((b) + 6, S) FNB_OP((b) + 7, S)
#define FNB_OP64(S)                                                     \
  FNB_OP8(0, S) FNB_OP8(8, S) FNB_OP8(16, S) FNB_OP8(24, S)             \
  FNB_OP8(32, S) FNB_OP8(40, S) FNB_OP8(48, S) FNB_OP8(56, S)

__attribute__((noinline)) uint64_t switchA(uint64_t v, int c) {
  switch (c & 63) {
    FNB_OP64(0x0001)
    default:
      v ^= 0xABCD;
      break;
  }
  return v + 0x11;
}

__attribute__((noinline)) uint64_t switchB(uint64_t v, int c) {
  switch (c & 63) {
    FNB_OP64(0x1357)
    default:
      v ^= 0xBEEF;
      break;
  }
  return (v << 1) + 0x22;
}

__attribute__((noinline)) uint64_t switchC(uint64_t v, int c) {
  switch (c & 63) {
    FNB_OP64(0x2468)
    default:
      v ^= 0xF00D;
      break;
  }
  return (v ^ (v >> 17)) + 0x33;
}

#undef FNB_OP64
#undef FNB_OP8
#undef FNB_OP

} // namespace

// Directly invoking a function, but with a memory- and branch-bound hot loop:
// dependent pointer-chasing loads through large permuted chains plus
// data-dependent indirect branches gated on the loaded values.
BENCHMARK(fn_invoke_pchase, iters) {
  static ChainNode* hot = makeChain(kHotLen);
  static ChainNode* cold = makeChain(kColdLen);
  ChainNode* hotPos = hot;
  ChainNode* coldPos = cold;
  uint64_t acc = 0;
  for (size_t n = 0; n < iters; ++n) {
    doNothing();
    hotPos = hotPos->next; // dependent L1d-missing load
    uint64_t payload = (uint64_t)(uintptr_t)hotPos;
    switch (n % 3) {
      case 0:
        acc = switchA(acc, (int)(payload & 0xFF));
        break;
      case 1:
        acc = switchB(acc, (int)(payload & 0xFF));
        break;
      default:
        acc = switchC(acc, (int)(payload & 0xFF));
        break;
    }
    // Branchless cold-chain advance every 8th step → LLC misses without
    // polluting the branch predictor.
    uint64_t coldMask = -(uint64_t)((n & 7) == 0);
    coldPos = (ChainNode*)(((uintptr_t)coldPos->next & coldMask) |
                           ((uintptr_t)coldPos & ~coldMask));
    acc += (uint64_t)(uintptr_t)coldPos >> 6;
  }
  folly::doNotOptimizeAway(acc);
  folly::doNotOptimizeAway(hotPos);
  folly::doNotOptimizeAway(coldPos);
}

// Invoking a function through a function pointer
BENCHMARK(fn_ptr_invoke, iters) {
  BM_fn_ptr_invoke_impl(iters, doNothing);
}

// Invoking a function through a std::function object
BENCHMARK(std_function_invoke, iters) {
  BM_std_function_invoke_impl(iters, doNothing);
}

// Invoking a function through a folly::Function object
BENCHMARK(Function_invoke, iters) {
  BM_Function_invoke_impl(iters, doNothing);
}

// Invoking a member function through a member function pointer
BENCHMARK(mem_fn_invoke, iters) {
  TestClass tc;
  BM_mem_fn_invoke_impl(iters, &tc, &TestClass::doNothing);
}

// Invoke a function pointer through an inlined wrapper function
BENCHMARK(fn_ptr_invoke_through_inline, iters) {
  BM_fn_ptr_invoke_inlined_impl(iters, doNothing);
}

// Invoke a lambda that calls doNothing() through an inlined wrapper function
BENCHMARK(lambda_invoke_fn, iters) {
  BM_invoke_fn_template_impl(iters, [] { doNothing(); });
}

// Invoke a lambda that does nothing
BENCHMARK(lambda_noop, iters) {
  BM_invoke_fn_template_impl(iters, [] {});
}

// Invoke a lambda that modifies a local variable
BENCHMARK(lambda_local_var, iters) {
  uint32_t count1 = 0;
  uint32_t count2 = 0;
  BM_invoke_fn_template_impl(iters, [&] {
    // Do something slightly more complicated than just incrementing a
    // variable.  Otherwise gcc is smart enough to optimize the loop away.
    if (count1 & 0x1) {
      ++count2;
    }
    ++count1;
  });

  // Use the values we computed, so gcc won't optimize the loop away
  CHECK_EQ(iters, count1);
  CHECK_EQ(iters / 2, count2);
}

// Invoke a function pointer through the same wrapper used for lambdas
BENCHMARK(fn_ptr_invoke_through_template, iters) {
  BM_invoke_fn_template_impl(iters, doNothing);
}

// Invoking a virtual method
BENCHMARK(virtual_fn_invoke, iters) {
  VirtualClass vc;
  BM_virtual_fn_invoke_impl(iters, &vc);
}

// Creating a function pointer and invoking it
BENCHMARK(fn_ptr_create_invoke, iters) {
  for (size_t n = 0; n < iters; ++n) {
    void (*fn)() = doNothing;
    fn();
  }
}

// Creating a std::function object from a function pointer, and invoking it
BENCHMARK(std_function_create_invoke, iters) {
  for (size_t n = 0; n < iters; ++n) {
    std::function<void()> fn = doNothing;
    fn();
  }
}

// Creating a folly::Function object from a function pointer, and
// invoking it
BENCHMARK(Function_create_invoke, iters) {
  for (size_t n = 0; n < iters; ++n) {
    folly::Function<void()> fn = doNothing;
    fn();
  }
}

// Creating a pointer-to-member and invoking it
BENCHMARK(mem_fn_create_invoke, iters) {
  TestClass tc;
  for (size_t n = 0; n < iters; ++n) {
    void (TestClass::*memfn)() = &TestClass::doNothing;
    (tc.*memfn)();
  }
}

// Using std::bind to create a std::function from a member function,
// and invoking it
BENCHMARK(std_bind_create_invoke, iters) {
  TestClass tc;
  for (size_t n = 0; n < iters; ++n) {
    std::function<void()> fn = std::bind(&TestClass::doNothing, &tc);
    fn();
  }
}

// Using std::bind directly to invoke a member function
BENCHMARK(std_bind_direct_invoke, iters) {
  TestClass tc;
  for (size_t n = 0; n < iters; ++n) {
    auto fn = std::bind(&TestClass::doNothing, &tc);
    fn();
  }
}

// Using ScopeGuard to invoke a std::function
BENCHMARK(scope_guard_std_function, iters) {
  std::function<void()> fn(doNothing);
  for (size_t n = 0; n < iters; ++n) {
    auto g = makeGuard(fn);
    (void)g;
  }
}

// Using ScopeGuard to invoke a std::function,
// but create the ScopeGuard with an rvalue to a std::function
BENCHMARK(scope_guard_std_function_rvalue, iters) {
  for (size_t n = 0; n < iters; ++n) {
    auto g = makeGuard(std::function<void()>(doNothing));
    (void)g;
  }
}

// Using ScopeGuard to invoke a folly::Function,
// but create the ScopeGuard with an rvalue to a folly::Function
BENCHMARK(scope_guard_Function_rvalue, iters) {
  for (size_t n = 0; n < iters; ++n) {
    auto g = makeGuard(folly::Function<void()>(doNothing));
    (void)g;
  }
}

// Using ScopeGuard to invoke a function pointer
BENCHMARK(scope_guard_fn_ptr, iters) {
  for (size_t n = 0; n < iters; ++n) {
    auto g = makeGuard(doNothing);
    (void)g;
  }
}

// Using ScopeGuard to invoke a lambda that does nothing
BENCHMARK(scope_guard_lambda_noop, iters) {
  for (size_t n = 0; n < iters; ++n) {
    auto g = makeGuard([] {});
    (void)g;
  }
}

// Using ScopeGuard to invoke a lambda that invokes a function
BENCHMARK(scope_guard_lambda_function, iters) {
  for (size_t n = 0; n < iters; ++n) {
    auto g = makeGuard([] { doNothing(); });
    (void)g;
  }
}

// Using ScopeGuard to invoke a lambda that modifies a local variable
BENCHMARK(scope_guard_lambda_local_var, iters) {
  uint32_t count = 0;
  for (size_t n = 0; n < iters; ++n) {
    auto g = makeGuard([&] {
      // Increment count if n is odd.  Without this conditional check
      // (i.e., if we just increment count each time through the loop),
      // gcc is smart enough to optimize the entire loop away, and just set
      // count = iters.
      if (n & 0x1) {
        ++count;
      }
    });
    (void)g;
  }

  // Check that the value of count is what we expect.
  // This check is necessary: if we don't use count, gcc detects that count is
  // unused and optimizes the entire loop away.
  CHECK_EQ(iters / 2, count);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(throw_exception, iters) {
  for (size_t n = 0; n < iters; ++n) {
    try {
      folly::throw_exception<Exception>("this is a test");
    } catch (const std::exception&) {
    }
  }
}

BENCHMARK(catch_no_exception, iters) {
  for (size_t n = 0; n < iters; ++n) {
    try {
      doNothing();
    } catch (const std::exception&) {
    }
  }
}

BENCHMARK(return_exc_ptr, iters) {
  for (size_t n = 0; n < iters; ++n) {
    returnExceptionPtr();
  }
}

BENCHMARK(exc_ptr_param_return, iters) {
  for (size_t n = 0; n < iters; ++n) {
    std::exception_ptr ex;
    exceptionPtrReturnParam(&ex);
  }
}

BENCHMARK(exc_ptr_param_return_null, iters) {
  for (size_t n = 0; n < iters; ++n) {
    exceptionPtrReturnParam(nullptr);
  }
}

BENCHMARK(return_string, iters) {
  for (size_t n = 0; n < iters; ++n) {
    returnString();
  }
}

BENCHMARK(return_string_noexcept, iters) {
  for (size_t n = 0; n < iters; ++n) {
    returnStringNoExcept();
  }
}

BENCHMARK(return_code, iters) {
  for (size_t n = 0; n < iters; ++n) {
    returnCode(false);
  }
}

BENCHMARK(return_code_noexcept, iters) {
  for (size_t n = 0; n < iters; ++n) {
    returnCodeNoExcept(false);
  }
}

BENCHMARK_DRAW_LINE();

BENCHMARK(std_function_create_move_invoke, iters) {
  LargeClass a;
  for (size_t i = 0; i < iters; ++i) {
    std::function<void()> f(a);
    invoke(std::move(f));
  }
}

BENCHMARK(Function_create_move_invoke, iters) {
  LargeClass a;
  for (size_t i = 0; i < iters; ++i) {
    folly::Function<void()> f(a);
    invoke(std::move(f));
  }
}

BENCHMARK(std_function_create_move_invoke_small, iters) {
  for (size_t i = 0; i < iters; ++i) {
    std::function<void()> f(doNothing);
    invoke(std::move(f));
  }
}

BENCHMARK(Function_create_move_invoke_small, iters) {
  for (size_t i = 0; i < iters; ++i) {
    folly::Function<void()> f(doNothing);
    invoke(std::move(f));
  }
}

BENCHMARK(std_function_create_move_invoke_ref, iters) {
  LargeClass a;
  for (size_t i = 0; i < iters; ++i) {
    std::function<void()> f(std::ref(a));
    invoke(std::move(f));
  }
}

BENCHMARK(Function_create_move_invoke_ref, iters) {
  LargeClass a;
  for (size_t i = 0; i < iters; ++i) {
    folly::Function<void()> f(std::ref(a));
    invoke(std::move(f));
  }
}

BENCHMARK_DRAW_LINE();

BENCHMARK(function_ptr_move, iters) {
  auto f = &doNothing;
  for (size_t i = 0; i < iters; ++i) {
    auto f2 = std::move(f);
    folly::doNotOptimizeAway(f2);
    f = std::move(f2);
  }
}

BENCHMARK(std_function_move_small, iters) {
  std::shared_ptr<int> a(new int);
  std::function<void()> f([a]() { doNothing(); });
  for (size_t i = 0; i < iters; ++i) {
    auto f2 = std::move(f);
    folly::doNotOptimizeAway(f2);
    f = std::move(f2);
  }
}

BENCHMARK(Function_move_small, iters) {
  std::shared_ptr<int> a(new int);
  folly::Function<void()> f([a]() { doNothing(); });
  for (size_t i = 0; i < iters; ++i) {
    auto f2 = std::move(f);
    folly::doNotOptimizeAway(f2);
    f = std::move(f2);
  }
}

BENCHMARK(std_function_move_small_trivial, iters) {
  std::function<void()> f(doNothing);
  for (size_t i = 0; i < iters; ++i) {
    auto f2 = std::move(f);
    folly::doNotOptimizeAway(f2);
    f = std::move(f2);
  }
}

BENCHMARK(Function_move_small_trivial, iters) {
  folly::Function<void()> f(doNothing);
  for (size_t i = 0; i < iters; ++i) {
    auto f2 = std::move(f);
    folly::doNotOptimizeAway(f2);
    f = std::move(f2);
  }
}

BENCHMARK(std_function_move_large, iters) {
  LargeClass a;
  std::function<void()> f(a);
  for (size_t i = 0; i < iters; ++i) {
    auto f2 = std::move(f);
    folly::doNotOptimizeAway(f2);
    f = std::move(f2);
  }
}

BENCHMARK(Function_move_large, iters) {
  LargeClass a;
  folly::Function<void()> f(a);
  for (size_t i = 0; i < iters; ++i) {
    auto f2 = std::move(f);
    folly::doNotOptimizeAway(f2);
    f = std::move(f2);
  }
}

// Directly invoking a function inside a tight hot loop whose iteration count
// is itself data-dependent on an unpredictable pointer-chase load, so the
// surrounding loop-branch is mispredicted and the call cannot be hoisted.
BENCHMARK(fn_invoke_dependent_branch, iters) {
  static ChainNode* hot = makeChain(kHotLen);
  ChainNode* pos = hot;
  uint64_t acc = 0;
  for (size_t n = 0; n < iters; ++n) {
    pos = pos->next; // dependent, prefetcher-defeating load
    uint64_t payload = (uint64_t)(uintptr_t)pos;
    // Data-dependent, unpredictable inner-loop trip count in [1, 4].
    unsigned reps = (unsigned)((payload >> 6) & 0x3) + 1;
    for (unsigned r = 0; r < reps; ++r) {
      doNothing();
      acc = switchA(acc, (int)((payload >> (r * 8)) & 0xFF));
    }
  }
  folly::doNotOptimizeAway(acc);
  folly::doNotOptimizeAway(pos);
}

// main()

int main(int argc, char** argv) {
  folly::gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::runBenchmarks();
}
