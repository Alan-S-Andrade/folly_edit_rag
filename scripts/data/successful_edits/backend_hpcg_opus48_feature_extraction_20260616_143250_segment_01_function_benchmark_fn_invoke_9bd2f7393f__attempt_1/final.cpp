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

#include <glog/logging.h>

#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

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

// ---------------------------------------------------------------------------
// Local fn_invoke variant: same doNothing() invocation, but the timed hot path
// is augmented with a pointer-chase through a large permuted chain (exceeds
// L1d, fits L2) plus data-dependent indirect dispatch. This deliberately moves
// the workload from compute-bound toward memory/branch-bound so that L1d-load
// misses and branch misses rise while IPC falls. The pointer-chase dependency
// chain defeats hardware prefetchers; the data-dependent switch defeats the
// branch predictor.
// ---------------------------------------------------------------------------
namespace {

struct ChainNode {
  ChainNode* next;
  char pad[48];
};

// 16384 * 64B = 1 MB: exceeds L1d (48KB), fits L2 (2MB).
constexpr size_t kHotLen = 16384;

ChainNode* makeChain(size_t len, uint64_t seed) {
  auto* nodes = new ChainNode[len];
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), size_t{0});
  std::mt19937_64 rng(seed);
  for (size_t i = len - 1; i > 0; --i) {
    std::swap(perm[i], perm[rng() % (i + 1)]);
  }
  for (size_t i = 0; i < len; ++i) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % len]];
  }
  return nodes;
}

__attribute__((noinline)) uint64_t chaseSwitchA(uint64_t v, int c) {
  switch (c & 0x1f) {
    case 0: v += 0x1234; v ^= 0x5678; break;
    case 1: v = (v << 3) | (v >> 61); v += 0x9e37; break;
    case 2: v ^= v >> 7; v *= 0x2545; break;
    case 3: v += c; v ^= 0xa5a5; break;
    case 4: v = (v >> 2) ^ (v << 5); break;
    case 5: v += 0x0f0f; v ^= (v << 11); break;
    case 6: v *= 3; v += 0x55; break;
    case 7: v ^= 0xdead; v += (v >> 4); break;
    case 8: v += 0x77; v = (v << 1) ^ v; break;
    case 9: v ^= 0xbeef; v += c * 7; break;
    case 10: v = (v << 7) | (v >> 57); break;
    case 11: v += 0x13; v ^= (v >> 9); break;
    case 12: v *= 0x100000001b3ull; break;
    case 13: v ^= 0xc0ffee; v += 1; break;
    case 14: v += (v << 6); v ^= 0x33; break;
    case 15: v = ~v; v += c; break;
    default: v += (uint64_t)c * 0x9e3779b97f4a7c15ull; break;
  }
  return v;
}

__attribute__((noinline)) uint64_t chaseSwitchB(uint64_t v, int c) {
  switch (c & 0x1f) {
    case 0: v ^= 0x1111; v += (v >> 3); break;
    case 1: v *= 0x85eb; v ^= c; break;
    case 2: v += 0xface; v = (v << 4) ^ v; break;
    case 3: v ^= (v >> 11); v += 0x77; break;
    case 4: v = (v << 9) | (v >> 55); break;
    case 5: v += c * 13; v ^= 0xabcd; break;
    case 6: v *= 5; v += 0x3; break;
    case 7: v ^= 0xfeed; v += (v << 2); break;
    case 8: v += 0x99; v ^= (v >> 6); break;
    case 9: v = (v >> 5) ^ (v << 8); break;
    case 10: v += 0x2222; v *= 3; break;
    case 11: v ^= c; v += 0xb00b; break;
    case 12: v = (v << 13) | (v >> 51); break;
    case 13: v += (v >> 2); v ^= 0x44; break;
    case 14: v *= 0xc2b2ae35ull; break;
    case 15: v ^= 0x5a5a; v += c + 1; break;
    default: v += (uint64_t)c * 0xff51afd7ed558ccdull; break;
  }
  return v;
}

__attribute__((noinline)) uint64_t chaseSwitchC(uint64_t v, int c) {
  switch (c & 0x1f) {
    case 0: v += 0xaaaa; v ^= (v << 7); break;
    case 1: v ^= c; v *= 0x27d4; break;
    case 2: v = (v << 11) | (v >> 53); break;
    case 3: v += 0xd00d; v ^= (v >> 8); break;
    case 4: v *= 7; v += 0x9; break;
    case 5: v ^= 0x1357; v += c * 3; break;
    case 6: v += (v << 5); v ^= 0x88; break;
    case 7: v = ~v; v += 0x11; break;
    case 8: v ^= 0x2468; v += (v >> 7); break;
    case 9: v = (v >> 3) ^ (v << 6); break;
    case 10: v += c * 17; v ^= 0xcccc; break;
    case 11: v *= 0x165667b19e3779f9ull; break;
    case 12: v ^= 0x7777; v += 1; break;
    case 13: v += (v >> 4); v ^= c; break;
    case 14: v = (v << 15) | (v >> 49); break;
    case 15: v ^= 0x9696; v += c + 5; break;
    default: v += (uint64_t)c * 0xd6e8feb86659fd93ull; break;
  }
  return v;
}

} // namespace

// fn_invoke with an added pointer-chase + data-dependent dispatch hot path.
BENCHMARK(fn_invoke_chase, iters) {
  // Two independent walkers over distinct 1 MB permuted chains multiply L1d
  // pressure while keeping the working set inside L2.
  static ChainNode* hot1 = makeChain(kHotLen, 42);
  static ChainNode* hot2 = makeChain(kHotLen, 1337);
  ChainNode* p1 = hot1;
  ChainNode* p2 = hot2;
  uint64_t acc = 0;
  for (size_t n = 0; n < iters; ++n) {
    // Dependent pointer-chase loads: each address depends on the prior load.
    p1 = p1->next;
    p2 = p2->next;
    uint64_t payload =
        (uint64_t)(uintptr_t)p1 ^ ((uint64_t)(uintptr_t)p2 >> 6);
    // Data-dependent indirect dispatch: branch target gated on loaded value.
    switch (payload % 3) {
      case 0:
        acc = chaseSwitchA(acc, (int)(payload & 0xFF));
        break;
      case 1:
        acc = chaseSwitchB(acc, (int)(payload & 0xFF));
        break;
      default:
        acc = chaseSwitchC(acc, (int)(payload & 0xFF));
        break;
    }
    doNothing();
  }
  folly::doNotOptimizeAway(acc);
  folly::doNotOptimizeAway(p1);
  folly::doNotOptimizeAway(p2);
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

// main()

int main(int argc, char** argv) {
  folly::gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::runBenchmarks();
}

/*
============================================================================
folly/test/function_benchmark/main.cpp          relative  time/iter  iters/s
============================================================================
fn_invoke                                                    1.22ns  822.88M
fn_ptr_invoke                                                1.22ns  822.99M
std_function_invoke                                          2.73ns  365.78M
Function_invoke                                              2.73ns  365.79M
mem_fn_invoke                                                1.37ns  731.38M
fn_ptr_invoke_through_inline                                 1.22ns  822.95M
lambda_invoke_fn                                             1.22ns  822.88M
lambda_noop                                                  0.00fs  Infinity
lambda_local_var                                           182.49ps    5.48G
fn_ptr_invoke_through_template                               1.22ns  822.95M
virtual_fn_invoke                                            1.22ns  822.98M
fn_ptr_create_invoke                                         1.22ns  822.94M
std_function_create_invoke                                   3.88ns  257.83M
Function_create_invoke                                       2.73ns  365.73M
mem_fn_create_invoke                                         1.22ns  822.98M
std_bind_create_invoke                                      18.91ns   52.89M
std_bind_direct_invoke                                       1.22ns  822.98M
scope_guard_std_function                                     7.24ns  138.14M
scope_guard_std_function_rvalue                              6.44ns  155.23M
scope_guard_Function_rvalue                                  5.53ns  180.87M
scope_guard_fn_ptr                                         928.25ps    1.08G
scope_guard_lambda_noop                                      0.00fs  Infinity
scope_guard_lambda_function                                  1.22ns  822.97M
scope_guard_lambda_local_var                               101.27ps    9.87G
----------------------------------------------------------------------------
throw_exception                                              1.90us  524.98K
catch_no_exception                                           1.22ns  822.98M
return_exc_ptr                                               1.39us  719.84K
exc_ptr_param_return                                         1.41us  711.08K
exc_ptr_param_return_null                                    1.82ns  548.61M
return_string                                                2.43ns  411.48M
return_string_noexcept                                       2.43ns  411.48M
return_code                                                  1.22ns  822.98M
return_code_noexcept                                       943.51ps    1.06G
----------------------------------------------------------------------------
std_function_create_move_invoke                             48.74ns   20.52M
Function_create_move_invoke                                 50.21ns   19.92M
std_function_create_move_invoke_small                        6.78ns  147.58M
Function_create_move_invoke_small                            7.01ns  142.67M
std_function_create_move_invoke_ref                          6.67ns  150.03M
Function_create_move_invoke_ref                              6.88ns  145.35M
----------------------------------------------------------------------------
function_ptr_move                                            1.21ns  823.05M
std_function_move_small                                      5.77ns  173.20M
Function_move_small                                          7.60ns  131.58M
std_function_move_small_trivial                              5.77ns  173.27M
Function_move_small_trivial                                  5.47ns  182.86M
std_function_move_large                                      5.77ns  173.22M
Function_move_large                                          6.38ns  156.63M
============================================================================
 */
