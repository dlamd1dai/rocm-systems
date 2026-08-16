/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for the GIN-SDMA AllReduce reduction ops in
// projects/rccl-tests/src/gin_sdma_reduce.h (namespace gin_sdma_reduce): the
// preOp / combine / postOp reduction the ReduceScatter half of the AllReduce
// design applies on device.
//
// The header is device-only (`__device__ __forceinline__`, __half intrinsics),
// so to exercise the SAME control flow on the host with no GPU we compile it as
// plain host C++ behind a minimal shim: the attributes drop to no-ops and __half
// is a thin float carrier. That makes the generic (int/float/double) reduction
// templates and ALL of the op-dispatch / avg pre-post / integral-vs-floating
// SFINAE branches directly testable and coverable.
//
// SCOPE / what the shim does NOT test: the shim intentionally models __half as a
// float, and bf16/fp8 are compiled out (HAVE_BF16 / HAVE_FP8 = 0). The exact
// low-precision *rounding* (narrow-on-every-step) that must bit-match
// verifiable.cu is validated by the on-hardware datacheck sweeps
// (all_reduce_perf -c 1 across {half,bfloat16,float} x {sum,prod,avg,max,min}),
// not here. This file covers the reduction *logic/structure* on the host.

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

// ---- host shim: make the device-only reduce header compile as host C++ -------
#define __device__
#define __forceinline__ inline
#define ENABLE_DEVICE_API 1
#ifndef NCCL_VERSION
#define NCCL_VERSION(a, b, c) ((a) * 10000 + (b) * 100 + (c))
#endif
#ifndef NCCL_VERSION_CODE
#define NCCL_VERSION_CODE NCCL_VERSION(2, 30, 4)
#endif
#define HAVE_BF16 0
#define HAVE_FP8 0
enum ncclRedOp_t { ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3, ncclAvg = 4 };
// __half modelled as a float carrier (structure-only; real half rounding is
// device-/hardware-tested via verifiable.cu). Must be a non-integral class type
// so the SFINAE floating paths select correctly.
struct __half {
  float v;
  __half() : v(0.f) {}
  __half(float f) : v(f) {}
};
static inline float __half2float(__half h) { return h.v; }
static inline __half __float2half(float f) { return __half(f); }
static inline bool operator==(__half a, __half b) { return a.v == b.v; }

#include "gin_sdma_reduce.h"

using namespace gin_sdma_reduce;

namespace {

// ------------------------------- rsum / rprod ------------------------------
TEST(ReduceOps, SumGeneric) {
  EXPECT_EQ(rsum<int>(2, 3), 5);
  EXPECT_EQ(rsum<int64_t>(-4, 10), 6);
  EXPECT_EQ(rsum<float>(1.5f, 2.25f), 3.75f);
  EXPECT_EQ(rsum<double>(1.0, 2.0), 3.0);
}

TEST(ReduceOps, ProdGeneric) {
  EXPECT_EQ(rprod<int>(3, 4), 12);
  EXPECT_EQ(rprod<float>(2.0f, 2.5f), 5.0f);
  EXPECT_EQ(rprod<double>(3.0, 0.5), 1.5);
}

TEST(ReduceOps, MinMaxGeneric) {
  EXPECT_EQ(rmin<int>(2, 7), 2);
  EXPECT_EQ(rmax<int>(2, 7), 7);
  EXPECT_EQ(rmin<int>(7, 2), 2);   // exercise the other ternary arm
  EXPECT_EQ(rmax<int>(7, 2), 7);
  EXPECT_EQ(rmin<double>(-1.0, 1.0), -1.0);
  EXPECT_EQ(rmax<float>(-1.0f, 1.0f), 1.0f);
}

// __half overloads (as float via the shim): exercises the specialized-overload
// lines the low-precision path resolves to on device.
TEST(ReduceOps, HalfOverloadsResolve) {
  EXPECT_EQ(rsum(__half(1.5f), __half(2.0f)).v, 3.5f);
  EXPECT_EQ(rprod(__half(1.5f), __half(2.0f)).v, 3.0f);
  EXPECT_EQ(rmin(__half(1.5f), __half(2.0f)).v, 1.5f);
  EXPECT_EQ(rmax(__half(1.5f), __half(2.0f)).v, 2.0f);
  EXPECT_EQ(rmin(__half(2.0f), __half(1.5f)).v, 1.5f);
  EXPECT_EQ(rmax(__half(2.0f), __half(1.5f)).v, 2.0f);
}

// ------------------------------- avgFactor --------------------------------
TEST(ReduceOps, AvgFactor) {
  // Generic (float): 1/N in float.
  EXPECT_EQ(avgFactor<float>(4), 0.25f);
  // double specialization.
  EXPECT_EQ(avgFactor<double>(8), 0.125);
  // __half via generic path (float math).
  EXPECT_EQ(avgFactor<__half>(2).v, 0.5f);
}

// -------------------------------- combine ---------------------------------
TEST(ReduceOps, CombineDispatchAllArms) {
  // prod / max / min / default(sum) -- covers every switch arm.
  EXPECT_EQ(combine<int>(ncclProd, 3, 4), 12);
  EXPECT_EQ(combine<int>(ncclMax, 3, 4), 4);
  EXPECT_EQ(combine<int>(ncclMin, 3, 4), 3);
  EXPECT_EQ(combine<int>(ncclSum, 3, 4), 7);
  // ncclAvg accumulates by + (same default arm as sum).
  EXPECT_EQ(combine<int>(ncclAvg, 3, 4), 7);
  EXPECT_EQ(combine<float>(ncclProd, 2.0f, 2.5f), 5.0f);
}

// --------------------------------- preOp ----------------------------------
TEST(ReduceOps, PreOpFloatingAvgPremultiplies) {
  // Floating avg: premultiply by 1/N BEFORE summing.
  EXPECT_EQ(preOp<float>(ncclAvg, 8.0f, 4), 2.0f);   // 8 * 1/4
  EXPECT_EQ(preOp<double>(ncclAvg, 8.0, 2), 4.0);
  EXPECT_EQ(preOp<__half>(ncclAvg, 8.0f, 4).v, 2.0f);
}

TEST(ReduceOps, PreOpIdentityCases) {
  // Integral avg is identity in preOp (division happens in postOp).
  EXPECT_EQ(preOp<int>(ncclAvg, 8, 4), 8);
  // Non-avg op is identity for every type.
  EXPECT_EQ(preOp<int>(ncclSum, 5, 4), 5);
  EXPECT_EQ(preOp<float>(ncclProd, 5.0f, 4), 5.0f);
}

// --------------------------- postDivAvg / postOp ---------------------------
TEST(ReduceOps, PostOpIntegralAvgDivides) {
  // Integral avg finishes with an integer divide of the summed value.
  EXPECT_EQ(postOp<int>(ncclAvg, 8, 4), 2);
  EXPECT_EQ(postOp<int64_t>(ncclAvg, 9, 4), 2);  // integer truncation
}

TEST(ReduceOps, PostOpFloatingAvgIsIdentity) {
  // Floating avg already divided in preOp, so postOp is identity.
  EXPECT_EQ(postOp<float>(ncclAvg, 2.0f, 4), 2.0f);
  EXPECT_EQ(postOp<double>(ncclAvg, 4.0, 2), 4.0);
  EXPECT_EQ(postOp<__half>(ncclAvg, 2.0f, 4).v, 2.0f);
}

TEST(ReduceOps, PostOpNonAvgIsIdentity) {
  EXPECT_EQ(postOp<int>(ncclSum, 8, 4), 8);
  EXPECT_EQ(postOp<float>(ncclMax, 8.0f, 4), 8.0f);
}

// postDivAvg directly (both SFINAE overloads).
TEST(ReduceOps, PostDivAvgDirect) {
  EXPECT_EQ(postDivAvg<int>(10, 4), 2);        // integral overload
  EXPECT_EQ(postDivAvg<float>(10.0f, 4), 10.0f);  // non-integral overload: identity
}

// ---------------- end-to-end fold matching the verifier order ----------------
// Fold N ascending contributions with preOp -> combine chain -> postOp, the same
// order ncclVerifiablePrepareExpected uses. Structural check on the host types.
TEST(ReduceOps, EndToEndSumAndAvg) {
  const int N = 4;
  // ncclSum over [1,2,3,4] = 10.
  {
    int acc = preOp<int>(ncclSum, 1, N);
    for (int r = 2; r <= 4; ++r) acc = combine<int>(ncclSum, acc, preOp<int>(ncclSum, r, N));
    EXPECT_EQ(postOp<int>(ncclSum, acc, N), 10);
  }
  // Floating ncclAvg over [1,2,3,4] = 2.5 (premul each by 1/4, sum, postOp identity).
  {
    float acc = preOp<float>(ncclAvg, 1.0f, N);
    for (int r = 2; r <= 4; ++r) acc = combine<float>(ncclAvg, acc, preOp<float>(ncclAvg, (float)r, N));
    EXPECT_EQ(postOp<float>(ncclAvg, acc, N), 2.5f);
  }
  // Integral ncclAvg over [1,2,3,4] = sum 10 then /4 = 2.
  {
    int acc = preOp<int>(ncclAvg, 1, N);
    for (int r = 2; r <= 4; ++r) acc = combine<int>(ncclAvg, acc, preOp<int>(ncclAvg, r, N));
    EXPECT_EQ(postOp<int>(ncclAvg, acc, N), 2);
  }
}

// Per-instantiation sweep: drive EVERY op arm through combine/preOp/postOp for
// EVERY instantiated element type so each template instantiation exercises all
// of its lines/branches (rmin/rmax ternary both arms, every switch case, the
// avg premul vs identity split, integral vs floating postOp). Closes the
// per-instantiation gcov gaps that a single-type test leaves.
template <class T>
void sweepArithmeticType() {
  // rsum/rprod/rmin/rmax both ternary arms.
  EXPECT_EQ(rsum<T>((T)2, (T)3), (T)5);
  EXPECT_EQ(rprod<T>((T)2, (T)3), (T)6);
  EXPECT_EQ(rmin<T>((T)2, (T)3), (T)2);
  EXPECT_EQ(rmin<T>((T)3, (T)2), (T)2);
  EXPECT_EQ(rmax<T>((T)2, (T)3), (T)3);
  EXPECT_EQ(rmax<T>((T)3, (T)2), (T)3);
  // combine: every switch arm.
  EXPECT_EQ(combine<T>(ncclSum, (T)3, (T)4), (T)7);
  EXPECT_EQ(combine<T>(ncclProd, (T)3, (T)4), (T)12);
  EXPECT_EQ(combine<T>(ncclMax, (T)3, (T)4), (T)4);
  EXPECT_EQ(combine<T>(ncclMin, (T)3, (T)4), (T)3);
  EXPECT_EQ(combine<T>(ncclAvg, (T)3, (T)4), (T)7);  // + accumulate
  // preOp: non-avg identity (return x) for every type.
  EXPECT_EQ(preOp<T>(ncclSum, (T)5, 4), (T)5);
  EXPECT_EQ(preOp<T>(ncclProd, (T)5, 4), (T)5);
  // postOp: non-avg identity for every type.
  EXPECT_EQ(postOp<T>(ncclSum, (T)8, 4), (T)8);
  EXPECT_EQ(postOp<T>(ncclMax, (T)8, 4), (T)8);
  // avg pre/post: integral divides in postOp, floating premultiplies in preOp.
  if (std::is_integral<T>::value) {
    EXPECT_EQ(preOp<T>(ncclAvg, (T)8, 4), (T)8);   // identity
    EXPECT_EQ(postOp<T>(ncclAvg, (T)8, 4), (T)2);  // 8/4
  } else {
    EXPECT_EQ(preOp<T>(ncclAvg, (T)8, 4), (T)2);   // 8 * 1/4
    EXPECT_EQ(postOp<T>(ncclAvg, (T)8, 4), (T)8);  // identity
  }
}

TEST(ReduceOps, PerInstantiationSweep) {
  sweepArithmeticType<int>();
  sweepArithmeticType<int64_t>();
  sweepArithmeticType<float>();
  sweepArithmeticType<double>();
}

// __half sweep (float-carrier shim): drive the specialized __half overloads and
// the class-type (non-integral) preOp/postOp paths through every op arm.
TEST(ReduceOps, HalfPerInstantiationSweep) {
  EXPECT_EQ(combine<__half>(ncclSum, __half(3.f), __half(4.f)).v, 7.f);
  EXPECT_EQ(combine<__half>(ncclProd, __half(3.f), __half(4.f)).v, 12.f);
  EXPECT_EQ(combine<__half>(ncclMax, __half(3.f), __half(4.f)).v, 4.f);
  EXPECT_EQ(combine<__half>(ncclMin, __half(3.f), __half(4.f)).v, 3.f);
  EXPECT_EQ(combine<__half>(ncclAvg, __half(3.f), __half(4.f)).v, 7.f);
  EXPECT_EQ(preOp<__half>(ncclSum, __half(5.f), 4).v, 5.f);   // non-avg identity
  EXPECT_EQ(preOp<__half>(ncclAvg, __half(8.f), 4).v, 2.f);   // floating premul
  EXPECT_EQ(postOp<__half>(ncclSum, __half(8.f), 4).v, 8.f);  // non-avg identity
  EXPECT_EQ(postOp<__half>(ncclAvg, __half(8.f), 4).v, 8.f);  // floating identity
}

}  // namespace
