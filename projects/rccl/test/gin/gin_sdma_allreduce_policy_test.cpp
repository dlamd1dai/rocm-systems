/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for the pure GIN Anvil-SDMA AllReduce policy/decision logic in
// projects/rccl-tests/src/gin_sdma_collective_policy.h (namespace gin_sdma).
//
// The GIN-SDMA AllReduce design (deviceImpl 5 single-launch / 6 two-launch) is
// ReduceScatter -> device-wide barrier -> in-place AllGather. Its host launcher
// (all_reduce.cu: AllReduceGetDevCommRequirements / AllReduceRunColl /
// AllReduceDeviceTime) drives ALL of its size-/rank-dependent decisions through
// these pure functions:
//   * threshold resolution (parseSize + resolveThreshold, via
//     common.h::testResolveSdmaThreshold) for arThr / arOneShot,
//   * AllGather tier selection (allReduceKernelTier),
//   * devComm resource requirements (allReduceDevReqs -> allReduceSignalCount),
//   * two-shot slice stride (allReduceSliceStride) and scratch sizing
//     (allReduceScratchBytes).
//
// The header is compiled here as plain host C++ (GIN_SDMA_HOST_ONLY drops the
// __host__ __device__ attributes) so these tests need neither a built librccl
// nor a GPU: the SAME functions the __global__ kernels and the host launcher
// call are exercised here, so exercising them covers the real decision code.
//
// NOTE ON SCOPE: the __global__ AllReduce kernels (all_reduce.cu) and the
// __device__ reduce intrinsics (gin_sdma_reduce.h) are GPU device code and are
// validated by the on-hardware datacheck sweeps (all_reduce_perf -c 1), not by
// these host unit tests -- host coverage tooling (llvm-cov / gcov) instruments
// host code only. This file targets the host-side policy layer, which is where
// every branch of the AllReduce launch decision lives.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>

#define GIN_SDMA_HOST_ONLY 1
#include "gin_sdma_collective_policy.h"

using namespace gin_sdma;

namespace {

constexpr size_t kKiB = 1024ull;
constexpr size_t kMiB = 1024ull * kKiB;
constexpr size_t kGiB = 1024ull * kMiB;

// Round-up-to-16 reference (matches allReduceSliceStride's alignment).
size_t roundUp16(size_t x) { return (x + 15) & ~(size_t)15; }
// Round-up-to-128 reference (matches allReduceScratchBytes granularity).
size_t roundUp128(size_t x) { return (x + 127) & ~(size_t)127; }

// ------------------------------------------------------------------------
// Design constants: guard the measured defaults the launcher passes as the
// collDefault fallbacks so a silent retune is caught by review.
// ------------------------------------------------------------------------
TEST(ArPolicyConstants, DefaultsMatchDesign) {
  EXPECT_EQ(kAllReduceSdmaThresholdDefault, 16ull * kMiB);   // 16 MiB total
  EXPECT_EQ(kAllReduceOneShotThresholdDefault, 256ull * kKiB);  // 256 KiB total
  EXPECT_EQ(kAllReduceTwoShotMaxCtas, 16);
  EXPECT_EQ(kThresholdUnset, (size_t)-1);
}

// ------------------------------------------------------------------------
// parseSize: the raw-string size parser behind testParseSdmaThresholdEnv /
// testResolveSdmaThreshold. Every branch: null, empty, no-digits, plain,
// each recognised suffix (both cases), unknown suffix, and zero.
// ------------------------------------------------------------------------
TEST(ArParseSize, NullAndEmptyAreUnset) {
  EXPECT_EQ(parseSize(nullptr), kThresholdUnset);
  EXPECT_EQ(parseSize(""), kThresholdUnset);
}

TEST(ArParseSize, NoLeadingDigitsIsUnset) {
  EXPECT_EQ(parseSize("abc"), kThresholdUnset);
  EXPECT_EQ(parseSize("K"), kThresholdUnset);
}

TEST(ArParseSize, PlainDecimal) {
  EXPECT_EQ(parseSize("0"), 0u);
  EXPECT_EQ(parseSize("1"), 1u);
  EXPECT_EQ(parseSize("262144"), 262144u);
}

TEST(ArParseSize, BinarySuffixesBothCases) {
  EXPECT_EQ(parseSize("4K"), 4ull * kKiB);
  EXPECT_EQ(parseSize("4k"), 4ull * kKiB);
  EXPECT_EQ(parseSize("2M"), 2ull * kMiB);
  EXPECT_EQ(parseSize("2m"), 2ull * kMiB);
  EXPECT_EQ(parseSize("1G"), 1ull * kGiB);
  EXPECT_EQ(parseSize("1g"), 1ull * kGiB);
}

TEST(ArParseSize, UnknownSuffixIgnored) {
  // The default switch arm ignores an unrecognised trailing char (matches
  // common.h): the parsed numeric value is returned unscaled.
  EXPECT_EQ(parseSize("5x"), 5u);
  EXPECT_EQ(parseSize("16B"), 16u);
}

// ------------------------------------------------------------------------
// resolveThreshold: 3-way precedence collVal > sharedVal > collDefault.
// ------------------------------------------------------------------------
TEST(ArResolveThreshold, CollectiveValueWins) {
  EXPECT_EQ(resolveThreshold(4096, 8192, 262144), 4096u);
  // collVal wins even when it is 0 (a deliberate "force all-SDMA" knob).
  EXPECT_EQ(resolveThreshold(0, 8192, 262144), 0u);
}

TEST(ArResolveThreshold, SharedValueWhenCollUnset) {
  EXPECT_EQ(resolveThreshold(kThresholdUnset, 8192, 262144), 8192u);
  EXPECT_EQ(resolveThreshold(kThresholdUnset, 0, 262144), 0u);
}

TEST(ArResolveThreshold, DefaultWhenBothUnset) {
  EXPECT_EQ(resolveThreshold(kThresholdUnset, kThresholdUnset, 262144), 262144u);
  EXPECT_EQ(resolveThreshold(kThresholdUnset, kThresholdUnset,
                             kAllReduceSdmaThresholdDefault),
            kAllReduceSdmaThresholdDefault);
}

// End-to-end resolution mirroring testResolveSdmaThreshold(parseSize()...).
TEST(ArResolveThreshold, EndToEndViaParseSize) {
  // Unset env strings -> default.
  EXPECT_EQ(resolveThreshold(parseSize(nullptr), parseSize(nullptr),
                             kAllReduceSdmaThresholdDefault),
            kAllReduceSdmaThresholdDefault);
  // Shared "0" forces all-SDMA when the per-collective var is unset.
  EXPECT_EQ(resolveThreshold(parseSize(nullptr), parseSize("0"),
                             kAllReduceSdmaThresholdDefault),
            0u);
  // Per-collective "32M" overrides both.
  EXPECT_EQ(resolveThreshold(parseSize("32M"), parseSize("0"),
                             kAllReduceSdmaThresholdDefault),
            32ull * kMiB);
}

// ------------------------------------------------------------------------
// allReduceKernelTier: total-bytes split LSA (<= threshold) / Gin (>).
// ------------------------------------------------------------------------
TEST(ArKernelTier, BelowAndAtThresholdIsLsa) {
  EXPECT_EQ(allReduceKernelTier(0, 16 * kMiB), ARTier::LSA);
  EXPECT_EQ(allReduceKernelTier(1, 16 * kMiB), ARTier::LSA);
  EXPECT_EQ(allReduceKernelTier(16 * kMiB, 16 * kMiB), ARTier::LSA);  // boundary inclusive
}

TEST(ArKernelTier, AboveThresholdIsGin) {
  EXPECT_EQ(allReduceKernelTier(16 * kMiB + 1, 16 * kMiB), ARTier::Gin);
  EXPECT_EQ(allReduceKernelTier(2 * kGiB, 16 * kMiB), ARTier::Gin);
}

TEST(ArKernelTier, ThresholdZeroForcesGinExceptEmpty) {
  // threshold 0 => only a 0-byte message stays LSA; any real message is Gin.
  EXPECT_EQ(allReduceKernelTier(0, 0), ARTier::LSA);
  EXPECT_EQ(allReduceKernelTier(1, 0), ARTier::Gin);
}

// ------------------------------------------------------------------------
// allReduceSignalCount: floored at 2, else == ctas.
// ------------------------------------------------------------------------
TEST(ArSignalCount, FlooredAtTwo) {
  EXPECT_EQ(allReduceSignalCount(0), 2);
  EXPECT_EQ(allReduceSignalCount(1), 2);
}

TEST(ArSignalCount, PassthroughAtOrAboveTwo) {
  EXPECT_EQ(allReduceSignalCount(2), 2);
  EXPECT_EQ(allReduceSignalCount(8), 8);
  EXPECT_EQ(allReduceSignalCount(64), 64);
}

// ------------------------------------------------------------------------
// allReduceDevReqs: barrier=lsaBarrier=ctas, signal=signalCount(ctas),
// needsGin=supported=true. Exercises the ctas<2 flooring path too.
// ------------------------------------------------------------------------
TEST(ArDevReqs, ShapeForTypicalGrid) {
  DevReqs r = allReduceDevReqs(64);
  EXPECT_EQ(r.barrierCount, 64);
  EXPECT_EQ(r.lsaBarrierCount, 64);
  EXPECT_EQ(r.ginSignalCount, 64);
  EXPECT_TRUE(r.needsGin);
  EXPECT_TRUE(r.supported);
}

TEST(ArDevReqs, SignalFlooredForTinyGrid) {
  DevReqs r1 = allReduceDevReqs(1);
  EXPECT_EQ(r1.barrierCount, 1);
  EXPECT_EQ(r1.lsaBarrierCount, 1);
  EXPECT_EQ(r1.ginSignalCount, 2);  // floored
  EXPECT_TRUE(r1.needsGin);

  DevReqs r0 = allReduceDevReqs(0);
  EXPECT_EQ(r0.barrierCount, 0);
  EXPECT_EQ(r0.lsaBarrierCount, 0);
  EXPECT_EQ(r0.ginSignalCount, 2);  // floored
}

// The launcher registers for max(deviceCtaCount, kArMaxCtas)=64; assert that
// composition yields 64 signals (the value AllReduceGetDevCommRequirements uses).
TEST(ArDevReqs, LauncherRegistersMaxGrid) {
  const int kArMaxCtas = 64;
  for (int deviceCtaCount : {1, 8, 32, 64, 128}) {
    int reg = deviceCtaCount > kArMaxCtas ? deviceCtaCount : kArMaxCtas;
    DevReqs r = allReduceDevReqs(reg);
    EXPECT_EQ(r.ginSignalCount, reg);
    EXPECT_GE(r.ginSignalCount, 64);
  }
}

// ------------------------------------------------------------------------
// allReduceSliceStride: ceil(msgBytes/nRanks) rounded up to 16 B.
// ------------------------------------------------------------------------
TEST(ArSliceStride, NonPositiveRanksIsZero) {
  EXPECT_EQ(allReduceSliceStride(1 * kMiB, 0), 0u);
  EXPECT_EQ(allReduceSliceStride(1 * kMiB, -1), 0u);
}

TEST(ArSliceStride, CeilThenAlign16) {
  // Exact divide, already 16-aligned: 256/8 = 32 -> 32.
  EXPECT_EQ(allReduceSliceStride(256, 8), 32u);
  // Non-multiple ceil: 100/8 = 12.5 -> 13 -> round up 16.
  EXPECT_EQ(allReduceSliceStride(100, 8), 16u);
  // Already 16-aligned per-rank slice stays put: 1 MiB / 8 = 128 KiB.
  EXPECT_EQ(allReduceSliceStride(1 * kMiB, 8), 128u * kKiB);
  // Single rank: whole message rounded up to 16.
  EXPECT_EQ(allReduceSliceStride(17, 1), 32u);
  EXPECT_EQ(allReduceSliceStride(16, 1), 16u);
}

TEST(ArSliceStride, MatchesReferenceOverSweep) {
  for (int nRanks : {1, 2, 4, 7, 8, 16}) {
    for (size_t msg : {size_t(0), size_t(1), size_t(15), size_t(16), size_t(17),
                       size_t(1000), 1 * kMiB, 128 * kMiB + 7, 4 * kGiB + 3}) {
      size_t per = (msg + (size_t)nRanks - 1) / (size_t)nRanks;
      EXPECT_EQ(allReduceSliceStride(msg, nRanks), roundUp16(per))
          << "msg=" << msg << " nRanks=" << nRanks;
    }
  }
}

// ------------------------------------------------------------------------
// allReduceScratchBytes: N * sliceStride(maxMsg,N) rounded up to 128 B;
// zero for empty message or non-positive ranks.
// ------------------------------------------------------------------------
TEST(ArScratchBytes, ZeroWhenEmptyOrNoRanks) {
  EXPECT_EQ(allReduceScratchBytes(0, 8), 0u);
  EXPECT_EQ(allReduceScratchBytes(1 * kMiB, 0), 0u);
  EXPECT_EQ(allReduceScratchBytes(1 * kMiB, -3), 0u);
  EXPECT_EQ(allReduceScratchBytes(0, 0), 0u);
}

TEST(ArScratchBytes, StagesNSlicesRoundedTo128) {
  // 8 ranks, 1 MiB: stride = 128 KiB, total = 1 MiB, already 128-aligned.
  EXPECT_EQ(allReduceScratchBytes(1 * kMiB, 8), 1u * kMiB);
  // 3 ranks, 100 B: stride ceil(100/3)=34 -> round16 = 48; total 3*48 = 144
  // -> round128 = 256.
  EXPECT_EQ(allReduceScratchBytes(100, 3), 256u);
  // 7 ranks, 1 B: stride ceil(1/7)=1 -> round16 = 16; total 7*16 = 112
  // -> round128 = 128.
  EXPECT_EQ(allReduceScratchBytes(1, 7), 128u);
}

TEST(ArScratchBytes, MatchesReferenceOverSweep) {
  for (int nRanks : {1, 2, 7, 8, 16}) {
    for (size_t msg : {size_t(1), size_t(127), size_t(128), size_t(129),
                       1 * kMiB, 16 * kMiB + 5, 2 * kGiB + 1}) {
      size_t stride = allReduceSliceStride(msg, nRanks);
      size_t expect = roundUp128(stride * (size_t)nRanks);
      EXPECT_EQ(allReduceScratchBytes(msg, nRanks), expect)
          << "msg=" << msg << " nRanks=" << nRanks;
    }
  }
}

// Monotonicity: sizing scratch at the max message covers every smaller message
// (the property the launcher relies on to size the window once).
TEST(ArScratchBytes, MonotonicInMessageSize) {
  const int nRanks = 8;
  size_t prev = 0;
  for (size_t msg : {size_t(1), 1 * kMiB, 16 * kMiB, 128 * kMiB, 1 * kGiB, 4 * kGiB}) {
    size_t s = allReduceScratchBytes(msg, nRanks);
    EXPECT_GE(s, prev) << "msg=" << msg;
    prev = s;
  }
}

}  // namespace
