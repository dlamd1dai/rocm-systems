/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for the pure GIN-SDMA collective policy logic shared by the
// Broadcast / AllGather / AllToAll device kernels and their host dispatch
// (projects/rccl-tests/src/gin_sdma_collective_policy.h). These cover the
// tier-selection, threshold/env resolution, chunk/alignment and devComm
// requirement decisions to high line+branch coverage WITHOUT a GPU: the header
// is compiled as plain host C++, and the same functions are called by the
// __global__ kernels on device, so exercising them here exercises the real
// decision code. GPU data-movement correctness (the branch bodies) is covered
// separately by the ctest `gpu_functional` matrix.

#include <gtest/gtest.h>

#include "gin_sdma_collective_policy.h"

using namespace gin_sdma;

namespace {

constexpr size_t kUnset = kThresholdUnset;

// --------------------------------- parseSize ---------------------------------

TEST(ParseSize, NullAndEmptyAreUnset) {
  EXPECT_EQ(parseSize(nullptr), kUnset);
  EXPECT_EQ(parseSize(""), kUnset);
}

TEST(ParseSize, NonNumericIsUnset) {
  EXPECT_EQ(parseSize("abc"), kUnset);  // no digits consumed
  EXPECT_EQ(parseSize("K"), kUnset);    // suffix only, no digits
}

TEST(ParseSize, PlainDecimal) {
  EXPECT_EQ(parseSize("0"), 0u);
  EXPECT_EQ(parseSize("1024"), 1024u);
  EXPECT_EQ(parseSize("262144"), 262144u);
}

TEST(ParseSize, BinarySuffixesBothCases) {
  EXPECT_EQ(parseSize("4K"), 4u * 1024);
  EXPECT_EQ(parseSize("4k"), 4u * 1024);
  EXPECT_EQ(parseSize("2M"), 2u * 1024 * 1024);
  EXPECT_EQ(parseSize("2m"), 2u * 1024 * 1024);
  EXPECT_EQ(parseSize("1G"), 1024u * 1024 * 1024);
  EXPECT_EQ(parseSize("1g"), 1024u * 1024 * 1024);
}

TEST(ParseSize, UnknownSuffixIgnored) {
  EXPECT_EQ(parseSize("512B"), 512u);
  EXPECT_EQ(parseSize("10X"), 10u);
}

// ------------------------------ resolveThreshold -----------------------------

TEST(ResolveThreshold, CollectiveValueWins) {
  EXPECT_EQ(resolveThreshold(4096, 8192, 262144), 4096u);
}

TEST(ResolveThreshold, SharedUsedWhenCollUnset) {
  EXPECT_EQ(resolveThreshold(kUnset, 8192, 262144), 8192u);
}

TEST(ResolveThreshold, DefaultWhenBothUnset) {
  EXPECT_EQ(resolveThreshold(kUnset, kUnset, 262144), 262144u);
}

TEST(ResolveThreshold, ExplicitZeroIsHonored) {
  // Zero is a valid "force GIN/SDMA always" value, distinct from unset.
  EXPECT_EQ(resolveThreshold(0, kUnset, 262144), 0u);
  EXPECT_EQ(resolveThreshold(kUnset, 0, 262144), 0u);
}

// -------------------------------- resolveLLCap -------------------------------

TEST(ResolveLLCap, UnsetUsesDefaultThenAligns) {
  EXPECT_EQ(resolveLLCap(kUnset, kBroadcastLLDefaultMaxBytes, kBroadcastLLMaxBytes),
            2048u);
  // AllToAll default is 0 (LL off).
  EXPECT_EQ(resolveLLCap(kUnset, kAllToAllLLDefaultMaxBytes, kAllToAllLLMaxBytes),
            0u);
}

TEST(ResolveLLCap, ClampsToMax) {
  EXPECT_EQ(resolveLLCap(1u << 20, kBroadcastLLDefaultMaxBytes, kBroadcastLLMaxBytes),
            kBroadcastLLMaxBytes);
}

TEST(ResolveLLCap, RoundsDownToEightBytes) {
  EXPECT_EQ(resolveLLCap(1001, 0, 65536), 1000u);  // 1001 -> 1000
  EXPECT_EQ(resolveLLCap(1000, 0, 65536), 1000u);  // already aligned
  EXPECT_EQ(resolveLLCap(7, 0, 65536), 0u);        // below one slot
}

// ------------------------------ alignChunkCount ------------------------------

TEST(AlignChunkCount, GuardsInvalidInputs) {
  EXPECT_EQ(alignChunkCount(1000, 0, 4), 0u);
  EXPECT_EQ(alignChunkCount(1000, -1, 4), 0u);
  EXPECT_EQ(alignChunkCount(1000, 8, 0), 0u);
}

TEST(AlignChunkCount, MasksPerElementSize) {
  // eltSize 4 -> mask ~3: 1000/8=125 -> 124.
  EXPECT_EQ(alignChunkCount(1000, 8, 4), 124u);
  // eltSize 8 -> mask ~1: 125 -> 124.
  EXPECT_EQ(alignChunkCount(1000, 8, 8), 124u);
  // eltSize 1 -> mask ~15: 125 -> 112.
  EXPECT_EQ(alignChunkCount(1000, 8, 1), 112u);
  // eltSize 16 -> mask all ones: unchanged.
  EXPECT_EQ(alignChunkCount(1000, 8, 16), 125u);
  // eltSize 2 -> mask ~7: 125 -> 120.
  EXPECT_EQ(alignChunkCount(1000, 8, 2), 120u);
}

// -------------------------- bcastUseScatterAllgather -------------------------

TEST(BcastScatterAllgather, DisabledWhenSagMinZero) {
  EXPECT_FALSE(bcastUseScatterAllgather(8u << 20, 1u << 20, 8, 0));
}

TEST(BcastScatterAllgather, RequiresAtLeastTwoRanks) {
  EXPECT_FALSE(bcastUseScatterAllgather(8u << 20, 1u << 20, 1, 2u << 20));
}

TEST(BcastScatterAllgather, RequiresMessageAtOrAboveMin) {
  const size_t sagMin = 2u << 20;
  EXPECT_FALSE(bcastUseScatterAllgather(sagMin - 1, 1u << 20, 8, sagMin));
  EXPECT_TRUE(bcastUseScatterAllgather(sagMin, 1u << 20, 8, sagMin));
}

TEST(BcastScatterAllgather, RequiresCountAtLeastNRanks) {
  EXPECT_FALSE(bcastUseScatterAllgather(8u << 20, 7, 8, 1024));
  EXPECT_TRUE(bcastUseScatterAllgather(8u << 20, 8, 8, 1024));
}

// ------------------------------ bcast LL / tier ------------------------------

TEST(BcastLLEligible, AllGates) {
  const size_t cap = kBroadcastLLMaxBytes;
  EXPECT_FALSE(bcastLLEligible(64, 0, cap));        // no slots
  EXPECT_FALSE(bcastLLEligible(60, 1000, cap));     // not 8-aligned
  EXPECT_FALSE(bcastLLEligible(cap + 8, 100000, cap));  // above ceiling
  EXPECT_FALSE(bcastLLEligible(800, 99, cap));      // 800/8=100 slots > 99
  EXPECT_TRUE(bcastLLEligible(800, 100, cap));      // exactly fits
  EXPECT_TRUE(bcastLLEligible(8, 1, cap));
}

TEST(BcastKernelTier, LadderSelection) {
  const size_t thr = 262144;
  const size_t cap = kBroadcastLLMaxBytes;
  EXPECT_EQ(bcastKernelTier(thr + 1, thr, 100000, cap), BcastTier::Flat);
  EXPECT_EQ(bcastKernelTier(800, thr, 100000, cap), BcastTier::LL);
  // <=thr but LL not configured -> LSA direct.
  EXPECT_EQ(bcastKernelTier(800, thr, 0, cap), BcastTier::LSADirect);
  // <=thr, LL configured but message above LL ceiling -> LSA direct.
  EXPECT_EQ(bcastKernelTier(cap + 8, thr, 1u << 20, cap), BcastTier::LSADirect);
}

TEST(BcastSignalCount, FloorOfTwo) {
  EXPECT_EQ(bcastSignalCount(0), 2);
  EXPECT_EQ(bcastSignalCount(1), 2);
  EXPECT_EQ(bcastSignalCount(2), 2);
  EXPECT_EQ(bcastSignalCount(8), 8);
}

// --------------------------------- sagChunk ----------------------------------

TEST(SagChunk, GuardsNonPositiveRanks) {
  Chunk c = sagChunk(800, 0, 0);
  EXPECT_EQ(c.count, 0u);
  EXPECT_EQ(c.eltOffset, 0u);
}

TEST(SagChunk, EvenSplit) {
  Chunk r0 = sagChunk(800, 8, 0);
  EXPECT_EQ(r0.count, 100u);
  EXPECT_EQ(r0.eltOffset, 0u);
  Chunk r3 = sagChunk(800, 8, 3);
  EXPECT_EQ(r3.count, 100u);
  EXPECT_EQ(r3.eltOffset, 300u);
  Chunk r7 = sagChunk(800, 8, 7);
  EXPECT_EQ(r7.count, 100u);
  EXPECT_EQ(r7.eltOffset, 700u);
}

TEST(SagChunk, RemainderFoldedIntoLast) {
  // 803 / 8 = base 100, last gets 803 - 700 = 103.
  Chunk r0 = sagChunk(803, 8, 0);
  EXPECT_EQ(r0.count, 100u);
  Chunk r7 = sagChunk(803, 8, 7);
  EXPECT_EQ(r7.count, 103u);
  EXPECT_EQ(r7.eltOffset, 700u);
  // Slices must exactly tile the whole message.
  size_t total = 0;
  for (int r = 0; r < 8; ++r) total += sagChunk(803, 8, r).count;
  EXPECT_EQ(total, 803u);
}

// ------------------------------ AllGather policy -----------------------------

TEST(AgLLEligible, AllGates) {
  const size_t cap = kAllGatherLLMaxBytes;
  EXPECT_FALSE(agLLEligible(64, 0, 8, cap));           // no slots
  EXPECT_FALSE(agLLEligible(60, 1000, 8, cap));        // not 8-aligned
  EXPECT_FALSE(agLLEligible(cap + 8, 100000, 8, cap)); // above ceiling
  // nRanks * chunkU64 must fit: 8 * (800/8)=800 slots.
  EXPECT_FALSE(agLLEligible(800, 799, 8, cap));
  EXPECT_TRUE(agLLEligible(800, 800, 8, cap));
}

TEST(AgKernelTier, FullLadder) {
  const size_t thr = 262144;
  const size_t cap = kAllGatherLLMaxBytes;
  const size_t single = kAllGatherLsaSingleCtaMax;
  EXPECT_EQ(agKernelTier(thr + 1, thr, 100000, 8, cap, single), AGTier::Gin);
  EXPECT_EQ(agKernelTier(800, thr, 1u << 20, 8, cap, single), AGTier::LL);
  // <=thr, LL off, <= single-CTA max -> single CTA.
  EXPECT_EQ(agKernelTier(single, thr, 0, 8, cap, single), AGTier::LSASingleCta);
  // <=thr, LL off, > single-CTA max -> multi CTA.
  EXPECT_EQ(agKernelTier(single + 1, thr, 0, 8, cap, single), AGTier::LSAMultiCta);
}

// ------------------------------ AllToAll policy ------------------------------

TEST(A2aLLEligible, AllGates) {
  const size_t cap = kAllToAllLLMaxBytes;
  EXPECT_FALSE(a2aLLEligible(64, 0, 8, cap));           // no slots
  EXPECT_FALSE(a2aLLEligible(60, 1000, 8, cap));        // not 8-aligned
  EXPECT_FALSE(a2aLLEligible(cap + 8, 1u << 20, 8, cap)); // above ceiling
  EXPECT_FALSE(a2aLLEligible(800, 799, 8, cap));        // 8*100 slots needed
  EXPECT_TRUE(a2aLLEligible(800, 800, 8, cap));
}

TEST(A2aKernelTier, FullLadder) {
  const size_t thr = 262144;
  const size_t cap = kAllToAllLLMaxBytes;
  EXPECT_EQ(a2aKernelTier(thr + 1, thr, 1u << 20, 8, cap), A2ATier::Gin);
  EXPECT_EQ(a2aKernelTier(800, thr, 1u << 20, 8, cap), A2ATier::LL);
  EXPECT_EQ(a2aKernelTier(800, thr, 0, 8, cap), A2ATier::LSA);  // LL off
}

TEST(A2aDevReqs, PerDeviceImpl) {
  const int cta = 8;

  DevReqs r1 = a2aDevReqs(1, cta);
  EXPECT_TRUE(r1.supported);
  EXPECT_FALSE(r1.needsGin);
  EXPECT_EQ(r1.lsaBarrierCount, cta);
  EXPECT_EQ(r1.barrierCount, 0);
  EXPECT_EQ(r1.ginSignalCount, 0);

  DevReqs r2 = a2aDevReqs(2, cta);
  EXPECT_TRUE(r2.supported);
  EXPECT_FALSE(r2.needsGin);
  EXPECT_EQ(r2.lsaBarrierCount, cta);

  // F1: case 3 sizes the barrier/signal pools for the largest grid the
  // size-adaptive LSA launch can request (max(deviceCtaCount, kA2aLsaMaxCtas)),
  // so a small -V (here cta=8) still allocates kA2aLsaMaxCtas barriers and any
  // adaptive grid uses only a subset.
  const int r3Expected = cta > kA2aLsaMaxCtas ? cta : kA2aLsaMaxCtas;
  DevReqs r3 = a2aDevReqs(3, cta);
  EXPECT_TRUE(r3.supported);
  EXPECT_TRUE(r3.needsGin);
  EXPECT_EQ(r3.barrierCount, r3Expected);
  EXPECT_EQ(r3.lsaBarrierCount, r3Expected);
  EXPECT_EQ(r3.ginSignalCount, r3Expected);

  // A large -V (> kA2aLsaMaxCtas) is honored as-is.
  DevReqs r3big = a2aDevReqs(3, kA2aLsaMaxCtas + 16);
  EXPECT_EQ(r3big.barrierCount, kA2aLsaMaxCtas + 16);
  EXPECT_EQ(r3big.lsaBarrierCount, kA2aLsaMaxCtas + 16);
  EXPECT_EQ(r3big.ginSignalCount, kA2aLsaMaxCtas + 16);

  DevReqs r4 = a2aDevReqs(4, cta);
  EXPECT_TRUE(r4.supported);
  EXPECT_TRUE(r4.needsGin);
  EXPECT_EQ(r4.barrierCount, 1);
  EXPECT_EQ(r4.lsaBarrierCount, cta - 1);
  EXPECT_EQ(r4.ginSignalCount, 1);

  DevReqs r0 = a2aDevReqs(0, cta);
  EXPECT_FALSE(r0.supported);
  DevReqs r5 = a2aDevReqs(5, cta);
  EXPECT_FALSE(r5.supported);
}

TEST(A2aLsaCtaCount, SizeLadder) {
  const int cap = kA2aLsaMaxCtas;  // 64
  // F3-tuned ladder (per-peer bytes): <=32K -> 8, <=64K -> 16, <=512K -> 32, else 48.
  EXPECT_EQ(a2aLsaCtaCount(0, cap), 8);
  EXPECT_EQ(a2aLsaCtaCount(32u * 1024, cap), 8);            // tiny boundary
  EXPECT_EQ(a2aLsaCtaCount(32u * 1024 + 1, cap), 16);
  EXPECT_EQ(a2aLsaCtaCount(64u * 1024, cap), 16);
  EXPECT_EQ(a2aLsaCtaCount(64u * 1024 + 1, cap), 32);
  EXPECT_EQ(a2aLsaCtaCount(512u * 1024, cap), 32);          // 4 MiB total: 32 optimum
  EXPECT_EQ(a2aLsaCtaCount(512u * 1024 + 1, cap), 48);      // >512K/peer -> 48
  EXPECT_EQ(a2aLsaCtaCount(2u * 1024 * 1024, cap), 48);     // 16 MiB total: top LSA rung
  EXPECT_EQ(a2aLsaCtaCount(64u * 1024 * 1024, cap), 48);
  // Cap clamps the ladder when maxCtas is below a rung.
  EXPECT_EQ(a2aLsaCtaCount(512u * 1024 + 1, 16), 16);
  EXPECT_EQ(a2aLsaCtaCount(0, 4), 4);
}

// -------------------- Scatter/Gather/SendRecv (movement) --------------------

TEST(MoveKernelTier, ThresholdSplit) {
  const size_t thr = 262144;
  EXPECT_EQ(moveKernelTier(thr, thr), MoveTier::LSA);      // at threshold -> LSA
  EXPECT_EQ(moveKernelTier(thr - 1, thr), MoveTier::LSA);
  EXPECT_EQ(moveKernelTier(thr + 1, thr), MoveTier::Gin);  // above -> GIN
  EXPECT_EQ(moveKernelTier(0, thr), MoveTier::LSA);
  EXPECT_EQ(moveKernelTier(1, 0), MoveTier::Gin);          // threshold 0 -> always GIN
}

TEST(MoveDevReqs, UniformCtaCountsAndGin) {
  const int cta = 8;
  DevReqs r = moveDevReqs(cta);
  EXPECT_TRUE(r.supported);
  EXPECT_TRUE(r.needsGin);
  EXPECT_EQ(r.barrierCount, cta);
  EXPECT_EQ(r.lsaBarrierCount, cta);
  EXPECT_EQ(r.ginSignalCount, cta);
}

TEST(MoveThresholdDefaults, TunedPerCollective) {
  // Tuned on 8x MI355X (2026-07-27): Scatter LSA is root-egress-bound so GIN
  // wins for chunks >=256 KiB (cutover 128 KiB); Gather/SendRecv distribute
  // writes so LSA wins to 512 MiB (LSA-always via a 1 GiB cutover).
  EXPECT_EQ(kScatterSdmaThresholdDefault, 131072u);        // 128 KiB
  EXPECT_EQ(kGatherSdmaThresholdDefault, 1073741824u);     // 1 GiB (LSA-always)
  EXPECT_EQ(kSendRecvSdmaThresholdDefault, 1073741824u);   // 1 GiB (LSA-always)
}

TEST(SendRecvLLEligible, AllGates) {
  const size_t cap = kSendRecvLLMaxBytes;
  EXPECT_FALSE(sendRecvLLEligible(64, 0, cap));           // no slots
  EXPECT_FALSE(sendRecvLLEligible(60, 1000, cap));        // not 8-aligned
  EXPECT_FALSE(sendRecvLLEligible(cap + 8, 100000, cap)); // above ceiling
  EXPECT_FALSE(sendRecvLLEligible(800, 99, cap));         // 800/8=100 slots > 99
  EXPECT_TRUE(sendRecvLLEligible(800, 100, cap));         // exactly fits
  EXPECT_TRUE(sendRecvLLEligible(8, 1, cap));
}

TEST(SendRecvKernelTier, LadderSelection) {
  const size_t thr = 262144;
  const size_t cap = kSendRecvLLMaxBytes;
  EXPECT_EQ(sendRecvKernelTier(thr + 1, thr, 100000, cap), SendRecvTier::Gin);
  EXPECT_EQ(sendRecvKernelTier(800, thr, 100000, cap), SendRecvTier::LL);
  // <=thr but LL not configured -> LSA.
  EXPECT_EQ(sendRecvKernelTier(800, thr, 0, cap), SendRecvTier::LSA);
  // <=thr, LL configured but message above LL ceiling -> LSA.
  EXPECT_EQ(sendRecvKernelTier(cap + 8, thr, 1u << 20, cap), SendRecvTier::LSA);
}

TEST(SendRecvLLDefaults, OnByDefault2KiB) {
  EXPECT_EQ(kSendRecvLLMaxBytes, 65536u);        // 64 KiB compile ceiling
  EXPECT_EQ(kSendRecvLLDefaultMaxBytes, 2048u);  // 2 KiB default cutover (on)
}

TEST(ScatterLLEligible, AllGates) {
  const size_t cap = kScatterLLMaxBytes;
  EXPECT_FALSE(scatterLLEligible(64, 0, cap));           // no slots
  EXPECT_FALSE(scatterLLEligible(60, 1000, cap));        // not 8-aligned
  EXPECT_FALSE(scatterLLEligible(cap + 8, 100000, cap)); // above ceiling
  EXPECT_FALSE(scatterLLEligible(800, 99, cap));         // 800/8=100 slots > 99
  EXPECT_TRUE(scatterLLEligible(800, 100, cap));         // exactly fits
  EXPECT_TRUE(scatterLLEligible(8, 1, cap));
}

TEST(ScatterKernelTier, LadderSelection) {
  // Compared against the per-rank chunk bytes.
  const size_t thr = 131072;  // scatter default
  const size_t cap = kScatterLLMaxBytes;
  EXPECT_EQ(scatterKernelTier(thr + 1, thr, 100000, cap), ScatterTier::Gin);
  EXPECT_EQ(scatterKernelTier(800, thr, 100000, cap), ScatterTier::LL);
  // <=thr but LL not configured -> LSA.
  EXPECT_EQ(scatterKernelTier(800, thr, 0, cap), ScatterTier::LSA);
  // <=thr, LL configured but chunk above LL ceiling -> LSA.
  EXPECT_EQ(scatterKernelTier(cap + 8, thr, 1u << 20, cap), ScatterTier::LSA);
}

TEST(ScatterLLDefaults, OnByDefault2KiB) {
  EXPECT_EQ(kScatterLLMaxBytes, 65536u);        // 64 KiB compile ceiling
  EXPECT_EQ(kScatterLLDefaultMaxBytes, 2048u);  // 2 KiB default cutover (on)
}

TEST(GatherLLEligible, AllGates) {
  // All-to-one fan-in: the ROOT holds nRanks chunks, so the slot demand is
  // nRanks*(chunk/8) (the AllGather form, not Scatter's single-message form).
  const size_t cap = kGatherLLMaxBytes;
  const int nRanks = 8;
  EXPECT_FALSE(gatherLLEligible(64, 0, nRanks, cap));            // no slots
  EXPECT_FALSE(gatherLLEligible(60, 100000, nRanks, cap));       // not 8-aligned
  EXPECT_FALSE(gatherLLEligible(cap + 8, 1u << 20, nRanks, cap));// above ceiling
  // 800 B chunk => 100 u64/chunk => root needs 8*100 = 800 slots.
  EXPECT_FALSE(gatherLLEligible(800, 799, nRanks, cap));         // 800 > 799
  EXPECT_TRUE(gatherLLEligible(800, 800, nRanks, cap));          // exactly fits
  EXPECT_TRUE(gatherLLEligible(8, nRanks, nRanks, cap));         // 1 u64/chunk * N
  EXPECT_FALSE(gatherLLEligible(8, nRanks - 1, nRanks, cap));    // N slots short
}

TEST(GatherKernelTier, LadderSelection) {
  // Compared against the per-rank chunk bytes; default threshold is LSA-always.
  const size_t thr = 1073741824u;  // gather default (1 GiB)
  const size_t cap = kGatherLLMaxBytes;
  const int nRanks = 8;
  // Chunk within LL cap and slots sized for N chunks -> LL.
  EXPECT_EQ(gatherKernelTier(800, thr, 8u << 10, nRanks, cap), GatherTier::LL);
  // <=thr but LL not configured -> LSA.
  EXPECT_EQ(gatherKernelTier(800, thr, 0, nRanks, cap), GatherTier::LSA);
  // <=thr, LL configured but chunk above LL ceiling -> LSA.
  EXPECT_EQ(gatherKernelTier(cap + 8, thr, 1u << 20, nRanks, cap), GatherTier::LSA);
  // Above the (forced-low) threshold -> GIN.
  EXPECT_EQ(gatherKernelTier(4096, 2048, 1u << 20, nRanks, cap), GatherTier::Gin);
}

TEST(GatherLLDefaults, OnByDefault2KiB) {
  EXPECT_EQ(kGatherLLMaxBytes, 65536u);        // 64 KiB compile ceiling
  EXPECT_EQ(kGatherLLDefaultMaxBytes, 2048u);  // 2 KiB provisional default (on)
}

// The Anvil-SDMA linear-copy count field is 30 bits and 1-based (count = bytes-1),
// so the largest safe single put is exactly 2^30 = 1 GiB (count = 2^30-1 fills the
// field). The chunk ceiling MUST NOT exceed 2^30, or seg-1 overflows 30 bits and
// the copy silently truncates. It is currently set to the hardware max (1 GiB).
TEST(GinPutMaxBytes, AtSdma30BitLimit) {
  EXPECT_EQ(kGinPutMaxBytes, 1073741824u);         // 1 GiB (2^30, HW max)
  // Hard correctness bound: seg <= kGinPutMaxBytes => count = seg-1 <= 2^30-1.
  EXPECT_LE(kGinPutMaxBytes, 1073741824ull);       // <= 2^30
  EXPECT_LE(kGinPutMaxBytes - 1u, 0x3FFFFFFFull);  // count fits 30-bit field
  EXPECT_EQ(kGinPutMaxBytes % 32u, 0u);            // 32 B copy-length aligned
}

// ---------------------------- ReduceScatter (P2) ----------------------------

TEST(ReduceScatterKernelTier, LadderSelection) {
  // Compared against the per-rank output-slice bytes (count*sizeof(T)).
  const size_t thr = kReduceScatterSdmaThresholdDefault;  // 256 KiB
  // At/below the threshold -> LSA read-reduce.
  EXPECT_EQ(reduceScatterKernelTier(1024, thr), RSTier::LSA);
  EXPECT_EQ(reduceScatterKernelTier(thr, thr), RSTier::LSA);       // boundary inclusive
  // Above the threshold -> put-partials GIN/SDMA + SM reduce.
  EXPECT_EQ(reduceScatterKernelTier(thr + 1, thr), RSTier::Gin);
  EXPECT_EQ(reduceScatterKernelTier(4u << 20, thr), RSTier::Gin);
  // A forced-low threshold (env override to 0) pushes everything to GIN.
  EXPECT_EQ(reduceScatterKernelTier(8, 0), RSTier::Gin);
}

TEST(ReduceScatterDevReqs, BarrierShapeAndGin) {
  const int cta = 12;
  DevReqs r = reduceScatterDevReqs(cta);
  EXPECT_TRUE(r.supported);
  EXPECT_TRUE(r.needsGin);
  EXPECT_EQ(r.barrierCount, cta);
  EXPECT_EQ(r.lsaBarrierCount, cta);
  EXPECT_EQ(r.ginSignalCount, cta);
}

TEST(ReduceScatterScratchBytes, RoundsUpTo128) {
  EXPECT_EQ(reduceScatterScratchBytes(0), 0u);         // no message -> no scratch
  EXPECT_EQ(reduceScatterScratchBytes(128), 128u);     // already aligned
  EXPECT_EQ(reduceScatterScratchBytes(1), 128u);       // rounds up to the 128 B unit
  EXPECT_EQ(reduceScatterScratchBytes(129), 256u);
  EXPECT_EQ(reduceScatterScratchBytes(4096), 4096u);
  // The full per-rank send buffer (N * maxChunk) must fit.
  EXPECT_GE(reduceScatterScratchBytes(1u << 20), (size_t)(1u << 20));
}

TEST(ReduceScatterDefaults, Threshold256KiB) {
  EXPECT_EQ(kReduceScatterSdmaThresholdDefault, 262144u);  // 256 KiB/rank slice
}

}  // namespace
