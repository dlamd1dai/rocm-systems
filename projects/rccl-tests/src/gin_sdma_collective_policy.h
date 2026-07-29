/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared, pure (no-GPU, no-RCCL) policy logic for the GIN-SDMA collective
// designs (Broadcast / AllGather / AllToAll, deviceImpl == 3). This is the
// single source of truth for:
//   - per-collective LSA<->GIN threshold + LL-cap env resolution,
//   - the tier-selection predicates the device kernels branch on,
//   - the chunk/offset and alignment math, and
//   - the devComm resource-requirement computation.
//
// Every function here is deliberately free of `ncclDevComm`, GPU intrinsics and
// getenv() so it can be (a) unit-tested on the host with GoogleTest to high
// line+branch coverage, and (b) called from the __global__ kernels so the SAME
// decision code is exercised on device. Attributes are guarded so the header
// also compiles as plain host C++ in the test binary.
//
// See ddai-artifacts/docs/gin-anvil-sdma-broadcast-design-plan.md (and the
// AllGather/AllToAll notes) for the measured basis of each default.

#ifndef GIN_SDMA_COLLECTIVE_POLICY_H_
#define GIN_SDMA_COLLECTIVE_POLICY_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// The device kernels get __host__ __device__; plain host builds (and the host
// unit test, which defines GIN_SDMA_HOST_ONLY) get no attribute so the header
// compiles as ordinary C++ with no device codegen.
#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(GIN_SDMA_HOST_ONLY)
#define GIN_SDMA_HD __host__ __device__
#else
#define GIN_SDMA_HD
#endif

namespace gin_sdma {

// Sentinel meaning "env var unset/empty/unparseable" (mirrors
// TEST_SDMA_THRESHOLD_UNSET in common.h so the two agree).
static constexpr size_t kThresholdUnset = (size_t)-1;

// ---- Design constants (single source of truth; the .cu files alias these) ----

// Broadcast LSA<->GIN default; compared against the full message bytes.
static constexpr size_t kBroadcastSdmaThresholdDefault = 262144;        // 256 KiB
// Broadcast scatter+allgather (large tier) default; 0 disables the tier.
static constexpr size_t kBroadcastScatterAgMinDefault  = 2ull * 1024 * 1024;  // 2 MiB
// Broadcast LL fast path: compile ceiling + default runtime cutover.
static constexpr size_t kBroadcastLLMaxBytes           = 65536;         // 64 KiB
static constexpr size_t kBroadcastLLDefaultMaxBytes    = 2048;          // 2 KiB

// AllGather LSA<->GIN default; compared against the per-rank chunk bytes.
static constexpr size_t kAllGatherSdmaThresholdDefault = 262144;        // 256 KiB/rank
static constexpr size_t kAllGatherLLMaxBytes           = 4096;          // 4 KiB/rank
// Per-rank chunk at/below which the LSA branch collapses to a single CTA.
static constexpr size_t kAllGatherLsaSingleCtaMax      = 8192;          // 8 KiB/rank

// AllToAll LSA<->GIN default; compared against the per-peer chunk bytes.
static constexpr size_t kAllToAllSdmaThresholdDefault  = 262144;        // 256 KiB/peer
static constexpr size_t kAllToAllLLMaxBytes            = 65536;         // 64 KiB/peer
// AllToAll LL is OFF by default (unset env -> 0 cap).
static constexpr size_t kAllToAllLLDefaultMaxBytes     = 0;

// SendRecv LL fast path: compile ceiling + default runtime cutover. Point-to-
// point carries a single message per receiver (like Broadcast, not AllGather),
// so a receiver needs just cap/8 u64 slots. On by default (2 KiB) since the ring
// is barrier-bound at tiny sizes and LL removes the exit barrier.
static constexpr size_t kSendRecvLLMaxBytes            = 65536;         // 64 KiB
static constexpr size_t kSendRecvLLDefaultMaxBytes     = 2048;          // 2 KiB

// Scatter LL fast path: compile ceiling + default runtime cutover. Scatter is
// one-to-all with a DISTINCT per-rank chunk, but each receiver still takes a
// single incoming message (its own chunk from the root), so a receiver needs
// just chunk/8 u64 slots -- the same single-message sizing as Broadcast/SendRecv
// (not the nRanks*... AllGather/AllToAll form). The LL cap is compared against
// the per-rank chunk bytes (matching the Scatter LSA<->GIN threshold). On by
// default (2 KiB) since tiny scatters are barrier-bound and LL drops the exit
// barrier; tune/disable via NCCL_GIN_ANVIL_SCATTER_LL_MAX_BYTES (0 = disable).
static constexpr size_t kScatterLLMaxBytes             = 65536;         // 64 KiB/chunk
static constexpr size_t kScatterLLDefaultMaxBytes      = 2048;          // 2 KiB/chunk

// Max bytes per single gin.put() on the Anvil-SDMA backend. The SDMA linear-copy
// descriptor count field is 30 bits and 1-based (HW encodes count = bytes - 1,
// see rocr-runtime amd_blit_sdma.cpp / sdma_registers.h), so the largest single
// packet is (2^30 - 1) + 1 = 2^30 = exactly 1 GiB. A put of >1 GiB silently
// truncates: a 2 GiB put encodes count = (2^31-1) & 0x3FFFFFFF = 2^30-1 and the
// HW copies only 1 GiB, corrupting the transfer. Every GIN-tier put must be
// split into segments of at most this size (see common.h::ginPutChunked).
//
// Set to exactly 1 GiB: this is the hardware maximum, so the segmentation clamp
// (seg <= kGinPutMaxBytes) guarantees count = seg-1 <= 2^30-1 = 0x3FFFFFFF, which
// fills the 30-bit field exactly with no truncation. Zero margin by design; do
// NOT raise above 2^30. 1 GiB is a multiple of 32 B, satisfying the copy
// descriptor's 32 B length alignment.
static constexpr size_t kGinPutMaxBytes                = 1024ull * 1024 * 1024;  // 1 GiB (2^30, HW max)

// ---------------------------- env / threshold ----------------------------

// Parse a size string ("123", "4K", "2M", "1G"; decimal only, optional single
// binary suffix). Returns kThresholdUnset for null/empty/non-numeric input.
// Pure: takes the raw string so it is testable without touching the process
// environment. (common.h::testParseSdmaThresholdEnv = this on getenv(name).)
// Host-only: it calls strtoull(), and only host dispatch/requirements code uses
// it, so it is not marked for device to keep the kernels device-clean.
inline size_t parseSize(const char* v) {
  if (v == nullptr || v[0] == '\0') return kThresholdUnset;
  char* end = nullptr;
  unsigned long long val = strtoull(v, &end, 10);
  if (end == v) return kThresholdUnset;  // no digits consumed
  if (*end) {  // trailing suffix char (end is always set by strtoull)
    switch (*end) {
      case 'k': case 'K': val *= 1024ULL; break;
      case 'm': case 'M': val *= 1024ULL * 1024ULL; break;
      case 'g': case 'G': val *= 1024ULL * 1024ULL * 1024ULL; break;
      default: break;  // unknown suffix ignored (matches common.h)
    }
  }
  return (size_t)val;
}

// Resolve a collective LSA<->GIN threshold from already-parsed candidates:
//   1. collective-specific value, if set;
//   2. shared NCCL_GIN_ANVIL_SDMA_THRESHOLD value, if set (global force knob);
//   3. the collective default.
GIN_SDMA_HD inline size_t resolveThreshold(size_t collVal, size_t sharedVal,
                                           size_t collDefault) {
  if (collVal != kThresholdUnset) return collVal;
  if (sharedVal != kThresholdUnset) return sharedVal;
  return collDefault;
}

// Resolve an LL cutover cap from an already-parsed env value:
//   unset -> unsetDefault (Broadcast: 2 KiB; AllToAll: 0 = off),
//   then clamp to the compile ceiling, then round down to 8 bytes.
GIN_SDMA_HD inline size_t resolveLLCap(size_t envVal, size_t unsetDefault,
                                       size_t maxCap) {
  size_t cap = (envVal == kThresholdUnset) ? unsetDefault : envVal;
  if (cap > maxCap) cap = maxCap;
  return (cap / 8) * 8;
}

// ------------------------------- alignment -------------------------------

// The rccl-tests per-rank/per-peer chunk count, aligned so a chunk is a whole
// number of 16-byte lines: (count/nRanks) & -(16/eltSize). Mirrors
// AllGather/AllToAll GetCollByteCount. eltSize must divide 16 (1,2,4,8,16).
GIN_SDMA_HD inline size_t alignChunkCount(size_t count, int nRanks, size_t eltSize) {
  if (nRanks <= 0 || eltSize == 0) return 0;
  size_t mask = (size_t)0 - (16 / eltSize);   // e.g. eltSize 4 -> mask ~3
  return (count / (size_t)nRanks) & mask;
}

// ------------------------------- Broadcast -------------------------------

// Host gate: use the scatter+allgather large tier instead of the flat/LSA/LL
// hybrid. Requires SAG enabled, >=2 ranks, message at/above the cutover, and at
// least one element per rank (the scatter needs a non-empty slice each).
GIN_SDMA_HD inline bool bcastUseScatterAllgather(size_t msgBytes, size_t count,
                                                 int nRanks, size_t sagMin) {
  return sagMin != 0 && nRanks >= 2 && msgBytes >= sagMin &&
         count >= (size_t)nRanks;
}

// In-kernel LL eligibility (inside the msgBytes<=sdmaThreshold branch):
// LL configured, 8-byte aligned, within the compile ceiling, and it fits the
// pre-sized slot count (a broadcast receiver needs msgBytes/8 u64 slots).
GIN_SDMA_HD inline bool bcastLLEligible(size_t msgBytes, int llSlots,
                                        size_t llMaxBytes) {
  return llSlots != 0 && (msgBytes % 8 == 0) && msgBytes <= llMaxBytes &&
         (msgBytes / 8) <= (size_t)llSlots;
}

enum class BcastTier { LL, LSADirect, Flat };

// Tier chosen by GinHybridBroadcastKernel (assumes the host already decided this
// is NOT the scatter+allgather tier). Uses bcastLLEligible so the kernel and the
// tests share the decision.
GIN_SDMA_HD inline BcastTier bcastKernelTier(size_t msgBytes, size_t sdmaThreshold,
                                             int llSlots, size_t llMaxBytes) {
  if (msgBytes <= sdmaThreshold) {
    if (bcastLLEligible(msgBytes, llSlots, llMaxBytes)) return BcastTier::LL;
    return BcastTier::LSADirect;
  }
  return BcastTier::Flat;
}

// ginSignalCount for Broadcast case 3: >=2 so the SAG two-signal scheme works.
GIN_SDMA_HD inline int bcastSignalCount(int deviceCtaCount) {
  return (deviceCtaCount < 2) ? 2 : deviceCtaCount;
}

// Even split with the remainder folded into the last rank's slice (SAG).
struct Chunk {
  size_t count;      // elements in this rank's slice
  size_t eltOffset;  // element offset of this rank's slice
};
GIN_SDMA_HD inline Chunk sagChunk(size_t totalCount, int nRanks, int rank) {
  Chunk c{0, 0};
  if (nRanks <= 0) return c;
  size_t base = totalCount / (size_t)nRanks;
  size_t tail = totalCount - base * (size_t)(nRanks - 1);
  c.count = (rank == nRanks - 1) ? tail : base;
  c.eltOffset = (size_t)rank * base;
  return c;
}

// ------------------------------- AllGather -------------------------------

GIN_SDMA_HD inline bool agLLEligible(size_t chunkBytes, int llSlots, int nRanks,
                                     size_t llMaxBytes) {
  return llSlots != 0 && (chunkBytes % 8 == 0) && chunkBytes <= llMaxBytes &&
         (size_t)nRanks * (chunkBytes / 8) <= (size_t)llSlots;
}

enum class AGTier { LL, LSASingleCta, LSAMultiCta, Gin };

GIN_SDMA_HD inline AGTier agKernelTier(size_t chunkBytes, size_t sdmaThreshold,
                                       int llSlots, int nRanks, size_t llMaxBytes,
                                       size_t singleCtaMax) {
  if (chunkBytes <= sdmaThreshold) {
    if (agLLEligible(chunkBytes, llSlots, nRanks, llMaxBytes)) return AGTier::LL;
    if (chunkBytes <= singleCtaMax) return AGTier::LSASingleCta;
    return AGTier::LSAMultiCta;
  }
  return AGTier::Gin;
}

// ------------------------------- AllToAll -------------------------------

GIN_SDMA_HD inline bool a2aLLEligible(size_t size, int llSlots, int nRanks,
                                      size_t llMaxBytes) {
  return llSlots != 0 && (size % 8 == 0) && size <= llMaxBytes &&
         (size_t)nRanks * (size / 8) <= (size_t)llSlots;
}

enum class A2ATier { LL, LSA, Gin };

GIN_SDMA_HD inline A2ATier a2aKernelTier(size_t size, size_t sdmaThreshold,
                                         int llSlots, int nRanks, size_t llMaxBytes) {
  if (size <= sdmaThreshold) {
    if (a2aLLEligible(size, llSlots, nRanks, llMaxBytes)) return A2ATier::LL;
    return A2ATier::LSA;
  }
  return A2ATier::Gin;
}

// devComm resource requirements per AllToAll deviceImpl (host-side). `supported`
// is false for unknown deviceImpls; `needsGin` marks the impls that fail when
// the communicator has no GIN backend.
struct DevReqs {
  int barrierCount;
  int lsaBarrierCount;
  int ginSignalCount;
  bool needsGin;
  bool supported;
};
GIN_SDMA_HD inline DevReqs a2aDevReqs(int deviceImpl, int deviceCtaCount) {
  DevReqs r{0, 0, 0, false, true};
  switch (deviceImpl) {
    case 1:  // NvlAlltoAllKernel
    case 2:  // NvlAlltoAllKernelOptimized
      r.lsaBarrierCount = deviceCtaCount;
      return r;
    case 3:  // GinHybridAlltoAllKernel
      r.barrierCount = deviceCtaCount;
      r.lsaBarrierCount = deviceCtaCount;
      r.ginSignalCount = deviceCtaCount;
      r.needsGin = true;
      return r;
    case 4:  // HybridAlltoAllKernel: CTA 0 = GIN, CTAs 1..N = LSA
      r.barrierCount = 1;
      r.lsaBarrierCount = deviceCtaCount - 1;
      r.ginSignalCount = 1;
      r.needsGin = true;
      return r;
    default:
      r.supported = false;
      return r;
  }
}

// ---------------- Scatter / Gather / SendRecv (pure movement) ----------------
//
// Phase-1 single-phase movement collectives. They share one tier predicate and
// one devComm-requirement shape: a size-hybrid LSA(small) / GIN-SDMA(large) split
// at a per-collective LSA<->GIN threshold (default 256 KiB, retuned by
// measurement per the design plan), and barrier = lsaBarrier = ginSignal =
// deviceCtaCount with needsGin. No scratch. SendRecv additionally has an
// opt-out LL tiny-message tier (below); Scatter/Gather stay LSA/GIN for now.

// Compared against the transfer bytes (Scatter/Gather: per-rank chunk; SendRecv:
// full message). Tuned on 8x MI355X (gfx950, NCCL_GIN_TYPE=6, -V 32, 2026-07-27;
// LSA-forced vs GIN-forced sweeps, gin-sdma-phase1-tune.bash):
//
//   * Scatter LSA is root-egress-bound (only the root SM-stores all N-1 peer
//     chunks, ~64 GB/s ceiling), so GIN/SDMA wins decisively once the per-rank
//     chunk is large: crossover at a 256 KiB chunk (2 MiB total), and by 512 MiB
//     GIN is 390 vs 64 GB/s. Default 128 KiB puts the cutover just below the
//     measured crossover (<=128 KiB chunk -> LSA, >=256 KiB -> GIN).
//   * Gather and SendRecv distribute the writes across all ranks (every rank
//     stores its own slice), so direct LSA beats GIN/SDMA at *every* measured
//     size up to 512 MiB (Gather 432 vs 419, SendRecv 62.4 vs 61.1 GB/s at
//     512 MiB). Their default is set high so LSA is used across all practical
//     sizes; the GIN tier remains available as an env-tunable escape hatch
//     (NCCL_GIN_ANVIL_SDMA_THRESHOLD_{GATHER,SENDRECV}=0 forces it).
static constexpr size_t kScatterSdmaThresholdDefault  = 131072;      // 128 KiB/rank chunk
static constexpr size_t kGatherSdmaThresholdDefault   = 1073741824;  // 1 GiB: LSA-always
static constexpr size_t kSendRecvSdmaThresholdDefault = 1073741824;  // 1 GiB: LSA-always

enum class MoveTier { LSA, Gin };

// Tier chosen by the single-phase movement kernels: LSA below/at the threshold,
// GIN/SDMA above it.
GIN_SDMA_HD inline MoveTier moveKernelTier(size_t bytes, size_t sdmaThreshold) {
  return (bytes <= sdmaThreshold) ? MoveTier::LSA : MoveTier::Gin;
}

// devComm requirements for the -D 3 movement kernels (scatter/gather/sendrecv):
// one barrier + one lsaBarrier + one signal per CTA, GIN required.
GIN_SDMA_HD inline DevReqs moveDevReqs(int deviceCtaCount) {
  DevReqs r{deviceCtaCount, deviceCtaCount, deviceCtaCount, true, true};
  return r;
}

// SendRecv LL eligibility (inside the msgBytes<=sdmaThreshold branch): LL
// configured (nSlots>0), 8-byte aligned, within the compile ceiling, and it
// fits the pre-sized slot count. Single incoming message per receiver, so the
// slot demand is msgBytes/8 (mirrors bcastLLEligible, not the nRanks*... forms).
GIN_SDMA_HD inline bool sendRecvLLEligible(size_t msgBytes, int llSlots,
                                           size_t llMaxBytes) {
  return llSlots != 0 && (msgBytes % 8 == 0) && msgBytes <= llMaxBytes &&
         (msgBytes / 8) <= (size_t)llSlots;
}

enum class SendRecvTier { LL, LSA, Gin };

// Tier chosen by GinSendRecvKernel: LL (tiny, one barrier removed) below the LL
// cap, else LSA below/at the SDMA threshold, else GIN/SDMA. Shared by the kernel
// and the host unit tests.
GIN_SDMA_HD inline SendRecvTier sendRecvKernelTier(size_t msgBytes,
                                                   size_t sdmaThreshold,
                                                   int llSlots,
                                                   size_t llMaxBytes) {
  if (msgBytes <= sdmaThreshold) {
    if (sendRecvLLEligible(msgBytes, llSlots, llMaxBytes)) return SendRecvTier::LL;
    return SendRecvTier::LSA;
  }
  return SendRecvTier::Gin;
}

// Scatter LL eligibility (inside the chunkBytes<=sdmaThreshold branch): LL
// configured (nSlots>0), 8-byte aligned, within the compile ceiling, and it
// fits the pre-sized slot count. Each receiver takes ONE chunk from the root, so
// the slot demand is chunkBytes/8 (mirrors bcast/sendRecvLLEligible, not the
// nRanks*... forms). Compared against the per-rank chunk bytes.
GIN_SDMA_HD inline bool scatterLLEligible(size_t chunkBytes, int llSlots,
                                          size_t llMaxBytes) {
  return llSlots != 0 && (chunkBytes % 8 == 0) && chunkBytes <= llMaxBytes &&
         (chunkBytes / 8) <= (size_t)llSlots;
}

enum class ScatterTier { LL, LSA, Gin };

// Tier chosen by GinScatterKernel: LL (tiny, exit barrier removed) below the LL
// cap, else LSA below/at the SDMA threshold, else GIN/SDMA. Compared against the
// per-rank chunk bytes. Shared by the kernel and the host unit tests.
GIN_SDMA_HD inline ScatterTier scatterKernelTier(size_t chunkBytes,
                                                 size_t sdmaThreshold,
                                                 int llSlots,
                                                 size_t llMaxBytes) {
  if (chunkBytes <= sdmaThreshold) {
    if (scatterLLEligible(chunkBytes, llSlots, llMaxBytes)) return ScatterTier::LL;
    return ScatterTier::LSA;
  }
  return ScatterTier::Gin;
}

}  // namespace gin_sdma

#endif  // GIN_SDMA_COLLECTIVE_POLICY_H_
