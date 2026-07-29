/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
#include "rccl_vector_types.h"
#endif

// LL (low-latency, packed data+flag) tiny-message Scatter fast path. Scatter is
// one-to-all with a DISTINCT per-rank chunk: the root writes each peer's chunk
// (as 8-byte units) into that peer's epoch-tagged LL scratch (including its own,
// so its slice arrives uniformly), and every rank polls its OWN scratch into
// recvbuff. The recv's epoch tag replaces the EXIT barrier; a single ENTRY LSA
// barrier is kept because -- like Broadcast/SendRecv, unlike AllGather -- only
// the root writes, so the ring lacks mutual-recv backpressure and a fast root
// could otherwise run >=2 epochs ahead of a laggard and deadlock the
// double-buffered LL scratch. Each rank writes only its own recvbuff, so it is
// immune to the initData recvbuff-memset race. On by default up to
// SCATTER_LL_DEFAULT_MAX_BYTES; tune/disable via
// NCCL_GIN_ANVIL_SCATTER_LL_MAX_BYTES (0 = disable). Constants live in
// gin_sdma_collective_policy.h so the kernel branch, the requirements sizing and
// the host unit tests share one source. nSlots==0 => LL not configured => the
// kernel uses the direct-LSA store.
#if (defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)) || NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
#define SC_HAVE_LL 1
static size_t g_scLLMaxBytes = 0;  // resolved from env at requirements time
static ncclLLA2AHandle g_scLLHandle = {};
static ncclDevResourceRequirements g_scLLReq = {};
#endif

void ScatterGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *recvcount = (count/nranks) & -(16/eltSize);
  *sendcount = (*recvcount)*nranks;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = *recvcount;
  *paramcount = *recvcount;
}

testResult_t ScatterInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    if (rank == root) TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, rep, 1, 0));
    TESTCHECK(InitData(args->expected[i], recvcount, rank*recvcount, type, ncclSum, rep, 1, 0));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

void ScatterGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t ScatterGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;
  switch(deviceImpl) {
    case 3: { // GinScatterKernel: root LSA stores (small) + root GIN puts (large)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      gin_sdma::DevReqs dr = gin_sdma::moveDevReqs(deviceCtaCount);
      reqs->barrierCount = dr.barrierCount;
      reqs->lsaBarrierCount = dr.lsaBarrierCount;
      reqs->ginSignalCount = dr.ginSignalCount;
      // LL scratch for the tiny-message fast path (single CTA => nBlocks=1), on
      // by default up to SCATTER_LL_DEFAULT_MAX_BYTES, tunable via
      // NCCL_GIN_ANVIL_SCATTER_LL_MAX_BYTES (bytes; 0 = disable). Each receiver
      // takes one chunk, so a receiver needs just cap/8 u64 slots (like
      // Broadcast/SendRecv, not the nRanks*... AllGather form).
      g_scLLMaxBytes = gin_sdma::resolveLLCap(
          testParseSdmaThresholdEnv("NCCL_GIN_ANVIL_SCATTER_LL_MAX_BYTES"),
          gin_sdma::kScatterLLDefaultMaxBytes, gin_sdma::kScatterLLMaxBytes);
      if (g_scLLMaxBytes > 0) {
        int llMaxElts = (int)(g_scLLMaxBytes / 8);
        int nSlots = ncclLLA2ACalcSlots(llMaxElts, /*maxEltSize=*/8);
        ncclLLA2ACreateRequirement(/*nBlocks=*/1, nSlots, &g_scLLHandle, &g_scLLReq);
        g_scLLReq.next = reqs->resourceRequirementsList;
        reqs->resourceRequirementsList = &g_scLLReq;
      }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,7)
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
      reqs->ginForceEnable = true;
#endif
      return testSuccess;
    }
    default:
      return testNotImplemented;
  }
}
#elif defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
bool ScatterGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));
  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: {
      gin_sdma::DevReqs dr = gin_sdma::moveDevReqs(deviceCtaCount);
      reqs->barrierCount = dr.barrierCount;
      reqs->lsaBarrierCount = dr.lsaBarrierCount;
      reqs->ginSignalCount = dr.ginSignalCount;
      return true;
    }
#endif
    default:
      return false;
  }
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
__device__ size_t ScatterGetSdmaThreshold(struct ncclDevComm const& devComm) {
  using nccl::utility::loadConst;
  if (devComm.ginConnectionCount == 0 || devComm.ginHandles[0] == nullptr) {
    return NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT;
  }
  ncclGinAnvilSdmaGPUContext* rsCtx =
      (ncclGinAnvilSdmaGPUContext*)devComm.ginHandles[0];
  if (rsCtx == nullptr ||
      loadConst(&rsCtx->layoutMagic) != NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC) {
    return NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT;
  }
  return loadConst(&rsCtx->sdmaThreshold);
}

// Simple local copy (16-byte-vectorized when aligned) used for the root's own
// slice on the out-of-place path.
template <typename T>
__device__ void ScatterLocalCopy(T* dst, const T* src, size_t count, int tid, int nthreads) {
  const size_t bytes = count * sizeof(T);
  const uintptr_t da = (uintptr_t)dst, sa = (uintptr_t)src;
  if ((bytes % sizeof(uint4)) == 0 && (da % sizeof(uint4)) == 0 && (sa % sizeof(uint4)) == 0) {
    uint4* d4 = (uint4*)dst; const uint4* s4 = (const uint4*)src;
    const size_t n4 = bytes / sizeof(uint4);
    for (size_t i = tid; i < n4; i += nthreads) d4[i] = s4[i];
  } else {
    for (size_t i = tid; i < count; i += nthreads) dst[i] = src[i];
  }
}

#if defined(SC_HAVE_LL)
// LL Scatter for tiny messages, single CTA. The root sends each peer r its
// distinct chunk (as 8-byte units) into peer r's epoch-tagged LL scratch --
// including its OWN chunk, so every rank (root included) uniformly polls its own
// scratch into recvbuff. A single entry LSA barrier bounds the epoch skew (only
// the root writes, so like Broadcast/SendRecv the ring lacks mutual-recv
// backpressure; see file header note); the recv's epoch tag replaces the exit
// barrier, and each rank writes only its own recvbuff, so it is immune to the
// initData recvbuff-memset race. In-place uses the same per-rank recv slot as
// the LSA path (recvoffset = dstBase + rank*chunk), and the root reconstructs
// the shared base to read chunk r.
__device__ void ScatterLLImpl(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset,
                              size_t chunkU64, int root, struct ncclDevComm const& devComm, ncclTeam lsa,
                              ncclLLA2AHandle llHandle) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int rank = devComm.rank, nRanks = devComm.nRanks;
  const bool inPlace = (sendwin == recvwin);
  const size_t chunkBytes = chunkU64 * 8;
  // Shared source base on the root (bytes): OOP reads sendoffset + r*chunk;
  // in-place reads its own recvbuff base (dstBase) + r*chunk, where
  // dstBase = recvoffset - root*chunk (recvoffset is this root's own slot).
  const size_t dstBase = inPlace ? (recvoffset - (size_t)rank * chunkBytes) : recvoffset;
  const size_t srcBase = inPlace ? dstBase : sendoffset;

  uint64_t* dst = (uint64_t*)ncclGetLocalPointer(recvwin, recvoffset);

  // Entry barrier: bounds root-vs-laggard epoch skew (see file header note).
  ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, /*block=*/0 };
  lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  ncclLLA2ASession<ncclCoopCta> ll { ncclCoopCta(), devComm, lsa, llHandle, /*block=*/0,
                                     /*maxElts=*/(int)chunkU64 };

  // Root sends each peer its distinct chunk into that peer's scratch [0..chunkU64).
  // Flatten the (peer, slot) fan-out across all threads with a PEER-MAJOR stride
  // (r = w % nRanks) so consecutive threads target DISTINCT peers -- i.e. all
  // xGMI egress links fire concurrently instead of one peer at a time. The old
  // "for r { for j=tid }" nesting left only chunkU64 threads active and walked
  // the peers serially, so the root's ~nRanks back-to-back remote stores were
  // the dominant term in scatter LL latency; this turns them into one concurrent
  // burst. Semantically identical (same slot, same epoch tag per (peer,slot)).
  if (rank == root) {
    const uint64_t* src = (const uint64_t*)ncclGetLocalPointer(sendwin, srcBase);
    const int totalSends = nRanks * (int)chunkU64;
    for (int w = tid; w < totalSends; w += nthreads) {
      const int r = w % nRanks;        // peer-major: distinct peers fire first
      const int j = w / nRanks;        // slot index within this peer's chunk
      ll.send(r, j, src[(size_t)r * chunkU64 + j]);
    }
  }
  // Every rank gathers its own chunk out of its own scratch.
  for (size_t j = tid; j < chunkU64; j += nthreads) {
    dst[j] = ll.recv<uint64_t>((int)j);
  }
  ll.endEpoch(ncclCoopCta());
}
#endif

// Root fan-out for the LSA tier, A/B-selectable via the lsaInterleave flag
// (env NCCL_GIN_ANVIL_SCATTER_LSA_INTERLEAVE, default on). The root alone stores
// all N per-rank chunks over xGMI, so its layout decides how well the root's
// egress links are used:
//   interleave==0 (sequential, historical): every CTA/thread walks the peers in
//     order (peer loop outer, chunk stride inner over the global tid), so all
//     SMs write ONE peer -- i.e. one xGMI link -- at a time, then the next. Root
//     egress is single-link-limited.
//   interleave!=0 (peer-interleaved): map each CTA to a peer slot
//     (blockIdx % nSlots, nSlots = min(gridDim, nRanks)) so all peers' links are
//     driven concurrently. Extra CTAs beyond nRanks become sub-blocks that split
//     a peer's chunk for more threads/link; when there are fewer CTAs than peers
//     each slot rotates over several peers. Every element is written exactly
//     once for any gridDim/nRanks combination.
template <typename T>
__device__ void ScatterLsaFanout(ncclWindow_t sendwin, size_t sendoffset,
                                 ncclWindow_t recvwin, size_t recvoffset,
                                 size_t count, size_t chunkBytes, int root,
                                 int nRanks, bool inPlace, size_t dstBase,
                                 bool interleave) {
  const T* src = (const T*)ncclGetLocalPointer(sendwin, sendoffset);
  if (!interleave) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nthreads = blockDim.x * gridDim.x;
    for (int r = 0; r < nRanks; r++) {
      if (inPlace && r == root) continue;  // own chunk already in place
      const size_t dstOff = inPlace ? (dstBase + (size_t)r * chunkBytes) : recvoffset;
      T* dst = (T*)ncclGetLsaPointer(recvwin, dstOff, r);
      const T* s = src + (size_t)r * count;
      for (size_t i = tid; i < count; i += nthreads) dst[i] = s[i];
    }
    return;
  }
  const int grid = (int)gridDim.x;
  const int blk = (int)blockIdx.x;
  const int nSlots = (grid < nRanks) ? grid : nRanks;   // active peer-slots
  const int peerSlot = blk % nSlots;
  const int subBlk = blk / nSlots;                       // sub-block within a slot
  const int nSub = (grid - peerSlot - 1) / nSlots + 1;   // sub-blocks for THIS slot
  const int localTid = subBlk * (int)blockDim.x + (int)threadIdx.x;
  const int localNthreads = nSub * (int)blockDim.x;
  for (int r = peerSlot; r < nRanks; r += nSlots) {
    if (inPlace && r == root) continue;
    const size_t dstOff = inPlace ? (dstBase + (size_t)r * chunkBytes) : recvoffset;
    T* dst = (T*)ncclGetLsaPointer(recvwin, dstOff, r);
    const T* s = src + (size_t)r * count;
    for (size_t i = localTid; i < count; i += localNthreads) dst[i] = s[i];
  }
}

// Single-node hybrid Scatter (-D 3): the root distributes a distinct per-rank
// chunk r (elements [r*count .. (r+1)*count) of its send buffer) to rank r.
//   chunkBytes <= LL cap (opt-out): LL packed data+flag, single CTA (tiny, exit
//                                   barrier removed vs LSA).
//   chunkBytes <= sdmaThreshold: root LSA-stores each peer's chunk (all CTAs).
//   chunkBytes >  sdmaThreshold: root issues one GIN put per non-self peer.
// Completion is receiver-side: each non-root receives exactly one put and waits
// base+1; the root receives none and only flushes (mirror of flat broadcast).
//
// In-place handling. Out-of-place: every rank's recv slot is recvoffset (the
// same symmetric offset on every peer). In-place: recvInplaceOffset = perRank
// chunk, so rank r's recv slot is at recvoffset base + r*chunk and the root's
// own chunk already sits in place. Detected by sendwin == recvwin (startColl
// passes recvwin as the send window when in-place); the root reconstructs the
// shared base offset from its own recvoffset (= base + root*chunk).
template <typename T>
__global__ void GinScatterKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, ncclLLA2AHandle llHandle, int lsaInterleave) {
  const size_t chunkBytes = count * sizeof(T);
  const size_t sdmaThreshold = (sdmaThresholdOverride != TEST_SDMA_THRESHOLD_UNSET)
                                   ? sdmaThresholdOverride
                                   : ScatterGetSdmaThreshold(devComm);
  const int rank = devComm.rank, nRanks = devComm.nRanks;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;

  const bool inPlace = (sendwin == recvwin);
  // Shared destination base offset across peers (bytes). OOP: recvoffset on all
  // ranks. In-place: recvoffset = base + rank*chunk, so base = recvoffset -
  // rank*chunk (computed on the root; only the root addresses peers).
  const size_t dstBase = inPlace ? (recvoffset - (size_t)rank * chunkBytes) : recvoffset;

  if (chunkBytes <= sdmaThreshold) {
    ncclTeam lsa = ncclTeamLsa(devComm);

#if defined(SC_HAVE_LL)
    // Tiny messages: LL packed data+flag path (single CTA). Keeps one entry
    // barrier (only the root writes, so the fan-out is not mutually
    // back-pressured) and drops the exit barrier; memset-race-immune (cross-rank
    // traffic stays in the LL scratch). Used when LL is configured (nSlots>0),
    // the per-rank chunk is 8-byte aligned, and it fits the pre-sized slots.
    if (gin_sdma::scatterLLEligible(chunkBytes, llHandle.nSlots, gin_sdma::kScatterLLMaxBytes)) {
      if (blockIdx.x != 0) return;  // single CTA
      ScatterLLImpl(sendwin, sendoffset, recvwin, recvoffset, chunkBytes / 8, root, devComm, lsa, llHandle);
      return;
    }
#endif

    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    if (rank == root) {
      // Root fan-out: peer-interleaved (all xGMI links concurrent) or the
      // historical sequential loop (one link at a time), A/B via lsaInterleave.
      ScatterLsaFanout<T>(sendwin, sendoffset, recvwin, recvoffset, count,
                          chunkBytes, root, nRanks, inPlace, dstBase,
                          lsaInterleave != 0);
    }
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  const unsigned int signalIndex = 0;
  ncclGin gin { devComm, /*context=*/0 };
  const uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  if (rank == root) {
    // Root's own slice: OOP needs a local copy into recvoffset; in-place it is
    // already in place.
    if (!inPlace) {
      T* lsrc = (T*)ncclGetLocalPointer(sendwin, sendoffset) + (size_t)root * count;
      T* ldst = (T*)ncclGetLocalPointer(recvwin, recvoffset);
      ScatterLocalCopy<T>(ldst, lsrc, count, tid, nthreads);
    }
    // Flat scatter: one put per non-self peer, each issued by a distinct thread
    // (tid=r). This GIN fan-out is already peer-concurrent by construction: the
    // Anvil-SDMA backend routes each peer's put to its own per-peer queue
    // (handles[r*numChannels + ch]), so the N-1 copies run concurrently on
    // independent SDMA engines. Adding SDMA channels (NUM_CHANNELS>1) or slicing
    // each peer's chunk into multiple sub-puts does NOT help -- a single put
    // already saturates its per-peer xGMI link, so both are perf-neutral-to-
    // negative (measured 2026-07-29 on 8x MI355X: NC=1/2/4 flat; segmenting each
    // peer into 2/4 sub-puts regressed 15-40% by adding SDMA descriptor
    // overhead). Hence one put per peer, unsegmented.
    // Chunked to <=1 GiB segments to avoid the 30-bit SDMA copy-count overflow
    // on >1 GiB per-rank chunks; the signal rides the final segment.
    for (int r = tid; r < nRanks; r += nthreads) {
      if (r == root) continue;
      const size_t dstOff = inPlace ? (dstBase + (size_t)r * chunkBytes) : recvoffset;
      ginPutChunked(gin, ncclTeamWorld(devComm), r,
          recvwin, dstOff,
          sendwin, sendoffset + (size_t)r * chunkBytes,
          chunkBytes, ncclGin_SignalInc{signalIndex});
    }
    gin.flush(ncclCoopCta());
  } else {
    // Each non-root receives exactly one scatter put.
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + 1);
  }
}
#endif

testResult_t ScatterRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (deviceImpl == 0) {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    size_t rankOffset = count * wordSize(type);
    if (count == 0) return testSuccess;

    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
    NCCLCHECK(ncclScatter(sptr, rptr, count, type, root, comm, stream));
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,7,0)
    NCCLCHECK(ncclGroupStart());
    if (rank == root) {
      for (int r=0; r<nRanks; r++) {
        NCCLCHECK(ncclSend(sptr + r * rankOffset, count, type, r, comm, stream));
      }
    }
    NCCLCHECK(ncclRecv(rptr, count, type, root, comm, stream));
    NCCLCHECK(ncclGroupEnd());
#else
    printf("NCCL 2.7 or later is needed for scatter. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#endif
  } else {
    switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3: {
        // Scatter-specific LSA<->GIN threshold (compared against the per-rank
        // chunk). Default 128 KiB: LSA is root-egress-bound so GIN/SDMA wins for
        // chunks >=256 KiB (8x MI355X, 2026-07-27; 512 MiB: 390 vs 64 GB/s).
        // Override with NCCL_GIN_ANVIL_SDMA_THRESHOLD_SCATTER or the shared
        // NCCL_GIN_ANVIL_SDMA_THRESHOLD.
        static const size_t scThr = testResolveSdmaThreshold("NCCL_GIN_ANVIL_SDMA_THRESHOLD_SCATTER", gin_sdma::kScatterSdmaThresholdDefault);
        // LSA-tier root fan-out layout: peer-interleaved (all xGMI links
        // concurrent) by default; NCCL_GIN_ANVIL_SCATTER_LSA_INTERLEAVE=0 forces
        // the historical sequential loop (one link at a time) for A/B.
        static const int scLsaInterleave = []() {
          size_t v = testParseSdmaThresholdEnv("NCCL_GIN_ANVIL_SCATTER_LSA_INTERLEAVE");
          return (v == TEST_SDMA_THRESHOLD_UNSET) ? 1 : (v != 0 ? 1 : 0);
        }();
        TESTCHECK(testLaunchDeviceKernelThresholdLLFlag(SPECIALIZE_KERNEL(GinScatterKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, scThr, g_scLLHandle, scLsaInterleave));
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

struct testColl scatterTest = {
  "Scatter",
  ScatterGetCollByteCount,
  ScatterInitData,
  ScatterGetBw,
  ScatterRunColl,
  NULL,
  NULL
};

void ScatterGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  ScatterGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t ScatterRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &scatterTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;
  int begin_root, end_root;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if (root != -1) {
    begin_root = end_root = root;
  } else {
    begin_root = 0;
    end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=begin_root; j<=end_root; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", j));
    }
  }
  return testSuccess;
}

struct testEngine ncclTestEngine = {
  .getBuffSize = ScatterGetBuffSize,
  .runTest = ScatterRunTest,
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = ScatterGetDevCommRequirements
#endif
};
