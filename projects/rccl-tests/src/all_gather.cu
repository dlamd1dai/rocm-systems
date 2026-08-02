
/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
#include "rccl_vector_types.h"
#include "gin_sdma_devtime.h"  // shared device-side (wall_clock64) timing scaffold
#endif

// LL (low-latency, packed data+flag) small-message AllGather path. The device
// types (ncclLLA2AHandle, ncclDevResourceRequirements) come from nccl_device.h,
// which common.h includes whenever the device API is on (>=2.28) or on any
// >=2.29 build. Gate the LL wiring on exactly that availability.
#if (defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)) || NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
#define AG_HAVE_LL 1
// Per-rank chunk size (bytes) at/below which the small-message AllGather uses
// the LL path lives in gin_sdma_collective_policy.h (kAllGatherLLMaxBytes,
// 4 KiB) so the kernel branch, requirement sizing and unit tests agree. LL
// doubles wire volume (8 B data + 8 B epoch tags per 16 B line) so it only pays
// off for tiny, latency-bound sizes; above this the vectorized LSA path is used.
// Tuned on 8x MI355X (2026-07-26): LL beats vectorized single-CTA LSA up to
// 4 KiB/rank (32 KiB total: 12.1 vs 13.2 us) but loses badly at 8 KiB/rank
// (64 KiB total: 19.0 vs 13.5 us) where the 2x volume dominates.
// LL scratch handle: bufHandle is assigned during ncclDevCommCreate; nSlots is
// set immediately by ncclLLA2ACreateRequirement. nSlots==0 means LL was not
// configured, so the kernel falls back to the vectorized LSA path.
static ncclLLA2AHandle g_agLLHandle = {};
static ncclDevResourceRequirements g_agLLReq = {};
#endif

void AllGatherGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  size_t base = gin_sdma::alignChunkCount(count, nranks, eltSize);
  *sendcount = base;
  *recvcount = base*nranks;
  *sendInplaceOffset = base;
  *recvInplaceOffset = 0;
  *paramcount = base;
}

testResult_t AllGatherInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? ((char*)args->recvbuffs[i])+rank*args->sendBytes : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0));
    for (int j=0; j<nranks; j++) {
      TESTCHECK(InitData((char*)args->expected[i] + args->sendBytes*j, sendcount, 0, type, ncclSum, 33*rep + j, 1, 0));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

testResult_t  AllGatherGetAlgoProtoChannels(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetAlgoInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetAlgoInfo(comm, ncclFunc_t::ncclFuncAllGather , count, type , 0, 0, 1, algo, proto, nchannels));
  return testSuccess;
}

testResult_t  AllGatherGetSymkInfo(ncclComm_t comm, size_t count, ncclDataType_t type, ncclRedOp_t op, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetSymkInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetSymkInfo(comm, ncclFunc_t::ncclFuncAllGather , count, type , op, algo, proto, nchannels));
  return testSuccess;
}

void AllGatherGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1))/((double)nranks);
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t AllGatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;

  switch(deviceImpl) {
    case 3: { // GinHybridAllGatherKernel: LSA direct (small) + direct GIN puts (large)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      reqs->barrierCount = deviceCtaCount;
      reqs->lsaBarrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      // LL scratch for the tiny-message fast path (single CTA => nBlocks=1).
      // Sized for the LSA team broadcasting up to ALLGATHER_LL_MAX_BYTES/rank:
      // a receiver holds nRanks chunks of (ALLGATHER_LL_MAX_BYTES/8) u64 slots.
      {
        int llMaxElts = commProperties->nRanks * (int)(gin_sdma::kAllGatherLLMaxBytes / 8);
        int nSlots = ncclLLA2ACalcSlots(llMaxElts, /*maxEltSize=*/8);
        ncclLLA2ACreateRequirement(/*nBlocks=*/1, nSlots, &g_agLLHandle, &g_agLLReq);
        g_agLLReq.next = reqs->resourceRequirementsList;
        reqs->resourceRequirementsList = &g_agLLReq;
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
bool AllGatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: // GinHybridAllGatherKernel: LSA direct (small) + direct GIN puts (large)
      reqs->barrierCount = deviceCtaCount;
      reqs->lsaBarrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      return true;
#endif
    default:
      return false;
  }
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
__device__ size_t AllGatherGetSdmaThreshold(struct ncclDevComm const& devComm) {
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

// Per-rank chunk size (bytes) at/below which the LSA branch collapses to a
// single CTA. Tiny AllGather is barrier-bound, not bandwidth-bound: using all
// deviceCtaCount CTAs multiplies the cross-rank LSA barrier traffic by the CTA
// count for no copy-throughput gain. One CTA (blockDim.x threads) saturates the
// local stores for chunks this small, so we drop the extra barriers. Tuned on
// 8x MI355X (2026-07-26): the single-CTA latency floor (~13 us) holds up to
// ~8 KiB/rank (64 KiB total); at 16 KiB/rank a single CTA turns
// bandwidth-bound (131072 total: 17.1 us single-CTA vs 14.9 us at 262144 with
// all CTAs), so hand off to the multi-CTA path there.
static const size_t ALLGATHER_LSA_SINGLE_CTA_MAX = gin_sdma::kAllGatherLsaSingleCtaMax;

// Vectorized LSA AllGather copy: read the local send chunk once (wide vector +
// unroll) and broadcast each value to every peer's recvbuff slot [rank*count].
// Because AllGather has a single source chunk (unlike AllToAll's per-peer
// chunks), we hoist the load out of the peer loop. Falls back to a scalar copy
// when the buffers are not vector-aligned.
template <typename T>
__device__ void AllGatherLsaVectorized(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int nRanks, int tid, int nthreads) {
  using TN = typename VectorTypeMapping<T>::Type;
  constexpr int VECTOR_FACTOR = sizeof(TN) / sizeof(T);
  constexpr int UNROLL_FACTOR = 128/sizeof(TN);
  constexpr int PEER_UNROLL = 2;

  T* src = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  const size_t dstOff = (size_t)rank * count;

  // dstOff*sizeof(T) is a multiple of sizeof(TN) whenever count*sizeof(T) is,
  // so per-peer dst pointers keep the same alignment as the recv base.
  bool canVectorize = (sizeof(TN) > sizeof(T)) &&
                      (reinterpret_cast<uintptr_t>(src) % sizeof(TN) == 0) &&
                      ((count * sizeof(T)) % sizeof(TN) == 0);

  if (canVectorize) {
    size_t vector_count = count / VECTOR_FACTOR;
    int elements_per_iteration = nthreads * UNROLL_FACTOR;
    TN* srcVec = (TN*)src;

    size_t aligned_vector_count = (vector_count / elements_per_iteration) * elements_per_iteration;
    for (size_t base_offset = tid; base_offset < aligned_vector_count; base_offset += elements_per_iteration) {
      // load the source vectors once; they are reused for every peer
      TN values[UNROLL_FACTOR];
      #pragma unroll
      for (int i = 0; i < UNROLL_FACTOR; i++) {
        values[i] = srcVec[base_offset + i * nthreads];
      }
      for (int peerBase = 0; peerBase < nRanks; peerBase += PEER_UNROLL) {
        int peersInGroup = min(PEER_UNROLL, nRanks - peerBase);
        #pragma unroll
        for (int p = 0; p < peersInGroup; p++) {
          int peer = peerBase + p;
          TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + dstOff);
          #pragma unroll
          for (int i = 0; i < UNROLL_FACTOR; i++) {
            size_t offset = base_offset + i * nthreads;
            recvVecPtr[offset] = values[i];
          }
        }
      }
    }

    // remaining vectorized elements outside the unrolled span
    for (size_t base_offset = aligned_vector_count + tid; base_offset < vector_count; base_offset += nthreads) {
      TN value = srcVec[base_offset];
      for (int peer = 0; peer < nRanks; peer++) {
        TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + dstOff);
        recvVecPtr[base_offset] = value;
      }
    }

    // scalar tail for counts not divisible by the vector factor
    size_t scalar_start = vector_count * VECTOR_FACTOR;
    for (size_t offset = scalar_start + tid; offset < count; offset += nthreads) {
      T value = src[offset];
      for (int peer = 0; peer < nRanks; peer++) {
        T* dst = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + dstOff;
        dst[offset] = value;
      }
    }
  } else {
    // scalar fallback for unaligned buffers
    for (size_t i = tid; i < count; i += nthreads) {
      T value = src[i];
      for (int lp = 0; lp < nRanks; lp++) {
        T* dst = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp) + dstOff;
        dst[i] = value;
      }
    }
  }
}

// LL (low-latency) AllGather for tiny messages, single CTA. Each rank broadcasts
// its chunk (as chunkU64 8-byte units) into the epoch-tagged LL scratch, then
// polls every rank's chunk out of its own scratch and writes it to its local
// recvbuff. There is NO cross-rank recvbuff write and NO barrier: cross-rank
// traffic is confined to the LL scratch and ordered purely by the per-slot
// epoch tag (ncclLLA2ASession), so it is immune to the initData recvbuff-memset
// race that a barrier-free direct-LSA copy would suffer. Requires 8-byte-aligned
// chunks (guaranteed by the test's 16/eltSize base alignment).
__device__ void AllGatherLLImpl(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset,
                                size_t chunkU64, struct ncclDevComm const& devComm, ncclTeam lsa,
                                ncclLLA2AHandle llHandle) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int nR = lsa.nRanks;
  const int myR = lsa.rank;
  const uint64_t* src = (const uint64_t*)ncclGetLocalPointer(sendwin, sendoffset);
  uint64_t* dst = (uint64_t*)ncclGetLocalPointer(recvwin, recvoffset);

  ncclLLA2ASession<ncclCoopCta> ll { ncclCoopCta(), devComm, lsa, llHandle, /*block=*/0,
                                     /*maxElts=*/(int)((size_t)nR * chunkU64) };

  // Broadcast my chunk into slot region [myR*chunkU64 ..].
  for (size_t j = tid; j < chunkU64; j += nthreads) {
    ll.bcast((int)((size_t)myR * chunkU64 + j), src[j]);
  }
  // Gather every rank's chunk (incl. my own bcast) into local recvbuff.
  for (int r = 0; r < nR; r++) {
    for (size_t j = tid; j < chunkU64; j += nthreads) {
      dst[(size_t)r * chunkU64 + j] = ll.recv<uint64_t>((int)((size_t)r * chunkU64 + j));
    }
  }
  ll.endEpoch(ncclCoopCta());
}

// Single-node hybrid AllGather (-D 3):
//   chunkBytes <= ALLGATHER_LL_MAX_BYTES:        LL packed data+flag, single CTA (tiny, no barrier).
//   chunkBytes <= ALLGATHER_LSA_SINGLE_CTA_MAX:  LSA on a single CTA (small, barrier-bound).
//   chunkBytes <= sdmaThreshold:                 vectorized LSA (all CTAs).
//   chunkBytes >  sdmaThreshold:                 direct all-peers GIN puts (proven MI355X path).
//
// sdmaThresholdOverride lets the AllGather-specific env var
// NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER tune this LSA<->GIN cutover
// independently; TEST_SDMA_THRESHOLD_UNSET falls back to the shared backend
// value (rsCtx->sdmaThreshold from NCCL_GIN_ANVIL_SDMA_THRESHOLD).
template <typename T>
__device__ __forceinline__ void ginAllGatherBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, struct ncclDevComm devComm, size_t sdmaThresholdOverride, ncclLLA2AHandle llHandle) {
  const size_t chunkBytes = count * sizeof(T);
  const size_t sdmaThreshold = (sdmaThresholdOverride != TEST_SDMA_THRESHOLD_UNSET)
                                   ? sdmaThresholdOverride
                                   : AllGatherGetSdmaThreshold(devComm);

  if (chunkBytes <= sdmaThreshold) {
    ncclTeam lsa = ncclTeamLsa(devComm);

    // Tiny messages: LL packed data+flag path (single CTA). Barrier-free and
    // immune to the recvbuff-memset race (cross-rank traffic stays in the
    // epoch-tagged LL scratch; only local recvbuff is written). Used when LL is
    // configured (nSlots>0), the chunk is 8-byte aligned, and it fits the
    // pre-sized slot count.
    if (gin_sdma::agLLEligible(chunkBytes, llHandle.nSlots, devComm.nRanks, gin_sdma::kAllGatherLLMaxBytes)) {
      if (blockIdx.x != 0) return;  // single CTA
      const size_t chunkU64 = chunkBytes / 8;
      AllGatherLLImpl(sendwin, sendoffset, recvwin, recvoffset, chunkU64, devComm, lsa, llHandle);
      return;
    }

    // Tiny messages: collapse to a single CTA to avoid multiplying the
    // cross-rank LSA barrier traffic by deviceCtaCount. All ranks take this
    // path in lockstep (identical size), so CTA 0's barrier slot stays
    // consistent across ranks; CTAs > 0 exit before touching any barrier.
    if (chunkBytes <= ALLGATHER_LSA_SINGLE_CTA_MAX) {
      if (blockIdx.x != 0) return;
      const int tid = threadIdx.x;
      const int nthreads = blockDim.x;
      ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, 0 };
      lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
      AllGatherLsaVectorized<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm.rank, devComm.nRanks, tid, nthreads);
      lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
      return;
    }

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nthreads = blockDim.x * gridDim.x;

    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    AllGatherLsaVectorized<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm.rank, devComm.nRanks, tid, nthreads);
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  const int ginContext = 0;
  const unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  const uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;

  // Chunked to <=1 GiB segments to avoid the 30-bit SDMA copy-count overflow
  // on >1 GiB per-rank chunks; the signal rides the final segment.
  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    ginPutChunked(gin, ncclTeamWorld(devComm), r,
        recvwin, recvoffset + (size_t)devComm.rank * chunkBytes,
        sendwin, sendoffset,
        chunkBytes, ncclGin_SignalInc{signalIndex});
  }

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());
}

// -D 3 production kernel: one AllGather (LL / single-CTA LSA / vectorized LSA / GIN puts).
template <typename T>
__global__ void GinHybridAllGatherKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, ncclLLA2AHandle llHandle) {
  (void)root;
  ginAllGatherBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm, sdmaThresholdOverride, llHandle);
}

// Device-timing kernel (shared gin_devtime methodology): run skip+loop back-to-back
// AllGather bodies under ONE persistent launch, bracketing only the timed region with
// wall_clock64() per CTA. Every body re-selects its tier and re-derives its sync at
// entry (LSA rebuilds its barrier; LL advances its epoch; the GIN tier re-reads the
// accumulated signal), so looping is correct with no extra bookkeeping. For the
// single-CTA tiers (LL / small LSA) only CTA 0 does work and CTAs>0 fall straight
// through to their end stamp, so min(start)..max(end) still captures CTA 0's window.
template <typename T>
__global__ void GinHybridAllGatherTimedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, ncclLLA2AHandle llHandle, int loop, int skip, long long* start_time, long long* end_time) {
  (void)root;
  for (int i = 0; i < skip + loop; i++) {
    if (i == skip) {
      __syncthreads();
      if (threadIdx.x == 0) start_time[blockIdx.x] = wall_clock64();
    }
    ginAllGatherBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm, sdmaThresholdOverride, llHandle);
  }
  __syncthreads();
  if (threadIdx.x == 0) end_time[blockIdx.x] = wall_clock64();
}
#endif
#endif

testResult_t AllGatherRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    NCCLCHECK(ncclAllGather(sptr, rptr, count, type, comm, stream));
  } else {
    switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3: {
        // AllGather-specific LSA<->GIN threshold. Compared against the per-rank
        // chunk (count*sizeof(T)). Default = 256 KiB/rank: on 8x MI355X
        // (NCCL_GIN_TYPE=6) LSA wins for a chunk <=256K (total <=2M) and GIN
        // wins >=512K/rank (total >=4M) (measured 2026-07-24). Override with
        // NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER, or the shared
        // NCCL_GIN_ANVIL_SDMA_THRESHOLD.
        static const size_t agThr = testResolveSdmaThreshold("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER", gin_sdma::kAllGatherSdmaThresholdDefault);
#if defined(AG_HAVE_LL)
        TESTCHECK(testLaunchDeviceKernelThresholdLL(SPECIALIZE_KERNEL(GinHybridAllGatherKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, agThr, g_agLLHandle));
#else
        TESTCHECK(testLaunchDeviceKernelThreshold(SPECIALIZE_KERNEL(GinHybridAllGatherKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, agThr));
#endif
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

// Device-side (in-kernel wall_clock64) timing for the GIN-SDMA AllGather (-D 3), via
// the shared gin_devtime scaffold. Opt-in through NCCL_GIN_ANVIL_DEVICE_TIMING (legacy
// NCCL_GIN_ANVIL_A2A_DEVICE_TIMING): 1=augment (print an extra #[ag-devtime] line next
// to the graph numbers), 2=device-time-only (report the in-kernel latency as the
// out-of-place metric; in-place keeps normal timing). loop/skip via
// NCCL_GIN_ANVIL_AG_DEVTIME_LOOP/_SKIP (default 10/10).
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
testResult_t AllGatherDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  static const int loop = gin_devtime::envInt("NCCL_GIN_ANVIL_AG_DEVTIME_LOOP", 10);
  static const int skip = gin_devtime::envInt("NCCL_GIN_ANVIL_AG_DEVTIME_SKIP", 10);

  const size_t count = args->nbytes / wordSize(type);   // per-rank send-chunk count
  if (count == 0 || loop < 1) return testSuccess;

  static const size_t agThr = testResolveSdmaThreshold("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER", gin_sdma::kAllGatherSdmaThresholdDefault);

  auto kernel = SPECIALIZE_KERNEL(GinHybridAllGatherTimedKernel, type, op);
  if (kernel == nullptr) return testSuccess;

  const int gridCtas = (deviceCtaCount > 0) ? deviceCtaCount : 16;
  double devUs = 0.0;
  TESTCHECK(gin_devtime::measure(args, gridCtas, loop,
      [&](int i, long long* d_start, long long* d_end) {
        ncclDevComm* devComm = args->devComms + i;
        ncclWindow_t sendwin = (ncclWindow_t)(in_place ? args->recvRegHandles[i] : args->sendRegHandles[i]);
        ncclWindow_t recvwin = (ncclWindow_t)args->recvRegHandles[i];
        size_t sendoff = in_place ? args->sendInplaceOffset * (size_t)devComm->rank : 0;
        kernel<<<gridCtas, 512, 0, args->streams[i]>>>(sendwin, sendoff, recvwin, 0, count, root, *devComm,
                 agThr, g_agLLHandle, loop, skip, d_start, d_end);
      },
      &devUs));

  if (outDeltaSec != nullptr) { *outDeltaSec = devUs * 1.0e-6; return testSuccess; }

  if (args->proc == 0 && args->thread == 0 && devUs > 0.0) {
    int nRanksGlobal = args->nProcs * args->nThreads * args->nGpus;
    const size_t totalBytes = count * wordSize(type) * (size_t)nRanksGlobal;
    double sec = devUs * 1.0e-6;
    double algBw = (double)totalBytes / 1.0e9 / sec;
    double busBw = algBw * ((double)(nRanksGlobal - 1) / (double)nRanksGlobal);
    printf("#[ag-devtime] size %12zu B  ctas %2d  loop %2d skip %2d  devtime %10.2f us  algbw %8.2f GB/s  busbw %8.2f GB/s\n",
           totalBytes, gridCtas, loop, skip, devUs, algBw, busBw);
    fflush(stdout);
  }
  return testSuccess;
}
#else
testResult_t AllGatherDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  return testSuccess;  // device API path not available in this build
}
#endif

struct testColl allGatherTest = {
  "AllGather",
  AllGatherGetCollByteCount,
  AllGatherInitData,
  AllGatherGetBw,
  AllGatherRunColl,
  AllGatherGetAlgoProtoChannels,
  AllGatherGetSymkInfo,
  AllGatherDeviceTime
};

void AllGatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllGatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AllGatherRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &allGatherTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

struct testEngine ncclTestEngine = {
  .getBuffSize = AllGatherGetBuffSize,
  .runTest = AllGatherRunTest,
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = AllGatherGetDevCommRequirements
#endif
};
