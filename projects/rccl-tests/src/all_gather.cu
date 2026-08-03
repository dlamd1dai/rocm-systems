
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

// Vectorized LSA AllGather copy, PULL model: each rank READS every peer's
// (read-only) send chunk over LSA and writes it into its OWN recvbuff slot
// [peer*count]. No rank writes a peer's recvbuff, so recvbuff is touched locally
// only -- structurally immune to the initData recvbuff-memset race the push
// variant suffered, and needing only an ENTRY barrier (order peers' sendbuf
// fill) with NO exit barrier (see call sites; mirrors the AllToAll pull tier).
// Reads N peer chunks instead of 1 local chunk, so it issues PEER_UNROLL peer
// loads up front to hide xGMI read latency, then drains the local stores. Falls
// back to a scalar copy when the buffers are not vector-aligned.
//
// In-place vs out-of-place source: peer p's chunk lives at DIFFERENT offsets in
// the two modes -- out-of-place it is at peer p's sendbuf base (all ranks share
// one sendoffset), but IN-PLACE the sendbuf overlaps recvbuf so peer p's chunk
// sits at peer p's recvbuf[p*count] (a rank-dependent offset). Reading a peer's
// send window with THIS rank's sendoffset is only correct out-of-place; in-place
// we must source peer p's own recvbuf slot. We detect the mode by pointer
// identity and pick the peer-source window accordingly. Either way each rank
// writes ONLY its own recvbuf and the [p*count] slots it reads stay
// idempotently stable, so a single entry barrier (no exit) is sufficient.
template <typename T>
__device__ void AllGatherLsaVectorized(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int nRanks, int tid, int nthreads) {
  using TN = typename VectorTypeMapping<T>::Type;
  constexpr int VECTOR_FACTOR = sizeof(TN) / sizeof(T);
  constexpr int UNROLL_FACTOR = 128/sizeof(TN);
  constexpr int PEER_UNROLL = 2;

  T* recvBase = (T*)ncclGetLocalPointer(recvwin, recvoffset);
  const T* sendLocal = (const T*)ncclGetLocalPointer(sendwin, sendoffset);
  const bool inPlace = (sendLocal == recvBase + (size_t)rank * count);

  // Per-peer source base pointer (peer p's own chunk), mode-aware (see header).
  auto peerSrc = [&](int peer) -> const T* {
    return inPlace ? ((const T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + (size_t)peer * count)
                   : ((const T*)ncclGetLsaPointer(sendwin, sendoffset, peer));
  };

  // recvBase[peer*count] keeps the recv-base alignment whenever count*sizeof(T)
  // is a multiple of sizeof(TN); every peer chunk shares the recv/send
  // registration alignment, so the local send base is a valid proxy for them.
  bool canVectorize = (sizeof(TN) > sizeof(T)) &&
                      (reinterpret_cast<uintptr_t>(recvBase) % sizeof(TN) == 0) &&
                      (reinterpret_cast<uintptr_t>(sendLocal) % sizeof(TN) == 0) &&
                      ((count * sizeof(T)) % sizeof(TN) == 0);

  if (canVectorize) {
    size_t vector_count = count / VECTOR_FACTOR;
    int elements_per_iteration = nthreads * UNROLL_FACTOR;
    size_t aligned_vector_count = (vector_count / elements_per_iteration) * elements_per_iteration;

    for (size_t base_offset = tid; base_offset < aligned_vector_count; base_offset += elements_per_iteration) {
      for (int peerBase = 0; peerBase < nRanks; peerBase += PEER_UNROLL) {
        int peersInGroup = min(PEER_UNROLL, nRanks - peerBase);
        TN values[PEER_UNROLL][UNROLL_FACTOR];
        TN* recvVecPtrs[PEER_UNROLL];
        // issue all peer loads first (PEER_UNROLL*UNROLL_FACTOR reads in flight)
        #pragma unroll
        for (int p = 0; p < peersInGroup; p++) {
          int peer = peerBase + p;
          const TN* peerSendVec = (const TN*)peerSrc(peer);
          recvVecPtrs[p] = (TN*)(recvBase + (size_t)peer * count);
          #pragma unroll
          for (int i = 0; i < UNROLL_FACTOR; i++)
            values[p][i] = peerSendVec[base_offset + i * nthreads];
        }
        #pragma unroll
        for (int p = 0; p < peersInGroup; p++) {
          #pragma unroll
          for (int i = 0; i < UNROLL_FACTOR; i++)
            recvVecPtrs[p][base_offset + i * nthreads] = values[p][i];
        }
      }
    }

    // remaining vectorized elements outside the unrolled span
    for (size_t base_offset = aligned_vector_count + tid; base_offset < vector_count; base_offset += nthreads) {
      for (int peer = 0; peer < nRanks; peer++) {
        const TN* peerSendVec = (const TN*)peerSrc(peer);
        TN* recvVecPtr = (TN*)(recvBase + (size_t)peer * count);
        recvVecPtr[base_offset] = peerSendVec[base_offset];
      }
    }

    // scalar tail for counts not divisible by the vector factor
    size_t scalar_start = vector_count * VECTOR_FACTOR;
    for (size_t offset = scalar_start + tid; offset < count; offset += nthreads) {
      for (int peer = 0; peer < nRanks; peer++) {
        recvBase[(size_t)peer * count + offset] = peerSrc(peer)[offset];
      }
    }
  } else {
    // scalar fallback for unaligned buffers
    for (size_t i = tid; i < count; i += nthreads) {
      for (int peer = 0; peer < nRanks; peer++) {
        recvBase[(size_t)peer * count + i] = peerSrc(peer)[i];
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
//   chunkBytes <= ALLGATHER_LSA_SINGLE_CTA_MAX:  pull LSA on a single CTA (small, entry barrier only).
//   chunkBytes <= sdmaThreshold:                 pull LSA (all CTAs, entry barrier only).
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
    // PULL tier: single ENTRY barrier orders every peer's sendbuf initData fill
    // before we read it; NO exit barrier -- recvbuf is written locally only and
    // sendbuf is not modified within the call, so kernel/stream completion
    // suffices (mirrors the AllToAll pull tier).
    if (chunkBytes <= ALLGATHER_LSA_SINGLE_CTA_MAX) {
      if (blockIdx.x != 0) return;
      const int tid = threadIdx.x;
      const int nthreads = blockDim.x;
      ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, 0 };
      lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
      AllGatherLsaVectorized<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm.rank, devComm.nRanks, tid, nthreads);
      return;
    }

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nthreads = blockDim.x * gridDim.x;

    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    AllGatherLsaVectorized<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm.rank, devComm.nRanks, tid, nthreads);
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

// AllGather is pure data movement: the element type only sets the per-element
// byte width (no reduction arithmetic), so bf16 (2B) and the fp8 types (1B) are
// gathered bit-exactly by the already-instantiated same-width specializations
// (half / uint8_t). The shared SPECIALIZE_KERNEL deliberately omits bf16/fp8
// because it is also used by the reduction collectives (e.g. AllReduce), whose
// kernels would then need real bf16/fp8 arithmetic instantiations. Handling
// them locally lets `-d all` cover bf16/fp8 for AllGather (matching RS, which
// gets them via SPECIALIZE_REDUCE_KERNEL) without aborting the sweep on the
// first bf16 launch, and without touching the arithmetic-bearing kernels.
#if HAVE_BF16
#define AG_BF16_CASE(kernel, type) (type) == ncclBfloat16 ? kernel<half> :
#else
#define AG_BF16_CASE(kernel, type)
#endif
#if HAVE_FP8
#define AG_FP8_CASE(kernel, type) (type) == ncclFloat8e4m3 ? kernel<uint8_t> : (type) == ncclFloat8e5m2 ? kernel<uint8_t> :
#else
#define AG_FP8_CASE(kernel, type)
#endif
#define SPECIALIZE_AG_KERNEL(kernel, type, op) \
  ( (op) != ncclSum ? nullptr : \
    AG_BF16_CASE(kernel, type) \
    AG_FP8_CASE(kernel, type) \
    SPECIALIZE_KERNEL(kernel, type, op) )

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
        TESTCHECK(testLaunchDeviceKernelThresholdLL(SPECIALIZE_AG_KERNEL(GinHybridAllGatherKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, agThr, g_agLLHandle));
#else
        TESTCHECK(testLaunchDeviceKernelThreshold(SPECIALIZE_AG_KERNEL(GinHybridAllGatherKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, agThr));
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

  auto kernel = SPECIALIZE_AG_KERNEL(GinHybridAllGatherTimedKernel, type, op);
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
