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
#endif

// LL (low-latency, packed data+flag) small-message AllToAll path. Mirrors the
// AllGather LL fast path, but uses the LL A2A session's point-to-point send()
// (scatter) instead of bcast(): AllToAll delivers a *distinct* per-peer chunk,
// whereas AllGather broadcasts one chunk to all peers. The device types
// (ncclLLA2AHandle, ncclDevResourceRequirements) come from nccl_device.h, which
// common.h includes whenever the device API is on (>=2.28) or on any >=2.29
// build. Gate the LL wiring on exactly that availability.
#if (defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)) || NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
#define A2A_HAVE_LL 1
// Compile ceiling on the per-peer chunk (bytes) the LL path can serve lives in
// gin_sdma_collective_policy.h (kAllToAllLLMaxBytes, 64 KiB); the LL scratch is
// sized for this cap at devComm creation. The *actual* LL<->LSA cutover is
// runtime-gated (see below): env NCCL_GIN_ANVIL_A2A_LL_MAX_BYTES.
// LL scratch handle: bufHandle is assigned during ncclDevCommCreate; nSlots is
// set by ncclLLA2ACreateRequirement only when the LL path is enabled. nSlots==0
// means LL was not configured, so the kernel uses the vectorized LSA path.
//
// Unlike AllGather (where LL robustly wins ~7-23% for tiny messages by removing
// both LSA barriers), the A2A direct-LSA all-CTA scatter is already at a ~11 us
// fixed-overhead floor on 8x MI355X that LL cannot beat: LL still needs every
// rank to poll all nRanks source slots (an implicit all-to-all sync) and moves
// 2x the wire volume. A careful A/B (50 iters x 3 reps, 2026-07-27) put LL at
// best a marginal, within-noise ~2-3% faster at the very smallest sizes
// (<=32 B/peer) and neutral-to-slightly-worse above that; run-to-run variance is
// itself ~2-3%. So the LL path is OFF by default and opt-in via
// NCCL_GIN_ANVIL_A2A_LL_MAX_BYTES=<per-peer bytes> (0/unset = disabled), kept for
// experimentation and future tuning. The env value is clamped to
// ALLTOALL_LL_MAX_BYTES and only enables LL for per-peer chunks at or below it
// (and only when the chunk fits the pre-sized slot count).
static size_t g_a2aLLMaxBytes = 0;  // resolved from env at requirements time
static ncclLLA2AHandle g_a2aLLHandle = {};
static ncclDevResourceRequirements g_a2aLLReq = {};

// Option A (barrier-free LSA completion) scratch: a per-rank flag inbox in the
// resource window. Allocated only when NCCL_GIN_ANVIL_A2A_SYNC_MODE==2. Layout
// per slab (uint64): [epoch[c]] for c in [0,kA2aLsaMaxCtas) then
// [done[c][s][slot]] indexed (kA2aLsaMaxCtas + ((c*nRanks + s)*kA2aFlagSlots + slot)).
static int g_a2aSyncMode = 0;                      // resolved from env at requirements time
static ncclDevResourceHandle_t g_a2aFlagHandle = 0;
static ncclDevResourceRequirements g_a2aFlagReq = {};

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// System-scope release store / acquire load of a single 64-bit flag to/from a
// peer's (or our own) resource-window inbox. System scope + release/acquire give
// the cross-GPU ordering (data writes -> fence -> done flag; peer: acquire flag
// -> data visible) without a collective barrier.
__device__ __forceinline__ void a2aFlagStore(uint64_t* p, uint64_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint64_t a2aFlagLoad(uint64_t* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
#endif
#endif

void AlltoAllGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *paramcount = gin_sdma::alignChunkCount(count, nranks, eltSize);
  *sendcount = nranks*(*paramcount);
  *recvcount = *sendcount;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
}

testResult_t AlltoAllInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0));
    for (int j=0; j<nranks; j++) {
      size_t partcount = sendcount/nranks;
      TESTCHECK(InitData((char*)args->expected[i] + j*partcount*wordSize(type), partcount, rank*partcount, type, ncclSum, 33*rep + j, 1, 0));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place alltoall
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// set devComm reqs for alltoall device kernels
testResult_t AlltoAllGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;

  switch(deviceImpl) {
    case 1: // NvlAlltoAllKernel
    case 2: // NvlAlltoAllKernelOptimized
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    case 3: { // GinHybridAlltoAllKernel: LSA direct (small) + all-peers GIN puts (large)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      {
        gin_sdma::DevReqs dr = gin_sdma::a2aDevReqs(3, deviceCtaCount);
        reqs->barrierCount = dr.barrierCount;
        reqs->lsaBarrierCount = dr.lsaBarrierCount;
        reqs->ginSignalCount = dr.ginSignalCount;
      }
      // LL scratch for the tiny-message fast path (multi-CTA barrier-free LL:
      // nBlocks=kA2aLLCtas, each CTA owns its own scratch block), opt-in via
      // NCCL_GIN_ANVIL_A2A_LL_MAX_BYTES (per-peer bytes; 0/unset = disabled).
      // Each block is sized for the full LSA-team scatter: a receiver holds
      // nRanks source-chunks of (cap/8) u64 slots (same layout as AllGather).
      // Per-block over-provisioning (each CTA uses only its chunk slice) keeps
      // the slot indexing identical to the single-CTA path and the scratch small
      // (nBlocks * 2 * nRanks * cap/8 * 16 B ~ 33 MiB at the 64 KiB/peer cap).
      {
        g_a2aLLMaxBytes = gin_sdma::resolveLLCap(
            testParseSdmaThresholdEnv("NCCL_GIN_ANVIL_A2A_LL_MAX_BYTES"),
            gin_sdma::kAllToAllLLDefaultMaxBytes, gin_sdma::kAllToAllLLMaxBytes);
        if (g_a2aLLMaxBytes > 0) {
          int llMaxElts = commProperties->nRanks * (int)(g_a2aLLMaxBytes / 8);
          int nSlots = ncclLLA2ACalcSlots(llMaxElts, /*maxEltSize=*/8);
          ncclLLA2ACreateRequirement(/*nBlocks=*/gin_sdma::kA2aLLCtas, nSlots, &g_a2aLLHandle, &g_a2aLLReq);
          g_a2aLLReq.next = reqs->resourceRequirementsList;
          reqs->resourceRequirementsList = &g_a2aLLReq;
        }
      }
      // Option A: allocate the barrier-free completion flag inbox when sync mode 2
      // is selected. Sized for kA2aLsaMaxCtas CTAs x nRanks sources x kA2aFlagSlots.
      {
        const char* e = getenv("NCCL_GIN_ANVIL_A2A_SYNC_MODE");
        g_a2aSyncMode = (e && *e) ? atoi(e) : 3;  // default: single exit barrier
        if (g_a2aSyncMode == 2) {
          size_t nFlags = (size_t)gin_sdma::kA2aLsaMaxCtas *
                          (1 + (size_t)commProperties->nRanks * gin_sdma::kA2aFlagSlots);
          memset(&g_a2aFlagReq, 0, sizeof(g_a2aFlagReq));
          g_a2aFlagReq.bufferSize = nFlags * sizeof(uint64_t);
          g_a2aFlagReq.bufferAlign = 16;
          g_a2aFlagReq.outBufferHandle = &g_a2aFlagHandle;
          g_a2aFlagReq.next = reqs->resourceRequirementsList;
          reqs->resourceRequirementsList = &g_a2aFlagReq;
        }
      }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,7)
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
      reqs->ginForceEnable = true;
#endif
      return testSuccess;
    }
    case 4: // HybridAlltoAllKernel: CTA 0 = GIN (1 barrier), CTAs 1..N = LSA
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      reqs->barrierCount = 1;
      reqs->lsaBarrierCount = deviceCtaCount - 1;
      reqs->ginSignalCount = 1;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,7)
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
      reqs->ginForceEnable = true;
#endif
      return testSuccess;
    default:
      return testNotImplemented;
  }
}
#elif defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
// set devComm reqs for alltoall device kernels
bool AlltoAllGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
    case 1: // NvlAlltoAllKernel
    case 2: // NvlAlltoAllKernelOptimized
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: // GinHybridAlltoAllKernel: LSA direct (small) + all-peers GIN puts (large)
      reqs->barrierCount = deviceCtaCount;
      reqs->lsaBarrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      return true;
    case 4: // HybridAlltoAllKernel: CTA 0 = GIN (1 barrier), CTAs 1..N = LSA
      reqs->barrierCount = 1;
      reqs->lsaBarrierCount = deviceCtaCount - 1;
      reqs->ginSignalCount = 1;
      return true;
#endif
    default:
      return false;
  }
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
// shared scalar AlltoAll implementation used by both kernels
template <typename T>
__device__ void AlltoAllScalarImpl(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int nRanks, int tid, int nthreads) {
  T* sendPtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, rank);

  // F2+F3 peer schedule: peer = (rank + blockIdx.x + pp) % nRanks. F2 rotates by
  // rank so different ranks target distinct peers at the same step (no cross-rank
  // xGMI incast). F3 adds blockIdx.x so different CTAs of the *same* rank are on
  // different peers at the same instant -- otherwise every CTA marches through
  // the peers in lock-step and the rank drives only one xGMI egress link at a
  // time, cycling the 7 links serially; the phase shift spreads the CTAs across
  // all peers so every link runs concurrently (host-RING channel parallelism).
  // Coverage is unchanged: each thread still visits every peer once per offset.
  for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int pp = 0; pp < nRanks; pp++) {
      int peer = (rank + blockIdx.x + pp) % nRanks;
      T value = sendPtr[peer * count + offset];
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[rank * count + offset] = value;
    }
  }
}

// shared vectorized+unrolled LSA AlltoAll body (falls back to scalar when the
// data is not vector-aligned). Used by NvlAlltoAllKernelOptimized and by the
// LSA branch of the size-hybrid GinHybridAlltoAllKernel.
template <typename T>
__device__ void AlltoAllVectorizedImpl(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int nRanks, int tid, int nthreads) {
  using TN = typename VectorTypeMapping<T>::Type;
  constexpr int VECTOR_FACTOR = sizeof(TN) / sizeof(T);
  constexpr int UNROLL_FACTOR = 128/sizeof(TN);
  constexpr int PEER_UNROLL = 2;

  T* sendPtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, rank);

  // alignment check: can we use vectorized operations?
  bool canVectorize = (sizeof(TN) > sizeof(T)) &&  // Only if vectorization helps
                      (reinterpret_cast<uintptr_t>(sendPtr) % sizeof(TN) == 0) &&  // Base aligned
                      ((count * sizeof(T)) % sizeof(TN) == 0);  // Stride compatible

  if (canVectorize) {
    size_t vector_count = count / VECTOR_FACTOR;
    int elements_per_iteration = nthreads * UNROLL_FACTOR;

    // process aligned vectorized elements without bounds checks
    size_t aligned_vector_count = (vector_count / elements_per_iteration) * elements_per_iteration;
    for (size_t base_offset = tid; base_offset < aligned_vector_count; base_offset += elements_per_iteration) {
      // unroll a limited number of peers at a time
      for (int peerBase = 0; peerBase < nRanks; peerBase += PEER_UNROLL) {
        int peersInGroup = min(PEER_UNROLL, nRanks - peerBase);

        #pragma unroll
        for (int p = 0; p < peersInGroup; p++) {
          // F2+F3: rotate by rank (cross-rank de-incast) AND by blockIdx.x so
          // different CTAs of this rank hit different peers concurrently, keeping
          // all xGMI egress links busy instead of one at a time. See
          // AlltoAllScalarImpl for the full rationale.
          int peer = (rank + blockIdx.x + peerBase + p) % nRanks;
          TN* sendVecPtr = (TN*)(sendPtr + peer * count);
          TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + rank * count);
          TN values[UNROLL_FACTOR];

          // split load/store into separate loops for better overlap and ILP
          #pragma unroll
          for (int i = 0; i < UNROLL_FACTOR; i++) {
            size_t offset = base_offset + i * nthreads;
            values[i] = sendVecPtr[offset];
          }
          #pragma unroll
          for (int i = 0; i < UNROLL_FACTOR; i++) {
            size_t offset = base_offset + i * nthreads;
            recvVecPtr[offset] = values[i];
          }
        }
      }
    }

    // handle remaining vectorized elements that didn't fit in aligned chunks
    for (size_t base_offset = aligned_vector_count + tid; base_offset < vector_count; base_offset += nthreads) {
      for (int pp = 0; pp < nRanks; pp++) {
        int peer = (rank + blockIdx.x + pp) % nRanks;  // F2+F3: rotated + per-CTA phase
        TN* sendVecPtr = (TN*)(sendPtr + peer * count);
        TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + rank * count);
        recvVecPtr[base_offset] = sendVecPtr[base_offset];
      }
    }

    // handle any remaining elements not divisible by vectorization factor
    size_t scalar_start = vector_count * VECTOR_FACTOR;
    for (size_t offset = scalar_start + tid; offset < count; offset += nthreads) {
      for (int pp = 0; pp < nRanks; pp++) {
        int peer = (rank + blockIdx.x + pp) % nRanks;  // F2+F3: rotated + per-CTA phase
        T value = sendPtr[peer * count + offset];
        T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
        recvPtr[rank * count + offset] = value;
      }
    }
  } else {
    // simple scalar fallback for unaligned data (identical to simple kernel)
    AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);
  }
}

// Device implementation #1 - simple NVL kernel
template <typename T>
__global__ void NvlAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  int rank = devComm.rank, nRanks = devComm.nRanks;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;

  AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

// Device implementation #2 - optimized NVL kernel using vectorization and unrolling
template <typename T>
__global__ void NvlAlltoAllKernelOptimized(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  int rank = devComm.rank, nRanks = devComm.nRanks;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;

  AlltoAllVectorizedImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
__device__ size_t AlltoAllGetSdmaThreshold(struct ncclDevComm const& devComm) {
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

// LL (low-latency) AllToAll for tiny messages, MULTI-CTA. Unlike AllGather
// (one chunk broadcast to all peers), AllToAll scatters a distinct chunk to
// each peer, so this uses the LL A2A session's point-to-point send(): rank myR
// sends its chunk destined for peer p into peer p's epoch-tagged scratch at the
// slot region [myR*chunkU64 ..], i.e. keyed by the *source* rank. Each rank then
// polls all nRanks source regions out of its own scratch and writes them to its
// local recvbuff at [s*chunkU64 ..]. There is NO cross-rank recvbuff write and
// NO barrier: cross-rank traffic is confined to the LL scratch and ordered
// purely by the per-slot epoch tag (ncclLLA2ASession), so it is immune to the
// initData recvbuff-memset race that a barrier-free direct-LSA copy would suffer.
//
// Multi-CTA: the single-CTA LL scatter/gather can't drive the xGMI inbox writes
// fast enough above ~32 KiB total, so each CTA (blockIdx.x, gridDim.x total)
// owns its own scratch block and processes the contiguous chunk slice
// [cta*slice .. ) of every peer's chunk. The slot index keeps the *global*
// element offset (myR*chunkU64 + e), so distinct blocks address disjoint slots
// within their (over-provisioned) block and the layout matches the single-CTA
// path exactly. Sender CTA c and receiver CTA c derive the same slice from
// blockIdx.x, so the epoch/flag handshake stays consistent per block.
// Requires 8-byte-aligned per-peer chunks (guaranteed by the test's
// 16/eltSize base alignment).
__device__ void AlltoAllLLImpl(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset,
                               size_t chunkU64, struct ncclDevComm const& devComm, ncclTeam lsa,
                               ncclLLA2AHandle llHandle) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int nR = lsa.nRanks;
  const int myR = lsa.rank;
  const int cta = blockIdx.x;
  const int nCtas = gridDim.x;
  const uint64_t* src = (const uint64_t*)ncclGetLocalPointer(sendwin, sendoffset);
  uint64_t* dst = (uint64_t*)ncclGetLocalPointer(recvwin, recvoffset);

  // This CTA's contiguous slice of every peer's per-peer chunk.
  const size_t slice = (chunkU64 + (size_t)nCtas - 1) / (size_t)nCtas;
  const size_t myStart = (size_t)cta * slice;
  const size_t myLen = (myStart >= chunkU64) ? 0
                       : ((chunkU64 - myStart < slice) ? (chunkU64 - myStart) : slice);

  ncclLLA2ASession<ncclCoopCta> ll { ncclCoopCta(), devComm, lsa, llHandle, /*block=*/(uint32_t)cta,
                                     /*maxElts=*/(int)((size_t)nR * chunkU64) };

  // Scatter my slice to peer p, keyed by my (source) rank + global element index.
  for (int p = 0; p < nR; p++) {
    for (size_t j = tid; j < myLen; j += nthreads) {
      const size_t e = myStart + j;
      ll.send(p, (int)((size_t)myR * chunkU64 + e), src[(size_t)p * chunkU64 + e]);
    }
  }
  // Gather every source rank's slice-for-me out of my scratch block into recvbuff.
  for (int s = 0; s < nR; s++) {
    for (size_t j = tid; j < myLen; j += nthreads) {
      const size_t e = myStart + j;
      dst[(size_t)s * chunkU64 + e] = ll.recv<uint64_t>((int)((size_t)s * chunkU64 + e));
    }
  }
  ll.endEpoch(ncclCoopCta());
}

// Single-node size-hybrid AlltoAll (-D 3):
//   per-peer chunkBytes <= LL cap (opt-in):       LL packed data+flag, single CTA (tiny, no barrier).
//   per-peer chunkBytes <= sdmaThreshold:         direct LSA all-peers copy (all CTAs),
//                                                 latency-optimal for small messages.
//   per-peer chunkBytes >  sdmaThreshold:         all-peers GIN puts (SDMA copy engine),
//                                                 bandwidth-optimal for large messages.
// The LL tier is OFF unless NCCL_GIN_ANVIL_A2A_LL_MAX_BYTES>0 (llHandle.nSlots
// stays 0 otherwise); on 8x MI355X it did not beat the direct-LSA copy.
//
// sdmaThresholdOverride lets the AlltoAll-specific env var
// NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALL tune this LSA<->SDMA cutover
// independently; TEST_SDMA_THRESHOLD_UNSET falls back to the shared backend
// value (rsCtx->sdmaThreshold from NCCL_GIN_ANVIL_SDMA_THRESHOLD).
template <typename T>
__global__ void GinHybridAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, ncclLLA2AHandle llHandle, int a2aSyncMode, ncclDevResourceHandle_t a2aFlagBuf) {
  const size_t size = count * sizeof(T);  // per-peer chunk bytes
  const size_t sdmaThreshold = (sdmaThresholdOverride != TEST_SDMA_THRESHOLD_UNSET)
                                   ? sdmaThresholdOverride
                                   : AlltoAllGetSdmaThreshold(devComm);

  if (size <= sdmaThreshold) {
    ncclTeam lsa = ncclTeamLsa(devComm);

    // Tiny messages: LL packed data+flag path (single CTA). Barrier-free and
    // immune to the recvbuff-memset race (cross-rank traffic stays in the
    // epoch-tagged LL scratch; only local recvbuff is written). Used when LL is
    // configured (nSlots>0), the per-peer chunk is 8-byte aligned, and it fits
    // the pre-sized slot count.
    if (gin_sdma::a2aLLEligible(size, llHandle.nSlots, devComm.nRanks, gin_sdma::kAllToAllLLMaxBytes)) {
      // Multi-CTA: every launched CTA participates (one scratch block each).
      const size_t chunkU64 = size / 8;
      AlltoAllLLImpl(sendwin, sendoffset, recvwin, recvoffset, chunkU64, devComm, lsa, llHandle);
      return;
    }

    /* small messages: direct LSA all-peers copy (all CTAs). a2aSyncMode selects
     * the cross-rank sync:
     *   3 = DEFAULT: single exit barrier only. The exit barrier both completes the
     *       call (all ranks done writing before any returns) and guards the *next*
     *       call's writes into this recvbuf, so a separate entry barrier is
     *       redundant; the first call's memset race is covered by the one-time
     *       connect/init sync. +9-17% busbw across the LSA band vs mode 0.
     *   0 = the two LSA barriers (legacy: redundant entry barrier + exit barrier).
     *   1 = NONE (diagnostic: barrier-free ceiling of the 1-pass copy; correct
     *       only under the harness's external per-iteration rank sync).
     *   2 = per-call point-to-point done-flag completion. Each CTA writes its
     *       slice, fences (release, system scope), signals every peer one done
     *       flag, then waits for all sources -> me. Correct, but measured ~= the
     *       two-barrier cost: a 1-pass copy's completion is inherently a
     *       cross-rank sync (barrier-free completion would need in-band flags,
     *       i.e. a second pass). Kept as a documented diagnostic. */
    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    if (a2aSyncMode == 0) lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    const int rank = devComm.rank, nRanks = devComm.nRanks;
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const int nthreads = blockDim.x * gridDim.x;

    // Mode 2: read the per-CTA persistent epoch (raw+1 so epoch>=1 never matches
    // a zero-initialized flag slot) and pick the double-buffer slot.
    __shared__ uint64_t s_a2aEpoch;
    uint64_t a2aEpoch = 0;
    int a2aSlot = 0;
    uint64_t* a2aMyFlags = nullptr;
    if (a2aSyncMode == 2) {
      a2aMyFlags = (uint64_t*)ncclGetResourceBufferLocalPointer(devComm, a2aFlagBuf);
      if (threadIdx.x == 0) s_a2aEpoch = a2aMyFlags[blockIdx.x] + 1;
      __syncthreads();
      a2aEpoch = s_a2aEpoch;
      a2aSlot = (int)(a2aEpoch % (uint64_t)gin_sdma::kA2aFlagSlots);
    }

    AlltoAllVectorizedImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);

    if (a2aSyncMode == 0 || a2aSyncMode == 3) {
      // mode 3: exit barrier only (skip entry). One optimized team barrier per
      // call provides both completion and the *next* call's readiness guard; the
      // very first call's memset race is covered by the harness's initial sync.
      lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    } else if (a2aSyncMode == 2) {
      // Publish this CTA's writes to system scope before signaling any done flag:
      // syncthreads (all block writes complete) -> per-thread system fence ->
      // syncthreads (all fences retired) so no done flag is stored before every
      // thread's recvbuf writes are globally visible.
      __syncthreads();
      __threadfence_system();
      __syncthreads();
      const int maxCtas = gin_sdma::kA2aLsaMaxCtas;
      const int SLOTS = gin_sdma::kA2aFlagSlots;
      const size_t doneBase = (size_t)maxCtas;
      // One lane per peer/source: lane l signals "I am done writing peer l" into
      // peer l's inbox, then waits for "source l is done writing me" in my inbox.
      if (threadIdx.x < nRanks) {
        const int l = threadIdx.x;
        uint64_t* peerFlags = (uint64_t*)ncclGetResourceBufferPeerPointer(devComm, a2aFlagBuf, lsa, l);
        a2aFlagStore(peerFlags + doneBase + ((size_t)blockIdx.x * nRanks + rank) * SLOTS + a2aSlot, a2aEpoch);
        uint64_t* mine = a2aMyFlags + doneBase + ((size_t)blockIdx.x * nRanks + l) * SLOTS + a2aSlot;
        while (a2aFlagLoad(mine) < a2aEpoch) { /* spin */ }
      }
      __syncthreads();
      if (threadIdx.x == 0) a2aMyFlags[blockIdx.x] = a2aEpoch;  // persist for next call
    }
    return;
  }

  /* large messages: all-peers GIN puts (SDMA copy engine) */
  int ginContext = 0;
  unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  //TODO: this contains a cross-node barrier with all ranks, essentially doubling latency
  //      is it however a valid requirement that we do not start writting to the dest buffer before
  //      the remote is ready, which is what this barrier achieves, need to think if a better alltoall
  //      could avoid this requirement (shmem does not require it).
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  /* send to all peers via GIN. Chunked to <=1 GiB segments to avoid the
     30-bit SDMA copy-count overflow on >1 GiB per-peer chunks; the signal
     rides the final segment. */
  for (int r=tid; r<devComm.nRanks; r+=nthreads) {
    ginPutChunked(gin, ncclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());

  //TODO: this fence presumed redundant because: RDMA dest buffer visible after waitsignal; remote done writting after waitSignal; local done writting after flush, so we are already peerwise quiet with all peers, no need for a secondary barrier to enforce it.
  //bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

// Hybrid LSA+GIN alltoall: CTA 0 handles remote peers via GIN,
// CTAs 1..N handle intra-node peers via LSA.
// GIN barrier is scoped to CTA 0 only (barrierCount=1), costing
// O(nRanks) signals once, not O(nCTAs x nRanks).
// LSA CTAs use their own lsaBarrier (pure intra-node, no GIN signals).
template <typename T>
__global__ void HybridAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa = ncclTeamLsa(devComm);
  const int startLsa = world.rank - lsa.rank;
  const int lsaSize  = lsa.nRanks;
  const size_t size = count * sizeof(T);
  int numRemotePeers = world.nRanks - lsa.nRanks;

  if (blockIdx.x == 0) {
    /* CTA 0: remote peers via GIN */
    int ginContext = 0;
    unsigned int signalIndex = 0;
    ncclGin gin { devComm, ginContext };
    uint64_t signalValue = gin.readSignal(signalIndex);

    ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, 0 };
    //TODO: this contains a cross-node barrier with all ranks, essentially doubling latency
    //      is it however a valid requirement that we do not start writting to the dest buffer before
    //      the remote is ready, which is what this barrier achieves, need to think if a better alltoall
    //      could avoid this requirement (shmem does not require it).
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Chunked to <=1 GiB segments to avoid the 30-bit SDMA copy-count overflow
    // on >1 GiB per-peer chunks; the signal rides the final segment.
    for (int r = tid; r < startLsa; r += nthreads) {
      ginPutChunked(gin, world, r,
          recvwin, recvoffset + world.rank * size,
          sendwin, sendoffset + r * size,
          size, ncclGin_SignalInc{signalIndex});
    }
    for (int r = startLsa + lsaSize + tid; r < world.nRanks; r += nthreads) {
      ginPutChunked(gin, world, r,
          recvwin, recvoffset + world.rank * size,
          sendwin, sendoffset + r * size,
          size, ncclGin_SignalInc{signalIndex});
    }

    if (numRemotePeers > 0) {
      gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + numRemotePeers);
    }
    gin.flush(ncclCoopCta());

    //TODO: this fence presumed redundant because: RDMA dest buffer visible after waitsignal; remote done writting after waitSignal; local done writting after flush, so we are already peerwise quiet with all peers, no need for a secondary barrier to enforce it.
    //bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
  } else {
    /* CTAs 1..N: local peers via LSA */
    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x - 1 };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    int tid = threadIdx.x + (blockIdx.x - 1) * blockDim.x;
    int nthreads = blockDim.x * (gridDim.x - 1);

    T* sendLocal = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    for (size_t offset = tid; offset < count; offset += nthreads) {
      for (int lp = 0; lp < lsa.nRanks; lp++) {
        int wr = startLsa + lp;
        T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp);
        recvPtr[world.rank * count + offset] = sendLocal[wr * count + offset];
      }
    }

    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
  }
}
#endif
#endif

testResult_t AlltoAllRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
    if (test_ncclVersion >= NCCL_VERSION(2,28,0)) {
      NCCLCHECK(ncclAlltoAll(sptr, rptr, count, type, comm, stream));
      return testSuccess;
    }
    if (test_ncclVersion >= NCCL_VERSION(2,19,0)) {
      NCCLCHECK(ncclAllToAll(sptr, rptr, count, type, comm, stream));
      return testSuccess;
    }
    printf("RCCL 2.19 or later is needed for alltoall API path. This test was compiled with %d.%d, but is running with RCCL %d.\n",
           NCCL_MAJOR, NCCL_MINOR, test_ncclVersion);
    return testNcclError;
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,7,0)
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = count * wordSize(type);
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(ncclSend(sptr+r*rankOffset, count, type, r, comm, stream));
      NCCLCHECK(ncclRecv(rptr+r*rankOffset, count, type, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
#else
    printf("NCCL 2.7 or later is needed for alltoall. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#endif
  } else {
    switch(deviceImpl) {
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
      case 1:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(NvlAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
      case 2:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(NvlAlltoAllKernelOptimized, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3: {
        // AlltoAll-specific LSA<->SDMA threshold. Compared against the per-peer
        // chunk (count*sizeof(T) = total/nRanks). Default = 2 MiB/peer: on 8x
        // MI355X (NCCL_GIN_TYPE=6, F1 adaptive-CTA + F2 rotated + F3 per-CTA peer
        // phase) the direct LSA copy wins through 2 MiB/peer (16 MiB total, 251 vs
        // 244 GB/s) and GIN/SDMA wins from 4 MiB/peer (32 MiB total: 311 vs 265,
        // scaling to ~426) (measured 2026-07-31). Override with
        // NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALL, or the shared
        // NCCL_GIN_ANVIL_SDMA_THRESHOLD.
        static const size_t a2aThr = testResolveSdmaThreshold("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALL", gin_sdma::kAllToAllSdmaThresholdDefault);
        // F1: pick the grid (CTA count) per call from the per-peer chunk. The host
        // computes the same tier the kernel will take (LL / LSA / GIN-SDMA) from
        // the same threshold, then sizes the grid: LSA scales CTAs with the chunk
        // (a2aLsaCtaCount) to match host-RING channel parallelism; the SDMA tier
        // uses a small fixed grid (only nRanks threads issue puts); LL is 1 CTA.
        const size_t perPeerBytes = count * wordSize(type);
        // NOTE: for device impls `comm` is really a ncclDevComm* (see the launch
        // wrappers, which cast it). Read nRanks straight off it -- calling the
        // host ncclCommCount() on a devComm pointer fails with "invalid argument".
        int nRanks = ((ncclDevComm*)comm)->nRanks;
#if defined(A2A_HAVE_LL)
        gin_sdma::A2ATier tier = gin_sdma::a2aKernelTier(perPeerBytes, a2aThr, g_a2aLLHandle.nSlots, nRanks, gin_sdma::kAllToAllLLMaxBytes);
        int gridCtas = (tier == gin_sdma::A2ATier::Gin)
                           ? gin_sdma::kA2aSdmaCtas
                           : (tier == gin_sdma::A2ATier::LL)
                                 ? gin_sdma::kA2aLLCtas
                                 : gin_sdma::a2aLsaCtaCount(perPeerBytes, gin_sdma::kA2aLsaMaxCtas);
        // Tuning override for the LSA tier grid (F1): NCCL_GIN_ANVIL_A2A_LSA_CTAS,
        // if >0, forces the LSA-tier CTA count (clamped to kA2aLsaMaxCtas) so the
        // adaptive ladder can be swept without a rebuild. Does not affect the SDMA
        // or LL tiers. Unset/<=0 keeps the a2aLsaCtaCount() ladder.
        if (tier == gin_sdma::A2ATier::LSA) {
          static const int lsaCtaOverride = []() {
            const char* e = getenv("NCCL_GIN_ANVIL_A2A_LSA_CTAS");
            return (e && *e) ? atoi(e) : 0;
          }();
          if (lsaCtaOverride > 0)
            gridCtas = lsaCtaOverride < gin_sdma::kA2aLsaMaxCtas ? lsaCtaOverride : gin_sdma::kA2aLsaMaxCtas;
        }
        // Prototype knob: sweep the multi-CTA LL grid without a rebuild (clamped
        // to kA2aLLCtas, the number of scratch blocks allocated at init).
        if (tier == gin_sdma::A2ATier::LL) {
          static const int llCtaOverride = []() {
            const char* e = getenv("NCCL_GIN_ANVIL_A2A_LL_CTAS");
            return (e && *e) ? atoi(e) : 0;
          }();
          if (llCtaOverride > 0)
            gridCtas = llCtaOverride < gin_sdma::kA2aLLCtas ? llCtaOverride : gin_sdma::kA2aLLCtas;
        }
        // LSA-tier cross-rank sync mode (NCCL_GIN_ANVIL_A2A_SYNC_MODE):
        //   3 (DEFAULT) = single exit barrier (the exit barrier both completes the
        //       call and guards the next call's writes; the first call's memset
        //       race is covered by the one-time connect/init sync). +9-17% busbw
        //       across the LSA band vs the legacy two-barrier path, #wrong==0.
        //   0 = two LSA barriers (legacy: redundant entry barrier + exit barrier).
        //   1 = none (diagnostic ceiling: barrier-free 1-pass copy; correct only
        //       under an external per-iteration sync -- beats host by 10-21%).
        //   2 = point-to-point done-flag completion (diagnostic: a correct per-call
        //       barrier-free-completion attempt; measured ~= the two-barrier cost,
        //       since a 1-pass copy's completion is inherently a cross-rank sync).
        static const int a2aSyncMode = []() {
          const char* e = getenv("NCCL_GIN_ANVIL_A2A_SYNC_MODE");
          return (e && *e) ? atoi(e) : 3;
        }();
        TESTCHECK(testLaunchDeviceKernelThresholdLLCtasFlag(SPECIALIZE_KERNEL(GinHybridAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, a2aThr, g_a2aLLHandle, gridCtas, a2aSyncMode, g_a2aFlagHandle));
#else
        TESTCHECK(testLaunchDeviceKernelThreshold(SPECIALIZE_KERNEL(GinHybridAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, a2aThr));
#endif
        return testSuccess;
      }
      case 4:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(HybridAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

struct testColl alltoAllTest = {
  "AlltoAll",
  AlltoAllGetCollByteCount,
  AlltoAllInitData,
  AlltoAllGetBw,
  AlltoAllRunColl,
  NULL,
  NULL
};

void AlltoAllGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AlltoAllRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &alltoAllTest;
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
  .getBuffSize = AlltoAllGetBuffSize,
  .runTest = AlltoAllRunTest,
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = AlltoAllGetDevCommRequirements
#endif
};
