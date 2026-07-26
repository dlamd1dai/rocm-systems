
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

void AllGatherGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  size_t base = (count/nranks) & -(16/eltSize);
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
static const size_t ALLGATHER_LSA_SINGLE_CTA_MAX = 8192;

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

// Single-node hybrid AllGather (-D 3):
//   chunkBytes <= ALLGATHER_LSA_SINGLE_CTA_MAX: LSA on a single CTA (tiny, barrier-bound).
//   chunkBytes <= sdmaThreshold:                vectorized LSA (all CTAs).
//   chunkBytes >  sdmaThreshold:                direct all-peers GIN puts (proven MI355X path).
//
// sdmaThresholdOverride lets the AllGather-specific env var
// NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER tune this LSA<->GIN cutover
// independently; TEST_SDMA_THRESHOLD_UNSET falls back to the shared backend
// value (rsCtx->sdmaThreshold from NCCL_GIN_ANVIL_SDMA_THRESHOLD).
template <typename T>
__global__ void GinHybridAllGatherKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride) {
  const size_t chunkBytes = count * sizeof(T);
  const size_t sdmaThreshold = (sdmaThresholdOverride != TEST_SDMA_THRESHOLD_UNSET)
                                   ? sdmaThresholdOverride
                                   : AllGatherGetSdmaThreshold(devComm);

  if (chunkBytes <= sdmaThreshold) {
    ncclTeam lsa = ncclTeamLsa(devComm);

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

  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + (size_t)devComm.rank * chunkBytes,
        sendwin, sendoffset,
        chunkBytes, ncclGin_SignalInc{signalIndex});
  }

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());
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
        static const size_t agThr = testResolveSdmaThreshold("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER", (size_t)262144);
        TESTCHECK(testLaunchDeviceKernelThreshold(SPECIALIZE_KERNEL(GinHybridAllGatherKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, agThr));
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

struct testColl allGatherTest = {
  "AllGather",
  AllGatherGetCollByteCount,
  AllGatherInitData,
  AllGatherGetBw,
  AllGatherRunColl,
  AllGatherGetAlgoProtoChannels,
  AllGatherGetSymkInfo
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
