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

// Single-node hybrid Scatter (-D 3): the root distributes a distinct per-rank
// chunk r (elements [r*count .. (r+1)*count) of its send buffer) to rank r.
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
__global__ void GinScatterKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride) {
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
    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    if (rank == root) {
      const T* src = (const T*)ncclGetLocalPointer(sendwin, sendoffset);
      for (int r = 0; r < nRanks; r++) {
        if (inPlace && r == root) continue;  // own chunk already in place
        const size_t dstOff = inPlace ? (dstBase + (size_t)r * chunkBytes) : recvoffset;
        T* dst = (T*)ncclGetLsaPointer(recvwin, dstOff, r);
        const T* s = src + (size_t)r * count;
        for (size_t i = tid; i < count; i += nthreads) dst[i] = s[i];
      }
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
    // Flat scatter: one put per non-self peer (issued once, by the thread tid=r).
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
        TESTCHECK(testLaunchDeviceKernelThreshold(SPECIALIZE_KERNEL(GinScatterKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, scThr));
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
