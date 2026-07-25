/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
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

void BroadcastGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t BroadcastInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    if (rank == root) TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, rep, 1, 0));
    TESTCHECK(InitData(args->expected[i], recvcount, 0, type, ncclSum, rep, 1, 0));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

testResult_t  BroadcastGetAlgoProtoChannels(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetAlgoInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetAlgoInfo(comm, ncclFuncBroadcast , count, type , 0, 0, 1, algo, proto, nchannels));
  return testSuccess;
}

void BroadcastGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t BroadcastGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;

  switch(deviceImpl) {
    case 3: { // GinHybridBroadcastKernel: LSA direct (small) + root GIN puts (large)
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
bool BroadcastGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: // GinHybridBroadcastKernel: LSA direct (small) + root GIN puts (large)
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
__device__ size_t BroadcastGetSdmaThreshold(struct ncclDevComm const& devComm) {
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

// Local send->recv copy on the root (out-of-place). Uses 16-byte vector stores
// when both buffers and the byte count are 16-byte aligned, else scalar.
template <typename T>
__device__ void BroadcastLocalCopy(T* dst, const T* src, size_t count, int tid, int nthreads) {
  const size_t bytes = count * sizeof(T);
  const uintptr_t da = (uintptr_t)dst;
  const uintptr_t sa = (uintptr_t)src;
  if ((bytes % sizeof(uint4)) == 0 && (da % sizeof(uint4)) == 0 && (sa % sizeof(uint4)) == 0) {
    uint4* d4 = (uint4*)dst;
    const uint4* s4 = (const uint4*)src;
    const size_t n4 = bytes / sizeof(uint4);
    for (size_t i = tid; i < n4; i += nthreads) d4[i] = s4[i];
  } else {
    for (size_t i = tid; i < count; i += nthreads) dst[i] = src[i];
  }
}

// Root reads one send slice and writes the same recv offset on every LSA peer
// (including itself, which also covers the out-of-place self copy).
template <typename T>
__device__ void BroadcastLsaDirect(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int nRanks, int tid, int nthreads) {
  T* src = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  for (size_t i = tid; i < count; i += nthreads) {
    T value = src[i];
    for (int lp = 0; lp < nRanks; lp++) {
      T* dst = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp);
      dst[i] = value;
    }
  }
}

// Single-node hybrid Broadcast (-D 3): flat / star fan-out from root.
//   msgBytes <= sdmaThreshold: root writes every peer's recvbuff via LSA.
//   msgBytes >  sdmaThreshold: root issues one GIN put per non-self peer
//                              (Anvil picks IPC or SDMA by size).
// Non-root ranks only participate in the entry/exit barriers.
template <typename T>
__global__ void GinHybridBroadcastKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  const size_t msgBytes = count * sizeof(T);
  const size_t sdmaThreshold = BroadcastGetSdmaThreshold(devComm);
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;

  if (msgBytes <= sdmaThreshold) {
    ncclTeam lsa = ncclTeamLsa(devComm);
    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    if (devComm.rank == root) {
      BroadcastLsaDirect<T>(sendwin, sendoffset, recvwin, recvoffset, count, lsa.nRanks, tid, nthreads);
    }
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  const int ginContext = 0;
  const unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  // ncclGin_SignalInc is a *remote* action: each put increments the receiving
  // peer's signal (confirmed vs AllGather/AlltoAll, where a rank receives N puts
  // and waits base+N). So the root -- which receives no puts -- must not wait on
  // its own signal; instead every non-root receives exactly one put and waits
  // base+1. flush() alone does not guarantee remote data has settled, so this
  // receiver-side waitSignal is what makes the payload visible on non-roots.
  const uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  // Entry barrier: every rank's recv buffer quiescent before root starts writing.
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  if (devComm.rank == root) {
    // Out-of-place self copy (no-op when in-place: local src == local dst).
    T* lsrc = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* ldst = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    if (lsrc != ldst) {
      BroadcastLocalCopy<T>(ldst, lsrc, count, tid, nthreads);
    }

    // Flat fan-out: one put per non-self peer; skip self.
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
      if (r == root) continue;
      gin.put(ncclTeamWorld(devComm), r,
          recvwin, recvoffset,
          sendwin, sendoffset,
          msgBytes, ncclGin_SignalInc{signalIndex});
    }

    // flush(): all source buffers safe to reuse once the puts are pushed.
    gin.flush(ncclCoopCta());
  } else {
    // Each non-root receives exactly one put from root; wait for it to settle.
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + 1);
  }
}
#endif
#endif

testResult_t BroadcastRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (deviceImpl == 0) {
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));

    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_MAJOR >= 2 && NCCL_MINOR >= 2
    NCCLCHECK(ncclBroadcast(sptr, rptr, count, type, root, comm, stream));
#else
    if (rank == root) {
      NCCLCHECK(ncclBcast(sptr, count, type, root, comm, stream));
    } else {
      NCCLCHECK(ncclBcast(rptr, count, type, root, comm, stream));
    }
#endif
  } else {
    switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(GinHybridBroadcastKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

struct testColl broadcastTest = {
  "Broadcast",
  BroadcastGetCollByteCount,
  BroadcastInitData,
  BroadcastGetBw,
  BroadcastRunColl,
  BroadcastGetAlgoProtoChannels,
  NULL
};

void BroadcastGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  BroadcastGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t BroadcastRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &broadcastTest;
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
  .getBuffSize = BroadcastGetBuffSize,
  .runTest = BroadcastRunTest,
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = BroadcastGetDevCommRequirements
#endif
};
