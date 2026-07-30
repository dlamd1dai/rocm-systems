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
#if HAVE_FP8
#include "rccl_float8.h"
#endif
#include "gin_sdma_reduce.h"  // device Apply<op,T>, mirrors verifiable.cu exactly

// ReduceScatter (-D 3): the first reduction collective. Two size tiers keyed on
// the per-rank output-slice bytes (chunk = count*sizeof(T)):
//   * chunk <= threshold: LSA read-reduce. Every rank reads its owned slice
//     [rank*count] from EVERY peer's sendbuff (ncclGetLsaPointer), folds with
//     gin_sdma_reduce (ascending source-rank order, matching the verifier), and
//     writes its local recvbuff. Balanced egress, no scratch, no signals -- entry
//     + exit LSA barrier only.
//   * chunk >  threshold: put-partials + SM reduce. Each rank gin.puts its slice
//     p (its contribution to peer p's output) into peer p's scratch window at
//     slot [rank*chunk] (SignalInc{0}); its own slice is copied locally into its
//     own scratch slot [rank*chunk]. After waitSignal(base + N-1) each rank
//     SM-reduces the N staged contributions into recvbuff. Balanced egress
//     across all N ranks (the ReduceScatter roofline).
// The reduction op is passed at launch and switched on at runtime by
// gin_sdma_reduce; the kernel template varies only on T. Scratch is a generic
// resource-buffer window (ncclGetResourceBuffer*), sized once at the worst-case
// message in ReduceScatterGetDevCommRequirements. PreMulSum/mulsum is deferred
// (SPECIALIZE_REDUCE_KERNEL returns nullptr -> testNotImplemented); fp8 prod is
// excluded there and by the ReduceScatterRunTest skip.
static ncclDevResourceHandle g_rsScratchHandle = 0;      // assigned in ncclDevCommCreate
static ncclDevResourceRequirements g_rsScratchReq = {};
#endif

void ReduceScatterGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  size_t base = (count/nranks) & -(16/eltSize);
  *sendcount = base*nranks;
  *recvcount = base;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = base;
  *paramcount = base;
}

testResult_t ReduceScatterInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, op, rep, nranks, rank));
    CUDACHECK(cudaMemcpy(args->expected[i], args->recvbuffs[i], args->expectedBytes, cudaMemcpyDefault));
    TESTCHECK(InitDataReduce(args->expected[i], recvcount, rank*recvcount, type, op, rep, nranks));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

testResult_t  ReduceScatterGetAlgoProtoChannels(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetAlgoInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetAlgoInfo(comm, ncclFuncReduceScatter , count, type , 0, 0, 1, algo, proto, nchannels));
  return testSuccess;
}

testResult_t  ReduceScatterGetSymkInfo(ncclComm_t comm, size_t count, ncclDataType_t type, ncclRedOp_t op, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetSymkInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetSymkInfo(comm, ncclFuncReduceScatter , count, type , op, algo, proto, nchannels));
  return testSuccess;
}

void ReduceScatterGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1))/((double)nranks);
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t ReduceScatterGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;
  switch(deviceImpl) {
    case 3: { // GinReduceScatterKernel: LSA read-reduce (small) + put-partials + SM reduce (large)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      gin_sdma::DevReqs dr = gin_sdma::reduceScatterDevReqs(deviceCtaCount);
      reqs->barrierCount = dr.barrierCount;
      reqs->lsaBarrierCount = dr.lsaBarrierCount;
      reqs->ginSignalCount = dr.ginSignalCount;
      // Scratch window for the large put-partials tier: each rank stages N
      // per-source partials, each up to the largest per-rank slice, so it needs
      // the full per-rank send-buffer worth (N*maxChunk == maxSendBytesPerRank).
      // maxBytes (the run's -e ceiling) upper-bounds the per-rank send bytes for
      // ReduceScatter (sendBytes ~= maxBytes), so size the scratch from it once.
      size_t scratchBytes = gin_sdma::reduceScatterScratchBytes(maxBytes);
      if (scratchBytes > 0) {
        memset(&g_rsScratchReq, 0, sizeof(g_rsScratchReq));
        g_rsScratchReq.bufferSize = scratchBytes;
        g_rsScratchReq.bufferAlign = 128;
        g_rsScratchReq.outBufferHandle = &g_rsScratchHandle;
        g_rsScratchReq.next = reqs->resourceRequirementsList;
        reqs->resourceRequirementsList = &g_rsScratchReq;
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
bool ReduceScatterGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));
  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: { // barriers only; the large-tier scratch needs the 2.29 requirements form
      gin_sdma::DevReqs dr = gin_sdma::reduceScatterDevReqs(deviceCtaCount);
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
__device__ size_t ReduceScatterGetSdmaThreshold(struct ncclDevComm const& devComm) {
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

// Single-node hybrid ReduceScatter (-D 3). count is the per-rank output-slice
// element count; the send buffer holds nRanks such slices ([p*count]).
template <typename T>
__global__ void GinReduceScatterKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, int redOp, ncclDevResourceHandle scratchHandle) {
  const size_t chunkBytes = count * sizeof(T);
  const size_t sdmaThreshold = (sdmaThresholdOverride != TEST_SDMA_THRESHOLD_UNSET)
                                   ? sdmaThresholdOverride
                                   : ReduceScatterGetSdmaThreshold(devComm);
  const int rank = devComm.rank, nRanks = devComm.nRanks;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;

  if (chunkBytes <= sdmaThreshold) {
    // ---- small/med: LSA read-reduce (balanced pull, no scratch/signals) ----
    ncclTeam lsa = ncclTeamLsa(devComm);
    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    T* dst = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    const size_t myBase = (size_t)rank * count;  // my owned slice within each peer's sendbuff
    for (size_t i = tid; i < count; i += nthreads) {
      // Register accumulator (written to recvbuff once) is in-place safe: our own
      // input slice [rank*count] is read before recvbuff is overwritten. Folded
      // in ascending source-rank order to match the verifier bit-for-bit.
      const T* s0 = (const T*)ncclGetLsaPointer(sendwin, sendoffset, 0);
      T acc = gin_sdma_reduce::preOp(redOp, s0[myBase + i], nRanks);
      for (int s = 1; s < nRanks; s++) {
        const T* sp = (const T*)ncclGetLsaPointer(sendwin, sendoffset, s);
        acc = gin_sdma_reduce::combine(redOp, acc, gin_sdma_reduce::preOp(redOp, sp[myBase + i], nRanks));
      }
      dst[i] = gin_sdma_reduce::postOp(redOp, acc, nRanks);
    }

    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  // ---- large: put-partials to peer scratch + SM reduce ----
  // Mirrors the proven all-to-all GIN sequence (alltoall.cu): put to EVERY rank
  // (self included, via GIN loopback), then waitSignal(base + nRanks), then flush.
  // Putting my own slice through GIN too (instead of a separate SM memcpy) keeps
  // the signal count uniform and avoids an SM-write / SDMA-write ordering race on
  // the scratch buffer that intermittently deadlocked the earlier flush-then-wait
  // + local-copy variant. In-place is safe: my own slice is staged into scratch
  // before recvbuff is overwritten by the reduce below.
  const unsigned int signalIndex = 0;
  ncclGin gin { devComm, /*context=*/0 };
  const uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  const size_t scratchByteBase = ncclGetResourceBufferOffset(scratchHandle);
  T* scratch = (T*)ncclGetResourceBufferLocalPointer(devComm, scratchHandle);

  // Stage my contribution to peer p's output (my send slice p) into peer p's
  // scratch slot [rank*chunk]. p == rank is a local GIN loopback into my own slot.
  ncclTeam world = ncclTeamWorld(devComm);
  for (int p = tid; p < nRanks; p += nthreads) {
    ginPutChunked(gin, world, p,
        devComm.resourceWindow, scratchByteBase + (size_t)rank * chunkBytes,
        sendwin, sendoffset + (size_t)p * chunkBytes,
        chunkBytes, ncclGin_SignalInc{signalIndex});
  }
  // Receive exactly one put from each of the N ranks (N-1 peers + my loopback).
  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + (uint64_t)nRanks);
  gin.flush(ncclCoopCta());

  // SM reduce: fold the N staged contributions (scratch slot [s*count]) into
  // recvbuff, ascending source-rank order (matches the verifier).
  T* dst = (T*)ncclGetLocalPointer(recvwin, recvoffset);
  for (size_t i = tid; i < count; i += nthreads) {
    T acc = gin_sdma_reduce::preOp(redOp, scratch[i], nRanks);  // s == 0
    for (int s = 1; s < nRanks; s++) {
      acc = gin_sdma_reduce::combine(redOp, acc, gin_sdma_reduce::preOp(redOp, scratch[(size_t)s * count + i], nRanks));
    }
    dst[i] = gin_sdma_reduce::postOp(redOp, acc, nRanks);
  }
}
#endif

testResult_t ReduceScatterRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    NCCLCHECK(ncclReduceScatter(sptr, rptr, count, type, op, comm, stream));
  } else {
    switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3: {
        if (count == 0) return testSuccess;
        // ReduceScatter-specific LSA<->GIN threshold (compared against the
        // per-rank output slice = count*sizeof(T)). Default 256 KiB/rank
        // (provisional; retune by A/B sweep per the design plan). Override with
        // NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCESCATTER or the shared
        // NCCL_GIN_ANVIL_SDMA_THRESHOLD.
        static const size_t rsThr = testResolveSdmaThreshold("NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCESCATTER", gin_sdma::kReduceScatterSdmaThresholdDefault);
        TESTCHECK(testLaunchDeviceKernelThresholdScratch(SPECIALIZE_REDUCE_KERNEL(GinReduceScatterKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, rsThr, g_rsScratchHandle));
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

struct testColl reduceScatterTest = {
  "ReduceScatter",
  ReduceScatterGetCollByteCount,
  ReduceScatterInitData,
  ReduceScatterGetBw,
  ReduceScatterRunColl,
  ReduceScatterGetAlgoProtoChannels,
  ReduceScatterGetSymkInfo
};

void ReduceScatterGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  ReduceScatterGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t ReduceScatterRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &reduceScatterTest;
  ncclDataType_t *run_types;
  ncclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if ((int)op != -1) {
    run_ops = &op;
    run_opnames = &opName;
    op_count = 1;
  } else {
    op_count = test_opnum;
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=0; j<op_count; j++) {
      // The GIN device path (deviceImpl != 0) has no PreMulSum ("mulsum") kernel
      // for any type (deferred), and no prod kernel for fp8; SPECIALIZE_REDUCE_KERNEL
      // returns nullptr for those, which would abort the op x type matrix on
      // testNotImplemented. Skip them here so the device sweep only exercises the
      // implemented combos. The host path (deviceImpl == 0) supports them all via
      // ncclReduceScatter, so it keeps full coverage.
      if (deviceImpl != 0) {
        if (strcmp(run_opnames[j], "mulsum") == 0) continue;
#if defined(RCCL_FLOAT8)
        if ((run_types[i] == ncclFloat8e4m3 || run_types[i] == ncclFloat8e5m2) && run_ops[j] == ncclProd)
          continue;
#endif
      }
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1));
    }
  }
  return testSuccess;
}

struct testEngine ncclTestEngine = {
  .getBuffSize = ReduceScatterGetBuffSize,
  .runTest = ReduceScatterRunTest,
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = ReduceScatterGetDevCommRequirements
#endif
};
