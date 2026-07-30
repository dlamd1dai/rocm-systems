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

// ReduceScatter (-D 3): the first reduction collective. Single tier -- balanced
// LSA read-reduce for all sizes. Every rank reads its owned output slice
// [rank*count] from EVERY peer's sendbuff (cached, ncclMemAlloc'd) via
// ncclGetLsaPointer, folds the N contributions with gin_sdma_reduce (ascending
// source-rank order, matching the verifier bit-for-bit) and writes its local
// recvbuff. Balanced egress, no scratch/signals -- entry + exit LSA barrier only.
// Reads are 128-bit packed (see GinReduceScatterKernel) for bandwidth.
//
// An earlier size-hybrid design added a large-tier "put-partials + SM reduce"
// path that staged into the GIN resource window; that window is UNCACHED
// (dev_runtime.cc), so the SM read-back of the staged partials dominated the
// kernel and capped it near the scalar-read ceiling (~95 GB/s). The vectorized
// LSA read of peers' CACHED sendbuffs is ~2.5-3x faster at large sizes and needs
// no scratch, so it now serves every size. The launch still carries the
// (unused) sdmaThreshold/scratch args for ABI stability; scratch is no longer
// registered. PreMulSum/mulsum is deferred (SPECIALIZE_REDUCE_KERNEL returns
// nullptr -> testNotImplemented); fp8 prod is excluded there and by the
// ReduceScatterRunTest skip.
static ncclDevResourceHandle g_rsScratchHandle = 0;  // unused (no scratch); passed to kernel as 0
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
    case 3: { // GinReduceScatterKernel: single-tier LSA read-reduce (no scratch)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      gin_sdma::DevReqs dr = gin_sdma::reduceScatterDevReqs(deviceCtaCount);
      reqs->barrierCount = dr.barrierCount;
      reqs->lsaBarrierCount = dr.lsaBarrierCount;
      reqs->ginSignalCount = dr.ginSignalCount;
      // No resource/scratch window: the LSA read-reduce reads peers' sendbuffs
      // directly, so nothing needs to be staged.
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
    case 3: { // single-tier LSA read-reduce: barriers only, no scratch
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
// Single-node ReduceScatter (-D 3). count is the per-rank output-slice element
// count; the send buffer holds nRanks such slices ([p*count]).
// Single tier: balanced LSA read-reduce. Each rank reads its owned output slice
// [rank*count] from EVERY peer's sendbuff (cached, ncclMemAlloc'd) via
// ncclGetLsaPointer, folds the N contributions in ascending source-rank order
// (matching verifiable.cu bit-for-bit via gin_sdma_reduce), and writes its local
// recvbuff. Entry + exit LSA barrier only -- no scratch, signals, or GIN puts.
//
// Reads are 128-bit packed (Pack = 16 bytes = VEC elements). count is always a
// multiple of 16/sizeof(T) (see ReduceScatterGetCollByteCount), so the owned
// slice base (rank*count) and length tile exactly into 16B-aligned packs with no
// scalar tail. This mirrors RCCL's vectorized LSA AllReduce kernel and is much
// faster than a scalar element loop (wider coalesced loads + per-pack ILP).
//
// This replaces an earlier size-hybrid design whose large tier staged partials
// via GIN put into the resource window and SM-reduced them. That window is
// UNCACHED (dev_runtime.cc allocates it hipMemAllocationTypeUncached), so the SM
// read-back dominated the kernel (~99% of time; ~144 GB/s vs HBM peak). Reading
// peers' cached sendbuffs directly is strictly faster at every measured size and
// removes the scratch round-trip, so one LSA tier now covers all sizes. The
// sdmaThreshold/scratch launch args are retained for ABI compatibility but
// unused.
template <typename T>
__global__ void GinReduceScatterKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, int redOp, ncclDevResourceHandle scratchHandle) {
  (void)sdmaThresholdOverride;
  (void)scratchHandle;
  const int nRanks = devComm.nRanks;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;

  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
  lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  // 128-bit packed read-reduce over the owned slice. In-place safe: each thread
  // reads its own input element (source s == rank) before writing the same recv
  // element, and each thread owns a disjoint pack index.
  constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
  struct alignas(16) Pack { T e[VEC]; };
  T* dst = (T*)ncclGetLocalPointer(recvwin, recvoffset);
  const size_t nPacks = count / (size_t)VEC;
  const size_t myBaseP = ((size_t)devComm.rank * count) / (size_t)VEC;  // pack idx of my slice

  for (size_t pk = tid; pk < nPacks; pk += nthreads) {
    Pack v = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, 0))[myBaseP + pk];  // s == 0
    T acc[VEC];
    #pragma unroll
    for (int e = 0; e < VEC; e++) acc[e] = gin_sdma_reduce::preOp(redOp, v.e[e], nRanks);
    for (int s = 1; s < nRanks; s++) {
      Pack vs = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[myBaseP + pk];
      #pragma unroll
      for (int e = 0; e < VEC; e++)
        acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, vs.e[e], nRanks));
    }
    Pack out;
    #pragma unroll
    for (int e = 0; e < VEC; e++) out.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
    ((Pack*)dst)[pk] = out;
  }

  lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
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
        // Single-tier LSA read-reduce: the kernel no longer branches on size, so
        // the threshold/scratch launch args are inert (kept for ABI stability).
        TESTCHECK(testLaunchDeviceKernelThresholdScratch(SPECIALIZE_REDUCE_KERNEL(GinReduceScatterKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, TEST_SDMA_THRESHOLD_UNSET, g_rsScratchHandle));
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
