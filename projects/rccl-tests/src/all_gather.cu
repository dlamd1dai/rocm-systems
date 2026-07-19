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
  NCCLCHECK(rcclTestsGetSymkInfo(comm, ncclFuncAllGather , count, type , op, algo, proto, nchannels));
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
    case 3: // GinAllGatherKernel: all CTAs participate, one barrier per CTA
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      reqs->barrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
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
bool AllGatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: // GinAllGatherKernel: all CTAs, one barrier per CTA
      reqs->barrierCount = deviceCtaCount;
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
template <typename T>
__global__ void GinAllGatherKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  /* broadcast this rank's send slice to all peers' recv slot [rank] */
  const size_t size = count * sizeof(T);
  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset,
        size, ncclGin_SignalInc{signalIndex});
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
      case 3:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(GinAllGatherKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
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
