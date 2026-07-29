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

// LL (low-latency, packed data+flag) tiny-message SendRecv fast path. The ring
// pairs rank r with (r+1)%N (send) and (r-1+N)%N (recv); each rank sends its
// payload into its send-peer's epoch-tagged LL scratch and polls its OWN scratch
// (written by its recv-peer) into recvbuff. The recv's epoch tag replaces the
// EXIT barrier; a single ENTRY LSA barrier is kept because the ring is not
// mutually back-pressured (rank r receives from r-1, not from its reader r+1),
// so -- like Broadcast, unlike AllGather -- a fast rank could otherwise run >=2
// epochs ahead of its reader and deadlock the double-buffered LL scratch. On by
// default up to SENDRECV_LL_DEFAULT_MAX_BYTES; tune/disable via
// NCCL_GIN_ANVIL_SENDRECV_LL_MAX_BYTES (0 = disable). Constants live in
// gin_sdma_collective_policy.h so the kernel branch, the requirements sizing and
// the host unit tests share one source. nSlots==0 => LL not configured => the
// kernel uses the direct-LSA store.
#if (defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)) || NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
#define SR_HAVE_LL 1
static size_t g_srLLMaxBytes = 0;  // resolved from env at requirements time
static ncclLLA2AHandle g_srLLHandle = {};
static ncclDevResourceRequirements g_srLLReq = {};
#endif

void SendRecvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t SendRecvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, rank*sendcount, type, ncclSum, rep, 1, 0));
    int peer = (rank-1+nranks)%nranks;
    TESTCHECK(InitData(args->expected[i], recvcount, peer*recvcount, type, ncclSum, rep, 1, 0));
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place sendrecv
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void SendRecvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t SendRecvGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;
  switch(deviceImpl) {
    case 3: { // GinSendRecvKernel: LSA store (small) + single GIN put (large)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      gin_sdma::DevReqs dr = gin_sdma::moveDevReqs(deviceCtaCount);
      reqs->barrierCount = dr.barrierCount;
      reqs->lsaBarrierCount = dr.lsaBarrierCount;
      reqs->ginSignalCount = dr.ginSignalCount;
      // LL scratch for the tiny-message fast path (single CTA => nBlocks=1), on
      // by default up to SENDRECV_LL_DEFAULT_MAX_BYTES, tunable via
      // NCCL_GIN_ANVIL_SENDRECV_LL_MAX_BYTES (bytes; 0 = disable). One message
      // per receiver, so a receiver needs just cap/8 u64 slots (like Broadcast).
      g_srLLMaxBytes = gin_sdma::resolveLLCap(
          testParseSdmaThresholdEnv("NCCL_GIN_ANVIL_SENDRECV_LL_MAX_BYTES"),
          gin_sdma::kSendRecvLLDefaultMaxBytes, gin_sdma::kSendRecvLLMaxBytes);
      if (g_srLLMaxBytes > 0) {
        int llMaxElts = (int)(g_srLLMaxBytes / 8);
        int nSlots = ncclLLA2ACalcSlots(llMaxElts, /*maxEltSize=*/8);
        ncclLLA2ACreateRequirement(/*nBlocks=*/1, nSlots, &g_srLLHandle, &g_srLLReq);
        g_srLLReq.next = reqs->resourceRequirementsList;
        reqs->resourceRequirementsList = &g_srLLReq;
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
bool SendRecvGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
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
__device__ size_t SendRecvGetSdmaThreshold(struct ncclDevComm const& devComm) {
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

#if defined(SR_HAVE_LL)
// LL ring SendRecv for tiny messages, single CTA. Each rank sends its payload
// (as 8-byte units) into its send-peer's epoch-tagged LL scratch at slots
// [0..chunkU64), then polls its OWN scratch (written by its recv-peer) into
// recvbuff. A single entry LSA barrier bounds the epoch skew (the ring lacks the
// mutual-recv backpressure AllGather has; see file header note); the recv's
// epoch tag replaces the exit barrier, and each rank writes only its own
// recvbuff, so it is immune to the initData recvbuff-memset race.
__device__ void SendRecvLLImpl(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset,
                               size_t chunkU64, int sendPeer, struct ncclDevComm const& devComm, ncclTeam lsa,
                               ncclLLA2AHandle llHandle) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const uint64_t* src = (const uint64_t*)ncclGetLocalPointer(sendwin, sendoffset);
  uint64_t* dst = (uint64_t*)ncclGetLocalPointer(recvwin, recvoffset);

  // Entry barrier: bounds send-peer-vs-laggard epoch skew (see file header note).
  ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, /*block=*/0 };
  lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  ncclLLA2ASession<ncclCoopCta> ll { ncclCoopCta(), devComm, lsa, llHandle, /*block=*/0,
                                     /*maxElts=*/(int)chunkU64 };

  // Send my message into the send-peer's scratch slots [0..chunkU64).
  for (size_t j = tid; j < chunkU64; j += nthreads) {
    ll.send(sendPeer, (int)j, src[j]);
  }
  // Gather my incoming message (written by my recv-peer) out of my own scratch.
  for (size_t j = tid; j < chunkU64; j += nthreads) {
    dst[j] = ll.recv<uint64_t>((int)j);
  }
  ll.endEpoch(ncclCoopCta());
}
#endif

// Single-node ring SendRecv (-D 3): every rank sends its buffer to (rank+1)%N
// and receives from (rank-1+N)%N, so each rank issues exactly one put and
// receives exactly one put (symmetric completion, waitSignal base+1). The GIN
// destination offset is uniform across ranks (out-of-place only; in-place
// sendrecv is not validated), so the peer's recvbuff is addressed at recvoffset
// directly.
//   msgBytes <= LL cap (opt-out): LL packed data+flag, single CTA (tiny, one
//                                 barrier removed vs LSA).
//   msgBytes <= sdmaThreshold:    LSA store to the send peer's recvbuff.
//   msgBytes >  sdmaThreshold:    one GIN put to the send peer (Anvil picks SDMA).
template <typename T>
__global__ void GinSendRecvKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, ncclLLA2AHandle llHandle) {
  const size_t msgBytes = count * sizeof(T);
  const size_t sdmaThreshold = (sdmaThresholdOverride != TEST_SDMA_THRESHOLD_UNSET)
                                   ? sdmaThresholdOverride
                                   : SendRecvGetSdmaThreshold(devComm);
  const int rank = devComm.rank, nRanks = devComm.nRanks;
  const int sendPeer = (rank + 1) % nRanks;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;

  if (msgBytes <= sdmaThreshold) {
    ncclTeam lsa = ncclTeamLsa(devComm);

#if defined(SR_HAVE_LL)
    // Tiny messages: LL packed data+flag path (single CTA). Keeps one entry
    // barrier (the ring is not mutually back-pressured) and drops the exit
    // barrier; memset-race-immune (cross-rank traffic stays in the LL scratch).
    // Used when LL is configured (nSlots>0), the message is 8-byte aligned, and
    // it fits the pre-sized slot count.
    if (gin_sdma::sendRecvLLEligible(msgBytes, llHandle.nSlots, gin_sdma::kSendRecvLLMaxBytes)) {
      if (blockIdx.x != 0) return;  // single CTA
      SendRecvLLImpl(sendwin, sendoffset, recvwin, recvoffset, msgBytes / 8, sendPeer, devComm, lsa, llHandle);
      return;
    }
#endif

    // LSA tier: store my send buffer into the send peer's recvbuff. Entry
    // barrier keeps every recvbuff quiescent past initData's memset before a
    // peer writes it; exit barrier makes the write visible before I read the
    // slice my recv peer wrote into my recvbuff.
    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    const T* src = (const T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* dst = (T*)ncclGetLsaPointer(recvwin, recvoffset, sendPeer);
    for (size_t i = tid; i < count; i += nthreads) dst[i] = src[i];

    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  const unsigned int signalIndex = 0;
  ncclGin gin { devComm, /*context=*/0 };
  const uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  // One put to the send peer (issue once, by the single global thread 0).
  // Chunked to <=1 GiB segments so a >1 GiB message does not overflow the
  // 30-bit SDMA copy-count; the signal rides the final segment.
  if (tid == 0) {
    ginPutChunked(gin, ncclTeamWorld(devComm), sendPeer,
        recvwin, recvoffset,
        sendwin, sendoffset,
        msgBytes, ncclGin_SignalInc{signalIndex});
  }
  // Each rank receives exactly one put (from its recv peer).
  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + 1);
  gin.flush(ncclCoopCta());
}
#endif

testResult_t SendRecvRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (deviceImpl == 0) {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    int recvPeer = (rank-1+nRanks) % nRanks;
    int sendPeer = (rank+1) % nRanks;

    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclSend(sptr, count, type, sendPeer, comm, stream));
    NCCLCHECK(ncclRecv(rptr, count, type, recvPeer, comm, stream));
    NCCLCHECK(ncclGroupEnd());
  } else {
    switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3: {
        // SendRecv-specific LSA<->GIN threshold (compared against the full
        // message). Default 1 GiB (LSA-always): the ring's writes are spread
        // across all ranks so direct LSA beats GIN/SDMA at all measured sizes to
        // 512 MiB (8x MI355X, 2026-07-27; 62.4 vs 61.1 GB/s). Set
        // NCCL_GIN_ANVIL_SDMA_THRESHOLD_SENDRECV=0 to force the GIN tier.
        static const size_t srThr = testResolveSdmaThreshold("NCCL_GIN_ANVIL_SDMA_THRESHOLD_SENDRECV", gin_sdma::kSendRecvSdmaThresholdDefault);
        TESTCHECK(testLaunchDeviceKernelThresholdLL(SPECIALIZE_KERNEL(GinSendRecvKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, srThr, g_srLLHandle));
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
static void SendRecvInitCommConfig(ncclConfig_t* config) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,30,0)
  // Ring sendrecv uses two distinct peers per rank (send to rank+1, recv from rank-1).
  config->maxP2pPeers = 2;
#else
  (void)config;
#endif
}
#endif

struct testColl sendRecvTest = {
  "SendRecv",
  SendRecvGetCollByteCount,
  SendRecvInitData,
  SendRecvGetBw,
  SendRecvRunColl,
  NULL,
  NULL
};

void SendRecvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  SendRecvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t SendRecvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &sendRecvTest;
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
    op_count = 1;
    run_ops = &op;
    run_opnames = &opName;
  } else {
    op_count = test_opnum;
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=0; j<op_count; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1));
    }
  }
  return testSuccess;
}

struct testEngine ncclTestEngine = {
  .getBuffSize = SendRecvGetBuffSize,
  .runTest = SendRecvRunTest,
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
  .initCommConfig = SendRecvInitCommConfig,
#endif
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = SendRecvGetDevCommRequirements,
#endif
};
