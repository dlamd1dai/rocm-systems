/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>

#define USE_RCCL_GATHER_SCATTER

// Build a deterministic sparse "send-size" matrix M[i][j] (sender i -> receiver j),
// identical on every rank, where every row and every column sums to exactly `total`.
//
// Method (Birkhoff--von Neumann): a doubly-stochastic matrix is a weighted sum of
// permutation matrices. Each random permutation contributes its weight to exactly one
// cell per row and per column, so all row/column sums equal the sum of the weights.
// Using integer weights that sum to `total` makes every row/column sum exactly `total`,
// which keeps the send/recv buffers exactly filled.
//
// Two tuning knobs (overridable via environment, but defaulted so all ranks agree):
//   RCCL_TESTS_A2AV_SPARSITY   fraction of (i,j) pairs forced to zero  [default 0.5]
//   RCCL_TESTS_A2AV_SIZESPREAD max/min ratio knob for nonzero weights  [default 3.0]
//   RCCL_TESTS_A2AV_SEED       shared RNG seed                         [default 2602]
//   RCCL_TESTS_A2AV_VERBOSE    if set, rank 0 prints the size matrix once
static void AlltoAllvGenSizeMatrix(int nranks, int rank, size_t total, std::vector<size_t>& M) {
  M.assign((size_t)nranks * nranks, 0);
  if (nranks <= 0 || total == 0) return;
  if (nranks == 1) { M[0] = total; return; }

  double sparsity = 0.5;
  double sizeSpread = 3.0;
  unsigned long seed = 2602;
  const char* s;
  if ((s = getenv("RCCL_TESTS_A2AV_SPARSITY")))   sparsity   = atof(s);
  if ((s = getenv("RCCL_TESTS_A2AV_SIZESPREAD"))) sizeSpread = atof(s);
  if ((s = getenv("RCCL_TESTS_A2AV_SEED")))       seed       = strtoul(s, NULL, 0);

  if (sparsity < 0.0)  sparsity = 0.0;
  if (sparsity > 0.95) sparsity = 0.95;
  if (sizeSpread < 1.0) sizeSpread = 1.0;

  // Number of permutation terms controls density. With k independent random
  // permutations, the expected fraction of zero cells is (1 - 1/N)^k, so
  // k ~= ln(sparsity) / ln(1 - 1/N) hits the requested sparsity.
  int k;
  if (sparsity > 0.0) {
    k = (int)std::lround(std::log(sparsity) / std::log(1.0 - 1.0 / nranks));
  } else {
    k = 4 * nranks; // effectively dense
  }
  if (k < 1) k = 1;

  std::mt19937_64 rng(seed);

  // Raw weights spread geometrically so max/min ~= sizeSpread.
  std::vector<double> raw(k);
  double rawSum = 0.0;
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  for (int t = 0; t < k; t++) {
    raw[t] = std::pow(sizeSpread, uni(rng)); // in [1, sizeSpread]
    rawSum += raw[t];
  }

  // Integer weights summing exactly to `total`.
  std::vector<size_t> w(k, 0);
  size_t assigned = 0;
  for (int t = 0; t < k; t++) {
    w[t] = (size_t)std::floor(raw[t] / rawSum * (double)total);
    assigned += w[t];
  }
  for (size_t rem = total - assigned, t = 0; rem > 0; rem--, t = (t + 1) % k) w[t] += 1;

  // Accumulate weighted random permutation matrices.
  std::vector<int> perm(nranks);
  for (int i = 0; i < nranks; i++) perm[i] = i;
  for (int t = 0; t < k; t++) {
    for (int i = nranks - 1; i > 0; i--) { // Fisher-Yates
      std::uniform_int_distribution<int> d(0, i);
      std::swap(perm[i], perm[d(rng)]);
    }
    for (int i = 0; i < nranks; i++) M[(size_t)i * nranks + perm[i]] += w[t];
  }

  if (rank == 0 && getenv("RCCL_TESTS_A2AV_VERBOSE")) {
    static bool printed = false;
    if (!printed) {
      printed = true;
      size_t nz = 0, mn = (size_t)-1, mx = 0;
      for (size_t e = 0; e < M.size(); e++) {
        if (M[e]) { nz++; if (M[e] < mn) mn = M[e]; if (M[e] > mx) mx = M[e]; }
      }
      if (mn == (size_t)-1) mn = 0;
      printf("AlltoAllv size matrix (rows=sender, cols=receiver):\n");
      printf("config:    total=%zu  requestedSparsity=%.1f%%  sizeSpread=%.2f  N=%d  seed=%lu\n",
             total, 100.0 * sparsity, sizeSpread, nranks, seed);
      printf("realized:  sparsity=%.1f%%  sizeSpreadRatio(max/min)=%.2f  nonzero[min..max]=[%zu..%zu]  terms=%d\n",
             100.0 * (1.0 - (double)nz / M.size()),
             mn ? (double)mx / (double)mn : 0.0, mn, mx, k);
      for (int i = 0; i < nranks; i++) {
        for (int j = 0; j < nranks; j++) printf("%10zu", M[(size_t)i * nranks + j]);
        printf("\n");
      }
    }
  }
}

void AlltoAllvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  if (count < nranks*nranks/2) {
    *sendcount = 0;
    *recvcount = 0;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = 0;
  } else {
    *paramcount = (count/nranks) & -(16/eltSize);
    *sendcount = nranks*(*paramcount);
    *recvcount = *sendcount;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
  }
}

testResult_t AlltoAllvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep+rank, 1, 0));

#if 0
    int *dataHost = (int *)malloc(args->sendBytes);
    cudaMemcpy(dataHost, data, args->sendBytes, cudaMemcpyDeviceToHost);
    printf(" Rank [%d] Original: ", rank);
    for(int j=0; j<sendcount; j++) {
	    printf("%d:%d ", j, dataHost[j]);
    }
    printf("\n");
    free(dataHost);
#endif

    // Shared sparse size matrix: M[j][r] = bytes sender j sends to receiver r.
    // `expected` for this rank is column `rank`, concatenated in sender order.
    std::vector<size_t> M;
    AlltoAllvGenSizeMatrix(nranks, rank, sendcount, M);

    size_t rdisp = 0;
    for (int j=0; j<nranks; j++) {
      size_t rcount = M[(size_t)j*nranks + rank];
      // Displacement of this chunk inside sender j's send buffer (prefix sum of row j).
      size_t sdisp = 0;
      for (int d=0; d<rank; d++) sdisp += M[(size_t)j*nranks + d];
      TESTCHECK(InitData(((char*)args->expected[i])+rdisp*wordSize(type), rcount, sdisp, type, ncclSum, 33*rep+j, 1, 0));
      rdisp += rcount;
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place alltoall
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// set devComm reqs for the GIN-SDMA AllToAllv device kernel
testResult_t AlltoAllvGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;
  switch(deviceImpl) {
    case 3: { // GinHybridAlltoAllvKernel: LSA push (small) + all-peers GIN puts (large)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      // AllToAllv shares the pure-movement devComm shape (barrier = lsaBarrier =
      // ginSignal = deviceCtaCount, GIN required); its push tiers need no scratch
      // and no LL window (the variable per-peer sizes have no tiny fast path yet).
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
bool AlltoAllvGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
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
__device__ size_t AlltoAllvGetSdmaThreshold(struct ncclDevComm const& devComm) {
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

// Copy `bytes` bytes peer<-local in `T` units (both offsets/lengths are exact
// multiples of sizeof(T) by construction: every per-peer size/displacement is an
// element count times eltSize). uint4-vectorized when the T pointers and length
// are 16 B aligned, else element-wise. tid/nthreads span the CTA-local threads.
template <typename T>
__device__ __forceinline__ void A2AvCopy(T* dst, const T* src, size_t nElts, int tid, int nthreads) {
  const size_t bytes = nElts * sizeof(T);
  const uintptr_t da = (uintptr_t)dst, sa = (uintptr_t)src;
  if ((bytes % sizeof(uint4)) == 0 && (da % sizeof(uint4)) == 0 && (sa % sizeof(uint4)) == 0) {
    uint4* d4 = (uint4*)dst; const uint4* s4 = (const uint4*)src;
    const size_t n4 = bytes / sizeof(uint4);
    for (size_t i = tid; i < n4; i += nthreads) d4[i] = s4[i];
  } else {
    for (size_t i = tid; i < nElts; i += nthreads) dst[i] = src[i];
  }
}

// Single-node size-hybrid AllToAllv (-D 3). Variable-count AllToAll: rank `rank`
// sends sendBytes[p] bytes to peer p, taken from its sendbuf at srcByteOff[p] and
// landing in peer p's recvbuf at dstByteOff[p] (the receiver-side column-p
// prefix). Empty (sendBytes==0) pairs are skipped. Both tiers are PUSH:
//   nominal per-peer chunk <= sdmaThreshold: direct LSA store into each peer's
//                                            recvbuf (entry+exit LSA barrier).
//   nominal per-peer chunk >  sdmaThreshold: one GIN put per non-empty peer
//                                            (SDMA copy engine), receiver-side
//                                            signal completion.
// The tier is chosen ONCE per collective from the nominal per-peer chunk
// (count*sizeof(T)), since the actual per-peer sizes vary. nIncoming is the
// number of non-empty puts this rank receives (its GIN waitSignal target).
//
// Metadata arrays (sendBytes/srcByteOff/dstByteOff) are per-rank device pointers
// in BYTE units, computed host-side from the deterministic, all-ranks-identical
// size matrix (no cross-rank exchange needed). The LSA tier maps each peer to a
// CTA (peer = blockIdx.x, blockIdx.x+gridDim.x, ...) so distinct peers' xGMI
// egress links run concurrently; the GIN tier issues one put per peer from a
// distinct thread (tid == p), which the backend routes to independent per-peer
// SDMA queues.
template <typename T>
__device__ void ginHybridAlltoAllvBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, struct ncclDevComm devComm, size_t sdmaThresholdOverride, const size_t* sendBytes, const size_t* srcByteOff, const size_t* dstByteOff, int nIncoming) {
  const int rank = devComm.rank, nRanks = devComm.nRanks;
  const size_t nominalPeerBytes = count * sizeof(T);  // tier selector (avg per-peer slice)
  const size_t sdmaThreshold = (sdmaThresholdOverride != TEST_SDMA_THRESHOLD_UNSET)
                                   ? sdmaThresholdOverride
                                   : AlltoAllvGetSdmaThreshold(devComm);

  if (gin_sdma::a2avKernelTier(nominalPeerBytes, sdmaThreshold) == gin_sdma::MoveTier::LSA) {
    // Small: direct LSA push. Entry barrier orders every peer's recvbuf memset
    // (initData) and sendbuf fill before any cross-rank write; exit barrier
    // guarantees all senders finished writing this rank's recvbuf before it is
    // read/verified. Each rank writes only PEER recvbufs (+ its own self-chunk).
    ncclTeam lsa = ncclTeamLsa(devComm);
    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    const char* srcBase = (const char*)ncclGetLocalPointer(sendwin, sendoffset);
    // Peer-interleaved fan-out: CTA b serves peers b, b+gridDim, ... so distinct
    // xGMI egress links fire concurrently (one CTA's 512 threads per peer chunk).
    for (int p = blockIdx.x; p < nRanks; p += gridDim.x) {
      const size_t bytes = sendBytes[p];
      if (bytes == 0) continue;
      T* dst = (T*)ncclGetLsaPointer(recvwin, recvoffset + dstByteOff[p], p);
      const T* s = (const T*)(srcBase + srcByteOff[p]);
      A2AvCopy<T>(dst, s, bytes / sizeof(T), (int)threadIdx.x, (int)blockDim.x);
    }
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  // Large: all-peers GIN puts (SDMA copy engine), receiver-side signal.
  const unsigned int signalIndex = 0;
  ncclGin gin { devComm, /*context=*/0 };
  const uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;

  // One put per non-empty peer, each issued by exactly one thread (tid == p), so
  // each (rank->p) pair carries exactly one receiver-side SignalInc. Chunked to
  // <=1 GiB segments (signal on the final segment) for the 30-bit SDMA limit.
  for (int p = tid; p < nRanks; p += nthreads) {
    const size_t bytes = sendBytes[p];
    if (bytes == 0) continue;
    ginPutChunked(gin, ncclTeamWorld(devComm), p,
        recvwin, recvoffset + dstByteOff[p],
        sendwin, sendoffset + srcByteOff[p],
        bytes, ncclGin_SignalInc{signalIndex});
  }

  // This rank receives exactly nIncoming puts (non-empty senders to its column).
  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + (uint64_t)nIncoming);
  gin.flush(ncclCoopCta());
}

template <typename T>
__global__ void GinHybridAlltoAllvKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, const size_t* sendBytes, const size_t* srcByteOff, const size_t* dstByteOff, int nIncoming) {
  ginHybridAlltoAllvBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm, sdmaThresholdOverride, sendBytes, srcByteOff, dstByteOff, nIncoming);
}
#endif

testResult_t AlltoAllvRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  char* sptr = (char*)sendbuff + sendoffset;
  char* rptr = (char*)recvbuff + recvoffset;

  int nranks, rank;
  if (deviceImpl == 0) {
    NCCLCHECK(ncclCommCount(comm, &nranks));
    NCCLCHECK(ncclCommUserRank(comm, &rank));
  } else {
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
    // For device impls `comm` is actually a ncclDevComm* (see the launch
    // wrappers, which cast it): read the topology straight off it. Calling the
    // host ncclCommCount()/ncclCommUserRank() on a devComm pointer fails with
    // "invalid argument".
    nranks = ((ncclDevComm*)comm)->nRanks;
    rank   = ((ncclDevComm*)comm)->rank;
#else
    return testNotImplemented;
#endif
  }

  if (count == 0) return testSuccess;

  std::vector<size_t> sendcounts, recvcounts, sdispls, rdispls;
  try {
    sendcounts = std::vector<size_t>(nranks * nranks);
    recvcounts = std::vector<size_t>(nranks * nranks);
    sdispls = std::vector<size_t>(nranks * nranks);
    rdispls = std::vector<size_t>(nranks * nranks);
  } catch (const std::bad_alloc&) {
    printf("failed to allocate buffers for alltoallv\n");
    return testNcclError;
  }

  // Shared sparse size matrix: row `rank` = what we send, column `rank` = what we recv.
  std::vector<size_t> M;
  AlltoAllvGenSizeMatrix(nranks, rank, count*nranks, M);

  size_t sdisp = 0, rdisp = 0;
  for (int i = 0; i < nranks; i++) {
      size_t scount = M[(size_t)rank*nranks + i]; // rank -> i
      size_t rcount = M[(size_t)i*nranks + rank]; // i -> rank
      sendcounts[i+rank*nranks] = scount;
      recvcounts[i+rank*nranks] = rcount;
      sdispls[i+rank*nranks] = sdisp;
      rdispls[i+rank*nranks] = rdisp;
      sdisp += scount;
      rdisp += rcount;
      //printf("%d->%d: send %zu @ %zu, recv %zu @ %zu\n", rank, i, scount, sdispls[i+rank*nranks], rcount, rdispls[i+rank*nranks]);
  }

  // Device kernels (-D >0). AllToAllv only implements the GIN-SDMA hybrid at
  // case 3; other impls are not provided.
  if (deviceImpl != 0) {
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    switch (deviceImpl) {
      case 3: {
        const size_t eltSize = wordSize(type);
        // Per-rank device metadata (BYTE units), derived from the shared size
        // matrix M with no cross-rank exchange (every rank knows all of M):
        //   sendBytes[p] = M[rank][p]*eltSize           (bytes rank sends to p)
        //   srcByteOff[p] = (row-r prefix)*eltSize       (== sdispls; sender-side)
        //   dstByteOff[p] = (Sum_{s<rank} M[s][p])*eltSize (receiver-side column
        //                   prefix: where this rank's chunk lands in p's recvbuf)
        // nIncoming = #{ s : M[s][rank] != 0 }           (non-empty puts received)
        std::vector<size_t> hMeta((size_t)3 * nranks, 0);
        int nIncoming = 0;
        for (int p = 0; p < nranks; p++) {
          hMeta[p]                       = M[(size_t)rank * nranks + p] * eltSize;
          hMeta[(size_t)nranks + p]      = sdispls[p + rank * nranks] * eltSize;
          size_t col = 0;
          for (int s = 0; s < rank; s++) col += M[(size_t)s * nranks + p];
          hMeta[(size_t)2 * nranks + p]  = col * eltSize;
          if (M[(size_t)p * nranks + rank] != 0) nIncoming++;
        }

        // Per-device cached staging: a persistent host+device [3*N] buffer keyed
        // by device ordinal (so an nGpus>1 process keeps a distinct buffer per
        // GPU). The values depend only on (nranks, count, type), which are
        // constant across a TimeTest's iterations, and BenchTime stream-syncs
        // between sizes -- so reusing one persistent host buffer per device is
        // race-free even with an in-flight async upload.
        struct A2AvMeta { size_t* d; size_t* h; int capN; };
        static std::map<int, A2AvMeta> g_meta;
        int dev = 0;
        CUDACHECK(cudaGetDevice(&dev));
        A2AvMeta& m = g_meta[dev];
        if (m.capN < nranks) {
          if (m.d) CUDACHECK(cudaFree(m.d));
          free(m.h);
          CUDACHECK(cudaMalloc(&m.d, sizeof(size_t) * 3 * nranks));
          m.h = (size_t*)malloc(sizeof(size_t) * 3 * nranks);
          m.capN = nranks;
        }
        memcpy(m.h, hMeta.data(), sizeof(size_t) * 3 * nranks);
        CUDACHECK(cudaMemcpyAsync(m.d, m.h, sizeof(size_t) * 3 * nranks, cudaMemcpyHostToDevice, stream));
        const size_t* dSendBytes = m.d;
        const size_t* dSrcOff    = m.d + nranks;
        const size_t* dDstOff    = m.d + 2 * nranks;

        // AllToAllv LSA<->GIN threshold (compared against the nominal per-peer
        // chunk count*eltSize). Unmeasured 256 KiB starting default; override
        // with NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALLV or the shared
        // NCCL_GIN_ANVIL_SDMA_THRESHOLD.
        static const size_t a2avThr = testResolveSdmaThreshold("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALLV", gin_sdma::kAllToAllvSdmaThresholdDefault);
        TESTCHECK(testLaunchDeviceKernelA2Av(SPECIALIZE_KERNEL(GinHybridAlltoAllvKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, a2avThr, dSendBytes, dSrcOff, dDstOff, nIncoming));
        return testSuccess;
      }
      default:
        return testNotImplemented;
    }
#else
    return testNotImplemented;
#endif
  }

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
  printf("NCCL 2.7 or later is needed for alltoallv. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
  return testNcclError;
#else
#if defined(RCCL_ALLTOALLV) && defined(USE_RCCL_GATHER_SCATTER) && NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
  if (test_ncclVersion >= NCCL_VERSION(2,28,0)) {
    NCCLCHECK(ncclAlltoAllv(sptr, sendcounts.data()+rank*nranks, sdispls.data()+rank*nranks, rptr, recvcounts.data()+rank*nranks, rdispls.data()+rank*nranks, type, comm, stream));
    return testSuccess;
  }
  if (test_ncclVersion >= NCCL_VERSION(2,19,0)) {
    NCCLCHECK(ncclAllToAllv(sptr, sendcounts.data()+rank*nranks, sdispls.data()+rank*nranks, rptr, recvcounts.data()+rank*nranks, rdispls.data()+rank*nranks, type, comm, stream));
    return testSuccess;
  }
  printf("RCCL 2.19 or later is needed for RCCL_ALLTOALLV. This test was compiled with %d.%d, but is running with RCCL %d.\n", NCCL_MAJOR, NCCL_MINOR, test_ncclVersion);
  return testNcclError;
#else
  NCCLCHECK(ncclGroupStart());
  for (int r=0; r<nranks; r++) {
    if (sendcounts[r+rank*nranks] != 0) {
      NCCLCHECK(ncclSend(
          sptr + sdispls[r+rank*nranks] * wordSize(type),
          sendcounts[r+rank*nranks],
          type,
          r,
          comm,
          stream));
    }
    if (recvcounts[r+rank*nranks] != 0) {
      NCCLCHECK(ncclRecv(
          rptr + rdispls[r+rank*nranks] * wordSize(type),
          recvcounts[r+rank*nranks],
          type,
          r,
          comm,
          stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());
#endif
#endif
  return testSuccess;
}

struct testColl alltoAllTest = {
  "AlltoAllv",
  AlltoAllvGetCollByteCount,
  AlltoAllvInitData,
  AlltoAllvGetBw,
  AlltoAllvRunColl,
  NULL,
  NULL
};

void AlltoAllvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AlltoAllvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &alltoAllTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = ncclNumTypes;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

struct testEngine ncclTestEngine = {
  .getBuffSize = AlltoAllvGetBuffSize,
  .runTest = AlltoAllvRunTest,
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = AlltoAllvGetDevCommRequirements
#endif
};
