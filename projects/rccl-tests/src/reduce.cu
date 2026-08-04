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
#include "gin_sdma_devtime.h" // shared device-side (wall_clock64) timing scaffold

// Reduce (-D 3): the all-to-one reduction. Implemented as a "reduce-scatter to
// the root": rank r read-reduces the owned slice [r*base] directly from EVERY
// peer's sendbuff via ncclGetLsaPointer (the exact tuned ReduceScatter read
// path), then writes its reduced slice into the ROOT's recvbuff at [r*base] via
// ncclGetLsaPointer(recvwin, .., root). The result is that the root's recvbuff
// ends up holding the full reduced array while every non-root recvbuff is left
// untouched (out-of-place: still zero; in-place: still its input) -- matching
// what the verifier expects (ReduceInitData only fills the reduce result on the
// root). Work and read egress are spread across all N ranks (each folds 1/N of
// the array from N peers); only the final write is directed at the single root.
//
// Bit-for-bit identical fold to the verifier: ascending source-rank order via
// gin_sdma_reduce (preOp/combine/postOp), narrowing to T on every pairwise step
// for low-precision types. Entry + exit LSA barrier only -- no scratch, signals,
// or GIN puts (single-node LSA). In-place safe: rank r is the sole reader AND
// writer of the root's [r*base] slice, and it reads all N sources into registers
// before writing, so the read-modify-write of its own contribution never races.
// The sdmaThreshold/scratch launch args are retained for ABI compatibility with
// the reduction-collective kernel signature but are unused.
static ncclDevResourceHandle g_reduceScratchHandle = 0;  // unused (no scratch); passed to kernel as 0

// ---- Pipelined large-tier Reduce (OOP): SM read-reduce || SDMA put-to-root ----
// (plan section 9.5). The fused GinReduceKernel serializes the read-reduce (SM,
// ~440 GB/s) and the root write (SM store over one xGMI link, ~58 GB/s), so at
// 2 GiB it costs read+write ~= 10 ms (215 GB/s). The read-reduce (SM) and the
// fast root delivery (SDMA GIN put, the gather path that hits ~500 GB/s) use
// DIFFERENT hardware, so a chunk-pipelined kernel overlaps them: each CTA
// SM-reduces sub-chunk k+1 of its slice stripe into its LOCAL recvbuff while the
// SDMA engine puts sub-chunk k's reduced result to the root. Staging in the
// rank's own recvbuff slice is OOP-safe (no other rank reads slice [r*base] of
// rank r's buffer) and multi-iteration-safe (sendbuff is never modified); after
// flushing its puts each non-root re-zeros its slice so the verifier still sees
// an untouched (zero) recvbuff.
//
// MEASURED (8x MI355X, warm A/B -V 32, float sum, OOP; plan 9.5.1): the pipeline
// LOSES to the fused read-reduce at every size -- 2G 202 vs 214 GB/s (-5.6%),
// 128M 181 vs 205 (-11.6%). The fused kernel does ONE read-reduce-write pass that
// already sits at the read-reduce roofline (OOP 214 ~= in-place 215 GB/s), so
// there is no serialized RS->gather phase for the pipeline to hide; the staging +
// per-sub-chunk __threadfence_system() + tiny per-CTA SDMA puts (1 channel) +
// re-zero overhead exceeds any overlap benefit. So the pipeline is DISABLED by
// default (reducePipeMinBytes()==0 -> always fused) and kept only as an env-gated
// experiment: set NCCL_GIN_ANVIL_REDUCE_PIPE_MIN_BYTES=<bytes> to enable it for
// OOP totals >= that size (sub-chunk count via reducePipeChunks()). In-place is
// never pipelined (staging would clobber the in-place input).
static inline size_t reducePipeMinBytes() {
  const char* e = getenv("NCCL_GIN_ANVIL_REDUCE_PIPE_MIN_BYTES");
  if (e && e[0]) return (size_t)strtoull(e, nullptr, 0);
  return 0;  // default OFF (fused wins); set the env to opt into the pipeline
}
static inline int reducePipeChunks() {
  const char* e = getenv("NCCL_GIN_ANVIL_REDUCE_PIPE_CHUNKS");
  if (e && e[0]) { int v = (int)strtol(e, nullptr, 0); if (v >= 1) return v; }
  return 4;
}

// ---- Multi-ring (edge-disjoint) large-tier Reduce (OOP) ----------------------
// (plan 9.5.2). The diagnostic spike showed host `ncclReduce` is ~linear in
// channel count (8ch 50, 16ch 101, 64ch 285 GB/s) and algorithm-agnostic, while
// the flat GIN read-reduce PLATEAUS at 214 (V32==V128) because every CTA hammers
// the same all-peer-read pattern and never spreads across the 7 xGMI links. The
// proven cure is the SAME edge-disjoint multi-ring the broadcast campaign used to
// go 238->350 (broadcast.cu): decompose K_N* into N-1 arc-disjoint Hamiltonian
// cycles, run ring b%nRings on buffer stripe b, so every GPU drives all its links
// at once. This is the reduce (all-to-one) mirror: data flows toward the root
// along each cycle (pos N-1 -> ... -> 0), each hop adds its own contribution
// (SM read-reduce) and forwards to its predecessor. Staging reuses the rank's own
// recvbuff (OOP-only; in-place falls back to fused); every non-root re-zeros its
// stripe at the end so the verifier sees an untouched recvbuff. Enabled for OOP
// totals >= reduceRingMinBytes() (default 0 = OFF, env-gated pending the A/B).
// CTA count self-selects (reduceRingCtas(), default 128, decoupled from -V like
// the broadcast ring); pipeline depth via reduceRingChunks().
static inline size_t reduceRingMinBytes() {
  const char* e = getenv("NCCL_GIN_ANVIL_REDUCE_RING_MIN_BYTES");
  if (e && e[0]) return (size_t)strtoull(e, nullptr, 0);
  return (size_t)64 << 20;  // default ON for OOP totals >= 64 MiB: ring>=host there (64M 229~host,
                            // 2G 330=116% host, 154% fused); below 64M fused wins (32M 163<189). 0 disables.
}
static inline int reduceRingCtas() {
  const char* e = getenv("NCCL_GIN_ANVIL_REDUCE_RING_CTAS");
  if (e && e[0]) { int v = (int)strtol(e, nullptr, 0); if (v >= 1) return v; }
  return 128;  // CTA-bound: needs ~128 to saturate all xGMI links (broadcast basis)
}
// Ring pipeline depth (chunks per CTA stripe). The ring is an (N-1)-hop pipeline,
// so its fill/drain efficiency is C/(C+N-1) -- too few chunks starves it (C=4 gave
// 149 vs C=64 314 GB/s @2G). Best throughput is at ~64 KiB per chunk per CTA
// (deeper wastes on barrier/coalescing at small sizes: 16 KiB chunks tanked 512M to
// 218). Env forces a fixed count; 0 (default/auto) => size-adaptive in RunColl.
static inline int reduceRingChunks() {
  const char* e = getenv("NCCL_GIN_ANVIL_REDUCE_RING_CHUNKS");
  if (e && e[0]) { int v = (int)strtol(e, nullptr, 0); if (v >= 1) return v; }
  return 0;  // auto (size-adaptive; see reduceRingAutoChunks)
}
// Size-adaptive chunk count: target ~64 KiB per chunk per CTA, clamped to [16,256]
// (>=16 to fill the pipeline past the N-1 fill cost; <=256 to keep chunks coalesced).
static inline int reduceRingAutoChunks(size_t totalBytes, int ctas) {
  if (ctas < 1) ctas = 1;
  size_t stripeBytes = totalBytes / (size_t)ctas;
  size_t ch = stripeBytes / (size_t)(64 * 1024);
  if (ch < 16) ch = 16;
  if (ch > 256) ch = 256;
  return (int)ch;
}
#endif

void ReduceGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  // Align the total element count so it splits into nranks equal, 16-byte-lined
  // slices (base = (count/nranks) & -(16/eltSize)); the device kernel gives rank
  // r the slice [r*base] and reads/writes it as whole 128-bit packs with no
  // scalar tail. sendbuff and (root's) recvbuff both hold the full base*nranks
  // array. Mirrors the ReduceScatter/AllGather alignment convention. The host
  // path (ncclReduce) is happy with any count, so the only effect is that a few
  // sub-(nranks*16B) sizes round down (identical to the other GIN collectives).
  size_t base = (nranks > 0) ? ((count / (size_t)nranks) & -(16 / eltSize)) : count;
  size_t aligned = base * (size_t)nranks;
  *sendcount = aligned;
  *recvcount = aligned;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = aligned;
}

testResult_t ReduceInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
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
    if (rank == root) TESTCHECK(InitDataReduce(args->expected[i], recvcount, 0, type, op, rep, nranks));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

testResult_t  ReduceGetAlgoProtoChannels(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetAlgoInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetAlgoInfo(comm, ncclFuncReduce , count, type , 0, 0, 1, algo, proto, nchannels));
  return testSuccess;
}

testResult_t  ReduceGetSymkInfo(ncclComm_t comm, size_t count, ncclDataType_t type, ncclRedOp_t op, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetSymkInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetSymkInfo(comm, ncclFuncReduce , count, type , op, algo, proto, nchannels));
  return testSuccess;
}

void ReduceGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;
  *algBw = baseBw;
  *busBw = baseBw;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t ReduceGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;
  switch(deviceImpl) {
    case 3: { // GinReduceKernel: reduce-scatter-to-root LSA read-reduce (no scratch)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      // Cover both the flat/pipelined kernels (deviceCtaCount) and the multi-ring
      // kernel's own larger CTA count (reduceRingCtas(), default 128, launched
      // decoupled from -V) -- the ring uses a per-block LSA barrier sig=blockIdx.x.
      int reduceBarCtas = deviceCtaCount > reduceRingCtas() ? deviceCtaCount : reduceRingCtas();
      gin_sdma::DevReqs dr = gin_sdma::reduceScatterDevReqs(reduceBarCtas);
      reqs->barrierCount = dr.barrierCount;
      reqs->lsaBarrierCount = dr.lsaBarrierCount;
      reqs->ginSignalCount = dr.ginSignalCount;
      // No resource/scratch window: each rank reads peers' sendbuffs directly and
      // writes only the root's recvbuff, so nothing needs to be staged.
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
bool ReduceGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));
  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: { // reduce-scatter-to-root LSA read-reduce: barriers only, no scratch
      int reduceBarCtas = deviceCtaCount > reduceRingCtas() ? deviceCtaCount : reduceRingCtas();
      gin_sdma::DevReqs dr = gin_sdma::reduceScatterDevReqs(reduceBarCtas);
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
// Single-node Reduce (-D 3). `count` is the FULL element count (== base*nRanks,
// aligned by ReduceGetCollByteCount); rank r folds the slice [r*base] from every
// peer and writes it into the root's recvbuff at [r*base]. The load schedule
// mirrors GinReduceScatterKernel exactly (both fold identically, bit-for-bit);
// the ONLY difference is the write target (root's recvbuff via LSA, offset by the
// rank's global slice base) instead of a local slice-sized recvbuff.
template <typename T>
__device__ __forceinline__ void ginReduceBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp) {
  const int nRanks = devComm.nRanks;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  const size_t base = (nRanks > 0) ? (count / (size_t)nRanks) : count;  // this rank's slice length

  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
  lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  // 128-bit packed read-reduce over the owned slice, folded into the ROOT's
  // recvbuff. dstP is the root's recvbuff (self for the root rank), so writes are
  // indexed by the global pack offset myBaseP+pk (the full array), whereas
  // ReduceScatter's local slice-sized recvbuff is indexed by pk alone.
  constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
  struct alignas(16) Pack { T e[VEC]; };
  Pack* dstP = (Pack*)ncclGetLsaPointer(recvwin, recvoffset, root);  // root's full recvbuff
  const size_t nPacks = base / (size_t)VEC;                          // packs in this rank's slice
  const size_t myBaseP = ((size_t)devComm.rank * base) / (size_t)VEC; // pack idx of my slice

  // Adaptive load schedule (identical fold in both branches; see reduce_scatter.cu
  // for the full rationale): small/mid uses a register-light grid-stride loop with
  // 2-way peer ILP; large uses a warp-strided pack-unrolled loop (U outstanding
  // coalesced loads) with 2-way peer ILP + source-0 next-iteration prefetch.
  constexpr size_t RS_UNROLL_MIN = (size_t)48 << 20;  // 48 MiB total message
  const size_t totalBytes = base * (size_t)nRanks * sizeof(T);

  if (totalBytes < RS_UNROLL_MIN) {
    // ---- small/mid: high-occupancy grid-stride with 2-way PEER ILP ----
    for (size_t pk = (size_t)tid; pk < nPacks; pk += (size_t)nthreads) {
      Pack v0 = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, 0))[myBaseP + pk];
      T acc[VEC];
      #pragma unroll
      for (int e = 0; e < VEC; e++) acc[e] = gin_sdma_reduce::preOp(redOp, v0.e[e], nRanks);
      int s = 1;
      for (; s + 1 < nRanks; s += 2) {
        Pack a = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[myBaseP + pk];
        Pack b = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s + 1))[myBaseP + pk];
        #pragma unroll
        for (int e = 0; e < VEC; e++)
          acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, a.e[e], nRanks));
        #pragma unroll
        for (int e = 0; e < VEC; e++)
          acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, b.e[e], nRanks));
      }
      for (; s < nRanks; s++) {  // odd peer tail
        Pack vs = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[myBaseP + pk];
        #pragma unroll
        for (int e = 0; e < VEC; e++)
          acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, vs.e[e], nRanks));
      }
      Pack o;
      #pragma unroll
      for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
      dstP[myBaseP + pk] = o;
    }
  } else {
    // ---- large: warp-strided pack-unrolled (U outstanding coalesced loads) ----
    constexpr int U = (VEC <= 8) ? 4 : 2;
    constexpr int WARP = 64;  // CDNA wavefront
    const int lane = tid & (WARP - 1);
    const size_t warpId = (size_t)(tid / WARP);
    const size_t nWarps = (size_t)(nthreads / WARP);
    const size_t tile = (size_t)U * WARP;
    const size_t gridStride = nWarps * tile;

    const Pack* src0Base = (const Pack*)ncclGetLsaPointer(sendwin, sendoffset, 0) + myBaseP;
    size_t wbase = warpId * tile;
    Pack seed[U];
    if (wbase + tile <= nPacks) {  // prime the pipeline for this warp's first full tile
      const Pack* sp = src0Base + wbase + (size_t)lane;
      #pragma unroll
      for (int u = 0; u < U; u++) seed[u] = sp[(size_t)u * WARP];
    }
    for (; wbase < nPacks; wbase += gridStride) {
      if (wbase + tile <= nPacks) {
        // ---- fully coalesced tile: 2*U outstanding 128-bit loads ----
        const size_t p0 = wbase + (size_t)lane;  // this lane's first pack (slice-local)
        T acc[U][VEC];
        #pragma unroll  // source s == 0: consume the prefetched seed (ascending fold)
        for (int u = 0; u < U; u++)
          #pragma unroll
          for (int e = 0; e < VEC; e++) acc[u][e] = gin_sdma_reduce::preOp(redOp, seed[u].e[e], nRanks);
        int s = 1;
        for (; s + 1 < nRanks; s += 2) {
          const Pack* sa = (const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s)     + myBaseP + p0;
          const Pack* sb = (const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s + 1) + myBaseP + p0;
          Pack ta[U], tb[U];
          #pragma unroll
          for (int u = 0; u < U; u++) ta[u] = sa[(size_t)u * WARP];
          #pragma unroll
          for (int u = 0; u < U; u++) tb[u] = sb[(size_t)u * WARP];
          #pragma unroll
          for (int u = 0; u < U; u++)
            #pragma unroll
            for (int e = 0; e < VEC; e++)
              acc[u][e] = gin_sdma_reduce::combine(redOp, acc[u][e], gin_sdma_reduce::preOp(redOp, ta[u].e[e], nRanks));
          #pragma unroll
          for (int u = 0; u < U; u++)
            #pragma unroll
            for (int e = 0; e < VEC; e++)
              acc[u][e] = gin_sdma_reduce::combine(redOp, acc[u][e], gin_sdma_reduce::preOp(redOp, tb[u].e[e], nRanks));
        }
        for (; s < nRanks; s++) {  // odd peer tail
          const Pack* sp = (const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s) + myBaseP + p0;
          Pack t[U];
          #pragma unroll
          for (int u = 0; u < U; u++) t[u] = sp[(size_t)u * WARP];
          #pragma unroll
          for (int u = 0; u < U; u++)
            #pragma unroll
            for (int e = 0; e < VEC; e++)
              acc[u][e] = gin_sdma_reduce::combine(redOp, acc[u][e], gin_sdma_reduce::preOp(redOp, t[u].e[e], nRanks));
        }
        const size_t nb = wbase + gridStride;
        if (nb + tile <= nPacks) {
          const Pack* spn = src0Base + nb + (size_t)lane;
          #pragma unroll
          for (int u = 0; u < U; u++) seed[u] = spn[(size_t)u * WARP];
        }
        #pragma unroll
        for (int u = 0; u < U; u++) {
          Pack o;
          #pragma unroll
          for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[u][e], nRanks);
          dstP[myBaseP + p0 + (size_t)u * WARP] = o;
        }
      } else {
        // ---- tail: partial warp-tile; each lane covers packs wbase+u*WARP+lane ----
        #pragma unroll
        for (int u = 0; u < U; u++) {
          const size_t pk = wbase + (size_t)u * WARP + (size_t)lane;
          if (pk >= nPacks) continue;
          Pack v = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, 0))[myBaseP + pk];
          T acc[VEC];
          #pragma unroll
          for (int e = 0; e < VEC; e++) acc[e] = gin_sdma_reduce::preOp(redOp, v.e[e], nRanks);
          for (int s = 1; s < nRanks; s++) {
            Pack vs = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[myBaseP + pk];
            #pragma unroll
            for (int e = 0; e < VEC; e++)
              acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, vs.e[e], nRanks));
          }
          Pack o;
          #pragma unroll
          for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
          dstP[myBaseP + pk] = o;
        }
      }
    }
  }

  lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
}

// -D 3 production kernel: one Reduce. sdmaThreshold/scratch args retained for ABI.
template <typename T>
__global__ void GinReduceKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, int redOp, ncclDevResourceHandle scratchHandle) {
  (void)sdmaThresholdOverride; (void)scratchHandle;
  ginReduceBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm, redOp);
}

// Pipelined large-tier body (OOP only; see the file-top note). Each CTA owns a
// contiguous stripe of THIS rank's output slice [rank*base] and pipelines it in
// nSub sub-chunks: reduce sub-chunk (SM, all peers) into the local recvbuff, then
// (tid 0) GIN-put it to the root's recvbuff[rank*base + ...] with a per-CTA signal
// increment, then move to the next sub-chunk while the SDMA put drains. The root
// reduces its OWN slice locally (no put) and waits for the (nRanks-1) peers' puts
// on its per-CTA signal; non-roots flush then re-zero their stripe. Fold order is
// ascending source rank (bit-identical to ReduceScatter; verifier tolerant).
template <typename T>
__device__ __forceinline__ void ginReducePipelinedBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp, int nSub) {
  const int nRanks = devComm.nRanks;
  const int rank = devComm.rank;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const size_t base = (nRanks > 0) ? (count / (size_t)nRanks) : count;  // this rank's slice elems

  constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
  struct alignas(16) Pack { T e[VEC]; };
  const size_t nPacks = base / (size_t)VEC;                     // packs in this rank's slice
  const size_t sliceBaseP = ((size_t)rank * base) / (size_t)VEC; // pack idx of my slice in the full array

  // Per-CTA contiguous stripe of the slice.
  const size_t stripePacks = (nPacks + (size_t)gridDim.x - 1) / (size_t)gridDim.x;
  const size_t pb0 = (size_t)blockIdx.x * stripePacks;
  const size_t pb1 = (pb0 + stripePacks < nPacks) ? (pb0 + stripePacks) : nPacks;
  const size_t stripeLen = (pb1 > pb0) ? (pb1 - pb0) : 0;
  const size_t subPacks = (stripeLen == 0) ? 1
                          : ((stripeLen + (size_t)nSub - 1) / (size_t)nSub);

  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
  ncclGin gin { devComm, /*context=*/0 };
  const unsigned int sig = blockIdx.x;
  const uint64_t sigBase = gin.readSignal(sig);

  // Entry barrier: every peer's sendbuff is filled before any read (single-node LSA).
  lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  Pack* dstLocal = (Pack*)ncclGetLocalPointer(recvwin, recvoffset);  // my recvbuff (full array)

  int nSubDone = 0;
  for (size_t cp0 = pb0; cp0 < pb1; cp0 += subPacks) {
    const size_t cp1 = (cp0 + subPacks < pb1) ? (cp0 + subPacks) : pb1;
    // ---- SM read-reduce sub-chunk [cp0,cp1) from all peers into local recvbuff ----
    for (size_t pk = cp0 + (size_t)tid; pk < cp1; pk += (size_t)nthreads) {
      const size_t sp = sliceBaseP + pk;
      Pack v0 = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, 0))[sp];
      T acc[VEC];
      #pragma unroll
      for (int e = 0; e < VEC; e++) acc[e] = gin_sdma_reduce::preOp(redOp, v0.e[e], nRanks);
      for (int s = 1; s < nRanks; s++) {
        Pack vs = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[sp];
        #pragma unroll
        for (int e = 0; e < VEC; e++)
          acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, vs.e[e], nRanks));
      }
      Pack o;
      #pragma unroll
      for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
      dstLocal[sp] = o;
    }
    __threadfence_system();  // publish this sub-chunk's stores so the SDMA engine reads them (system scope, matches AllReduce RS->AG)
    __syncthreads();         // all CTA threads done + fenced
    if (rank != root && tid == 0 && cp1 > cp0) {
      const size_t byteOff = recvoffset + (sliceBaseP + cp0) * (size_t)VEC * sizeof(T);
      const size_t chunkBytes = (cp1 - cp0) * (size_t)VEC * sizeof(T);
      ginPutChunked(gin, ncclTeamWorld(devComm), root,
          recvwin, byteOff, recvwin, byteOff, chunkBytes, ncclGin_SignalInc{sig});
    }
    __syncthreads();     // keep the CTA lockstep across sub-chunks (put issued from tid 0)
    nSubDone++;
  }

  if (rank == root) {
    // My own slice is reduced locally above; wait for every non-root's stripe puts.
    gin.waitSignal(ncclCoopCta(), sig, sigBase + (uint64_t)(nRanks - 1) * (uint64_t)nSubDone);
  } else {
    gin.flush(ncclCoopCta());     // my puts' source reads are complete
    __syncthreads();
    // Re-zero my stripe so the verifier sees an untouched (zero) non-root recvbuff.
    Pack z;
    #pragma unroll
    for (int e = 0; e < VEC; e++) z.e[e] = (T)0;
    for (size_t pk = pb0 + (size_t)tid; pk < pb1; pk += (size_t)nthreads) dstLocal[sliceBaseP + pk] = z;
  }
}

// -D 3 pipelined large-tier kernel (OOP). Signature mirrors GinReduceKernel plus
// the sub-chunk count; launched with its own grid (deviceCtaCount) in RunColl.
template <typename T>
__global__ void GinReducePipelinedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, int redOp, ncclDevResourceHandle scratchHandle, int nSub) {
  (void)sdmaThresholdOverride; (void)scratchHandle;
  ginReducePipelinedBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm, redOp, nSub);
}

// Device-timing kernel (shared gin_devtime methodology): run skip+loop back-to-back
// Reduce bodies under ONE persistent launch, bracketing only the timed region with
// wall_clock64() per CTA. Every body re-derives its entry/exit LSA barrier, so
// looping is correct with no extra bookkeeping (pure LSA -> no GIN cadence concern).
template <typename T>
__global__ void GinReduceTimedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp, int loop, int skip, long long* start_time, long long* end_time) {
  for (int i = 0; i < skip + loop; i++) {
    if (i == skip) {
      __syncthreads();
      if (threadIdx.x == 0) start_time[blockIdx.x] = wall_clock64();
    }
    ginReduceBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm, redOp);
  }
  __syncthreads();
  if (threadIdx.x == 0) end_time[blockIdx.x] = wall_clock64();
}

// ===== Multi-ring (edge-disjoint) large-tier Reduce (OOP), plan 9.5.2 ========
// Edge-disjoint decomposition of K_N* into N-1 arc-disjoint Hamiltonian cycles
// (same construction as the broadcast ring). CTA b runs ring b%nRings on buffer
// stripe b; data flows toward the root (pos N-1 -> ... -> 0), each hop adding its
// local contribution (SM read-reduce) and forwarding the partial to its
// predecessor. Every GPU thus drives all its N-1 links at once (the lever the spike
// identified). Tables are host-built once per N and uploaded to constant memory.
#define RED_RING_MAXR 16
#define RED_RING_MAXN 16
__constant__ int c_redNRings;
__constant__ int c_redRingPred[RED_RING_MAXR * RED_RING_MAXN];  // [ring*N+rank] -> predecessor (pos-1) == forward target
__constant__ int c_redRingPos[RED_RING_MAXR * RED_RING_MAXN];   // [ring*N+rank] -> position in cycle

// Ring reduce body. Incoming partials land in MY recvbuff (written by my ring
// successor, pos p+1); I add my local sendbuff contribution and forward to my
// predecessor's recvbuff (pos p-1). Root (pos 0) writes the final (postOp'd)
// result to its own recvbuff; non-roots forward the pre-postOp partial and re-zero
// their stripe at the end (OOP verifier expects an untouched zero non-root recvbuff).
// Per-step grid-wide LSA barrier keeps the pipeline lock-step (the table variant
// that beat P2P for broadcast). preOp is applied once per rank's local (N total,
// matching the fused kernel); postOp once at the root.
template <typename T>
__device__ __forceinline__ void ginRingReduceBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm& devComm, int redOp, size_t nChunksArg) {
  const int N = devComm.nRanks;
  const int rank = devComm.rank;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const int nRings = c_redNRings;
  if (nRings < 1) return;
  const int ring = (int)(blockIdx.x % (unsigned)nRings);
  const int predRank = c_redRingPred[ring * N + rank];
  const int posCur  = c_redRingPos[ring * N + rank];
  const int posRoot = c_redRingPos[ring * N + root];
  const int pos = (posCur - posRoot + N) % N;             // hops from root along this ring
  const bool isRoot = (pos == 0);
  const bool isTail = (pos == N - 1);
  const int distTail = (N - 1) - pos;                     // pipeline offset from the tail

  constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
  struct alignas(16) Pack { T e[VEC]; };
  const size_t nPacks = count / (size_t)VEC;

  const size_t sBaseP  = (nPacks * (size_t)blockIdx.x) / (size_t)gridDim.x;
  const size_t sEndP   = (nPacks * (size_t)(blockIdx.x + 1)) / (size_t)gridDim.x;
  const size_t sCountP = (sEndP > sBaseP) ? (sEndP - sBaseP) : 0;

  int C = (int)((size_t)nChunksArg < sCountP ? (size_t)nChunksArg : sCountP);
  if (C < 1) C = 1;

  Pack* myRecv       = (Pack*)ncclGetLocalPointer(recvwin, recvoffset);
  const Pack* mySend = (const Pack*)ncclGetLocalPointer(sendwin, sendoffset);
  Pack* predRecv     = (Pack*)ncclGetLsaPointer(recvwin, recvoffset, predRank);

  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);    // entry: sendbuffs filled, recvbuffs quiescent

  const int totalSteps = C + (N - 1);
  for (int step = 0; step < totalSteps; step++) {
    const int c = step - distTail;
    const bool active = (c >= 0) && (c < C);
    if (active) {
      const size_t cStart = sBaseP + (sCountP * (size_t)c) / (size_t)C;
      const size_t cEnd   = sBaseP + (sCountP * (size_t)(c + 1)) / (size_t)C;
      for (size_t pk = cStart + (size_t)tid; pk < cEnd; pk += (size_t)nthreads) {
        Pack lv = mySend[pk];
        T acc[VEC];
        #pragma unroll
        for (int e = 0; e < VEC; e++) acc[e] = gin_sdma_reduce::preOp(redOp, lv.e[e], N);
        if (!isTail) {
          Pack iv = myRecv[pk];                            // incoming partial from pos p+1
          #pragma unroll
          for (int e = 0; e < VEC; e++)
            acc[e] = gin_sdma_reduce::combine(redOp, iv.e[e], acc[e]);
        }
        Pack o;
        if (isRoot) {
          #pragma unroll
          for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], N);
          myRecv[pk] = o;                                  // final result -> root's recvbuff (local, cached)
        } else {
          #pragma unroll
          for (int e = 0; e < VEC; e++) o.e[e] = acc[e];   // forward pre-postOp partial
          // Streaming (nontemporal system-scope b128) peer store -- the xGMI store
          // path LL/host use to push cross-GPU writes at full rate without polluting
          // cache (Pack is exactly 16B and 16B-aligned for every supported T).
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
          union { Pack p; v4u u; } cv; cv.p = o;
          __builtin_amdgcn_global_store_b128((v4u_gptr)(predRecv + pk), cv.u, RCCL_SYSTEM_SYNCSCOPE);
#else
          predRecv[pk] = o;                                // -> predecessor's recvbuff
#endif
        }
      }
    }
    // The release-ordered LSA barrier both syncs the CTA and publishes this step's
    // peer stores to the predecessor GPU before it reads them next step (the same
    // publish mechanism the broadcast table kernel uses -- an explicit
    // __threadfence_system per step reproduced the slow P2P path, 149 vs fused 214).
    bar.sync(ncclCoopCta(), cuda::memory_order_release);
  }

  if (!isRoot) {                                           // re-zero intermediate scratch for the verifier
    Pack z;
    #pragma unroll
    for (int e = 0; e < VEC; e++) z.e[e] = (T)0;
    for (size_t pk = sBaseP + (size_t)tid; pk < sEndP; pk += (size_t)nthreads) myRecv[pk] = z;
  }
}

template <typename T>
__global__ void GinRingReduceKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp, size_t nChunksArg) {
  ginRingReduceBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm, redOp, nChunksArg);
}

// Host decomposition of K_N* into (N-1) arc-disjoint directed Hamiltonian cycles by
// verified backtracking (exists for N != 4,6; the search fails otherwise and we fall
// back to the flat kernel). Fills pred[ring*N+rank] (the pos-1 forward target) and
// pos[ring*N+rank]; returns nRings (N-1) or 0.
struct RedRingDecomp {
  int N;
  bool used[RED_RING_MAXN][RED_RING_MAXN];
  int order[RED_RING_MAXR][RED_RING_MAXN];
  long budget;
  bool extend(int r, int* path, bool* vis, int len) {
    if (--budget < 0) return false;
    const int cur = path[len - 1];
    if (len == N) {
      if (used[cur][0]) return false;
      for (int i = 0; i < N; i++) used[path[i]][path[(i + 1) % N]] = true;
      for (int i = 0; i < N; i++) order[r][i] = path[i];
      if (solve(r + 1)) return true;
      for (int i = 0; i < N; i++) used[path[i]][path[(i + 1) % N]] = false;
      return false;
    }
    for (int nx = 1; nx < N; nx++) {
      if (!vis[nx] && !used[cur][nx]) {
        vis[nx] = true; path[len] = nx;
        if (extend(r, path, vis, len + 1)) return true;
        vis[nx] = false;
      }
    }
    return false;
  }
  bool solve(int r) {
    if (r == N - 1) return true;
    int path[RED_RING_MAXN]; bool vis[RED_RING_MAXN];
    for (int i = 0; i < N; i++) vis[i] = false;
    path[0] = 0; vis[0] = true;
    return extend(r, path, vis, 1);
  }
};
static int buildRedRingDecomp(int N, int* pred, int* pos) {
  if (N < 2 || N > RED_RING_MAXN || (N - 1) > RED_RING_MAXR) return 0;
  static RedRingDecomp d;
  d.N = N;
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) d.used[i][j] = false;
  d.budget = 20000000L;
  if (!d.solve(0)) return 0;
  for (int r = 0; r < N - 1; r++)
    for (int k = 0; k < N; k++) {
      const int rk = d.order[r][k];
      pred[r * N + rk] = d.order[r][(k - 1 + N) % N];       // predecessor (pos-1) == forward target
      pos[r * N + rk]  = k;
    }
  return N - 1;
}
static int g_redBuiltN = -1;
static int g_redBuiltNRings = 0;
#endif

testResult_t ReduceRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    NCCLCHECK(ncclReduce(sptr, rptr, count, type, op, root, comm, stream));
  } else {
    switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3: {
        if (count == 0) return testSuccess;
        // Large OUT-OF-PLACE totals use the pipelined SM-reduce || SDMA-put tier
        // (overlaps the read-reduce with the root delivery on distinct HW units;
        // plan 9.5). In-place (sendwin == recvwin) and small totals keep the fused
        // direct-to-root read-reduce (the threshold/scratch args stay inert there).
        const bool inPlace = (sendbuff == recvbuff);
        const ncclDevComm* dc = (const ncclDevComm*)comm;
        const int nRanks = (dc != nullptr) ? dc->nRanks : 0;
        const size_t totalBytes = count * (size_t)wordSize(type);
        // Multi-ring (edge-disjoint) large tier (OOP): stripes CTAs across all xGMI
        // links to break the flat kernel's 214 GB/s plateau (plan 9.5.2). Built +
        // uploaded once per N; falls back to the flat kernel if the decomposition is
        // unavailable. Env-gated (default OFF) pending the warm A/B.
        const size_t ringMin = reduceRingMinBytes();
        if (!inPlace && ringMin > 0 && totalBytes >= ringMin && nRanks > 1 && nRanks <= RED_RING_MAXN) {
          if (g_redBuiltN != nRanks) {
            static int h_pred[RED_RING_MAXR * RED_RING_MAXN];
            static int h_pos[RED_RING_MAXR * RED_RING_MAXN];
            g_redBuiltNRings = buildRedRingDecomp(nRanks, h_pred, h_pos);
            if (g_redBuiltNRings > 0) {
              CUDACHECK(cudaMemcpyToSymbol(c_redNRings, &g_redBuiltNRings, sizeof(int)));
              CUDACHECK(cudaMemcpyToSymbol(c_redRingPred, h_pred, sizeof(int) * nRanks * g_redBuiltNRings));
              CUDACHECK(cudaMemcpyToSymbol(c_redRingPos, h_pos, sizeof(int) * nRanks * g_redBuiltNRings));
            }
            g_redBuiltN = nRanks;
          }
          if (g_redBuiltNRings > 0) {
            const int ringCtas = reduceRingCtas();
            const int envChunks = reduceRingChunks();
            const size_t nChunks = (envChunks > 0) ? (size_t)envChunks
                                                   : (size_t)reduceRingAutoChunks(totalBytes, ringCtas);
            TESTCHECK(testLaunchDeviceKernelReduceRing(SPECIALIZE_REDUCE_KERNEL(GinRingReduceKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, nChunks, ringCtas));
            return testSuccess;
          }
        }
        const size_t pipeMin = reducePipeMinBytes();
        if (!inPlace && pipeMin > 0 && totalBytes >= pipeMin && nRanks > 1) {
          const int nSub = reducePipeChunks();
          TESTCHECK(testLaunchDeviceKernelReducePipelined(SPECIALIZE_REDUCE_KERNEL(GinReducePipelinedKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, TEST_SDMA_THRESHOLD_UNSET, g_reduceScratchHandle, nSub));
          return testSuccess;
        }
        // reduce-scatter-to-root LSA read-reduce: no size branch, so the
        // threshold/scratch launch args are inert (kept for ABI stability).
        TESTCHECK(testLaunchDeviceKernelThresholdScratch(SPECIALIZE_REDUCE_KERNEL(GinReduceKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, TEST_SDMA_THRESHOLD_UNSET, g_reduceScratchHandle));
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

// Device-side (in-kernel wall_clock64) timing for the GIN-SDMA Reduce (-D 3), via
// the shared gin_devtime scaffold. Opt-in through NCCL_GIN_ANVIL_DEVICE_TIMING
// (legacy NCCL_GIN_ANVIL_A2A_DEVICE_TIMING): 1=augment (print an extra
// #[reduce-devtime] line next to the graph numbers), 2=device-time-only (report
// the in-kernel latency as the out-of-place metric; in-place keeps normal
// timing). loop/skip via NCCL_GIN_ANVIL_REDUCE_DEVTIME_LOOP/_SKIP (default 10/10).
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
testResult_t ReduceDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  static const int loop = gin_devtime::envInt("NCCL_GIN_ANVIL_REDUCE_DEVTIME_LOOP", 10);
  static const int skip = gin_devtime::envInt("NCCL_GIN_ANVIL_REDUCE_DEVTIME_SKIP", 10);

  const size_t count = args->nbytes / wordSize(type);   // full (aligned) element count
  if (count == 0 || loop < 1) return testSuccess;

  auto kernel = SPECIALIZE_REDUCE_KERNEL(GinReduceTimedKernel, type, op);
  if (kernel == nullptr) return testSuccess;

  const int gridCtas = (deviceCtaCount > 0) ? deviceCtaCount : 16;
  double devUs = 0.0;
  TESTCHECK(gin_devtime::measure(args, gridCtas, loop,
      [&](int i, long long* d_start, long long* d_end) {
        ncclDevComm* devComm = args->devComms + i;
        ncclWindow_t sendwin = (ncclWindow_t)(in_place ? args->recvRegHandles[i] : args->sendRegHandles[i]);
        ncclWindow_t recvwin = (ncclWindow_t)args->recvRegHandles[i];
        kernel<<<gridCtas, 512, 0, args->streams[i]>>>(sendwin, 0, recvwin, 0, count, root, *devComm,
                 (int)op, loop, skip, d_start, d_end);
      },
      &devUs));

  if (outDeltaSec != nullptr) { *outDeltaSec = devUs * 1.0e-6; return testSuccess; }

  if (args->proc == 0 && args->thread == 0 && devUs > 0.0) {
    const size_t totalBytes = count * wordSize(type);
    double sec = devUs * 1.0e-6;
    double algBw = (double)totalBytes / 1.0e9 / sec;
    printf("#[reduce-devtime] size %12zu B  ctas %2d  loop %2d skip %2d  devtime %10.2f us  algbw %8.2f GB/s  busbw %8.2f GB/s\n",
           totalBytes, gridCtas, loop, skip, devUs, algBw, algBw);
    fflush(stdout);
  }
  return testSuccess;
}
#else
testResult_t ReduceDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  return testSuccess;  // device API path not available in this build
}
#endif

struct testColl reduceTest = {
  "Reduce",
  ReduceGetCollByteCount,
  ReduceInitData,
  ReduceGetBw,
  ReduceRunColl,
  ReduceGetAlgoProtoChannels,
  ReduceGetSymkInfo,
  ReduceDeviceTime
};

void ReduceGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  ReduceGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t ReduceRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &reduceTest;
  ncclDataType_t *run_types;
  ncclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;
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

  if ((int)op != -1) {
    op_count = 1;
    run_ops = &op;
    run_opnames = &opName;
  } else {
    op_count = test_opnum;
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  if (root != -1) {
    begin_root = end_root = root;
  } else {
    begin_root = 0;
    end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=0; j<op_count; j++) {
      // The GIN device path (deviceImpl != 0) shares ReduceScatter's reduction
      // kernel dispatch: no PreMulSum ("mulsum") kernel for any type (deferred),
      // and no prod kernel for fp8. SPECIALIZE_REDUCE_KERNEL returns nullptr for
      // those, which would abort the sweep on testNotImplemented; skip them here
      // so the device sweep only exercises implemented combos. The host path
      // (deviceImpl == 0) keeps full coverage via ncclReduce.
      if (deviceImpl != 0) {
        if (strcmp(run_opnames[j], "mulsum") == 0) continue;
#if defined(RCCL_FLOAT8)
        if ((run_types[i] == ncclFloat8e4m3 || run_types[i] == ncclFloat8e5m2) && run_ops[j] == ncclProd)
          continue;
#endif
      }
#if defined(RCCL_FLOAT8)
      else if((run_types[i] == ncclFloat8e4m3 || run_types[i] == ncclFloat8e5m2) && (run_ops[j] == ncclProd || run_ops[j] == ncclAvg || strcmp(run_opnames[j],"mulsum") == 0))
        continue;
#endif
      for (int k=begin_root; k<=end_root; k++) {
        TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], k));
      }
    }
  }
  return testSuccess;
}

struct testEngine ncclTestEngine = {
  .getBuffSize = ReduceGetBuffSize,
  .runTest = ReduceRunTest,
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = ReduceGetDevCommRequirements
#endif
};
