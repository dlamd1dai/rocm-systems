/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * AllReduce Performance Test Implementation
 *
 * This file implements multiple AllReduce kernel variants optimized for different
 * use cases within CUDA P2P connectivity.
 * These kernels are designed to highlight the device API functionality. As well as how to optimize for best performance.
 *
 * IMPORTANT: All custom kernels require CUDA P2P connectivity since they require Load-Store Accessible (LSA) memory.
 *
 * Kernel Selection Strategy:
 * - deviceImpl = 0: NCCL's built-in AllReduce implementation (fallback)
 * - deviceImpl = 1: allReduceLsaKernel - Basic LSA implementation for demonstration and small message sizes.
 * - deviceImpl = 2: allReduceLsaVectorizedKernel - Vectorized LSA for demonstration to achieve performance for large message sizes.
 * - deviceImpl = 3: allReduceMultimemKernel - Multi-memory for hardware acceleration. Requires Multimem capable hardware but can offer better performance.
 * - deviceImpl = 4: allReduceMultimemVectorizedKernel - Vectorized multi-memory for best performance. Requires Multimem capable hardware but can offer better performance.
 */

#include "cuda_runtime.h"
#include "common.h"
#include <algorithm>
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
#include "rccl_vector_types.h"
#include "multimem_ops.h"
#if HAVE_FP8
#include "rccl_float8.h"
#endif
#include "gin_sdma_reduce.h"  // device preOp/combine/postOp, mirrors verifiable.cu exactly

// GIN-SDMA AllReduce (-D 5). The upstream LSA/multimem demo kernels keep -D 1..4;
// -D 5 is the size-hybrid one-shot(small)/two-shot(large) GIN-SDMA reduction.
// Two RS variants for the large/in-place two-shot path are prototyped and picked
// by measurement (see the design notes at GinAllReduceKernel):
//   (A) direct-LSA read-reduce RS + GIN-put AG (default; no scratch).
//   (B) put-partials into the resource-window scratch + SM reduce + GIN-put AG,
//       selected when the scratch window is registered (scratchHandle != 0),
//       which the host does only when NCCL_GIN_ANVIL_AR_RS_PUTPARTIALS is set.
// PreMulSum/mulsum is deferred (SPECIALIZE_REDUCE_KERNEL -> nullptr ->
// testNotImplemented); fp8 {prod,avg,mulsum} excluded (see AllReduceRunTest).
static ncclDevResourceRequirements g_arScratchReq = {};
static ncclDevResourceHandle g_arScratchHandle = 0;  // 0 => variant A (no scratch)
#endif

void AllReduceGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t AllReduceInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, op, rep, nranks, rank));
    TESTCHECK(InitDataReduce(args->expected[i], recvcount, 0, type, op, rep, nranks));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

testResult_t  AllReduceGetAlgoProtoChannels(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetAlgoInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetAlgoInfo(comm, ncclFuncAllReduce , count, type , 0, 0, 1, algo, proto, nchannels));
  return testSuccess;
}

testResult_t  AllReduceGetSymkInfo(ncclComm_t comm, size_t count, ncclDataType_t type, ncclRedOp_t op, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetSymkInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetSymkInfo(comm, ncclFuncAllReduce , count, type , op, algo, proto, nchannels));
  return testSuccess;
}

void AllReduceGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(2*(nranks - 1)))/((double)nranks);
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
 // set devComm reqs for allreduce device kernels
testResult_t AllReduceGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;

  switch(deviceImpl) {
    case 1: // allReduceLsaKernel
    case 2: // allReduceLsaVectorizedKernel
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    case 3: // allReduceMultimemKernel
    case 4: // allReduceMultimemVectorizedKernel
      if (!commProperties->multimemSupport) {
        fprintf(stderr, "This test requires multimem support, but multimem support is not enabled for this communicator.\n");
        return testInternalError;
      }
      reqs->lsaMultimem = true;
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 5: { // GinAllReduceKernel: one-shot(small)/two-shot(large) GIN-SDMA
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      gin_sdma::DevReqs dr = gin_sdma::allReduceDevReqs(deviceCtaCount);
      reqs->barrierCount = dr.barrierCount;
      reqs->lsaBarrierCount = dr.lsaBarrierCount;
      reqs->ginSignalCount = dr.ginSignalCount;
      // Variant B only: stage N per-source slice partials in the resource window.
      // Enabled by NCCL_GIN_ANVIL_AR_RS_PUTPARTIALS; otherwise variant A (no scratch).
      const char* pp = getenv("NCCL_GIN_ANVIL_AR_RS_PUTPARTIALS");
      if (pp && atoi(pp) != 0) {
        size_t sb = gin_sdma::allReduceScratchBytes(maxBytes, commProperties->nRanks);
        if (sb > 0) {
          memset(&g_arScratchReq, 0, sizeof(g_arScratchReq));
          g_arScratchReq.bufferSize = sb;
          g_arScratchReq.bufferAlign = 128;
          g_arScratchReq.outBufferHandle = &g_arScratchHandle;
          g_arScratchReq.next = reqs->resourceRequirementsList;
          reqs->resourceRequirementsList = &g_arScratchReq;
        }
      }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,7)
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
      reqs->ginForceEnable = true;
#endif
      return testSuccess;
    }
#endif
    default:
      return testNotImplemented;
  }
}
#elif defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
 bool AllReduceGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
   if (!reqs) return false;
   memset(reqs, 0, sizeof(*reqs));
   switch(deviceImpl) {
    case 1: // allReduceLsaKernel
    case 2: // allReduceLsaVectorizedKernel
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
    case 3: // allReduceMultimemKernel
    case 4: // allReduceMultimemVectorizedKernel
      reqs->lsaMultimem = true;
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 5: { // GinAllReduceKernel: barriers + per-CTA GIN signals (variant A; no scratch on 2.28)
      gin_sdma::DevReqs dr = gin_sdma::allReduceDevReqs(deviceCtaCount);
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

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
/*
 * Kernel 1: allReduceLsaKernel - Basic LSA-based AllReduce
 *
 * Purpose: Provides a simple, deterministic AllReduce implementation for small to
 * medium message sizes within CUDA P2P connectivity.
 *
 * Solution: Implements AllReduce using direct peer-to-peer memory access through
 * LSA windows. Each rank reads from all other ranks, performs reduction, and
 * writes the result back to all ranks using cooperative thread arrays.
 *
 * Key Optimizations:
 * - LSA barriers for faster synchronization than global barriers
 * - Global grid stride loop to distribute work across all ranks
 * - Direct peer access within CUDA P2P connectivity for optimal bandwidth
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - This kernel requires all participating
 * ranks to be within the same CUDA P2P connectivity.
 *
 * Use Case: Small to medium messages (< 1MB) where simplicity and determinism
 * are more important than maximum bandwidth.
 */
template <typename T>
__global__ void allReduceLsaKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
  const int rank = devComm.rank, nRanks = devComm.nRanks;

  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    T v = T{0};
    for (int peer=0; peer<nRanks; peer++) {
      T* sendPtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, peer);
      v += sendPtr[offset];
    }
    for (int peer=0; peer<nRanks; peer++) {
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[offset] = v;
    }
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

/*
 * Kernel 2: allReduceLsaVectorizedKernel - Vectorized LSA-based AllReduce
 *
 * Purpose: Enhanced AllReduce implementation using vectorized memory operations
 * and loop unrolling to maximize memory bandwidth utilization for large messages
 * within CUDA P2P connectivity.
 *
 * Solution: Builds upon the basic LSA approach but adds vectorized loads/stores
 * and aggressive loop unrolling to achieve higher memory bandwidth. Handles
 * misaligned data gracefully while maximizing vectorized throughput. Not necessarily optimal for small message sizes.
 *
 * Key Optimizations:
 * - Vectorized loads/stores for improved memory bandwidth (128-bit operations)
 * - Loop unrolling to reduce loop overhead and improve instruction-level parallelism
 * - Warp-coalesced memory access patterns for optimal memory controller utilization
 * - Graceful handling of misaligned data with scalar fallback, comes at the cost of higher latency if not required.
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - Same as basic LSA kernel. Requires
 * CUDA P2P connectivity due to LSA memory access patterns.
 *
 * Use Case: Large messages where maximum memory bandwidth is
 * critical and data alignment can be optimized.
 */
template <typename T>
__global__ void allReduceLsaVectorizedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  // Compile time vector type and vector size mapping
  using TN = typename VectorTypeMapping<T>::Type;
  constexpr int VECTOR_FACTOR = sizeof(TN)/sizeof(T);

  constexpr int UNROLL_FACTOR = 128/sizeof(TN); // Same as before 128 Bytes per thread

  const int rank = devComm.rank, nRanks = devComm.nRanks;

  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  // Since we use vector types, the non-vector allocated memory is not necessarily aligned.
  const size_t alignment_offset = (sendoffset % sizeof(TN)) / sizeof(T);
  const size_t aligned_count = count - alignment_offset;
  const size_t vector_count = aligned_count / VECTOR_FACTOR;
  const size_t remainder = aligned_count % VECTOR_FACTOR;

  // As before
  const int elements_per_block = globalNthreads * UNROLL_FACTOR;
  const int num_blocks = vector_count / elements_per_block;

  const int warp_id = globalTid / WARP_SIZE;
  const int lane_id = globalTid % WARP_SIZE;

  const int warp_offset = warp_id * WARP_SIZE * UNROLL_FACTOR;
  const int lane_offset = lane_id;
  const int warp_lane_offset = warp_offset + lane_offset;

  // Handle misaligned elements first using scalar operations. Grid stride loop with scalar handling
  if (alignment_offset > 0) {
    for (size_t offset = globalTid; offset < alignment_offset; offset += globalNthreads) {
      T v_scalar = T{0};

      for (int peer=0; peer<nRanks; peer++) {
        T* remotePtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, peer);
        v_scalar += remotePtr[offset];
      }

      for (int peer=0; peer<nRanks; peer++) {
        T* remotePtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
        remotePtr[offset] = v_scalar;
      }
    }
  }

  // Handle vectorized memory that can be handled in whole chunks (no if)
  for (int block = 0; block < num_blocks; block += 1) {
    TN v[UNROLL_FACTOR] = {TN{0}};
    const size_t block_offset = block * globalNthreads * UNROLL_FACTOR;
    for (int peer=0; peer<nRanks; peer++) {
#pragma unroll
      for (int i=0; i < UNROLL_FACTOR; i++) {
        const int stride_offset = i * WARP_SIZE;
        const size_t offset = warp_lane_offset + block_offset + stride_offset;
        // Uses TN* as pointer type for vectorized pointer arithmatic
        // The pointer is also adjusted for misalignment
        TN* remotePtr = (TN*)ncclGetLsaPointer(sendwin, sendoffset + alignment_offset * sizeof(T), peer);
        v[i] = vectorAdd(v[i], remotePtr[offset]);
      }
    }
    for (int peer=0; peer<nRanks; ++peer) {
#pragma unroll
      for (int i=0; i < UNROLL_FACTOR; i++) {
        const int stride_offset = i * WARP_SIZE;
        const size_t offset = warp_lane_offset + block_offset + stride_offset;
        TN* remotePtr = (TN*)ncclGetLsaPointer(recvwin, recvoffset + alignment_offset * sizeof(T), peer);
        remotePtr[offset] = v[i];
      }
    }
  }

    // Handle the last partial vectorized block, but with if conditions
  const int block = num_blocks;
  TN v[UNROLL_FACTOR] = {TN{0}};
  const size_t block_offset = block * globalNthreads * UNROLL_FACTOR;
  for (int peer=0; peer<nRanks; peer++) {
#pragma unroll
      for (int i=0; i < UNROLL_FACTOR; i++) {
        const int stride_offset = i * WARP_SIZE;
        const size_t offset = warp_lane_offset + block_offset + stride_offset;
        if (offset < vector_count) {
          TN* remotePtr = (TN*)ncclGetLsaPointer(sendwin, sendoffset + alignment_offset * sizeof(T), peer);
          v[i] = vectorAdd(v[i], remotePtr[offset]);
        }
      }
  }
  for (int peer=0; peer<nRanks; ++peer) {
#pragma unroll
      for(int i=0; i < UNROLL_FACTOR; i++){
        const int stride_offset = i * WARP_SIZE;
        const size_t offset = warp_lane_offset + block_offset + stride_offset;
        if (offset < vector_count) {
          TN* remotePtr = (TN*)ncclGetLsaPointer(recvwin, recvoffset + alignment_offset * sizeof(T), peer);
          remotePtr[offset] = v[i];
        }
      }
  }

  // Since the data doesn't have to be perfectly aligned with the vector size, we need to handle remaining elements.
  if (remainder > 0) {
    const size_t remainder_start = alignment_offset + vector_count * VECTOR_FACTOR;
    const int globalTid_remainder = globalTid;
    const int globalNthreads_remainder = globalNthreads;

    for (size_t offset = globalTid_remainder; offset < remainder; offset += globalNthreads_remainder) {
      T v_scalar = 0;
      const size_t actual_offset = remainder_start + offset;

      for (int peer=0; peer<nRanks; peer++) {
        T* remotePtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, peer);
        v_scalar += remotePtr[actual_offset];
      }

      for (int peer=0; peer<nRanks; peer++) {
        T* remotePtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
        remotePtr[actual_offset] = v_scalar;
      }
    }
  }

  // Sync
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

/*
 * Kernel 3: allReduceMultimemKernel - Multi-memory Hardware-Accelerated AllReduce
 *
 * Purpose: High-performance AllReduce implementation using multi-memory primitives
 * that leverage hardware acceleration for memory operations, significantly reducing
 * SM utilization while maintaining high bandwidth within CUDA P2P connectivity.
 *
 * Solution: Replaces the O(Nrank) peer loop approach with hardware-accelerated
 * multi-memory operations. The kernel initiates CUDA P2P reductions directly through
 * hardware, eliminating the need for explicit peer-to-peer communication loops.
 *
 * Key Optimizations:
 * - Multi-memory primitives for hardware-accelerated operations
 * - Eliminates O(Nrank) scaling by using hardware reduction capabilities
 * - Hardware-assisted memory synchronization and reduction
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - Requires CUDA P2P connectivity and
 * multi-memory support. Hardware acceleration is only available within the
 * same CUDA P2P connectivity where multi-memory operations can be performed.
 *
 * Use Case: Large CUDA P2P connectivity where scaling to more ranks is desired.
 *
 * Hardware Requirements: Hopper+ architecture with multi-memory support enabled.
 */
template <typename T>
__global__ void allReduceMultimemKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x, true };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = devComm.rank, nRanks = devComm.nRanks;

  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  T* send_ptr = reinterpret_cast<T*>(ncclGetLsaMultimemPointer(sendwin, sendoffset, devComm));
  T* recv_ptr = reinterpret_cast<T*>(ncclGetLsaMultimemPointer(recvwin, recvoffset, devComm));
  for (size_t offset=globalTid; offset < count; offset += globalNthreads) {
    if (offset < count) {
      T v = multimemLoadSum<T,T>(send_ptr + offset);
      multimemStore<T,T>(recv_ptr + offset, v);
    }
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

/*
 * Kernel 4: allReduceMultimemVectorizedKernel - Vectorized Multi-memory AllReduce
 *
 * Purpose: Ultimate performance AllReduce implementation combining multi-memory
 * hardware acceleration with vectorized operations and loop unrolling for maximum
 * bandwidth utilization within CUDA P2P connectivity.
 *
 * Solution: Combines the hardware acceleration benefits of multi-memory operations
 * with the bandwidth optimization techniques from vectorized kernels. This kernel
 * represents the highest performance option for large, aligned data sets.
 *
 * Key Optimizations:
 * - Multi-memory primitives for hardware-accelerated operations
 * - Vectorized loads/stores for maximum memory bandwidth (128-bit operations)
 * - Aggressive loop unrolling for improved instruction-level parallelism
 * - Warp-coalesced memory access patterns for optimal memory controller utilization
 * - Hardware-assisted memory synchronization and reduction
 * - Graceful handling of misaligned data with scalar fallback
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - Requires CUDA P2P connectivity and
 * multi-memory support. This kernel leverages both P2P locality and hardware
 * acceleration for optimal performance.
 *
 * Hardware Requirements: Hopper+ architecture with multi-memory support enabled.
 *
 * Performance Note: This kernel provides the best performance for large, aligned
 * data sets but requires careful data alignment for optimal vectorization benefits.
 */
template <typename T>
__global__ void allReduceMultimemVectorizedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x, true };

  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  using TN = typename VectorTypeMapping<T>::Type;
  constexpr int VECTOR_FACTOR = sizeof(TN)/sizeof(T);

  constexpr int UNROLL_FACTOR = 128/sizeof(TN);

  const int rank = devComm.rank, nRanks = devComm.nRanks;

  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  // Calculate alignment offset to handle misaligned elements first
  const size_t alignment_offset = (sendoffset % sizeof(TN)) / sizeof(T);
  const size_t aligned_count = count - alignment_offset;
  const size_t vector_count = aligned_count / VECTOR_FACTOR;
  const size_t remainder = aligned_count % VECTOR_FACTOR;

  const int elements_per_block = globalNthreads * UNROLL_FACTOR;
  const int num_blocks = vector_count / elements_per_block;

  const int warp_id = globalTid / WARP_SIZE;
  const int lane_id = globalTid % WARP_SIZE;

  const int warp_offset = warp_id * WARP_SIZE * UNROLL_FACTOR;
  const int lane_offset = lane_id;
  const int warp_lane_offset = warp_offset + lane_offset;

  // Multimem pointers that handle scalar access for misaligned and remainder elements
  T* send_ptr = reinterpret_cast<T*>(ncclGetLsaMultimemPointer(sendwin, sendoffset, devComm));
  T* recv_ptr = reinterpret_cast<T*>(ncclGetLsaMultimemPointer(recvwin, recvoffset, devComm));

  // Handle misaligned elements first using scalar operations
  if (alignment_offset > 0) {
    for (size_t offset = globalTid; offset < std::max(alignment_offset,count); offset += globalNthreads) {
      T v_scalar = multimemLoadSum<T,T>(send_ptr + offset);
      multimemStore<T,T>(recv_ptr+offset, v_scalar);
    }
  }

  // separate TN* for 2 reasons. a) alignment offset, b) pointer arithmetic with the vectorized type
  TN* send_ptrN = reinterpret_cast<TN*>(ncclGetLsaMultimemPointer(sendwin, sendoffset+alignment_offset*sizeof(T), devComm));
  TN* recv_ptrN = reinterpret_cast<TN*>(ncclGetLsaMultimemPointer(recvwin, recvoffset+alignment_offset*sizeof(T), devComm));

  // Handle vectorized memory that can be handled in whole chunks (no if)
  for (int block = 0; block < num_blocks; block += 1) {
    TN v[UNROLL_FACTOR] = {TN{0}};
    const size_t block_offset = block * globalNthreads * UNROLL_FACTOR;
#pragma unroll
    for (int i=0; i < UNROLL_FACTOR; i++) {
      const int stride_offset = i * WARP_SIZE;
      const size_t offset = warp_lane_offset + block_offset + stride_offset;
      v[i] = multimemLoadSum<T,TN>(reinterpret_cast<T*>(send_ptrN + offset));
    }

#pragma unroll
    for (int i=0; i < UNROLL_FACTOR; i++) {
      const int stride_offset = i * WARP_SIZE;
      const size_t offset = warp_lane_offset + block_offset + stride_offset;
      multimemStore<T,TN>(reinterpret_cast<T*>(recv_ptrN+offset), v[i]);
    }
  }

  // Handle the last partial vectorized block, but with if conditions
  const int block = num_blocks;
  TN v[UNROLL_FACTOR] = {TN{0}};
  const size_t block_offset = block * globalNthreads * UNROLL_FACTOR;
#pragma unroll
  for (int i=0; i < UNROLL_FACTOR; i++) {
    const int stride_offset = i * WARP_SIZE;
    const size_t offset = warp_lane_offset + block_offset + stride_offset;
    if (offset < vector_count) {
      v[i] = multimemLoadSum<T,TN>(reinterpret_cast<T*>(send_ptrN+offset));
    }
  }
#pragma unroll
  for (int i=0; i < UNROLL_FACTOR; i++) {
    const int stride_offset = i * WARP_SIZE;
    const size_t offset = warp_lane_offset + block_offset + stride_offset;
    if (offset < vector_count) {
      multimemStore<T,TN>(reinterpret_cast<T*>(recv_ptrN+offset), v[i]);
    }
  }

  // Handle remainder elements using scalar operations
  if (remainder > 0) {
    const size_t remainder_start = alignment_offset + vector_count * VECTOR_FACTOR;
    const int globalTid_remainder = globalTid;
    const int globalNthreads_remainder = globalNthreads;

    for (size_t offset = globalTid_remainder; offset < remainder; offset += globalNthreads_remainder) {
      const size_t actual_offset = remainder_start + offset;
      T v_scalar = multimemLoadSum<T,T>(send_ptr+actual_offset);
      multimemStore<T,T>(recv_ptr+actual_offset, v_scalar);
    }
  }

  // Sync
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// ---------------------------------------------------------------------------
// GIN-SDMA AllReduce (-D 5)
// ---------------------------------------------------------------------------
// Size-hybrid single-node AllReduce. `count` is the whole-message element count
// (send == recv == count). Reductions fold in ascending source-rank order via
// gin_sdma_reduce {preOp,combine,postOp} to match verifiable.cu bit-for-bit;
// low-precision accumulators stay in T (narrow every pairwise step).
//
// Tier (compares TOTAL message bytes):
//   * small (<= threshold) AND out-of-place: ONE-SHOT direct LSA read-reduce.
//     Every rank reads the whole message from each peer's sendbuf, folds, and
//     writes its OWN recvbuf. Entry+exit LSA barrier, no scratch/signals. This is
//     out-of-place only: a one-shot kernel where every rank writes the full
//     buffer cannot run in place (a peer would overwrite input another rank is
//     still reading), so in-place always takes the two-shot path below.
//   * large (> threshold) OR in-place: TWO-SHOT ReduceScatter + AllGather.
//     The whole message is split into N per-rank slices (allReduceSliceStride,
//     16 B aligned). Rank r owns slice r. Each slice is further tiled ACROSS CTAs
//     so every CTA is SELF-CONTAINED: CTA k reduces tile k of its owned slice and
//     then AllGather-puts exactly that same tile. This is deliberate -- a
//     single-kernel two-phase composition where the RS output is consumed by the
//     AG has a producer->consumer dependency, and the per-CTA-slot LSA/GIN
//     barriers only synchronize a CTA slot ACROSS ranks, not all CTAs within a
//     rank; a grid-wide sync is unavailable (plain <<<>>> launch, no cooperative
//     grid). Making CTA k both produce and consume tile k reduces the cross-phase
//     dependency to an intra-CTA __syncthreads + a system fence.
//     AG uses a PER-CTA GIN signal (index = blockIdx.x): sender CTA k -> receiver
//     signal k, so each CTA's completion count is independent and computed from
//     the (deterministic, rank-agnostic) tiling via arTile/arTileNonemptyPeers.
//
// In-place safety of the two-shot path: only rank r reads slice r (from every
// peer) during RS, sequenced before r's AG put of that slice (same rank), so no
// other rank reads a region r overwrites; every write target is r's own slice,
// disjoint across ranks. Holds for both variants.

// Element range [begElt,endElt) of rank s's owned slice handled by CTA k.
// Slice s = [s*stride, min((s+1)*stride,msg)). Tiled into nCTA contiguous,
// 16 B-aligned pieces (last CTA takes the remainder incl. any scalar tail).
// Identical on every rank, so senders and receivers agree on the split.
template <typename T>
__device__ __forceinline__ void arTile(size_t msgBytes, size_t strideBytes, int s, int k,
                                        int nCTA, size_t& begElt, size_t& endElt) {
  constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
  const size_t eltSize = sizeof(T);
  const size_t byteOff = (size_t)s * strideBytes;
  if (byteOff >= msgBytes) { begElt = endElt = 0; return; }
  const size_t byteCnt = (strideBytes < msgBytes - byteOff) ? strideBytes : (msgBytes - byteOff);
  const size_t sOff = byteOff / eltSize;   // strideBytes is 16 B aligned -> VEC aligned
  const size_t sCnt = byteCnt / eltSize;
  size_t b = ((size_t)k * sCnt) / (size_t)nCTA;         b -= b % (size_t)VEC;
  size_t e;
  if (k == nCTA - 1) e = sCnt;
  else { e = ((size_t)(k + 1) * sCnt) / (size_t)nCTA;   e -= e % (size_t)VEC; }
  begElt = sOff + b;
  endElt = sOff + e;
}

// Number of OTHER ranks whose CTA-k tile is non-empty (== incoming per-CTA-signal
// puts a receiver expects on signal k in a symmetric all-to-peers AllGather).
template <typename T>
__device__ __forceinline__ int arTileNonemptyPeers(size_t msgBytes, size_t strideBytes,
                                                    int self, int k, int nCTA, int nRanks) {
  int n = 0;
  for (int s = 0; s < nRanks; s++) {
    if (s == self) continue;
    size_t b, e; arTile<T>(msgBytes, strideBytes, s, k, nCTA, b, e);
    if (e > b) n++;
  }
  return n;
}

// CTA-cooperative direct-LSA read-reduce of [begElt,endElt) into local recvbuf,
// folding every peer's sendbuf slice in ascending source-rank order. 128-bit
// packed with a scalar tail (only the last CTA's tile can have one). begElt is
// VEC aligned. Threads are CTA-local (threadIdx.x / blockDim.x).
template <typename T>
__device__ __forceinline__ void arReduceTileLsa(ncclWindow_t sendwin, size_t sendoffset,
                                                 ncclWindow_t recvwin, size_t recvoffset,
                                                 size_t begElt, size_t endElt, int nRanks, int redOp) {
  if (endElt <= begElt) return;
  const int tid = threadIdx.x, nthreads = blockDim.x;
  constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
  struct alignas(16) Pack { T e[VEC]; };
  const size_t cnt = endElt - begElt;
  const size_t nPacks = cnt / (size_t)VEC;
  const size_t basePk = begElt / (size_t)VEC;
  Pack* dstP = (Pack*)ncclGetLocalPointer(recvwin, recvoffset);
  for (size_t pk = (size_t)tid; pk < nPacks; pk += (size_t)nthreads) {
    const size_t gp = basePk + pk;
    Pack v0 = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, 0))[gp];
    T acc[VEC];
    #pragma unroll
    for (int e = 0; e < VEC; e++) acc[e] = gin_sdma_reduce::preOp(redOp, v0.e[e], nRanks);
    for (int s = 1; s < nRanks; s++) {
      Pack vs = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[gp];
      #pragma unroll
      for (int e = 0; e < VEC; e++)
        acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, vs.e[e], nRanks));
    }
    Pack o;
    #pragma unroll
    for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
    dstP[gp] = o;
  }
  const size_t tailBeg = begElt + nPacks * (size_t)VEC;
  T* dstS = (T*)ncclGetLocalPointer(recvwin, recvoffset);
  for (size_t i = tailBeg + (size_t)tid; i < endElt; i += (size_t)nthreads) {
    T acc = gin_sdma_reduce::preOp(redOp, ((const T*)ncclGetLsaPointer(sendwin, sendoffset, 0))[i], nRanks);
    for (int s = 1; s < nRanks; s++)
      acc = gin_sdma_reduce::combine(redOp, acc,
              gin_sdma_reduce::preOp(redOp, ((const T*)ncclGetLsaPointer(sendwin, sendoffset, s))[i], nRanks));
    dstS[i] = gin_sdma_reduce::postOp(redOp, acc, nRanks);
  }
}

// CTA-cooperative reduce of MY tile [begElt,endElt) for variant B: source s != me
// comes from my local scratch slot [s*strideBytes + (begElt-mySliceOff)*eltSize],
// source s == me from my own sendbuf. Ascending source-rank fold. 128-bit packed
// + scalar tail. begElt/mySliceOff are VEC aligned.
template <typename T>
__device__ __forceinline__ void arReduceTileScratch(ncclWindow_t sendwin, size_t sendoffset,
                                                     ncclWindow_t recvwin, size_t recvoffset,
                                                     const char* scratchLocal, size_t strideBytes,
                                                     size_t mySliceOffElt, size_t begElt, size_t endElt,
                                                     int self, int nRanks, int redOp) {
  if (endElt <= begElt) return;
  const int tid = threadIdx.x, nthreads = blockDim.x;
  constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
  struct alignas(16) Pack { T e[VEC]; };
  const size_t cnt = endElt - begElt;
  const size_t nPacks = cnt / (size_t)VEC;
  const size_t inSliceByte = (begElt - mySliceOffElt) * sizeof(T);  // this tile's byte offset inside my slice
  Pack* dstP = (Pack*)ncclGetLocalPointer(recvwin, recvoffset);
  const Pack* myInP = (const Pack*)ncclGetLsaPointer(sendwin, sendoffset, self);
  const size_t myBaseP = begElt / (size_t)VEC;
  for (size_t pk = (size_t)tid; pk < nPacks; pk += (size_t)nthreads) {
    // source 0
    Pack v0 = (self == 0) ? myInP[myBaseP + pk]
                          : ((const Pack*)(scratchLocal + 0 * strideBytes + inSliceByte))[pk];
    T acc[VEC];
    #pragma unroll
    for (int e = 0; e < VEC; e++) acc[e] = gin_sdma_reduce::preOp(redOp, v0.e[e], nRanks);
    for (int s = 1; s < nRanks; s++) {
      Pack vs = (s == self) ? myInP[myBaseP + pk]
                            : ((const Pack*)(scratchLocal + (size_t)s * strideBytes + inSliceByte))[pk];
      #pragma unroll
      for (int e = 0; e < VEC; e++)
        acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, vs.e[e], nRanks));
    }
    Pack o;
    #pragma unroll
    for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
    dstP[myBaseP + pk] = o;
  }
  // scalar tail (last CTA only)
  const size_t tailBeg = begElt + nPacks * (size_t)VEC;
  const size_t tailInSliceByte = inSliceByte + nPacks * (size_t)VEC * sizeof(T);
  T* dstS = (T*)ncclGetLocalPointer(recvwin, recvoffset);
  const T* myInS = (const T*)ncclGetLsaPointer(sendwin, sendoffset, self);
  for (size_t off = (size_t)tid; tailBeg + off < endElt; off += (size_t)nthreads) {
    const size_t i = tailBeg + off;
    T v0 = (self == 0) ? myInS[i]
                       : ((const T*)(scratchLocal + 0 * strideBytes + tailInSliceByte))[off];
    T acc = gin_sdma_reduce::preOp(redOp, v0, nRanks);
    for (int s = 1; s < nRanks; s++) {
      T vs = (s == self) ? myInS[i]
                         : ((const T*)(scratchLocal + (size_t)s * strideBytes + tailInSliceByte))[off];
      acc = gin_sdma_reduce::combine(redOp, acc, gin_sdma_reduce::preOp(redOp, vs, nRanks));
    }
    dstS[i] = gin_sdma_reduce::postOp(redOp, acc, nRanks);
  }
}

template <typename T>
__global__ void GinAllReduceKernel(ncclWindow_t sendwin, size_t sendoffset,
                                    ncclWindow_t recvwin, size_t recvoffset, size_t count,
                                    int root, struct ncclDevComm devComm,
                                    size_t sdmaThresholdOverride, int redOp,
                                    ncclDevResourceHandle scratchHandle) {
  const int nRanks = devComm.nRanks;
  const int rank = devComm.rank;
  const size_t msgBytes = count * sizeof(T);
  const size_t thr = (sdmaThresholdOverride == gin_sdma::kThresholdUnset)
                         ? gin_sdma::kAllReduceSdmaThresholdDefault : sdmaThresholdOverride;
  const bool inPlace = (sendwin == recvwin) && (sendoffset == recvoffset);
  const bool oneShot = (gin_sdma::allReduceKernelTier(msgBytes, thr) == gin_sdma::ARTier::LSA) && !inPlace;

  // ---- ONE-SHOT small out-of-place: direct LSA read-reduce of the whole buffer.
  if (oneShot) {
    ncclTeam lsa = ncclTeamLsa(devComm);
    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    // Grid-stride whole-buffer reduce (each thread writes only packs it computes,
    // so no cross-CTA dependency). Uses global tid across all CTAs.
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nthreads = blockDim.x * gridDim.x;
    constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
    struct alignas(16) Pack { T e[VEC]; };
    const size_t nPacks = count / (size_t)VEC;
    Pack* dstP = (Pack*)ncclGetLocalPointer(recvwin, recvoffset);
    for (size_t pk = (size_t)tid; pk < nPacks; pk += (size_t)nthreads) {
      Pack v0 = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, 0))[pk];
      T acc[VEC];
      #pragma unroll
      for (int e = 0; e < VEC; e++) acc[e] = gin_sdma_reduce::preOp(redOp, v0.e[e], nRanks);
      for (int s = 1; s < nRanks; s++) {
        Pack vs = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[pk];
        #pragma unroll
        for (int e = 0; e < VEC; e++)
          acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, vs.e[e], nRanks));
      }
      Pack o;
      #pragma unroll
      for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
      dstP[pk] = o;
    }
    const size_t tailBeg = nPacks * (size_t)VEC;
    T* dstS = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    for (size_t i = tailBeg + (size_t)tid; i < count; i += (size_t)nthreads) {
      T acc = gin_sdma_reduce::preOp(redOp, ((const T*)ncclGetLsaPointer(sendwin, sendoffset, 0))[i], nRanks);
      for (int s = 1; s < nRanks; s++)
        acc = gin_sdma_reduce::combine(redOp, acc,
                gin_sdma_reduce::preOp(redOp, ((const T*)ncclGetLsaPointer(sendwin, sendoffset, s))[i], nRanks));
      dstS[i] = gin_sdma_reduce::postOp(redOp, acc, nRanks);
    }
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  // ---- TWO-SHOT (large OR in-place): per-CTA self-contained RS + AG.
  const int nCTA = gridDim.x;
  const int cta = blockIdx.x;
  const size_t strideBytes = gin_sdma::allReduceSliceStride(msgBytes, nRanks);
  const size_t mySliceOffElt = ((size_t)rank * strideBytes) / sizeof(T);
  size_t tBeg, tEnd; arTile<T>(msgBytes, strideBytes, rank, cta, nCTA, tBeg, tEnd);

  const int ginContext = 0;
  const unsigned int sig = (unsigned int)cta;       // per-CTA signal
  ncclGin gin { devComm, ginContext };

  if (scratchHandle == 0) {
    // ===== Variant A: direct-LSA read-reduce RS, then GIN-put AG =====
    const uint64_t sigBase = gin.readSignal(sig);   // baseline BEFORE the barrier (no puts yet)
    ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

    // RS: reduce my tile from every peer's sendbuf into my recvbuf.
    arReduceTileLsa<T>(sendwin, sendoffset, recvwin, recvoffset, tBeg, tEnd, nRanks, redOp);
    __syncthreads();
    __threadfence_system();  // publish this CTA's tile writes to the SDMA engine

    // AG: put my (reduced) tile to every other rank; wait for peers' tile-k puts.
    if (tEnd > tBeg && threadIdx.x == 0) {
      const size_t off = recvoffset + tBeg * sizeof(T);
      const size_t bytes = (tEnd - tBeg) * sizeof(T);
      for (int p = 0; p < nRanks; p++) {
        if (p == rank) continue;
        ginPutChunked(gin, ncclTeamWorld(devComm), p, recvwin, off, recvwin, off, bytes,
                      ncclGin_SignalInc{sig});
      }
    }
    const int expected = arTileNonemptyPeers<T>(msgBytes, strideBytes, rank, cta, nCTA, nRanks);
    gin.waitSignal(ncclCoopCta(), sig, sigBase + (uint64_t)expected);
    gin.flush(ncclCoopCta());
    return;
  }

  // ===== Variant B: put-partials into scratch + SM reduce, then GIN-put AG =====
  const size_t scratchOff = ncclGetResourceBufferOffset(scratchHandle);
  const uint64_t rsBase = gin.readSignal(sig);
  ncclBarrier<ncclCoopCta>(ncclCoopCta(), ncclTeamTagWorld(), gin, (uint32_t)blockIdx.x,
                           cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
  // RS phase 1a: CTA k sends tile k of EACH peer p's slice (from my sendbuf) into
  // p's scratch slot [rank*stride + (tile offset within p's slice)].
  if (threadIdx.x == 0) {
    for (int p = 0; p < nRanks; p++) {
      if (p == rank) continue;
      size_t pb, pe; arTile<T>(msgBytes, strideBytes, p, cta, nCTA, pb, pe);
      if (pe <= pb) continue;
      const size_t pSliceOffElt = ((size_t)p * strideBytes) / sizeof(T);
      const size_t inSlice = (pb - pSliceOffElt) * sizeof(T);
      ginPutChunked(gin, ncclTeamWorld(devComm), p,
                    devComm.resourceWindow, scratchOff + (size_t)rank * strideBytes + inSlice,
                    sendwin, sendoffset + pb * sizeof(T),
                    (pe - pb) * sizeof(T), ncclGin_SignalInc{sig});
    }
  }
  // Wait for peers' partials of MY tile k to land, then SM-reduce.
  {
    const int expIn = (tEnd > tBeg) ? (nRanks - 1) : 0;
    gin.waitSignal(ncclCoopCta(), sig, rsBase + (uint64_t)expIn);
    gin.flush(ncclCoopCta());
  }
  __threadfence_system();
  const char* scratchLocal = (const char*)ncclGetResourceBufferLocalPointer(devComm, scratchHandle);
  arReduceTileScratch<T>(sendwin, sendoffset, recvwin, recvoffset, scratchLocal, strideBytes,
                         mySliceOffElt, tBeg, tEnd, rank, nRanks, redOp);
  __syncthreads();
  __threadfence_system();

  // AG phase: same as variant A (re-baseline the per-CTA signal after RS).
  const uint64_t agBase = gin.readSignal(sig);
  ncclBarrier<ncclCoopCta>(ncclCoopCta(), ncclTeamTagWorld(), gin, (uint32_t)blockIdx.x,
                           cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
  if (tEnd > tBeg && threadIdx.x == 0) {
    const size_t off = recvoffset + tBeg * sizeof(T);
    const size_t bytes = (tEnd - tBeg) * sizeof(T);
    for (int p = 0; p < nRanks; p++) {
      if (p == rank) continue;
      ginPutChunked(gin, ncclTeamWorld(devComm), p, recvwin, off, recvwin, off, bytes,
                    ncclGin_SignalInc{sig});
    }
  }
  const int expAg = arTileNonemptyPeers<T>(msgBytes, strideBytes, rank, cta, nCTA, nRanks);
  gin.waitSignal(ncclCoopCta(), sig, agBase + (uint64_t)expAg);
  gin.flush(ncclCoopCta());
}
#endif

testResult_t AllReduceRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {

  char* sptr = (char*)sendbuff + sendoffset;
  char* rptr = (char*)recvbuff + recvoffset;

  switch (deviceImpl) {
  case 0:
    NCCLCHECK(ncclAllReduce(sptr, rptr, count, type, op, comm, stream));
    return testSuccess;
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  case 1:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(allReduceLsaKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
  case 2:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(allReduceLsaVectorizedKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
  case 3:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(allReduceMultimemKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
  case 4:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(allReduceMultimemVectorizedKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
  case 5: {
    if (count == 0) return testSuccess;
    // GIN-SDMA one-shot(small)/two-shot(large) reduction. Threshold compares
    // TOTAL message bytes; override via NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLREDUCE
    // or the shared NCCL_GIN_ANVIL_SDMA_THRESHOLD. Variant A (no scratch) unless
    // NCCL_GIN_ANVIL_AR_RS_PUTPARTIALS registered the scratch window (variant B).
    static const size_t arThr = testResolveSdmaThreshold("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLREDUCE", gin_sdma::kAllReduceSdmaThresholdDefault);
    TESTCHECK(testLaunchDeviceKernelThresholdScratch(SPECIALIZE_REDUCE_KERNEL(GinAllReduceKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, arThr, g_arScratchHandle));
    return testSuccess;
  }
#endif
#endif
  }

  return testNotImplemented;
}

struct testColl allReduceTest = {
  "AllReduce",
  AllReduceGetCollByteCount,
  AllReduceInitData,
  AllReduceGetBw,
  AllReduceRunColl,
  AllReduceGetAlgoProtoChannels,
  AllReduceGetSymkInfo
};

void AllReduceGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllReduceGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AllReduceRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &allReduceTest;
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
#if defined(RCCL_FLOAT8)
  if((run_types[i] == ncclFloat8e4m3 || run_types[i] == ncclFloat8e5m2) && (run_ops[j] == ncclProd || run_ops[j] == ncclAvg || strcmp(run_opnames[j],"mulsum") == 0))
    continue;
#endif
      // GIN device path (deviceImpl != 0) has no PreMulSum ("mulsum") kernel for
      // any type (deferred): SPECIALIZE_REDUCE_KERNEL returns nullptr, which would
      // abort the matrix on testNotImplemented. Skip it here; the host path
      // (deviceImpl == 0) keeps full coverage via ncclAllReduce.
      if (deviceImpl != 0 && strcmp(run_opnames[j], "mulsum") == 0) continue;
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1));
    }
  }
  return testSuccess;
}

struct testEngine ncclTestEngine = {
  .getBuffSize = AllReduceGetBuffSize,
  .runTest = AllReduceRunTest,
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = AllReduceGetDevCommRequirements
#endif
};
