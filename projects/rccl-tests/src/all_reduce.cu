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

// GIN-SDMA AllReduce (-D 5 / -D 6). The upstream LSA/multimem demo kernels keep -D 1..4.
// Both are the GIN-SDMA reduction (ReduceScatter then single-signal in-place AllGather,
// all sizes), differing only in the RS->AG sync: -D 5 (default) is ONE kernel with an
// in-kernel device-wide barrier (arGridBarrier); -D 6 is the TWO-LAUNCH alternative (RS
// kernel then AG kernel on the same stream, boundary = kernel-launch boundary). A one-shot
// small-message fast path was prototyped and dropped -- it re-triggered the cumulative GPU
// hang; see the notes above the kernels. PreMulSum/mulsum is deferred
// (SPECIALIZE_REDUCE_KERNEL -> nullptr -> testNotImplemented); fp8 {prod,avg,mulsum}
// excluded (see AllReduceRunTest).
static ncclDevResourceHandle g_arScratchHandle = 0;  // always 0: split kernels use no scratch
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

// Empirically-tuned CTA count for the GIN-SDMA AllReduce (-D 5/-D 6) on 8x MI355X (xGMI).
// The ReduceScatter reduction is memory-bandwidth bound and keeps scaling with CTAs up to
// ~64 for large messages, but small messages prefer FEW CTAs (barrier/launch overhead
// dominates and over-subscription hurts). So the grid is chosen per message size at launch,
// while barrier/signal slots are registered for the max (kArMaxCtas). Measured busbw
// (float,sum): 128M 313->369 GB/s, 64M 292->328, 1M 34->40, 64K 2.7->3.2 vs the old fixed
// 32-CTA grid. Beyond 64 CTAs large-message BW regresses (co-residency pressure).
static constexpr int kArMaxCtas = 64;
static inline int arTunedGridCtas(size_t msgBytes, int cap) {
  int g = (msgBytes < ((size_t)512 << 10)) ? 8      // <512 KiB
        : (msgBytes < ((size_t)8   << 20)) ? 16     // <8 MiB
        : (msgBytes < ((size_t)32  << 20)) ? 32     // <32 MiB
        : 64;                                       // >=32 MiB
  if (g > cap) g = cap;
  return (g < 1) ? 1 : g;
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
    case 5:   // GinAllReduceKernel:            single-launch RS + AllGather (all sizes)
    case 6: { // GinAllReduceRSKernel/AGKernel: two-launch  RS + AllGather (all sizes)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      // Register for the MAX grid we might launch (size-adaptive up to kArMaxCtas).
      gin_sdma::DevReqs dr = gin_sdma::allReduceDevReqs(deviceCtaCount > kArMaxCtas ? deviceCtaCount : kArMaxCtas);
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
    case 5:   // GinAllReduceKernel:            single-launch RS + AllGather (all sizes)
    case 6: { // GinAllReduceRSKernel/AGKernel: two-launch  RS + AllGather (all sizes)
      gin_sdma::DevReqs dr = gin_sdma::allReduceDevReqs(deviceCtaCount > kArMaxCtas ? deviceCtaCount : kArMaxCtas);
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
// GIN-SDMA AllReduce (-D 5 single-launch / -D 6 two-launch): RS + AllGather (all sizes)
// ---------------------------------------------------------------------------
// Single-node AllReduce. `count` is the whole-message element count (send == recv ==
// count). Reductions fold in ascending source-rank order via gin_sdma_reduce
// {preOp,combine,postOp} to match verifiable.cu bit-for-bit; low-precision
// accumulators stay in T (narrow every pairwise step).
//
// EVERY size goes through ONE kernel (GinAllReduceKernel) with two
// non-overlapping phases separated by a device-wide barrier: a ReduceScatter phase
// (LSA read-reduce of the owned slice) then an AllGather phase using the proven
// single-signal put pattern (O(N) puts/rank on one recycled signal). This replaced an
// earlier fused single-kernel two-shot (per-CTA self-contained RS+AG on per-CTA GIN
// signals) that hung under long dense sweeps, and then a two-launch RS/AG split (same
// GIN pattern, RS->AG boundary at the kernel-launch boundary); the device-wide barrier
// lets us keep that stable GIN pattern in a SINGLE launch. Full design at GinAllReduceKernel.
//
// NOTE (2026-08-01, 8x MI355X): a one-shot direct-LSA read-reduce fast path for the
// small out-of-place tier was prototyped and dropped. Interleaving those LSA-only
// launches between the GIN AllGather launches deterministically re-triggered the
// cumulative "GPU Hang" on the full type x op x size sweep (hung 2/2 runs), whereas
// split-for-all-sizes ran clean 3/3 (2x -D 6 + 1x -D 5 THRESHOLD=0, 1400 rows each,
// #wrong=0). The one-shot has no GIN of its own, so this is a backend GIN
// signal-completion-pipeline disturbance from the irregular LSA/GIN launch cadence,
// not a signal-op-volume effect (the one-shot mix issues FEWER AG launches yet hung).
// Keep AllReduce on the uniform split; do not reintroduce a non-GIN fast path without
// a backend fix for GIN completion recycling.

// ---------------------------------------------------------------------------
// GIN-SDMA AllReduce: ReduceScatter + AllGather (all sizes)
// ---------------------------------------------------------------------------
// The collective is a ReduceScatter followed by an in-place AllGather. There are TWO
// selectable realizations that share the SAME two phase bodies (factored below into
// arReduceScatterOwnedSlice + arAllGatherInPlace) and the SAME per-launch GIN signal
// pattern; they differ ONLY in how the RS->AG boundary is synchronized:
//
//   -D 5  SINGLE-LAUNCH (default): one kernel (GinAllReduceKernel) runs RS, then an
//         in-kernel device-wide barrier (arGridBarrier), then AG. Halves host launches.
//         Trade-off: the grid barrier busy-spins, so it REQUIRES every CTA co-resident
//         on the GPU -- fine for the small grids used here, but a grid that oversubscribes
//         the CUs would deadlock. (cg::this_grid()/hipLaunchCooperativeKernel would be the
//         "correct" primitive but SIGSEGVs on this ROCm build -- deferred kernel-binary
//         lookup fault, ROCm issue #2805 -- so the barrier is hand-rolled.)
//
//   -D 6  TWO-LAUNCH (alternative): two kernels (GinAllReduceRSKernel then
//         GinAllReduceAGKernel) back-to-back on the SAME stream, so the RS->AG boundary
//         is the kernel-launch boundary -- a true global sync via stream ordering that
//         fully drains the grid. No co-residency requirement and cannot deadlock on grid
//         size, at the cost of a second host launch per collective. Use this if the grid
//         must exceed device residency, or as a robustness fallback.
//
// Phase 1 -- ReduceScatter (arReduceScatterOwnedSlice): each rank LSA-read-reduces its
//   OWNED slice [rank*stride, ...) from every peer's sendbuf into its recvbuf slice,
//   folding in ascending source-rank order (bit-matching verifiable.cu via
//   gin_sdma_reduce). Pure intra-node LSA loads + local stores: NO GIN puts/signals.
//   In-place safe (each rank owns a disjoint output slice and only reads that slice index
//   across peers). Grid-strided across ALL CTAs -> the whole owned slice is produced
//   jointly by the launch, which is why the RS->AG boundary needs a grid-wide sync.
//
// Phase 2 -- AllGather (arAllGatherInPlace): the PROVEN, gate-stable
//   GinHybridAllGatherKernel signal pattern -- a SINGLE world-barrier entry, one put of
//   this rank's reduced slice to every OTHER peer's matching slot on a SINGLE GIN signal
//   (index 0), then waitSignal for the incoming slices. O(N) puts/rank on one recycled
//   signal (vs an earlier fused two-shot's O(nCTA*N) puts over nCTA per-CTA signals, the
//   source of the cumulative SDMA/GIN completion-recycling hang). The signal baseline uses
//   the persistent shadow ledger (deterministic across launches). `expected` = number of
//   OTHER ranks with a non-empty owned slice; my own slice is already in place from Phase 1.

// Device-wide (single-GPU, all-CTAs) sense-reversing barrier for the RS->AG phase
// boundary. Plain <<<>>> launches have no built-in grid sync and cooperative launch
// SIGSEGVs on this ROCm build, so we roll our own via two global counters. Called
// EXACTLY ONCE per kernel (single stream -> launches serialize -> no concurrent use;
// the last arriver resets the arrive counter and bumps a monotonic release sense, so
// no host-side reset is needed). REQUIRES every CTA co-resident -- true for the small
// grids used here (deviceCtaCount <= 128 @ 512 threads on MI355X); a plain launch that
// oversubscribes would deadlock here, same residency constraint cooperative launch has.
__device__ unsigned int g_arBarArrive = 0;
__device__ unsigned int g_arBarRelease = 0;
__device__ __forceinline__ void arGridBarrier() {
  __syncthreads();
  __threadfence();
  if (threadIdx.x == 0) {
    const unsigned int rel = atomicAdd(&g_arBarRelease, 0u);        // release sense before arriving
    const unsigned int arrived = atomicAdd(&g_arBarArrive, 1u) + 1u;
    if (arrived == gridDim.x) {
      atomicExch(&g_arBarArrive, 0u);                               // reset for the next launch
      __threadfence();
      atomicAdd(&g_arBarRelease, 1u);                              // release all waiters
    } else {
      while (atomicAdd(&g_arBarRelease, 0u) == rel) { /* spin */ }
    }
  }
  __syncthreads();
}

// Phase 1 body: reduce MY owned slice [rank*stride, ...) from every peer's sendbuf into
// my recvbuf slice. Grid-strided across ALL CTAs; NO barriers (callers bracket it with
// the appropriate entry/exit sync). In-place safe (disjoint owned slices).
template <typename T>
__device__ __forceinline__ void arReduceScatterOwnedSlice(
    ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset,
    size_t msgBytes, size_t strideBytes, int rank, int nRanks, int redOp) {
  const size_t begByte = (size_t)rank * strideBytes;
  if (begByte >= msgBytes) return;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  const size_t endByte = (strideBytes < msgBytes - begByte) ? begByte + strideBytes : msgBytes;
  const size_t begElt = begByte / sizeof(T);   // strideBytes is 16 B aligned -> VEC aligned
  const size_t endElt = endByte / sizeof(T);
  constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
  struct alignas(16) Pack { T e[VEC]; };
  const size_t nPacks = (endElt - begElt) / (size_t)VEC;
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
  // scalar tail (only the last owned slice can have one; its endByte == msgBytes)
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

// Phase 2 body: in-place AllGather of the N reduced slices. SINGLE world-barrier entry,
// one put of MY reduced slice to every OTHER peer on a SINGLE GIN signal, waitSignal for
// the incoming slices. `expected` = # of OTHER active ranks (each sends me its slice).
template <typename T>
__device__ __forceinline__ void arAllGatherInPlace(
    ncclWindow_t recvwin, size_t recvoffset, size_t msgBytes, size_t strideBytes,
    int rank, int nRanks, struct ncclDevComm devComm) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  const size_t begByte = (size_t)rank * strideBytes;

  int nActive = 0;                                    // ranks with a non-empty owned slice
  for (int s = 0; s < nRanks; s++) if ((size_t)s * strideBytes < msgBytes) nActive++;
  const bool iSend = begByte < msgBytes;
  const size_t myBytes = iSend ? ((strideBytes < msgBytes - begByte) ? strideBytes : (msgBytes - begByte)) : 0;
  const int expected = nActive - (iSend ? 1 : 0);     // slices I receive from OTHER active ranks

  const unsigned int sig = 0;                          // single shared signal (AllGather pattern)
  ncclGin gin { devComm, 0 };
  uint64_t* shadowPtr = gin.getSignalShadowPtr(sig);
  const uint64_t sigBase = *shadowPtr;

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  if (iSend) {
    // Each thread drives one destination peer; put MY reduced slice to that peer's
    // matching slot. Skip self (my slice is already in place from Phase 1).
    for (int p = tid; p < nRanks; p += nthreads) {
      if (p == rank) continue;
      ginPutChunked(gin, ncclTeamWorld(devComm), p,
                    recvwin, recvoffset + begByte, recvwin, recvoffset + begByte,
                    myBytes, ncclGin_SignalInc{sig});
    }
  }
  gin.waitSignal(ncclCoopCta(), sig, sigBase + (uint64_t)expected);
  gin.flush(ncclCoopCta());
  if (tid == 0) *shadowPtr = sigBase + (uint64_t)expected;   // persist ledger (single grid-wide writer)
}

// -D 5 SINGLE-LAUNCH: RS -> in-kernel device-wide barrier -> AG, all in one kernel.
template <typename T>
__global__ void GinAllReduceKernel(ncclWindow_t sendwin, size_t sendoffset,
                                   ncclWindow_t recvwin, size_t recvoffset, size_t count,
                                   int root, struct ncclDevComm devComm,
                                   size_t sdmaThresholdOverride, int redOp,
                                   ncclDevResourceHandle scratchHandle) {
  const int nRanks = devComm.nRanks;
  const int rank = devComm.rank;
  const size_t msgBytes = count * sizeof(T);
  const size_t strideBytes = gin_sdma::allReduceSliceStride(msgBytes, nRanks);

  // Entry cross-rank barrier: every rank's sendbuf is populated (LSA).
  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
  lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  arReduceScatterOwnedSlice<T>(sendwin, sendoffset, recvwin, recvoffset, msgBytes, strideBytes, rank, nRanks, redOp);

  // Phase boundary: device-wide barrier so RS is globally done before AG reads it.
  __threadfence_system();     // publish this rank's reduced-slice stores
  arGridBarrier();

  arAllGatherInPlace<T>(recvwin, recvoffset, msgBytes, strideBytes, rank, nRanks, devComm);
}

// -D 6 TWO-LAUNCH, phase 1: standalone ReduceScatter. The RS->AG boundary is the
// kernel-launch boundary (stream ordering), so no in-kernel grid barrier is needed;
// entry+exit LSA barrier only.
template <typename T>
__global__ void GinAllReduceRSKernel(ncclWindow_t sendwin, size_t sendoffset,
                                     ncclWindow_t recvwin, size_t recvoffset, size_t count,
                                     int root, struct ncclDevComm devComm,
                                     size_t sdmaThresholdOverride, int redOp,
                                     ncclDevResourceHandle scratchHandle) {
  const int nRanks = devComm.nRanks;
  const int rank = devComm.rank;
  const size_t msgBytes = count * sizeof(T);
  const size_t strideBytes = gin_sdma::allReduceSliceStride(msgBytes, nRanks);

  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
  lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  arReduceScatterOwnedSlice<T>(sendwin, sendoffset, recvwin, recvoffset, msgBytes, strideBytes, rank, nRanks, redOp);

  lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
}

// -D 6 TWO-LAUNCH, phase 2: standalone in-place AllGather (launched after the RS kernel
// on the same stream).
template <typename T>
__global__ void GinAllReduceAGKernel(ncclWindow_t sendwin, size_t sendoffset,
                                     ncclWindow_t recvwin, size_t recvoffset, size_t count,
                                     int root, struct ncclDevComm devComm,
                                     size_t sdmaThresholdOverride, int redOp,
                                     ncclDevResourceHandle scratchHandle) {
  const int nRanks = devComm.nRanks;
  const int rank = devComm.rank;
  const size_t msgBytes = count * sizeof(T);
  const size_t strideBytes = gin_sdma::allReduceSliceStride(msgBytes, nRanks);
  arAllGatherInPlace<T>(recvwin, recvoffset, msgBytes, strideBytes, rank, nRanks, devComm);
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
    // GIN-SDMA AllReduce, SINGLE-LAUNCH (default): ONE kernel with two non-overlapping
    // phases (ReduceScatter -> device-wide arGridBarrier() -> single-signal in-place
    // AllGather) for ALL sizes. Grid is size-adaptive (arTunedGridCtas). arThr is passed
    // for signature parity only; the kernel ignores it.
    static const size_t arThr = testResolveSdmaThreshold("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLREDUCE", gin_sdma::kAllReduceSdmaThresholdDefault);
    const int arGctas = arTunedGridCtas(count * wordSize(type), kArMaxCtas);
    TESTCHECK(testLaunchDeviceKernelThresholdScratchCtas(SPECIALIZE_REDUCE_KERNEL(GinAllReduceKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, arThr, g_arScratchHandle, arGctas));
    return testSuccess;
  }
  case 6: {
    if (count == 0) return testSuccess;
    // GIN-SDMA AllReduce, TWO-LAUNCH alternative: same two phases as -D 5 but as two
    // kernels back-to-back on the SAME stream (RS then AG), so the RS->AG boundary is the
    // kernel-launch boundary. No co-residency requirement / cannot deadlock on grid size,
    // at the cost of a second host launch. Same size-adaptive grid. arThr: parity only.
    static const size_t arThr = testResolveSdmaThreshold("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLREDUCE", gin_sdma::kAllReduceSdmaThresholdDefault);
    const int arGctas = arTunedGridCtas(count * wordSize(type), kArMaxCtas);
    TESTCHECK(testLaunchDeviceKernelAR2SplitCtas(SPECIALIZE_REDUCE_KERNEL(GinAllReduceRSKernel, type, op),
               SPECIALIZE_REDUCE_KERNEL(GinAllReduceAGKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, arThr, g_arScratchHandle, arGctas));
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
