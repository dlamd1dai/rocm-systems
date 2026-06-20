/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_H_

#include "../gin_device_common.h"
#include "gin_anvil_sdma_device_host_common.h"
#include "sdma/anvil_device.hpp"

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes, ncclGinSignalDescriptor signal,
                                      ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                      cuda::thread_scope given, uint32_t optFlags = ncclGinOptFlagsDefault) {
    using nccl::utility::loadConst;
    bool hasSignal = signal.type != NCCL_GIN_SIGNAL_TYPE_NONE;

    coop.sync();
    if (coop.thread_rank() == 0) {
      ncclGinAnvilSdmaGPUContext* rsCtx = (ncclGinAnvilSdmaGPUContext*)ctx.handle;
      int numCh = loadConst(&rsCtx->numChannels);
      auto** handles = (rocshmem::anvil::SdmaQueueDeviceHandle**)loadConst(&rsCtx->queueHandles);
      int eff_ch = 0;
      rocshmem::anvil::SdmaQueueDeviceHandle* handle =
          loadConst(handles + peer * numCh + eff_ch);

      if ((required == cuda::thread_scope_system) && (given > required)) {
        __threadfence_system();
      }

      if (hasWins && handle != nullptr) {
        ncclGinAnvilSdmaMemHandle* dstMh = (ncclGinAnvilSdmaMemHandle*)dstWin;
        ncclGinAnvilSdmaMemHandle* srcMh = (ncclGinAnvilSdmaMemHandle*)srcWin;
        uintptr_t dstAddr = loadConst(loadConst(&dstMh->remote_vas) + peer) + dstOff;
        uintptr_t srcAddr = loadConst(&srcMh->local_va) + srcOff;
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        rocshmem::anvil::put(*handle, (void*)dstAddr, (void*)srcAddr, bytes);
        uint64_t bit = 1ULL << (peer * numCh + eff_ch);
        __hip_atomic_fetch_or(rsCtx->sdmaDirty, bit, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }

      if (hasSignal || hasCounter) {
        if (hasCounter && handle != nullptr)
          rocshmem::anvil::quiet(*handle);
        else
          __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        if (hasSignal) {
          if (signalOp == ncclGinSignalInc) signalOpArg = 1;
          uintptr_t sigBase = loadConst(loadConst(&rsCtx->signal_peer_addrs) + peer);
          uint64_t* sigPtr =
              (uint64_t*)(sigBase + sizeof(uint64_t) * (size_t)signal.indexedSignal.signalId);
          if (handle != nullptr) rocshmem::anvil::signal(*handle, sigPtr);
        }
        if (hasCounter) {
          atomicAdd((unsigned long long*)&loadConst(&rsCtx->counters)[counterId], 1ULL);
        }
      }
    }
    coop.sync();
  }
};

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_ANVIL_SDMA> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal, ncclGinSignalDescriptor signal,
                                      ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                      cuda::thread_scope given, uint32_t optFlags = ncclGinOptFlagsDefault) {
    using nccl::utility::loadConst;
    bool hasSignal = signal.type != NCCL_GIN_SIGNAL_TYPE_NONE;
    coop.sync();
    if (coop.thread_rank() == 0) {
      ncclGinAnvilSdmaGPUContext* rsCtx = (ncclGinAnvilSdmaGPUContext*)ctx.handle;
      int numCh = loadConst(&rsCtx->numChannels);
      auto** handles = (rocshmem::anvil::SdmaQueueDeviceHandle**)loadConst(&rsCtx->queueHandles);
      rocshmem::anvil::SdmaQueueDeviceHandle* handle = loadConst(handles + peer * numCh + 0);
      ncclGinAnvilSdmaMemHandle* dstMh = (ncclGinAnvilSdmaMemHandle*)dstWin;
      uintptr_t dstAddr = loadConst(loadConst(&dstMh->remote_vas) + peer) + dstOff;
      T tmp = srcVal;
      if ((required == cuda::thread_scope_system) && (given > required)) {
        __threadfence_system();
      }
      if (handle != nullptr) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        rocshmem::anvil::put(*handle, (void*)dstAddr, (void*)&tmp, sizeof(T));
        uint64_t bit = 1ULL << (peer * numCh + 0);
        __hip_atomic_fetch_or(rsCtx->sdmaDirty, bit, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      if (hasSignal) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        if (signalOp == ncclGinSignalInc) signalOpArg = 1;
        uintptr_t sigBase = loadConst(loadConst(&rsCtx->signal_peer_addrs) + peer);
        uint64_t* sigPtr =
            (uint64_t*)(sigBase + sizeof(uint64_t) * (size_t)signal.indexedSignal.signalId);
        if (handle != nullptr) rocshmem::anvil::signal(*handle, sigPtr);
      }
    }
    coop.sync();
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_ANVIL_SDMA> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinAnvilSdmaGPUContext* rsCtx = (ncclGinAnvilSdmaGPUContext*)ctx.handle;
    return nccl::utility::loadConst(&rsCtx->counters) + counterId;
  }
};

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_ANVIL_SDMA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinAnvilSdmaGPUContext* rsCtx = (ncclGinAnvilSdmaGPUContext*)ctx.handle;
    nccl::utility::loadConst(&rsCtx->counters)[counterId] = 0;
  }
};

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_ANVIL_SDMA> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    ncclGinAnvilSdmaGPUContext* rsCtx = (ncclGinAnvilSdmaGPUContext*)ctx.handle;
    return nccl::utility::loadConst(&rsCtx->signals) + signalId;
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_ANVIL_SDMA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    ncclGinAnvilSdmaGPUContext* rsCtx = (ncclGinAnvilSdmaGPUContext*)ctx.handle;
    if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED)
      nccl::utility::loadConst(&rsCtx->signals)[signal.indexedSignal.signalId] = 0;
  }
};

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ANVIL_SDMA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, cuda::memory_order ord,
                                      uint32_t* abortFlag) {
    using nccl::utility::loadConst;
    ncclGinAnvilSdmaGPUContext* rsCtx = (ncclGinAnvilSdmaGPUContext*)ctx.handle;
    auto** handles = (rocshmem::anvil::SdmaQueueDeviceHandle**)loadConst(&rsCtx->queueHandles);
    int nr = ctx.nRanks;
    int numCh = loadConst(&rsCtx->numChannels);
#pragma unroll 1
    for (int peer = coop.thread_rank(); peer < nr; peer += coop.size()) {
      for (int ch = 0; ch < numCh; ++ch) {
        rocshmem::anvil::SdmaQueueDeviceHandle* h = loadConst(handles + peer * numCh + ch);
        if (h != nullptr) rocshmem::anvil::quiet(*h);
      }
    }
    coop.sync();
  }
};

#endif
