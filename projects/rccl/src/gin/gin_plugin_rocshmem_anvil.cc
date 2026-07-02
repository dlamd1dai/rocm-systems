/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

/**
 * GIN plugin: SDMA Anvil device path (NCCL_GIN_TYPE=6).
 * Host flow mirrors GIN-GDA: allgather per-rank VAs at regMrSym/createContext.
 * Data movement uses standalone Anvil SDMA (rocshmem_gin_anvil_create) plus
 * device-side IPC flat stores for small messages — no rocSHMEM runtime API.
 */

#include "gin/gin_host_rocshmem_common.h"
#include "gin/gin_host_rocshmem_anvil.h"
#include "comm.h"
#include "dev_runtime.h"
#include "bootstrap.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
#include <rocshmem/gin_anvil_factory.h>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>

struct ginAnvilCollCtx {
  int nranks;
  int rank;
  struct ncclComm* comm;
  rocshmem_gin_anvil_handle_t anvil;
  void** gpu_queue_handles;
  uint64_t* sdma_dirty_d;
  int numChannels;
  int sdmaChannelStride;
};

struct ginAnvilGinCtx {
  ncclNetDeviceHandle_v11_t* devHandle;
  ncclGinAnvilSdmaGPUContext* gpuCtxDev;
  ncclGinAnvilSdmaGPUContext gpuCtxHost;
  int nRanks;
  int rank;
  int nSignals;
  int nCounters;
  bool hasError;
  void** gpu_queue_handles;
  uint64_t* sdma_dirty_d;
  int numChannels;
  int sdmaChannelStride;
};

struct ginAnvilMemHandle {
  ncclGinAnvilSdmaMemHandle* devHandle;
  uintptr_t* remoteVasDev;
  void* addr;
  size_t size;
};

struct ginAnvilListenCtx {
  int dev;
};

static int ginAnvilBootstrapAllgather(void* ctx, void* buf, size_t perRankSize) {
  return (bootstrapAllGather(ctx, buf, (int)perRankSize) == ncclSuccess) ? 0 : -1;
}

static ncclResult_t ginAnvilInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  const char* gin_type = getenv("NCCL_GIN_TYPE");
  if (gin_type && atoi(gin_type) != NCCL_NET_DEVICE_GIN_ANVIL_SDMA) return ncclInternalError;
  if (rocshmem_gin_anvil_probe() <= 0) return ncclInternalError;
  struct ginRocshmemInitCtx* ictx = new ginRocshmemInitCtx{};
  *ctx = ictx;
  return ncclSuccess;
}

static ncclResult_t ginAnvilDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

static ncclResult_t ginAnvilGetProperties(int dev, ncclNetProperties_v12_t* props) {
  memset(props, 0, sizeof(*props));
  props->name = const_cast<char*>("gin-anvil-sdma");
  props->pciPath = nullptr;
  props->guid = 0;
  props->ptrSupport = NCCL_PTR_CUDA;
  props->netDeviceType = NCCL_NET_DEVICE_GIN_ANVIL_SDMA;
  props->netDeviceVersion = NCCL_GIN_ANVIL_SDMA_NET_VERSION;
  props->maxP2pBytes = 1ULL << 30;
  props->maxCollBytes = 1ULL << 30;
  return ncclSuccess;
}

static ncclResult_t ginAnvilListen(void* ctx, int dev, void* handle, void** listenComm) {
  auto* lctx = new ginAnvilListenCtx;
  lctx->dev = dev;
  *listenComm = lctx;
  memset(handle, 0, NCCL_NET_HANDLE_MAXSIZE);
  return ncclSuccess;
}

static int ginAnvilSdmaThresholdFromEnv() {
  const char* e = getenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD");
  if (!e || !e[0]) return (int)NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT;
  int v = atoi(e);
  return v > 0 ? v : (int)NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT;
}

static uint32_t ginAnvilFusedSignalFromEnv() {
  const char* e = getenv("NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL");
  if (!e || !e[0]) return NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL_DEFAULT;
  if (e[0] == '0' && e[1] == '\0') return 0;
  return atoi(e) != 0 ? 1u : 0u;
}

static ncclResult_t ginAnvilConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm,
                                    void** collComm) {
  struct ginRocshmemInitCtx* ictx = (struct ginRocshmemInitCtx*)ctx;
  auto* cctx = new ginAnvilCollCtx{};
  cctx->nranks = nranks;
  cctx->rank = rank;
  cctx->comm = ictx->comm;

  int localDev = 0;
  if (hipGetDevice(&localDev) != hipSuccess) {
    delete cctx;
    return ncclSystemError;
  }

  int numCh = 1;
  if (const char* e = getenv("NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS")) {
    int v = atoi(e);
    if (v >= 1 && v <= 8) numCh = v;
  }

  rocshmem_gin_anvil_handle_t h = nullptr;
  void* gpu_handles = nullptr;
  uint64_t* dirty = nullptr;
  if (rocshmem_gin_anvil_create(nranks, rank, localDev, ginAnvilBootstrapAllgather, cctx->comm->bootstrap,
                               numCh, &h, &gpu_handles, &dirty) != 0) {
    WARN("GIN anvil-sdma: rocshmem_gin_anvil_create failed");
    delete cctx;
    return ncclSystemError;
  }

  cctx->anvil = h;
  cctx->gpu_queue_handles = (void**)gpu_handles;
  cctx->sdma_dirty_d = dirty;
  cctx->numChannels = rocshmem_gin_anvil_get_num_channels(h);
  cctx->sdmaChannelStride = rocshmem_gin_anvil_get_channel_stride(h);

  INFO(NCCL_INIT, "GIN anvil-sdma: standalone SDMA queues (%d ranks, %d ch, spread=%d)", nranks,
       cctx->numChannels, cctx->sdmaChannelStride);
  *collComm = cctx;
  return ncclSuccess;
}

static ncclResult_t ginAnvilCloseListen(void* listenComm) {
  delete (ginAnvilListenCtx*)listenComm;
  return ncclSuccess;
}

static ncclResult_t ginAnvilCloseColl(void* collComm) {
  ginAnvilCollCtx* cctx = (ginAnvilCollCtx*)collComm;
  if (cctx) {
    if (cctx->anvil) rocshmem_gin_anvil_destroy(cctx->anvil);
    delete cctx;
  }
  return ncclSuccess;
}

static ncclResult_t ginAnvilFinalize(void* ctx) {
  delete (ginRocshmemInitCtx*)ctx;
  return ncclSuccess;
}

static ncclResult_t ginAnvilRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags,
                                     void** mhandle, void** ginHandle) {
  ginAnvilCollCtx* cctx = (ginAnvilCollCtx*)collComm;
  struct ncclDevrState* devr = &cctx->comm->devrState;
  ncclResult_t ret = ncclSuccess;
  uintptr_t* vasHost = nullptr;
  uintptr_t* vasDev = nullptr;

  ginAnvilMemHandle* mh = nullptr;
  NCCLCHECK(ncclCalloc(&mh, 1));

  uintptr_t lsaFlatBase = 0;
  uint32_t stride4G = 0;
  if (ncclDevrGetGinAnvilMemLayout(devr, data, &lsaFlatBase, &stride4G) != ncclSuccess) {
    WARN("GIN anvil-sdma: could not resolve LSA flat layout for %p", data);
    free(mh);
    return ncclSystemError;
  }

  mh->addr = data;
  mh->size = size;

  NCCLCHECKGOTO(ncclCalloc(&vasHost, cctx->nranks), ret, fail);
  vasHost[cctx->rank] = reinterpret_cast<uintptr_t>(data);
  NCCLCHECKGOTO(bootstrapAllGather(cctx->comm->bootstrap, vasHost, sizeof(uintptr_t)), ret, fail);

  if (hipMalloc(&vasDev, sizeof(uintptr_t) * cctx->nranks) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }
  if (hipMemcpy(vasDev, vasHost, sizeof(uintptr_t) * cctx->nranks, hipMemcpyHostToDevice) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }
  mh->remoteVasDev = vasDev;

  if (hipMalloc(&mh->devHandle, sizeof(ncclGinAnvilSdmaMemHandle)) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }

  ncclGinAnvilSdmaMemHandle hostMh;
  hostMh.lsaFlatBase = lsaFlatBase;
  hostMh.stride4G = stride4G;
  hostMh.remoteVas = vasDev;
  hostMh.nRanks = cctx->nranks;
  if (hipMemcpy(mh->devHandle, &hostMh, sizeof(ncclGinAnvilSdmaMemHandle), hipMemcpyHostToDevice) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }

  INFO(NCCL_INIT,
       "GIN anvil-sdma: registered addr=%p lsaFlatBase=%p stride4G=%u remoteVa[0]=%p remoteVa[1]=%p +%zu",
       data, (void*)hostMh.lsaFlatBase, hostMh.stride4G,
       cctx->nranks > 0 ? (void*)vasHost[0] : nullptr,
       cctx->nranks > 1 ? (void*)vasHost[1] : nullptr, size);

  free(vasHost);
  *mhandle = mh;
  *ginHandle = mh->devHandle;
  return ncclSuccess;

fail:
  if (vasDev) (void)hipFree(vasDev);
  free(vasHost);
  if (mh) {
    if (mh->devHandle) (void)hipFree(mh->devHandle);
    free(mh);
  }
  return ret;
}

static ncclResult_t ginAnvilRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset,
                                           int fd, uint64_t mrFlags, void** mhandle, void** ginHandle) {
  return ginAnvilRegMrSym(collComm, data, size, type, mrFlags, mhandle, ginHandle);
}

static ncclResult_t ginAnvilDeregMrSym(void* collComm, void* mhandle) {
  ginAnvilMemHandle* mh = (ginAnvilMemHandle*)mhandle;
  if (!mh) return ncclSuccess;

  if (mh->remoteVasDev) (void)hipFree(mh->remoteVasDev);
  if (mh->devHandle) (void)hipFree(mh->devHandle);
  free(mh);
  return ncclSuccess;
}

static ncclResult_t ginAnvilCreateContext(void* collComm, ncclGinConfig_v13_t* config, void** outGinCtx,
                                          ncclNetDeviceHandle_v11_t** outDevHandle) {
  ginAnvilCollCtx* cctx = (ginAnvilCollCtx*)collComm;
  ncclResult_t ret = ncclSuccess;
  uintptr_t* addrsBuf = nullptr;
  auto* ctx = new ginAnvilGinCtx{};
  ctx->nRanks = cctx->nranks;
  ctx->rank = cctx->rank;
  ctx->nSignals = config->nSignals;
  ctx->nCounters = config->nCounters;
  ctx->hasError = false;
  ctx->gpu_queue_handles = cctx->gpu_queue_handles;
  ctx->sdma_dirty_d = cctx->sdma_dirty_d;
  ctx->numChannels = cctx->numChannels;
  ctx->sdmaChannelStride = cctx->sdmaChannelStride;

  if (!cctx->gpu_queue_handles || !cctx->sdma_dirty_d) {
    WARN("GIN anvil-sdma: missing SDMA infrastructure (handles=%p dirty=%p)",
         cctx->gpu_queue_handles, (void*)cctx->sdma_dirty_d);
    ret = ncclSystemError;
    goto fail;
  }

  NCCLCHECK(ncclCalloc(&ctx->devHandle, 1));
  ctx->devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_ANVIL_SDMA;
  ctx->devHandle->netDeviceVersion = NCCL_GIN_ANVIL_SDMA_NET_VERSION;
  ctx->devHandle->needsProxyProgress = 0;

  if (hipMalloc(&ctx->gpuCtxDev, sizeof(ncclGinAnvilSdmaGPUContext)) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }

  memset(&ctx->gpuCtxHost, 0, sizeof(ncclGinAnvilSdmaGPUContext));
  ctx->gpuCtxHost.layoutMagic = NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC;
  ctx->gpuCtxHost.nRanks = ctx->nRanks;
  ctx->gpuCtxHost.rank = ctx->rank;
  ctx->gpuCtxHost.nSignals = config->nSignals;
  ctx->gpuCtxHost.nCounters = config->nCounters;
  ctx->gpuCtxHost.numChannels = ctx->numChannels;
  ctx->gpuCtxHost.sdmaChannel = 0;
  ctx->gpuCtxHost.sdmaChannelStride = ctx->sdmaChannelStride;
  ctx->gpuCtxHost.queueHandles = ctx->gpu_queue_handles;
  ctx->gpuCtxHost.sdmaDirty = ctx->sdma_dirty_d;
  ctx->gpuCtxHost.sdmaThreshold = (uint32_t)ginAnvilSdmaThresholdFromEnv();
  ctx->gpuCtxHost.fusedSdmaSignal = ginAnvilFusedSignalFromEnv();

  if (config->nSignals > 0) {
    if (hipExtMallocWithFlags((void**)&ctx->gpuCtxHost.signals, sizeof(uint64_t) * config->nSignals,
                              hipDeviceMallocFinegrained) != hipSuccess) {
      WARN("GIN anvil-sdma: hipExtMallocWithFlags failed for %d signals", config->nSignals);
      ret = ncclSystemError;
      goto fail;
    }
    if (hipMemset(ctx->gpuCtxHost.signals, 0, sizeof(uint64_t) * config->nSignals) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }

    if (hipMalloc(&ctx->gpuCtxHost.signalPeerAddrs, sizeof(uintptr_t) * ctx->nRanks) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }

    addrsBuf = (uintptr_t*)malloc(sizeof(uintptr_t) * ctx->nRanks);
    if (!addrsBuf) {
      ret = ncclSystemError;
      goto fail;
    }
    addrsBuf[ctx->rank] = (uintptr_t)ctx->gpuCtxHost.signals;
    NCCLCHECKGOTO(bootstrapAllGather(cctx->comm->bootstrap, addrsBuf, sizeof(uintptr_t)), ret, failAddrs);
    if (hipMemcpy(ctx->gpuCtxHost.signalPeerAddrs, addrsBuf, sizeof(uintptr_t) * ctx->nRanks,
                  hipMemcpyHostToDevice) != hipSuccess) {
      ret = ncclSystemError;
      goto failAddrs;
    }
    free(addrsBuf);
  }

  if (config->nCounters > 0) {
    if (hipExtMallocWithFlags((void**)&ctx->gpuCtxHost.counters, sizeof(uint64_t) * config->nCounters,
                              hipDeviceMallocFinegrained) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
    if (hipMemset(ctx->gpuCtxHost.counters, 0, sizeof(uint64_t) * config->nCounters) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
  }

  if (hipMemcpy(ctx->gpuCtxDev, &ctx->gpuCtxHost, sizeof(ncclGinAnvilSdmaGPUContext),
                hipMemcpyHostToDevice) != hipSuccess) {
    WARN("GIN anvil-sdma: hipMemcpy gpu context failed");
    ret = ncclSystemError;
    goto fail;
  }

  ctx->devHandle->handle = ctx->gpuCtxDev;
  ctx->devHandle->size = sizeof(ncclGinAnvilSdmaGPUContext);

  *outGinCtx = ctx;
  *outDevHandle = ctx->devHandle;
  INFO(NCCL_INIT,
       "GIN anvil-sdma: context created (v%d, %d signals, %d counters, sdmaThreshold=%u, spread=%d, "
       "fusedSignal=%u)",
       NCCL_GIN_ANVIL_SDMA_NET_VERSION, config->nSignals, config->nCounters,
       ctx->gpuCtxHost.sdmaThreshold, ctx->gpuCtxHost.sdmaChannelStride, ctx->gpuCtxHost.fusedSdmaSignal);
  return ncclSuccess;

failAddrs:
  if (addrsBuf) free(addrsBuf);
fail:
  if (ctx) {
    if (ctx->gpuCtxHost.signals) (void)hipFree(ctx->gpuCtxHost.signals);
    if (ctx->gpuCtxHost.signalPeerAddrs) (void)hipFree(ctx->gpuCtxHost.signalPeerAddrs);
    if (ctx->gpuCtxHost.counters) (void)hipFree(ctx->gpuCtxHost.counters);
    if (ctx->gpuCtxDev) (void)hipFree(ctx->gpuCtxDev);
    free(ctx->devHandle);
    delete ctx;
  }
  return ret;
}

static ncclResult_t ginAnvilDestroyContext(void* ginCtx) {
  ginAnvilGinCtx* ctx = (ginAnvilGinCtx*)ginCtx;
  if (!ctx) return ncclSuccess;
  if (ctx->gpuCtxHost.signals) (void)hipFree(ctx->gpuCtxHost.signals);
  if (ctx->gpuCtxHost.signalPeerAddrs) (void)hipFree(ctx->gpuCtxHost.signalPeerAddrs);
  if (ctx->gpuCtxHost.counters) (void)hipFree(ctx->gpuCtxHost.counters);
  if (ctx->gpuCtxDev) (void)hipFree(ctx->gpuCtxDev);
  free(ctx->devHandle);
  delete ctx;
  return ncclSuccess;
}

static ncclResult_t ginAnvilGinProgress(void* ginCtx) { return ncclSuccess; }

static ncclResult_t ginAnvilQueryLastError(void* ginCtx, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclGin_t ncclGinRocshmemAnvilPlugin = {
    .name = "gin-anvil-sdma",
    .init = ginAnvilInit,
    .devices = ginAnvilDevices,
    .getProperties = ginAnvilGetProperties,
    .listen = ginAnvilListen,
    .connect = ginAnvilConnect,
    .createContext = ginAnvilCreateContext,
    .regMrSym = ginAnvilRegMrSym,
    .regMrSymDmaBuf = ginAnvilRegMrSymDmaBuf,
    .deregMrSym = ginAnvilDeregMrSym,
    .destroyContext = ginAnvilDestroyContext,
    .closeColl = ginAnvilCloseColl,
    .closeListen = ginAnvilCloseListen,
    .iput = NULL,
    .iputSignal = NULL,
    .iget = NULL,
    .iflush = NULL,
    .test = NULL,
    .ginProgress = ginAnvilGinProgress,
    .queryLastError = ginAnvilQueryLastError,
    .finalize = ginAnvilFinalize,
};

#endif  // ENABLE_ROCSHMEM_GIN
