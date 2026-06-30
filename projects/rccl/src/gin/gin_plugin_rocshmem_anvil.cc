/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

/**
 * GIN plugin: SDMA Anvil device path (NCCL_GIN_TYPE=6).
 * Host flow mirrors GIN rocSHMEM API (regMrSym registers buffers with
 * rocshmem_buffer_register_vmm for constant-memory remote lookup). Data movement uses
 * a standalone Anvil SDMA stack (rocshmem_gin_anvil_create) — independent of the
 * rocSHMEM IPC SDMA transport, so GIN type 6 can outlive removal of GIN type 4.
 */

#include "gin/gin_host_rocshmem_api.h"
#include "gin/gin_host_rocshmem_anvil.h"
#include "comm.h"
#include "dev_runtime.h"
#include "bootstrap.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
#include <rocshmem/gin_anvil_factory.h>
#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <map>

// Refcount for buffer registration: ncclGinRegister calls regMrSym once per
// connection for the same buffer.
static std::map<void*, int> bufferRegRefcount;

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

  ginAnvilMemHandle* mh = nullptr;
  NCCLCHECK(ncclCalloc(&mh, 1));

  auto& refcount = bufferRegRefcount[data];

  void* lsaSelfAddr = nullptr;
  NCCLCHECK(ncclDevrGetLsaSelfAddr(devr, data, &lsaSelfAddr));
  if (lsaSelfAddr == nullptr) {
    WARN("GIN anvil-sdma: could not resolve LSA flat addr for %p", data);
    bufferRegRefcount.erase(data);
    free(mh);
    return ncclSystemError;
  }

  if (refcount == 0) {
    int rc = rocshmem::rocshmem_buffer_register_vmm(lsaSelfAddr, size, devr->lsaSelf, devr->lsaSize,
                                                    (ptrdiff_t)devr->bigSize);
    if (rc != 0) {
      WARN("GIN anvil-sdma: buffer register failed for %p (lsaSelf=%p) size %zu", data, lsaSelfAddr,
           size);
      bufferRegRefcount.erase(data);
      free(mh);
      return ncclSystemError;
    }
    INFO(NCCL_INIT, "GIN anvil-sdma: registered addr=%p lsaSelf=%p +%zu", data, lsaSelfAddr, size);
  }
  refcount++;

  mh->addr = data;
  mh->size = size;

  if (hipMalloc(&mh->devHandle, sizeof(ncclGinAnvilSdmaMemHandle)) != hipSuccess) {
    free(mh);
    return ncclSystemError;
  }

  ncclGinAnvilSdmaMemHandle hostMh;
  hostMh.baseAddr = (uintptr_t)lsaSelfAddr;
  (void)hipMemcpy(mh->devHandle, &hostMh, sizeof(ncclGinAnvilSdmaMemHandle), hipMemcpyHostToDevice);

  *mhandle = mh;
  *ginHandle = mh->devHandle;
  return ncclSuccess;
}

static ncclResult_t ginAnvilRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset,
                                           int fd, uint64_t mrFlags, void** mhandle, void** ginHandle) {
  return ginAnvilRegMrSym(collComm, data, size, type, mrFlags, mhandle, ginHandle);
}

static ncclResult_t ginAnvilDeregMrSym(void* collComm, void* mhandle) {
  ginAnvilMemHandle* mh = (ginAnvilMemHandle*)mhandle;
  if (!mh) return ncclSuccess;

  if (mh->addr) {
    auto& refcount = bufferRegRefcount[mh->addr];
    refcount--;
    if (refcount <= 0) {
      bufferRegRefcount.erase(mh->addr);
    }
  }
  if (mh->devHandle) (void)hipFree(mh->devHandle);
  free(mh);
  return ncclSuccess;
}

static ncclResult_t ginAnvilCreateContext(void* collComm, ncclGinConfig_v13_t* config, void** outGinCtx,
                                          ncclNetDeviceHandle_v11_t** outDevHandle) {
  ginAnvilCollCtx* cctx = (ginAnvilCollCtx*)collComm;
  ncclResult_t ret = ncclSuccess;
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
    ctx->gpuCtxHost.signals =
        (uint64_t*)rocshmem::rocshmem_malloc(sizeof(uint64_t) * config->nSignals);
    if (!ctx->gpuCtxHost.signals) {
      WARN("GIN anvil-sdma: rocshmem_malloc failed for %d signals", config->nSignals);
      ret = ncclSystemError;
      goto fail;
    }
    if (hipMemset(ctx->gpuCtxHost.signals, 0, sizeof(uint64_t) * config->nSignals) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
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

fail:
  if (ctx) {
    if (ctx->gpuCtxHost.signals) rocshmem::rocshmem_free(ctx->gpuCtxHost.signals);
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
  if (ctx->gpuCtxHost.signals) rocshmem::rocshmem_free(ctx->gpuCtxHost.signals);
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
