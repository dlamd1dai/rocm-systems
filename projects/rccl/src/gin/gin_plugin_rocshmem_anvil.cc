/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

/**
 * GIN plugin: SDMA Anvil device path (NCCL_GIN_TYPE=6).
 * Host flow mirrors GIN-GDA (connect builds transport resources, regMrSym exchanges
 * addressing metadata, createContext wires device context). Data movement uses
 * rocSHMEM's Anvil SDMA helpers (anvil::put / quiet / signal) directly — not rocshmem_putmem.
 */

#include "gin/gin_host_rocshmem_api.h"
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
  void** signal_ipc_peer_ptrs;
};

struct ginAnvilMemHandle {
  ncclGinAnvilSdmaMemHandle* devHandle;
  uintptr_t* remote_vas_dev;
};

struct ginAnvilListenCtx {
  int dev;
};

static int ginAnvilBootstrapAllgather(void* ctx, void* buf, size_t perRankSize) {
  return (bootstrapAllGather(ctx, buf, (int)perRankSize) == ncclSuccess) ? 0 : -1;
}

static void ginAnvilCloseSignalIpc(ginAnvilGinCtx* ctx) {
  if (!ctx || !ctx->signal_ipc_peer_ptrs) return;
  for (int p = 0; p < ctx->nRanks; ++p) {
    if (p != ctx->rank && ctx->signal_ipc_peer_ptrs[p]) {
      (void)hipIpcCloseMemHandle(ctx->signal_ipc_peer_ptrs[p]);
      ctx->signal_ipc_peer_ptrs[p] = nullptr;
    }
  }
  free(ctx->signal_ipc_peer_ptrs);
  ctx->signal_ipc_peer_ptrs = nullptr;
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

  INFO(NCCL_INIT, "GIN anvil-sdma: SDMA queues ready (%d ranks, %d ch)", nranks, cctx->numChannels);
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
  void* lsaSelfAddr = nullptr;
  NCCLCHECK(ncclDevrGetLsaSelfAddr(devr, data, &lsaSelfAddr));

  ginAnvilMemHandle* mh = nullptr;
  NCCLCHECK(ncclCalloc(&mh, 1));

  if (hipMalloc(&mh->devHandle, sizeof(ncclGinAnvilSdmaMemHandle)) != hipSuccess) {
    free(mh);
    return ncclSystemError;
  }

  uintptr_t* vas_buf = (uintptr_t*)malloc(sizeof(uintptr_t) * (size_t)cctx->nranks);
  if (!vas_buf) {
    (void)hipFree(mh->devHandle);
    free(mh);
    return ncclSystemError;
  }

  const bool use_symmetric_flat =
      lsaSelfAddr != nullptr && devr->bigSize > 1 && devr->lsaFlatBase != nullptr;
  if (use_symmetric_flat) {
    const int myLsaRank = devr->lsaSelf;
    for (int p = 0; p < cctx->nranks; ++p) {
      int peerLsaRank = 0;
      NCCLCHECK(ncclDevrWorldToLsaRank(cctx->comm, p, &peerLsaRank));
      vas_buf[p] = (uintptr_t)lsaSelfAddr +
                   (ptrdiff_t)(peerLsaRank - myLsaRank) * (ptrdiff_t)devr->bigSize;
    }
    INFO(NCCL_INIT, "GIN anvil-sdma: registered addr=%p lsaSelf=%p size %zu", data, lsaSelfAddr,
         size);
  } else {
    vas_buf[cctx->rank] = (uintptr_t)data;
    NCCLCHECK(bootstrapAllGather(cctx->comm->bootstrap, vas_buf, (int)sizeof(uintptr_t)));
    if (lsaSelfAddr == nullptr) {
      WARN("GIN anvil-sdma: buffer %p not in symmetric VA; using allgathered local pointers", data);
    }
  }

  if (hipMalloc(&mh->remote_vas_dev, sizeof(uintptr_t) * (size_t)cctx->nranks) != hipSuccess) {
    free(vas_buf);
    (void)hipFree(mh->devHandle);
    free(mh);
    return ncclSystemError;
  }
  (void)hipMemcpy(mh->remote_vas_dev, vas_buf, sizeof(uintptr_t) * (size_t)cctx->nranks, hipMemcpyHostToDevice);
  free(vas_buf);

  ncclGinAnvilSdmaMemHandle hostMh;
  hostMh.local_va = (uintptr_t)(lsaSelfAddr != nullptr ? lsaSelfAddr : data);
  hostMh.remote_vas = mh->remote_vas_dev;
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
  if (mh->remote_vas_dev) (void)hipFree(mh->remote_vas_dev);
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
  ctx->signal_ipc_peer_ptrs = nullptr;

  NCCLCHECK(ncclCalloc(&ctx->devHandle, 1));
  ctx->devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_ANVIL_SDMA;
  ctx->devHandle->netDeviceVersion = NCCL_GIN_ANVIL_SDMA_NET_VERSION;
  ctx->devHandle->needsProxyProgress = 0;

  if (hipMalloc(&ctx->gpuCtxDev, sizeof(ncclGinAnvilSdmaGPUContext)) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }

  memset(&ctx->gpuCtxHost, 0, sizeof(ncclGinAnvilSdmaGPUContext));
  ctx->gpuCtxHost.nRanks = ctx->nRanks;
  ctx->gpuCtxHost.rank = ctx->rank;
  ctx->gpuCtxHost.nSignals = config->nSignals;
  ctx->gpuCtxHost.nCounters = config->nCounters;
  ctx->gpuCtxHost.numChannels = ctx->numChannels;
  ctx->gpuCtxHost.queueHandles = ctx->gpu_queue_handles;
  ctx->gpuCtxHost.sdmaDirty = ctx->sdma_dirty_d;
  ctx->gpuCtxHost.sdmaThreshold = (uint32_t)ginAnvilSdmaThresholdFromEnv();

  if (config->nSignals > 0) {
    if (hipExtMallocWithFlags((void**)&ctx->gpuCtxHost.signals, sizeof(uint64_t) * config->nSignals,
                              hipDeviceMallocFinegrained) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
    (void)hipMemset(ctx->gpuCtxHost.signals, 0, sizeof(uint64_t) * config->nSignals);

    uintptr_t* addrs = (uintptr_t*)malloc(sizeof(uintptr_t) * (size_t)ctx->nRanks);
    hipIpcMemHandle_t* ipc_handles = nullptr;
    if (!addrs) {
      ret = ncclSystemError;
      goto fail;
    }

    ipc_handles = (hipIpcMemHandle_t*)malloc(sizeof(hipIpcMemHandle_t) * (size_t)ctx->nRanks);
    if (!ipc_handles) {
      free(addrs);
      ret = ncclSystemError;
      goto fail;
    }

    const bool signal_ipc_ok =
        hipIpcGetMemHandle(&ipc_handles[ctx->rank], ctx->gpuCtxHost.signals) == hipSuccess;
    if (signal_ipc_ok) {
      NCCLCHECKGOTO(bootstrapAllGather(cctx->comm->bootstrap, ipc_handles, sizeof(hipIpcMemHandle_t)),
                    ret, signal_fail);
      NCCLCHECKGOTO(ncclCalloc(&ctx->signal_ipc_peer_ptrs, ctx->nRanks), ret, signal_fail);
      addrs[ctx->rank] = (uintptr_t)ctx->gpuCtxHost.signals;
      for (int p = 0; p < ctx->nRanks; ++p) {
        if (p == ctx->rank) continue;
        void* peer_ptr = nullptr;
        hipError_t hip_ret =
            hipIpcOpenMemHandle(&peer_ptr, ipc_handles[p], hipIpcMemLazyEnablePeerAccess);
        if (hip_ret != hipSuccess) {
          WARN("GIN anvil-sdma: hipIpcOpenMemHandle(rank %d) failed: %s", p,
               hipGetErrorString(hip_ret));
          ret = ncclSystemError;
          goto signal_fail;
        }
        ctx->signal_ipc_peer_ptrs[p] = peer_ptr;
        addrs[p] = (uintptr_t)peer_ptr;
      }
    } else {
      WARN("GIN anvil-sdma: hipIpcGetMemHandle failed for signals; using allgathered pointers");
      addrs[ctx->rank] = (uintptr_t)ctx->gpuCtxHost.signals;
      NCCLCHECKGOTO(bootstrapAllGather(cctx->comm->bootstrap, addrs, (int)sizeof(uintptr_t)), ret,
                    signal_fail);
    }
    free(ipc_handles);
    ipc_handles = nullptr;

    if (hipMalloc(&ctx->gpuCtxHost.signal_peer_addrs, sizeof(uintptr_t) * (size_t)ctx->nRanks) !=
        hipSuccess) {
      free(addrs);
      ginAnvilCloseSignalIpc(ctx);
      ret = ncclSystemError;
      goto fail;
    }
    (void)hipMemcpy(ctx->gpuCtxHost.signal_peer_addrs, addrs, sizeof(uintptr_t) * (size_t)ctx->nRanks,
                    hipMemcpyHostToDevice);
    free(addrs);
    goto signal_done;

  signal_fail:
    if (ipc_handles) free(ipc_handles);
    free(addrs);
    ginAnvilCloseSignalIpc(ctx);
    goto fail;

  signal_done:;
  }

  if (config->nCounters > 0) {
    if (hipExtMallocWithFlags((void**)&ctx->gpuCtxHost.counters, sizeof(uint64_t) * config->nCounters,
                              hipDeviceMallocFinegrained) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
    (void)hipMemset(ctx->gpuCtxHost.counters, 0, sizeof(uint64_t) * config->nCounters);
  }

  (void)hipMemcpy(ctx->gpuCtxDev, &ctx->gpuCtxHost, sizeof(ncclGinAnvilSdmaGPUContext), hipMemcpyHostToDevice);

  ctx->devHandle->handle = ctx->gpuCtxDev;
  ctx->devHandle->size = sizeof(ncclGinAnvilSdmaGPUContext);

  *outGinCtx = ctx;
  *outDevHandle = ctx->devHandle;
  INFO(NCCL_INIT, "GIN anvil-sdma: context created (%d signals, %d counters, sdmaThreshold=%u)",
       config->nSignals, config->nCounters, ctx->gpuCtxHost.sdmaThreshold);
  return ncclSuccess;

fail:
  if (ctx) {
    ginAnvilCloseSignalIpc(ctx);
    if (ctx->gpuCtxHost.signals) (void)hipFree(ctx->gpuCtxHost.signals);
    if (ctx->gpuCtxHost.signal_peer_addrs) (void)hipFree(ctx->gpuCtxHost.signal_peer_addrs);
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
  ginAnvilCloseSignalIpc(ctx);
  if (ctx->gpuCtxHost.signals) (void)hipFree(ctx->gpuCtxHost.signals);
  if (ctx->gpuCtxHost.signal_peer_addrs) (void)hipFree(ctx->gpuCtxHost.signal_peer_addrs);
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
