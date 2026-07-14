/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#include "nccl_device/gin/anvil_sdma/gin_anvil_ipc_table.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace {

static bool ipcDebugEnabled() {
  static int cached = -1;
  if (cached < 0) {
    const char* e = std::getenv("NCCL_GIN_ANVIL_IPC_DEBUG");
    cached = (e && e[0] && !(e[0] == '0' && e[1] == '\0')) ? 1 : 0;
  }
  return cached != 0;
}

static void logIpcEntry(const char* action, const ncclGinAnvilIpcBufEntry* entry, int nRanks) {
  if (!ipcDebugEnabled() || entry == nullptr) return;
  std::fprintf(stderr,
               "[gin-anvil-ipc] %s local_base=0x%lx length=%zu (nRanks=%d)\n", action,
               static_cast<unsigned long>(entry->local_base), entry->length, nRanks);
  const int limit = nRanks < NCCL_GIN_ANVIL_IPC_MAX_RANKS ? nRanks : NCCL_GIN_ANVIL_IPC_MAX_RANKS;
  for (int pe = 0; pe < limit; ++pe) {
    std::fprintf(stderr, "[gin-anvil-ipc]   peer[%d] remote_base=0x%lx\n", pe,
                 static_cast<unsigned long>(entry->remote_bases[pe]));
  }
}

ncclGinAnvilIpcBufEntry masterEntries[NCCL_GIN_ANVIL_IPC_MAX_BUFS];
int masterEntryCount = 0;

ncclGinAnvilIpcBufEntry* d_ipcTable = nullptr;
int d_ipcTableCount = 0;

static void logIpcTableSnapshot(const char* action) {
  if (!ipcDebugEnabled()) return;
  std::fprintf(stderr, "[gin-anvil-ipc] %s table_count=%d d_ipcTable=%p\n", action, masterEntryCount,
               static_cast<void*>(d_ipcTable));
  for (int i = 0; i < masterEntryCount; ++i) {
    logIpcEntry("entry", &masterEntries[i], NCCL_GIN_ANVIL_IPC_MAX_RANKS);
  }
}

struct LiveGpuContext {
  ncclGinAnvilSdmaGPUContext* hostCtx;
  ncclGinAnvilSdmaGPUContext* devCtx;
  LiveGpuContext* next;
};

LiveGpuContext* liveContexts = nullptr;

static int ensureDeviceTableAllocated() {
  if (d_ipcTable != nullptr) return 0;
  hipError_t err =
      hipMalloc(&d_ipcTable, NCCL_GIN_ANVIL_IPC_MAX_BUFS * sizeof(ncclGinAnvilIpcBufEntry));
  return err == hipSuccess ? 0 : -1;
}

static void refreshAllLiveContexts() {
  for (LiveGpuContext* it = liveContexts; it != nullptr; it = it->next) {
    if (it->hostCtx == nullptr || it->devCtx == nullptr) continue;
    it->hostCtx->ipcTable = d_ipcTable;
    it->hostCtx->ipcTableCount = d_ipcTableCount;
    (void)hipMemcpy(it->devCtx, it->hostCtx, sizeof(ncclGinAnvilSdmaGPUContext), hipMemcpyHostToDevice);
  }
}

static int syncTableToDevice() {
  int count = masterEntryCount;
  if (count > NCCL_GIN_ANVIL_IPC_MAX_BUFS) count = NCCL_GIN_ANVIL_IPC_MAX_BUFS;

  if (count == 0) {
    d_ipcTableCount = 0;
    refreshAllLiveContexts();
    return 0;
  }

  if (ensureDeviceTableAllocated() != 0) return -1;

  hipError_t err =
      hipMemcpy(d_ipcTable, masterEntries, count * sizeof(ncclGinAnvilIpcBufEntry), hipMemcpyHostToDevice);
  if (err != hipSuccess) return -1;
  d_ipcTableCount = count;
  refreshAllLiveContexts();
  logIpcTableSnapshot("syncTableToDevice");
  return 0;
}

static int findEntryIndex(uintptr_t localBase) {
  for (int i = 0; i < masterEntryCount; i++) {
    if (masterEntries[i].local_base == localBase) return i;
  }
  return -1;
}

static int addEntry(const ncclGinAnvilIpcBufEntry* entry) {
  if (masterEntryCount >= NCCL_GIN_ANVIL_IPC_MAX_BUFS) return -1;
  masterEntries[masterEntryCount++] = *entry;
  return syncTableToDevice();
}

static int removeEntry(uintptr_t localBase) {
  int idx = findEntryIndex(localBase);
  if (idx < 0) return -1;
  if (idx < masterEntryCount - 1) {
    masterEntries[idx] = masterEntries[masterEntryCount - 1];
  }
  masterEntryCount--;
  return syncTableToDevice();
}

}  // namespace

extern "C" void ncclGinAnvilIpcTableGetDevice(const ncclGinAnvilIpcBufEntry** outTable, int* outCount) {
  if (outTable) *outTable = d_ipcTable;
  if (outCount) *outCount = d_ipcTableCount;
}

extern "C" void ncclGinAnvilIpcTableTrackContext(ncclGinAnvilSdmaGPUContext* hostCtx,
                                                 ncclGinAnvilSdmaGPUContext* devCtx) {
  if (!hostCtx || !devCtx) return;
  for (LiveGpuContext* it = liveContexts; it != nullptr; it = it->next) {
    if (it->hostCtx == hostCtx) return;
  }
  auto* node = new LiveGpuContext{hostCtx, devCtx, liveContexts};
  liveContexts = node;
  hostCtx->ipcTable = d_ipcTable;
  hostCtx->ipcTableCount = d_ipcTableCount;
  (void)hipMemcpy(devCtx, hostCtx, sizeof(ncclGinAnvilSdmaGPUContext), hipMemcpyHostToDevice);
  if (ipcDebugEnabled()) {
    std::fprintf(stderr,
                 "[gin-anvil-ipc] TrackContext hostCtx=%p devCtx=%p ipcTable=%p ipcTableCount=%d\n",
                 static_cast<void*>(hostCtx), static_cast<void*>(devCtx),
                 static_cast<void*>(d_ipcTable), d_ipcTableCount);
  }
}

extern "C" void ncclGinAnvilIpcTableUntrackContext(ncclGinAnvilSdmaGPUContext* hostCtx) {
  if (!hostCtx) return;
  LiveGpuContext** prev = &liveContexts;
  for (LiveGpuContext* it = liveContexts; it != nullptr; it = it->next) {
    if (it->hostCtx == hostCtx) {
      *prev = it->next;
      delete it;
      return;
    }
    prev = &it->next;
  }
}

extern "C" int ncclGinAnvilIpcTableRegisterVmm(void* localBase, size_t length, int myRank, int nRanks,
                                               ptrdiff_t strideBytes) {
  if (!localBase || length == 0 || nRanks < 1 || nRanks > NCCL_GIN_ANVIL_IPC_MAX_RANKS) return -1;
  if (findEntryIndex(reinterpret_cast<uintptr_t>(localBase)) >= 0) return 0;

  ncclGinAnvilIpcBufEntry entry = {};
  entry.local_base = reinterpret_cast<uintptr_t>(localBase);
  entry.length = length;
  for (int pe = 0; pe < nRanks; pe++) {
    entry.remote_bases[pe] = entry.local_base + static_cast<ptrdiff_t>(pe - myRank) * strideBytes;
  }
  if (ipcDebugEnabled()) {
    std::fprintf(stderr,
                 "[gin-anvil-ipc] RegisterVmm localBase=%p length=%zu myRank=%d nRanks=%d stride=%td\n",
                 localBase, length, myRank, nRanks, static_cast<ptrdiff_t>(strideBytes));
    logIpcEntry("RegisterVmm", &entry, nRanks);
  }
  return addEntry(&entry);
}

extern "C" int ncclGinAnvilIpcTableRegisterExplicit(void* localBase, const uintptr_t* remoteBases,
                                                    int nRanks, size_t length) {
  if (!localBase || !remoteBases || length == 0 || nRanks < 1 || nRanks > NCCL_GIN_ANVIL_IPC_MAX_RANKS)
    return -1;
  if (findEntryIndex(reinterpret_cast<uintptr_t>(localBase)) >= 0) return 0;

  ncclGinAnvilIpcBufEntry entry = {};
  entry.local_base = reinterpret_cast<uintptr_t>(localBase);
  entry.length = length;
  for (int pe = 0; pe < nRanks; pe++) {
    entry.remote_bases[pe] = remoteBases[pe];
  }
  return addEntry(&entry);
}

extern "C" int ncclGinAnvilIpcTableUnregister(void* localBase) {
  if (!localBase) return -1;
  return removeEntry(reinterpret_cast<uintptr_t>(localBase));
}

extern "C" void ncclGinAnvilIpcTableTestReset(void) {
  while (liveContexts != nullptr) {
    LiveGpuContext* next = liveContexts->next;
    delete liveContexts;
    liveContexts = next;
  }
  masterEntryCount = 0;
  d_ipcTableCount = 0;
  (void)syncTableToDevice();
}

#endif  // ENABLE_ROCSHMEM_GIN
