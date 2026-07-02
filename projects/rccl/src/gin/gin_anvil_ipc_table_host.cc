/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#include "nccl_device/gin/anvil_sdma/gin_anvil_ipc_table.h"
#include <hip/hip_runtime.h>

namespace {

ncclGinAnvilIpcBufEntry masterEntries[NCCL_GIN_ANVIL_IPC_MAX_BUFS];
int masterEntryCount = 0;

ncclGinAnvilIpcBufEntry* d_ipcTable = nullptr;
int d_ipcTableCount = 0;
int d_ipcTableCapacity = 0;

static int syncTableToDevice() {
  int count = masterEntryCount;
  if (count > NCCL_GIN_ANVIL_IPC_MAX_BUFS) count = NCCL_GIN_ANVIL_IPC_MAX_BUFS;

  if (count == 0) {
    d_ipcTableCount = 0;
    return 0;
  }

  if (count > d_ipcTableCapacity) {
    if (d_ipcTable) (void)hipFree(d_ipcTable);
    d_ipcTable = nullptr;
    d_ipcTableCapacity = 0;
    hipError_t err = hipMalloc(&d_ipcTable, count * sizeof(ncclGinAnvilIpcBufEntry));
    if (err != hipSuccess) return -1;
    d_ipcTableCapacity = count;
  }

  hipError_t err =
      hipMemcpy(d_ipcTable, masterEntries, count * sizeof(ncclGinAnvilIpcBufEntry), hipMemcpyHostToDevice);
  if (err != hipSuccess) return -1;
  d_ipcTableCount = count;
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

#endif  // ENABLE_ROCSHMEM_GIN
