/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#include "nccl_device/gin/anvil_sdma/gin_anvil_ipc_table.h"
#include <hip/hip_runtime.h>

__constant__ ncclGinAnvilIpcBufEntry nccl_gin_anvil_ipc_table[NCCL_GIN_ANVIL_IPC_MAX_BUFS];
__constant__ int nccl_gin_anvil_ipc_table_count = 0;

namespace {

ncclGinAnvilIpcBufEntry masterEntries[NCCL_GIN_ANVIL_IPC_MAX_BUFS];
int masterEntryCount = 0;

static int syncTableToDevice() {
  int count = masterEntryCount;
  if (count > NCCL_GIN_ANVIL_IPC_MAX_BUFS) count = NCCL_GIN_ANVIL_IPC_MAX_BUFS;
  hipError_t err =
      hipMemcpyToSymbol(HIP_SYMBOL(nccl_gin_anvil_ipc_table), masterEntries,
                        count * sizeof(ncclGinAnvilIpcBufEntry), 0, hipMemcpyHostToDevice);
  if (err != hipSuccess) return -1;
  err = hipMemcpyToSymbol(HIP_SYMBOL(nccl_gin_anvil_ipc_table_count), &count, sizeof(int), 0,
                          hipMemcpyHostToDevice);
  return err == hipSuccess ? 0 : -1;
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
