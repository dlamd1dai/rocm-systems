/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_IPC_TABLE_H_
#define _NCCL_DEVICE_GIN_ANVIL_IPC_TABLE_H_

#include <stddef.h>
#include <stdint.h>

/** GIN-owned IPC lookup table (replaces rocshmem ipc_user_buf_table for anvil-sdma). */
#define NCCL_GIN_ANVIL_IPC_MAX_BUFS 16
#define NCCL_GIN_ANVIL_IPC_MAX_RANKS 16

struct ncclGinAnvilIpcBufEntry {
  uintptr_t local_base;
  uintptr_t remote_bases[NCCL_GIN_ANVIL_IPC_MAX_RANKS];
  size_t length;
};

#if defined(__HIPCC__) || defined(__CUDACC__)

#include "../../hip_compat.h"

extern __constant__ ncclGinAnvilIpcBufEntry nccl_gin_anvil_ipc_table[NCCL_GIN_ANVIL_IPC_MAX_BUFS];
extern __constant__ int nccl_gin_anvil_ipc_table_count;

namespace nccl {
namespace gin {
namespace anvil {
namespace detail {

NCCL_DEVICE_INLINE void* ginAnvilResolvePeerVa(void* localAddr, int peer) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(localAddr);
  int count = nccl_gin_anvil_ipc_table_count;
  for (int b = 0; b < count; b++) {
    uintptr_t base = nccl_gin_anvil_ipc_table[b].local_base;
    size_t len = nccl_gin_anvil_ipc_table[b].length;
    if (addr >= base && addr < base + len) {
      uintptr_t off = addr - base;
      if (peer < 0 || peer >= NCCL_GIN_ANVIL_IPC_MAX_RANKS) return nullptr;
      return reinterpret_cast<void*>(nccl_gin_anvil_ipc_table[b].remote_bases[peer] + off);
    }
  }
  return nullptr;
}

}  // namespace detail
}  // namespace anvil
}  // namespace gin
}  // namespace nccl

#endif  // __HIPCC__ || __CUDACC__

#ifdef __cplusplus
extern "C" {
#endif

int ncclGinAnvilIpcTableRegisterVmm(void* localBase, size_t length, int myRank, int nRanks,
                                    ptrdiff_t strideBytes);
int ncclGinAnvilIpcTableRegisterExplicit(void* localBase, const uintptr_t* remoteBases, int nRanks,
                                         size_t length);
int ncclGinAnvilIpcTableUnregister(void* localBase);

#ifdef __cplusplus
}
#endif

#endif  // _NCCL_DEVICE_GIN_ANVIL_IPC_TABLE_H_
