/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_IPC_COPY_H_
#define _NCCL_DEVICE_GIN_ANVIL_IPC_COPY_H_

#include "../../hip_compat.h"
#include "util.hpp"

namespace nccl {
namespace gin {
namespace anvil {

// Scalar / tiny put (<= 8 B): flat system-scope stores to peer-mapped memory.
NCCL_DEVICE_INLINE void ipcPutScalar(void* dst, const void* src, size_t bytes) {
  if (bytes == 0) return;
  if (bytes == 8) {
    unsigned long long v;
    __builtin_memcpy(&v, src, 8);
    __hip_atomic_store(reinterpret_cast<unsigned long long*>(dst), v, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    return;
  }
  if (bytes == 4) {
    unsigned int v;
    __builtin_memcpy(&v, src, 4);
    __hip_atomic_store(reinterpret_cast<unsigned int*>(dst), v, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    return;
  }
  if (bytes == 2) {
    unsigned short v;
    __builtin_memcpy(&v, src, 2);
    __hip_atomic_store(reinterpret_cast<unsigned short*>(dst), v, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    return;
  }
  if (bytes == 1) {
    unsigned char v;
    __builtin_memcpy(&v, src, 1);
    __hip_atomic_store(reinterpret_cast<unsigned char*>(dst), v, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    return;
  }
  // Unaligned 3/5/6/7 B tail.
  auto* d = static_cast<uint8_t*>(dst);
  auto* s = static_cast<const uint8_t*>(src);
  for (size_t i = 0; i < bytes; ++i) {
    __hip_atomic_store(d + i, s[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// IPC load/store path for transfers below the SDMA threshold (mirrors rocSHMEM IpcSdmaImpl).
NCCL_DEVICE_INLINE void ipcPut(void* dst, const void* src, size_t bytes) {
  if (bytes == 0) return;
  if (bytes <= 8) {
    ipcPutScalar(dst, src, bytes);
    return;
  }
  rocshmem::memcpy_lane<rocshmem::MemcpyKind::Put>(dst, const_cast<void*>(src), bytes);
}

NCCL_DEVICE_INLINE void ipcSignal(uint64_t* sigPtr, uint64_t value) {
  __hip_atomic_fetch_add(sigPtr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

}  // namespace anvil
}  // namespace gin
}  // namespace nccl

#endif  // _NCCL_DEVICE_GIN_ANVIL_IPC_COPY_H_
