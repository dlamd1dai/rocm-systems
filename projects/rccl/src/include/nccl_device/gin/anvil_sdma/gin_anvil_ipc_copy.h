/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_IPC_COPY_H_
#define _NCCL_DEVICE_GIN_ANVIL_IPC_COPY_H_

#include "../../hip_compat.h"

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
// Self-contained: do not include rocshmem util.hpp here (symbol clash with queue_pair_device.h).
NCCL_DEVICE_INLINE void ipcPut(void* dst, const void* src, size_t bytes) {
  if (bytes == 0) return;
  if (bytes <= 8) {
    ipcPutScalar(dst, src, bytes);
    return;
  }
  auto* d = static_cast<uint8_t*>(dst);
  auto* s = static_cast<const uint8_t*>(src);
  size_t i = 0;
  for (; i + 8 <= bytes; i += 8) {
    unsigned long long v;
    __builtin_memcpy(&v, s + i, 8);
    __hip_atomic_store(reinterpret_cast<unsigned long long*>(d + i), v, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if (i < bytes) ipcPutScalar(d + i, s + i, bytes - i);
}

}  // namespace anvil
}  // namespace gin
}  // namespace nccl

#endif  // _NCCL_DEVICE_GIN_ANVIL_IPC_COPY_H_
