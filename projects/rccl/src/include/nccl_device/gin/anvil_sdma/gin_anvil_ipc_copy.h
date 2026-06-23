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
namespace detail {

// Local load + remote system-scope store (matches rocSHMEM memcpy_lane Put policies).
// Self-contained: do not include rocshmem util.hpp (symbol clash with queue_pair_device.h).

NCCL_DEVICE_INLINE void ipcFlatStoreSys8(void* dst, unsigned long long val) {
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("flat_store_dwordx2 %0, %1, sc0 sc1" : : "v"(dst), "v"(val) : "memory");
#elif defined(__gfx90a__) || defined(__gfx1100__)
  asm volatile("flat_store_dwordx2 %0, %1, glc slc" : : "v"(dst), "v"(val) : "memory");
#elif defined(__gfx1201__) || defined(__gfx1250__)
  asm volatile("flat_store_b64 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val) : "memory");
#else
  __hip_atomic_store(reinterpret_cast<unsigned long long*>(dst), val, __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_SYSTEM);
#endif
}

NCCL_DEVICE_INLINE void ipcFlatStoreSys4(void* dst, unsigned int val) {
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("flat_store_dword %0, %1, sc0 sc1" : : "v"(dst), "v"(val) : "memory");
#elif defined(__gfx90a__) || defined(__gfx1100__)
  asm volatile("flat_store_dword %0, %1, glc slc" : : "v"(dst), "v"(val) : "memory");
#elif defined(__gfx1201__) || defined(__gfx1250__)
  asm volatile("flat_store_b32 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val) : "memory");
#else
  __hip_atomic_store(reinterpret_cast<unsigned int*>(dst), val, __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_SYSTEM);
#endif
}

NCCL_DEVICE_INLINE void ipcFlatStoreSys2(void* dst, unsigned short val) {
#if defined(__gfx942__) || defined(__gfx950__)
  unsigned short val16 = val;
  asm volatile("flat_store_short %0, %1, sc0 sc1" : : "v"(dst), "v"(val16) : "memory");
#elif defined(__gfx90a__) || defined(__gfx1100__)
  unsigned short val16 = val;
  asm volatile("flat_store_short %0, %1, glc slc" : : "v"(dst), "v"(val16) : "memory");
#elif defined(__gfx1201__) || defined(__gfx1250__)
  unsigned short val16 = val;
  asm volatile("flat_store_b16 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val16) : "memory");
#else
  __hip_atomic_store(reinterpret_cast<unsigned short*>(dst), val, __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_SYSTEM);
#endif
}

NCCL_DEVICE_INLINE void ipcFlatStoreSys1(void* dst, unsigned char val) {
#if defined(__gfx942__) || defined(__gfx950__)
  unsigned short val16 = val;
  asm volatile("flat_store_byte %0, %1, sc0 sc1" : : "v"(dst), "v"(val16) : "memory");
#elif defined(__gfx90a__) || defined(__gfx1100__)
  unsigned short val16 = val;
  asm volatile("flat_store_byte %0, %1, glc slc" : : "v"(dst), "v"(val16) : "memory");
#elif defined(__gfx1201__) || defined(__gfx1250__)
  unsigned short val16 = val;
  asm volatile("flat_store_b8 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val16) : "memory");
#else
  __hip_atomic_store(reinterpret_cast<unsigned char*>(dst), val, __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_SYSTEM);
#endif
}

NCCL_DEVICE_INLINE unsigned long long ipcFlatLoad8(const void* src) {
#if defined(__gfx942__) || defined(__gfx950__)
  unsigned long long val;
  asm volatile("flat_load_dwordx2 %0, %1" : "=v"(val) : "v"(src) : "memory");
  return val;
#elif defined(__gfx90a__) || defined(__gfx1100__)
  unsigned long long val;
  asm volatile("flat_load_dwordx2 %0, %1" : "=v"(val) : "v"(src) : "memory");
  return val;
#elif defined(__gfx1201__) || defined(__gfx1250__)
  unsigned long long val;
  asm volatile("flat_load_b64 %0, %1, scope:SCOPE_SE" : "=v"(val) : "v"(src) : "memory");
  return val;
#else
  unsigned long long val;
  __builtin_memcpy(&val, src, 8);
  return val;
#endif
}

NCCL_DEVICE_INLINE void ipcFlatStoreSys16(void* dst, const void* src) {
#if defined(__gfx942__) || defined(__gfx950__)
  __int128_t val;
  asm volatile("flat_load_dwordx4 %0, %1" : "=v"(val) : "v"(src) : "memory");
  asm volatile("flat_store_dwordx4 %0, %1, sc0 sc1" : : "v"(dst), "v"(val) : "memory");
#elif defined(__gfx90a__) || defined(__gfx1100__)
  __int128_t val;
  asm volatile("flat_load_dwordx4 %0, %1" : "=v"(val) : "v"(src) : "memory");
  asm volatile("flat_store_dwordx4 %0, %1, glc slc" : : "v"(dst), "v"(val) : "memory");
#elif defined(__gfx1201__) || defined(__gfx1250__)
  __int128_t val;
  asm volatile("flat_load_b128 %0, %1, scope:SCOPE_SE" : "=v"(val) : "v"(src) : "memory");
  asm volatile("flat_store_b128 %0, %1, scope:SCOPE_SYS" : : "v"(dst), "v"(val) : "memory");
#else
  ipcFlatStoreSys8(dst, ipcFlatLoad8(src));
  ipcFlatStoreSys8(static_cast<uint8_t*>(dst) + 8, ipcFlatLoad8(static_cast<const uint8_t*>(src) + 8));
#endif
}

// Mirrors rocSHMEM copy_remainder<Standard, SystemScope>.
NCCL_DEVICE_INLINE void ipcPutRemainder(uint8_t* dst, uint8_t* src, int remainder) {
  if (remainder == 0) return;
  if (remainder & 1) {
    ipcFlatStoreSys1(dst, *src);
    if (remainder == 1) return;
    dst += 1;
    src += 1;
  }
  if (remainder & 2) {
    unsigned short v;
    __builtin_memcpy(&v, src, 2);
    ipcFlatStoreSys2(dst, v);
    if (remainder == 2) return;
    dst += 2;
    src += 2;
  }
  if (remainder & 4) {
    unsigned int v;
    __builtin_memcpy(&v, src, 4);
    ipcFlatStoreSys4(dst, v);
    if (remainder == 4) return;
    dst += 4;
    src += 4;
  }
  if (remainder & 8) {
    ipcFlatStoreSys8(dst, ipcFlatLoad8(src));
  }
}

}  // namespace detail

NCCL_DEVICE_INLINE void ipcPutScalar(void* dst, const void* src, size_t bytes) {
  detail::ipcPutRemainder(static_cast<uint8_t*>(dst), const_cast<uint8_t*>(static_cast<const uint8_t*>(src)),
                         static_cast<int>(bytes));
}

// IPC load/store path for transfers below the SDMA threshold (mirrors rocSHMEM memcpy_lane Put).
NCCL_DEVICE_INLINE void ipcPut(void* dst, const void* src, size_t bytes) {
  if (bytes == 0) return;
  if (bytes == 16) {
    detail::ipcFlatStoreSys16(dst, src);
    return;
  }
  if (bytes <= 8) {
    ipcPutScalar(dst, src, bytes);
    return;
  }
  auto* d = static_cast<uint8_t*>(dst);
  auto* s = const_cast<uint8_t*>(static_cast<const uint8_t*>(src));
  size_t i = 0;
  for (; i + 16 <= bytes; i += 16) {
    detail::ipcFlatStoreSys16(d + i, s + i);
  }
  if (i < bytes) detail::ipcPutRemainder(d + i, s + i, static_cast<int>(bytes - i));
}

}  // namespace anvil
}  // namespace gin
}  // namespace nccl

#endif  // _NCCL_DEVICE_GIN_ANVIL_IPC_COPY_H_
