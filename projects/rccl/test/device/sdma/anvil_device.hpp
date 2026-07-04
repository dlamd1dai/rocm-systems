/* Test stub: no-op gin_anvil::sdma device ops for template coverage tests. */
#pragma once

#include "sdma_opcodes.h"

namespace gin_anvil {
namespace sdma {

struct SdmaQueueDeviceHandle {
  int tag;
};

struct SdmaQueueSingleProducerDeviceHandle {
  int tag;
};

__device__ __forceinline__ void put(SdmaQueueDeviceHandle& handle, void* dst, void* src, size_t size) {
  (void)handle;
  (void)dst;
  (void)src;
  (void)size;
}

__device__ __forceinline__ void putSignal(SdmaQueueDeviceHandle& handle, void* dst, void* src, size_t size,
                                          uint64_t* signal) {
  (void)handle;
  (void)dst;
  (void)src;
  (void)size;
  (void)signal;
}

__device__ __forceinline__ void quiet(SdmaQueueDeviceHandle& handle) { (void)handle; }

}  // namespace sdma
}  // namespace gin_anvil
