// Phase-0 spike shared helper: HIP fabric-handle VMM allocation / export / import.
//
// Mirrors the exact API sequence used by rocSHMEM's vmm_fabric allocator
// (projects/rocshmem/src/memory/hip_allocator_vmm_common.hpp and
//  hip_allocator_vmm_fabric.cpp on origin/develop) so the spike faithfully
// exercises the same code path the production adaptation would use.
//
// The fabric API (hipMemFabricHandle_t / hipMemHandleTypeFabric /
// hipMemImportFromShareableHandle) is only present on ROCm builds that expose
// it (ROCm 7.14+ with AMD SMI fabric handle support; see ROCm/rocm-systems
// #2170). CMake defines SPIKE_HAS_FABRIC when the toolchain provides it. When
// it is absent, this header still compiles but fabric calls are stubbed so the
// probe can report "not available" cleanly.
#ifndef DDAI_SPIKE_FABRIC_VMM_HPP_
#define DDAI_SPIKE_FABRIC_VMM_HPP_

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(expr)                                                                     \
  do {                                                                                      \
    hipError_t _e = (expr);                                                                 \
    if (_e != hipSuccess) {                                                                 \
      std::fprintf(stderr, "HIP error %d (%s) at %s:%d -> %s\n", (int)_e,                    \
                   hipGetErrorString(_e), __FILE__, __LINE__, #expr);                        \
      std::abort();                                                                         \
    }                                                                                       \
  } while (0)

// Like HIP_CHECK but returns the error instead of aborting (for probing paths
// that are expected to fail on some topologies, e.g. cross-node copy engine).
#define HIP_TRY(expr, out_err)                                                              \
  do {                                                                                      \
    (out_err) = (expr);                                                                     \
    if ((out_err) != hipSuccess) {                                                          \
      std::fprintf(stderr, "HIP (non-fatal) %d (%s) at %s:%d -> %s\n", (int)(out_err),       \
                   hipGetErrorString(out_err), __FILE__, __LINE__, #expr);                   \
    }                                                                                       \
  } while (0)

namespace ddai_spike {

// Runtime probe: does this device support the fabric handle type?
inline bool deviceFabricSupported(int dev) {
#if defined(SPIKE_HAS_FABRIC)
  int supported = 0;
  hipError_t e = hipDeviceGetAttribute(&supported, hipDeviceAttributeHandleTypeFabricSupported, dev);
  return (e == hipSuccess) && (supported != 0);
#else
  (void)dev;
  return false;
#endif
}

#if defined(SPIKE_HAS_FABRIC)

struct FabricAlloc {
  void* ptr = nullptr;                          // mapped device VA on the owning device
  hipMemGenericAllocationHandle_t handle = 0;   // generic allocation handle
  size_t size = 0;                              // granularity-rounded size
};

struct FabricImport {
  void* ptr = nullptr;                          // mapped VA into the importer's address space
  hipMemGenericAllocationHandle_t handle = 0;   // imported handle
  size_t size = 0;
};

inline size_t fabricGranularity() {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypeUncached;
  prop.location.type = hipMemLocationTypeDevice;
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) return 0;
  prop.location.id = dev;
  prop.requestedHandleTypes = hipMemHandleTypeFabric;
  prop.allocFlags.gpuDirectRDMACapable = 1;
  size_t g = 0;
  if (hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum) != hipSuccess) return 0;
  return g ? g : 1;
}

// Allocate a fabric-exportable VMM buffer on the current device, mapped RW for
// both the device and host (host access lets the spike read results back).
inline void fabricAlloc(FabricAlloc& a, size_t size) {
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypeUncached;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = dev;
  prop.requestedHandleTypes = hipMemHandleTypeFabric;
  prop.allocFlags.gpuDirectRDMACapable = 1;

  size_t g = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum));
  if (g == 0) g = 1;
  const size_t asize = ((size + g - 1) / g) * g;

  HIP_CHECK(hipMemCreate(&a.handle, asize, &prop, 0));

  void* base = nullptr;
  HIP_CHECK(hipMemAddressReserve(&base, asize, 0, 0, 0));
  HIP_CHECK(hipMemMap(base, asize, 0, a.handle, 0));

  hipMemAccessDesc acc[2];
  acc[0].location.type = hipMemLocationTypeDevice;
  acc[0].location.id = dev;
  acc[0].flags = hipMemAccessFlagsProtReadWrite;
  acc[1].location.type = hipMemLocationTypeHost;
  acc[1].location.id = 0;
  acc[1].flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(base, asize, acc, 2));

  a.ptr = base;
  a.size = asize;
}

inline void fabricExport(hipMemFabricHandle_t& out, const FabricAlloc& a) {
  HIP_CHECK(hipMemExportToShareableHandle(&out, a.handle, hipMemHandleTypeFabric, 0));
}

// Import a peer's fabric handle and map it into the current device's address
// space with RW access. The returned VA is the peer's memory, addressable by
// this device's shader cores and (single-node) copy engine.
inline void fabricImport(FabricImport& imp, const hipMemFabricHandle_t& h, size_t size) {
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));

  HIP_CHECK(hipMemImportFromShareableHandle(&imp.handle, (void*)&h, hipMemHandleTypeFabric));

  void* base = nullptr;
  HIP_CHECK(hipMemAddressReserve(&base, size, 0, 0, 0));
  HIP_CHECK(hipMemMap(base, size, 0, imp.handle, 0));

  hipMemAccessDesc acc;
  acc.location.type = hipMemLocationTypeDevice;
  acc.location.id = dev;
  acc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(base, size, &acc, 1));

  imp.ptr = base;
  imp.size = size;
}

inline void fabricFree(FabricAlloc& a) {
  if (!a.ptr) return;
  (void)hipMemUnmap(a.ptr, a.size);
  (void)hipMemAddressFree(a.ptr, a.size);
  (void)hipMemRelease(a.handle);
  a = FabricAlloc{};
}

inline void fabricFreeImport(FabricImport& imp) {
  if (!imp.ptr) return;
  (void)hipMemUnmap(imp.ptr, imp.size);
  (void)hipMemAddressFree(imp.ptr, imp.size);
  (void)hipMemRelease(imp.handle);
  imp = FabricImport{};
}

#endif  // SPIKE_HAS_FABRIC

}  // namespace ddai_spike

#endif  // DDAI_SPIKE_FABRIC_VMM_HPP_
