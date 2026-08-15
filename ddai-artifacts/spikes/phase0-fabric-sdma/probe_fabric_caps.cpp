// Phase-0 spike #1: fabric capability probe.
//
// Answers work-plan Phase-0 item (1): "Confirm HIP_FABRIC_API on the MI455 SUT".
// Prints, per device: arch name, whether the fabric handle type is supported,
// VMM granularity, and whether peers can access each other (xGMI/PCIe local
// peer access, which is what gates the single-node SDMA path).
//
// Build/run: see README.md. This program never allocates fabric memory; it only
// queries attributes, so it is safe to run anywhere.
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <cstdio>

#include "fabric_vmm.hpp"

int main() {
  int nDev = 0;
  HIP_CHECK(hipGetDeviceCount(&nDev));
  std::printf("HIP_VERSION=%d  devices=%d\n", HIP_VERSION, nDev);

#if defined(SPIKE_HAS_FABRIC)
  std::printf("SPIKE_HAS_FABRIC=1 (toolchain exposes hipMemHandleTypeFabric / hipMemImportFromShareableHandle)\n");
#else
  std::printf("SPIKE_HAS_FABRIC=0 (fabric handle API NOT available in this ROCm build; see ROCm/rocm-systems #2170)\n");
#endif

  for (int d = 0; d < nDev; ++d) {
    hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, d));
    HIP_CHECK(hipSetDevice(d));

    bool fab = ddai_spike::deviceFabricSupported(d);
    size_t gran = 0;
#if defined(SPIKE_HAS_FABRIC)
    gran = ddai_spike::fabricGranularity();
#endif

    std::printf("dev %d: arch=%-12s fabricSupported=%d vmmGranularity=%zu\n",
                d, p.gcnArchName, (int)fab, gran);
  }

  // Local peer-access matrix: this is the property the rocSHMEM SDMA connect
  // loop relies on (hipDeviceCanAccessPeer). SDMA-over-fabric is only wired for
  // peers where this is true (node-local).
  std::printf("\nlocal peer-access matrix (hipDeviceCanAccessPeer), rows=src dst=cols:\n    ");
  for (int j = 0; j < nDev; ++j) std::printf("%3d", j);
  std::printf("\n");
  for (int i = 0; i < nDev; ++i) {
    std::printf("%3d:", i);
    for (int j = 0; j < nDev; ++j) {
      int can = 0;
      if (i != j) HIP_CHECK(hipDeviceCanAccessPeer(&can, i, j));
      else can = 1;
      std::printf("%3d", can);
    }
    std::printf("\n");
  }

  std::printf("\nInterpretation:\n");
  std::printf("  * fabricSupported=1 on gfx1250 => HIP_FABRIC_API is usable; proceed with sdma_over_fabric.\n");
  std::printf("  * fabricSupported=0 or SPIKE_HAS_FABRIC=0 => fabric path must stay behind #ifdef + fallback.\n");
  std::printf("  * peer-access=1 between two devices => single-node SDMA-over-fabric is expected to work for that pair.\n");
  return 0;
}
