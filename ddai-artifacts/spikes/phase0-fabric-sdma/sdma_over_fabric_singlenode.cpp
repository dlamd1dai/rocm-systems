// Phase-0 spike #2: single-node copy-engine + shader-core writes into a
// FABRIC-IMPORTED peer VA.
//
// Answers work-plan Phase-0 item (2): validate on the actual MI455 SUT that a
// copy-engine transfer (hipMemcpyAsync, the "cudaMemcpyAsync/SDMA-put" of the
// original question) AND a shader-core store can write across the UALoE fabric
// into a peer's fabric-imported memory, single-node. This is the load-bearing
// assumption (evidence E3 in the work plan) that lets us keep the SDMA transport
// on single-node MI455.
//
// Model (single process, multiple GPUs on one node):
//   For each ordered pair (W=writer device, T=target device), W != T:
//     1. On T: allocate a fabric buffer bufT, zero it, export its fabric handle.
//     2. On W: import T's handle -> impPtr (T's memory, mapped into W).
//     3. On W: allocate srcW, fill with a known pattern.
//     4. Test A (copy engine): hipMemcpyAsync(impPtr, srcW, N, D2D) on W, verify bufT.
//     5. Test B (shader core): kernel on W stores pattern into impPtr + threadfence_system, verify bufT.
//
// A "verify" reads bufT back from device T to host and compares.
//
// NOTE: hipMemcpyAsync D2D uses the same DMA copy-engine hardware that the Anvil
// SDMA queue drives; if this succeeds, the production path (Anvil SDMA put into
// the imported VA) is expected to succeed as well. A follow-up variant can wire
// the real projects/rocshmem/src/sdma Anvil queue for an exact match.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "fabric_vmm.hpp"

namespace {

constexpr uint32_t kMagic = 0xA5A50000u;

__global__ void fillPattern(uint32_t* p, size_t n, uint32_t base) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = base + (uint32_t)i;
}

__global__ void zeroBuf(uint32_t* p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0u;
}

// Shader-core store into a (possibly fabric-imported) peer VA, with a
// system-scope fence so the writes are visible beyond this agent -- mirrors the
// ordering the multi-node shader-core transport will need (PR #3 guard).
__global__ void storePatternSys(uint4* dst, const uint4* src, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) dst[i] = src[i];
  __threadfence_system();
}

#if defined(SPIKE_HAS_FABRIC)

bool verifyTarget(int devT, const ddai_spike::FabricAlloc& bufT, size_t n, uint32_t base) {
  HIP_CHECK(hipSetDevice(devT));
  std::vector<uint32_t> host(n, 0xdeadbeefu);
  HIP_CHECK(hipMemcpy(host.data(), bufT.ptr, n * sizeof(uint32_t), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < n; ++i) {
    if (host[i] != base + (uint32_t)i) {
      std::printf("      MISMATCH at [%zu]: got 0x%08x want 0x%08x\n", i, host[i], base + (uint32_t)i);
      return false;
    }
  }
  return true;
}

// Returns: bit0 = copy-engine test passed, bit1 = shader-core test passed.
int runPair(int devW, int devT, size_t n) {
  const size_t bytes = n * sizeof(uint32_t);
  int result = 0;

  // (1) target buffer on T
  HIP_CHECK(hipSetDevice(devT));
  ddai_spike::FabricAlloc bufT{};
  ddai_spike::fabricAlloc(bufT, bytes);
  {
    dim3 b(256), g((n + 255) / 256);
    hipLaunchKernelGGL(zeroBuf, g, b, 0, 0, (uint32_t*)bufT.ptr, n);
    HIP_CHECK(hipDeviceSynchronize());
  }
  hipMemFabricHandle_t handle;
  ddai_spike::fabricExport(handle, bufT);

  // (2) import on W
  HIP_CHECK(hipSetDevice(devW));
  // Best-effort local peer access (ignored if unavailable / already enabled).
  int can = 0;
  (void)hipDeviceCanAccessPeer(&can, devW, devT);
  if (can) {
    hipError_t pe = hipDeviceEnablePeerAccess(devT, 0);
    if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) {
      std::printf("      (note) hipDeviceEnablePeerAccess(%d): %s\n", devT, hipGetErrorString(pe));
    }
  }
  ddai_spike::FabricImport imp{};
  ddai_spike::fabricImport(imp, handle, bufT.size);

  // (3) source on W
  ddai_spike::FabricAlloc srcW{};  // ordinary VMM works; reuse the helper (host-readable)
  ddai_spike::fabricAlloc(srcW, bytes);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // ---- Test A: copy engine (hipMemcpyAsync) ----
  {
    const uint32_t base = kMagic | 0x1111u;
    dim3 b(256), g((n + 255) / 256);
    hipLaunchKernelGGL(fillPattern, g, b, 0, stream, (uint32_t*)srcW.ptr, n, base);
    hipError_t ce = hipSuccess;
    HIP_TRY(hipMemcpyAsync(imp.ptr, srcW.ptr, bytes, hipMemcpyDeviceToDevice, stream), ce);
    if (ce == hipSuccess) HIP_TRY(hipStreamSynchronize(stream), ce);
    bool ok = (ce == hipSuccess) && verifyTarget(devT, bufT, n, base);
    HIP_CHECK(hipSetDevice(devW));
    std::printf("    [W=%d -> T=%d] copy-engine (hipMemcpyAsync D2D): %s\n", devW, devT, ok ? "PASS" : "FAIL");
    if (ok) result |= 0x1;
  }

  // ---- Test B: shader-core store + threadfence_system ----
  {
    HIP_CHECK(hipSetDevice(devT));
    dim3 b0(256), g0((n + 255) / 256);
    hipLaunchKernelGGL(zeroBuf, g0, b0, 0, 0, (uint32_t*)bufT.ptr, n);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipSetDevice(devW));
    const uint32_t base = kMagic | 0x2222u;
    dim3 b(256), g((n + 255) / 256);
    hipLaunchKernelGGL(fillPattern, g, b, 0, stream, (uint32_t*)srcW.ptr, n, base);
    const size_t n4 = n / 4;  // n is a multiple of 4 (see main)
    dim3 g4((n4 + 255) / 256);
    hipLaunchKernelGGL(storePatternSys, g4, b, 0, stream, (uint4*)imp.ptr, (const uint4*)srcW.ptr, n4);
    hipError_t se = hipSuccess;
    HIP_TRY(hipStreamSynchronize(stream), se);
    bool ok = (se == hipSuccess) && verifyTarget(devT, bufT, n, base);
    HIP_CHECK(hipSetDevice(devW));
    std::printf("    [W=%d -> T=%d] shader-core store + threadfence_system: %s\n", devW, devT, ok ? "PASS" : "FAIL");
    if (ok) result |= 0x2;
  }

  HIP_CHECK(hipStreamDestroy(stream));
  ddai_spike::fabricFreeImport(imp);
  ddai_spike::fabricFree(srcW);
  HIP_CHECK(hipSetDevice(devT));
  ddai_spike::fabricFree(bufT);
  return result;
}

#endif  // SPIKE_HAS_FABRIC

}  // namespace

int main(int argc, char** argv) {
#if !defined(SPIKE_HAS_FABRIC)
  std::printf("SPIKE_HAS_FABRIC=0: fabric handle API not available in this ROCm build.\n");
  std::printf("Rebuild on a ROCm 7.14+ (MI455) toolchain that exposes hipMemHandleTypeFabric.\n");
  return 77;  // skip
#else
  size_t n = (argc > 1) ? (size_t)std::atoll(argv[1]) : (1u << 20);  // default 1M uint32 = 4 MiB
  n = (n / 4) * 4;  // uint4 alignment
  if (n == 0) n = 4;

  int nDev = 0;
  HIP_CHECK(hipGetDeviceCount(&nDev));
  if (nDev < 2) {
    std::printf("Need >= 2 GPUs on this node; found %d.\n", nDev);
    return 77;
  }

  // Confirm fabric support on all devices before testing.
  for (int d = 0; d < nDev; ++d) {
    if (!ddai_spike::deviceFabricSupported(d)) {
      std::printf("dev %d does not support fabric handles; aborting single-node test.\n", d);
      return 77;
    }
  }

  std::printf("single-node fabric transport test: %d GPUs, n=%zu uint32 (%zu MiB)\n\n",
              nDev, n, (n * sizeof(uint32_t)) >> 20);

  int pairs = 0, ceOk = 0, shOk = 0;
  for (int w = 0; w < nDev; ++w) {
    for (int t = 0; t < nDev; ++t) {
      if (w == t) continue;
      std::printf("  pair W=%d T=%d:\n", w, t);
      int r = runPair(w, t, n);
      ++pairs;
      if (r & 0x1) ++ceOk;
      if (r & 0x2) ++shOk;
    }
  }

  std::printf("\nSUMMARY: %d pairs | copy-engine PASS %d/%d | shader-core PASS %d/%d\n",
              pairs, ceOk, pairs, shOk, pairs);
  std::printf("GO/NO-GO: if copy-engine PASS on all pairs => keep SDMA transport for single-node (work-plan Phase 3).\n");
  std::printf("          if only shader-core PASSes => single-node also uses shader-core (Phases 3/9 simplify).\n");
  return (ceOk == pairs || shOk == pairs) ? 0 : 1;
#endif
}
