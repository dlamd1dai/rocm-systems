// Phase-0 spike #3: multi-node (MNNVL clique) fabric peer test.
//
// Answers work-plan Phase-0 item (3): confirm across nodes that (a) shader-core
// stores into a fabric-imported REMOTE-node peer VA land correctly with a
// system-scope fence, and (b) the copy engine is NOT a viable cross-node
// transport (optional, --try-ce), which is why the multi-node transport must be
// shader-core (evidence E4 in the work plan).
//
// Model (MPI, one rank per GPU, ranks span >= 2 nodes):
//   Ring exchange of fabric handles via MPI_Allgather. Each rank r:
//     - allocates + zeroes a fabric buffer, exports its handle
//     - imports its right neighbor's handle  dst = (r+1) % size
//     - shader-core stores a rank-keyed pattern into the neighbor's buffer + threadfence_system
//     - after a barrier, verifies its OWN buffer holds the pattern written by
//       its left neighbor src = (r-1+size) % size
//
// Requires an MPI build (USE_MPI) and the fabric API (SPIKE_HAS_FABRIC).
// GPU binding: uses OMPI_COMM_WORLD_LOCAL_RANK / SLURM_LOCALID if present,
// else falls back to (worldRank % deviceCount).
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fabric_vmm.hpp"

#if defined(USE_MPI)
#include <mpi.h>
#endif

namespace {

constexpr uint32_t kMagic = 0xB6B60000u;

__global__ void fillPattern(uint32_t* p, size_t n, uint32_t base) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = base + (uint32_t)i;
}
__global__ void zeroBuf(uint32_t* p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0u;
}
__global__ void storePatternSys(uint4* dst, const uint4* src, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) dst[i] = src[i];
  __threadfence_system();
}

int pickLocalDevice(int worldRank, int nDev) {
  const char* envs[] = {"OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "MV2_COMM_WORLD_LOCAL_RANK"};
  for (const char* e : envs) {
    const char* v = std::getenv(e);
    if (v && *v) return std::atoi(v) % nDev;
  }
  return worldRank % nDev;
}

}  // namespace

int main(int argc, char** argv) {
#if !defined(USE_MPI)
  (void)argc; (void)argv;
  std::printf("USE_MPI not defined: rebuild with -DSPIKE_ENABLE_MPI=ON to run the multi-node test.\n");
  return 77;
#elif !defined(SPIKE_HAS_FABRIC)
  MPI_Init(&argc, &argv);
  int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r);
  if (r == 0) std::printf("SPIKE_HAS_FABRIC=0: fabric handle API not available in this ROCm build.\n");
  MPI_Finalize();
  return 77;
#else
  MPI_Init(&argc, &argv);
  int worldRank = 0, worldSize = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  bool tryCE = false;
  for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--try-ce") == 0) tryCE = true;

  size_t n = (1u << 20);  // 4 MiB of uint32
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') { n = (size_t)std::atoll(argv[i]); break; }
  }
  n = (n / 4) * 4; if (n == 0) n = 4;
  const size_t bytes = n * sizeof(uint32_t);

  int nDev = 0; HIP_CHECK(hipGetDeviceCount(&nDev));
  int dev = pickLocalDevice(worldRank, nDev);
  HIP_CHECK(hipSetDevice(dev));

  if (!ddai_spike::deviceFabricSupported(dev)) {
    std::printf("[rank %d] dev %d lacks fabric support; skipping.\n", worldRank, dev);
    MPI_Finalize();
    return 77;
  }
  if (worldSize < 2) {
    std::printf("Need >= 2 ranks (ideally spanning >= 2 nodes).\n");
    MPI_Finalize();
    return 77;
  }

  // Allocate + zero our buffer, export handle.
  ddai_spike::FabricAlloc myBuf{};
  ddai_spike::fabricAlloc(myBuf, bytes);
  { dim3 b(256), g((n + 255) / 256); hipLaunchKernelGGL(zeroBuf, g, b, 0, 0, (uint32_t*)myBuf.ptr, n); HIP_CHECK(hipDeviceSynchronize()); }

  hipMemFabricHandle_t myHandle;
  ddai_spike::fabricExport(myHandle, myBuf);

  // Allgather all handles (fixed-size POD). Also gather the granularity-rounded size.
  std::vector<hipMemFabricHandle_t> allHandles(worldSize);
  MPI_Allgather(&myHandle, sizeof(hipMemFabricHandle_t), MPI_BYTE,
                allHandles.data(), sizeof(hipMemFabricHandle_t), MPI_BYTE, MPI_COMM_WORLD);
  unsigned long long mySize = (unsigned long long)myBuf.size;
  std::vector<unsigned long long> allSizes(worldSize);
  MPI_Allgather(&mySize, 1, MPI_UNSIGNED_LONG_LONG, allSizes.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

  const int dst = (worldRank + 1) % worldSize;
  const int src = (worldRank - 1 + worldSize) % worldSize;

  // Import the neighbor we write to.
  ddai_spike::FabricImport imp{};
  ddai_spike::fabricImport(imp, allHandles[dst], (size_t)allSizes[dst]);

  // Source buffer with our rank-keyed pattern.
  ddai_spike::FabricAlloc srcW{};
  ddai_spike::fabricAlloc(srcW, bytes);
  const uint32_t myBase = kMagic | (uint32_t)worldRank;
  { dim3 b(256), g((n + 255) / 256); hipLaunchKernelGGL(fillPattern, g, b, 0, 0, (uint32_t*)srcW.ptr, n, myBase); HIP_CHECK(hipDeviceSynchronize()); }

  hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream));

  // Optional: probe copy-engine cross-node (expected to be non-viable). Guarded
  // so it does not hang the ring test; run in isolation with --try-ce.
  int ceStatus = -1;  // -1 = not attempted
  if (tryCE) {
    hipError_t ce = hipSuccess;
    HIP_TRY(hipMemcpyAsync(imp.ptr, srcW.ptr, bytes, hipMemcpyDeviceToDevice, stream), ce);
    if (ce == hipSuccess) HIP_TRY(hipStreamSynchronize(stream), ce);
    ceStatus = (ce == hipSuccess) ? 1 : 0;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Shader-core store into neighbor's fabric-imported buffer + system fence.
  const size_t n4 = n / 4;
  { dim3 b(256), g4((n4 + 255) / 256); hipLaunchKernelGGL(storePatternSys, g4, b, 0, stream, (uint4*)imp.ptr, (const uint4*)srcW.ptr, n4); }
  HIP_CHECK(hipStreamSynchronize(stream));

  MPI_Barrier(MPI_COMM_WORLD);  // ensure all writers finished before verify

  // Verify our own buffer was written by src with base = kMagic | src.
  std::vector<uint32_t> host(n, 0xdeadbeefu);
  HIP_CHECK(hipMemcpy(host.data(), myBuf.ptr, bytes, hipMemcpyDeviceToHost));
  const uint32_t wantBase = kMagic | (uint32_t)src;
  int localOk = 1;
  for (size_t i = 0; i < n; ++i) {
    if (host[i] != wantBase + (uint32_t)i) {
      std::printf("[rank %d dev %d] shader-core MISMATCH at [%zu]: got 0x%08x want 0x%08x (writer=%d)\n",
                  worldRank, dev, i, host[i], wantBase + (uint32_t)i, src);
      localOk = 0;
      break;
    }
  }
  std::printf("[rank %d dev %d] recv-from %d shader-core: %s%s\n", worldRank, dev, src,
              localOk ? "PASS" : "FAIL",
              (ceStatus < 0) ? "" : (ceStatus ? "  | cross-node copy-engine: UNEXPECTEDLY OK"
                                              : "  | cross-node copy-engine: not viable (expected)"));

  int globalOk = 0;
  MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (worldRank == 0) {
    std::printf("\nSUMMARY: shader-core cross-fabric AllToAll-ring %s across %d ranks.\n",
                globalOk ? "PASS" : "FAIL", worldSize);
    std::printf("GO/NO-GO: shader-core PASS => multi-node transport confirmed (work-plan Phase 3 fallback).\n");
  }

  HIP_CHECK(hipStreamDestroy(stream));
  ddai_spike::fabricFreeImport(imp);
  ddai_spike::fabricFree(srcW);
  ddai_spike::fabricFree(myBuf);
  MPI_Finalize();
  return globalOk ? 0 : 1;
#endif
}
