/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * @file gin_allgather_verify.cu
 *
 * Standalone MPI verification for GinAllGatherKernel (-D 3, NCCL_GIN_TYPE=5).
 * Runs one AllGather iteration on 2–8 local GPUs and compares recv to golden.
 *
 * Run:
 *   NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=5 ROCSHMEM_SDMA_ENABLED=0 \
 *   mpirun -np 8 ./gin_allgather_verify
 */

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#define HIP_CHECK(cmd) do {                                      \
  hipError_t e = (cmd);                                          \
  if (e != hipSuccess) {                                         \
    fprintf(stderr, "[rank %d] HIP error %d at %s:%d\n",         \
            rank, (int)e, __FILE__, __LINE__);                   \
    MPI_Abort(MPI_COMM_WORLD, 1);                                \
  }                                                              \
} while(0)

#define NCCL_CHECK(cmd) do {                                     \
  ncclResult_t r = (cmd);                                        \
  if (r != ncclSuccess) {                                        \
    fprintf(stderr, "[rank %d] NCCL error %d at %s:%d\n",        \
            rank, (int)r, __FILE__, __LINE__);                   \
    MPI_Abort(MPI_COMM_WORLD, 1);                                \
  }                                                              \
} while(0)

static int rank = 0;
static int nranks = 0;

template <typename T>
__global__ void GinAllGatherVerifyKernel(ncclWindow_t sendwin, size_t sendoffset,
                                         ncclWindow_t recvwin, size_t recvoffset,
                                         size_t count, int /*root*/,
                                         struct ncclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;
  const size_t size = count * sizeof(T);

  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset,
        size, ncclGin_SignalInc{signalIndex});
  }

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());
}

static void fillPattern(int32_t* buf, size_t count, int seed) {
  for (size_t i = 0; i < count; i++) {
    buf[i] = (int32_t)(seed * 1315423911u + (uint32_t)i);
  }
}

static int verifyPattern(const int32_t* buf, size_t count, int seed) {
  for (size_t i = 0; i < count; i++) {
    int32_t expect = (int32_t)(seed * 1315423911u + (uint32_t)i);
    if (buf[i] != expect) return (int)i + 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  if (nranks < 2 || nranks > 8) {
    if (rank == 0) {
      fprintf(stderr, "gin_allgather_verify: requires 2 <= np <= 8 (got %d)\n", nranks);
    }
    MPI_Finalize();
    return 1;
  }

  const char* ginEnable = getenv("NCCL_GIN_ENABLE");
  const char* ginType = getenv("NCCL_GIN_TYPE");
  if (!ginEnable || atoi(ginEnable) != 1 || !ginType || atoi(ginType) != 5) {
    if (rank == 0) {
      fprintf(stderr, "gin_allgather_verify: set NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=5\n");
    }
    MPI_Finalize();
    return 1;
  }

  int nGpus = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpus));
  if (nGpus <= 0) {
    if (rank == 0) fprintf(stderr, "gin_allgather_verify: no HIP devices\n");
    MPI_Finalize();
    return 1;
  }
  HIP_CHECK(hipSetDevice(rank % nGpus));

  ncclUniqueId uid;
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&uid));
  MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, nranks, uid, rank));

  const size_t countPerRank = 4096; // 16 KiB int32 slice
  const size_t sendBytes = countPerRank * sizeof(int32_t);
  const size_t recvBytes = sendBytes * nranks;

  int32_t *sendDev = nullptr, *recvDev = nullptr;
  HIP_CHECK(hipMalloc(&sendDev, sendBytes));
  HIP_CHECK(hipMalloc(&recvDev, recvBytes));
  HIP_CHECK(hipMemset(recvDev, 0, recvBytes));

  std::vector<int32_t> sendHost(countPerRank);
  fillPattern(sendHost.data(), countPerRank, rank);
  HIP_CHECK(hipMemcpy(sendDev, sendHost.data(), sendBytes, hipMemcpyHostToDevice));

  ncclWindow_t sendWin = nullptr;
  ncclWindow_t recvWin = nullptr;
  NCCL_CHECK(ncclCommWindowRegister(comm, sendDev, sendBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
  NCCL_CHECK(ncclCommWindowRegister(comm, recvDev, recvBytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.barrierCount = 1;
  reqs.ginSignalCount = 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;

  ncclDevComm devComm;
  NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &devComm));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  GinAllGatherVerifyKernel<int32_t><<<1, 512, 0, stream>>>(
      sendWin, 0, recvWin, 0, countPerRank, 0, devComm);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int32_t> recvHost(recvBytes / sizeof(int32_t));
  HIP_CHECK(hipMemcpy(recvHost.data(), recvDev, recvBytes, hipMemcpyDeviceToHost));

  int localWrong = 0;
  for (int r = 0; r < nranks; r++) {
    const int32_t* slice = recvHost.data() + (size_t)r * countPerRank;
    int err = verifyPattern(slice, countPerRank, r);
    if (err != 0) localWrong++;
  }

  int totalWrong = 0;
  MPI_Allreduce(&localWrong, &totalWrong, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0) {
    if (totalWrong == 0) {
      printf("PASS: gin_allgather_verify np=%d count=%zu\n", nranks, countPerRank);
    } else {
      printf("FAIL: gin_allgather_verify wrong_slices=%d\n", totalWrong);
    }
  } else if (localWrong) {
    printf("[rank %d] FAIL: recv mismatch\n", rank);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  NCCL_CHECK(ncclDevCommDestroy(comm, &devComm));
  NCCL_CHECK(ncclCommWindowDeregister(comm, sendWin));
  NCCL_CHECK(ncclCommWindowDeregister(comm, recvWin));
  NCCL_CHECK(ncclCommDestroy(comm));
  HIP_CHECK(hipFree(sendDev));
  HIP_CHECK(hipFree(recvDev));

  MPI_Finalize();
  return totalWrong ? 1 : 0;
}
