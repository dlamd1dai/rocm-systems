/******************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include <rocshmem/gin_anvil_factory.h>

#include "anvil.hpp"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <algorithm>

#include "log.hpp"

struct rocshmem_gin_anvil_opaque {
  int nRanks;
  int numChannels;
  int myRank;
  int myDeviceId;
  int sdmaChannelStride;
  rocshmem::anvil::SdmaQueueDeviceHandle** deviceHandles_d;
  uint64_t* sdmaDirty_d;
};

extern "C" int rocshmem_gin_anvil_probe(void) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) return 0;
  return 1;
}

static void checkHip(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    LOG_ERROR("rocshmem_gin_anvil: %s: %s", what, hipGetErrorString(e));
  }
}

// Mirrors ROCSHMEM_SDMA_SPREAD_CHANNELS / default-context assignSdmaChannel:
// stride=1 spreads wavefronts across channels; stride=0 pins to sdmaChannel.
static int ginAnvilSpreadChannelsFromEnv() {
  const char* e = getenv("NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS");
  if (!e || !e[0]) return 1;
  if (e[0] == '0' && e[1] == '\0') return 0;
  return atoi(e) != 0 ? 1 : 0;
}

extern "C" int rocshmem_gin_anvil_create(int nRanks, int myRank, int my_device_id,
                                         int (*allgather)(void* ctx, void* buf, size_t bytes_per_rank),
                                         void* allgather_ctx, int num_channels,
                                         rocshmem_gin_anvil_handle_t* out_handle, void** out_gpu_handles,
                                         uint64_t** out_sdma_dirty) {
  if (!out_handle || !out_gpu_handles || !out_sdma_dirty || !allgather || nRanks < 1 || myRank < 0 ||
      myRank >= nRanks)
    return -1;

  const int numChannels = std::max(1, std::min(8, num_channels));

  std::vector<int> devs(static_cast<size_t>(nRanks), -1);
  devs[static_cast<size_t>(myRank)] = my_device_id;
  if (allgather(allgather_ctx, devs.data(), sizeof(int)) != 0) {
    LOG_ERROR("rocshmem_gin_anvil: allgather(device ids) failed");
    return -1;
  }

  for (int i = 0; i < nRanks; ++i) {
    if (devs[static_cast<size_t>(i)] < 0) {
      LOG_ERROR("rocshmem_gin_anvil: invalid device id for rank %d", i);
      return -1;
    }
  }

  const int myDev = devs[static_cast<size_t>(myRank)];

  rocshmem::anvil::anvil.init();

  // Standalone Anvil stack (independent of rocSHMEM IPC SDMA). Connection
  // pattern matches SdmaImpl::sdmaHostInit: one queue set per local peer index.
  for (int local_pe = 0; local_pe < nRanks; ++local_pe) {
    const int remoteDev = devs[static_cast<size_t>(local_pe)];
    if (rocshmem::anvil::anvil.getSdmaQueue(myDev, remoteDev, 0) != nullptr) continue;
    if (myDev != remoteDev) rocshmem::anvil::EnablePeerAccess(myDev, remoteDev);
    rocshmem::anvil::anvil.connect(myDev, remoteDev, numChannels);
  }

  const int total = nRanks * numChannels;
  std::vector<rocshmem::anvil::SdmaQueueDeviceHandle*> host_handles(static_cast<size_t>(total),
                                                                    nullptr);
  for (int local_pe = 0; local_pe < nRanks; ++local_pe) {
    const int remoteDev = devs[static_cast<size_t>(local_pe)];
    for (int c = 0; c < numChannels; ++c) {
      rocshmem::anvil::SdmaQueue* q = rocshmem::anvil::anvil.getSdmaQueue(myDev, remoteDev, c);
      host_handles[static_cast<size_t>(local_pe * numChannels + c)] =
          q ? reinterpret_cast<rocshmem::anvil::SdmaQueueDeviceHandle*>(q->deviceHandle())
            : nullptr;
    }
  }

  rocshmem::anvil::SdmaQueueDeviceHandle** dev_row = nullptr;
  checkHip(hipMalloc(&dev_row, static_cast<size_t>(total) * sizeof(void*)), "hipMalloc handles");
  checkHip(hipMemcpy(dev_row, host_handles.data(), static_cast<size_t>(total) * sizeof(void*),
                     hipMemcpyHostToDevice),
           "hipMemcpy handles");

  uint64_t* dirty = nullptr;
  checkHip(hipExtMallocWithFlags((void**)&dirty, sizeof(uint64_t), hipDeviceMallocFinegrained),
           "hipExtMallocWithFlags sdmaDirty");
  checkHip(hipMemset(dirty, 0, sizeof(uint64_t)), "hipMemset sdmaDirty");

  auto* impl = new rocshmem_gin_anvil_opaque{};
  impl->nRanks = nRanks;
  impl->numChannels = numChannels;
  impl->myRank = myRank;
  impl->myDeviceId = myDev;
  impl->sdmaChannelStride = ginAnvilSpreadChannelsFromEnv();
  impl->deviceHandles_d = dev_row;
  impl->sdmaDirty_d = dirty;

  *out_handle = impl;
  *out_gpu_handles = dev_row;
  *out_sdma_dirty = dirty;
  return 0;
}

extern "C" void rocshmem_gin_anvil_destroy(rocshmem_gin_anvil_handle_t handle) {
  if (!handle) return;
  auto* impl = handle;
  if (impl->deviceHandles_d) hipFree(impl->deviceHandles_d);
  if (impl->sdmaDirty_d) hipFree(impl->sdmaDirty_d);
  delete impl;
  /* Intentionally do not call anvil.disconnect(): other libraries in-process may share AnvilLib. */
}

extern "C" int rocshmem_gin_anvil_get_n_ranks(rocshmem_gin_anvil_handle_t handle) {
  return handle ? handle->nRanks : 0;
}

extern "C" int rocshmem_gin_anvil_get_num_channels(rocshmem_gin_anvil_handle_t handle) {
  return handle ? handle->numChannels : 0;
}

extern "C" int rocshmem_gin_anvil_get_channel_stride(rocshmem_gin_anvil_handle_t handle) {
  return handle ? handle->sdmaChannelStride : 0;
}
