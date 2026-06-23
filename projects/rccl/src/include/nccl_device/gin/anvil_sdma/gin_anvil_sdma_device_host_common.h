/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_DEVICE_HOST_COMMON_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_DEVICE_HOST_COMMON_H_

#include <stdint.h>

#define NCCL_GIN_ANVIL_SDMA_NET_VERSION 109

/** Default SDMA threshold (bytes). Transfers below this use IPC load/store. */
#define NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT 256u

struct ncclGinAnvilSdmaGPUContext {
  void** queueHandles;
  uint64_t* sdmaDirty;
  uint64_t* signals;
  uint64_t* counters;
  uint32_t nSignals;
  uint32_t nCounters;
  uint32_t sdmaThreshold;
  int nRanks;
  int rank;
  int numChannels;
};

struct ncclGinAnvilSdmaMemHandle {
  uintptr_t local_va;
  uintptr_t* remote_vas;
};

#endif
