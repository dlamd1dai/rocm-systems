/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GIN_DEVCOMM_ROLLBACK_H_
#define NCCL_GIN_DEVCOMM_ROLLBACK_H_

#include "nccl_device/impl/comm__types.h"

#include <cstring>

// Host-visible GIN fields after ncclGinDevCommSetup succeeded but
// ncclDevrCommCreateInternal failed. Pure state; no GPU.
inline void ncclGinDevCommClearGinFields(struct ncclDevComm* outDevComm) {
  if (outDevComm == nullptr) return;
  outDevComm->ginContextCount = 0;
  memset(outDevComm->ginHandles, 0, sizeof(outDevComm->ginHandles));
  memset(outDevComm->ginNetDeviceTypes, 0, sizeof(outDevComm->ginNetDeviceTypes));
  outDevComm->resourceWindow = nullptr;
  outDevComm->resourceWindow_inlined = {};
}

#endif
