/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared host helper for fabric AllToAll LL lane build + cross-rank agree.
// Used by gin_host.cc (lane table) and gin_plugin_anvil_sdma.cc (GPU context).

#ifndef NCCL_GIN_FABRIC_A2A_PUBLISH_H_
#define NCCL_GIN_FABRIC_A2A_PUBLISH_H_

#include "algorithms/dda/fabric/fabric_init.h"
#include "comm.h"
#include "debug.h"
#include "gin/gin_fabric_a2a_host.h"
#include "nccl_device/gin/anvil_sdma/gin_fabric_ll_policy.h"
#include "param.h"

#include <cstring>

struct GinFabricA2ALaneBuildResult {
  ncclGinFabricA2ALane lane;
  int localEnabled;
  int allEnabled;
};

inline gin::fabric::GinFabricA2ACommState ginFabricA2ACommStateFromComm(struct ncclComm* comm) {
  return gin::fabric::GinFabricA2ACommState{comm->ddaFabricMemHandler,
                                          (void**)comm->ddaPeerPtrsDev,
                                          comm->ddaLLEpochDev,
                                          comm->ddaScratch,
                                          comm->ddaScratchBytes,
                                          comm->ddaLLEpochLen,
                                          comm->nRanks};
}

inline ncclResult_t ginFabricA2ALaneBuildAndAgree(struct ncclComm* comm, bool ddaLLEnabled, size_t llThreshold,
                                                  GinFabricA2ALaneBuildResult* out) {
  if (out == nullptr) return ncclInvalidArgument;
  memset(out, 0, sizeof(*out));
  if (comm == nullptr) return ncclInvalidArgument;
  if (!ginAnvilUseFabricMem(comm)) return ncclSuccess;

  const gin::fabric::GinFabricA2ACommState commState = ginFabricA2ACommStateFromComm(comm);
  out->localEnabled =
      gin::fabric::ginFabricA2ALaneTryBuild(commState, ddaLLEnabled, llThreshold, &out->lane) ? 1 : 0;
  return ncclGinFabricA2ALaneAgreeEnabled(comm, out->localEnabled, &out->allEnabled);
}

inline void ginFabricA2AWarnIfLaneUnavailable(struct ncclComm* comm, int allEnabled) {
  if (allEnabled) return;
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaPeerPtrsDev == nullptr || comm->ddaLLEpochDev == nullptr ||
      comm->ddaScratch == nullptr) {
    WARN("GIN A2A: fabric small-msg lane unavailable: missing DDA fabric resources");
  }
}

#endif  // NCCL_GIN_FABRIC_A2A_PUBLISH_H_
