/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// STAGED / NOT-YET-WIRED prototype for the GIN-SDMA AllToAll fabric-heap path.
//
// Destination when integrated: projects/rccl/src/include/gin/gin_anvil_sdma_fabric.h
// (declaration) + projects/rccl/src/gin/gin_anvil_sdma_fabric.cc (definition),
// added to projects/rccl/src/CMakeLists.txt only behind HIP_FABRIC_API.
//
// This file is intentionally NOT part of any build target so it cannot break the
// SUT build before review. It has NOT been compiled. See
// ddai-artifacts/docs/gin-sdma-fabric-a2a-impl-design.md.
//
// Provides:
//   * ncclGinAnvilUseFabricPath(comm)  -- Phase 1 gating predicate.
//   * GinAnvilFabricPeers              -- per-registered-buffer cross-node fabric
//                                         peer-VA exchange, wrapping the proven
//                                         ncclFabricMemHandler (full-comm allgather).
#ifndef NCCL_GIN_ANVIL_SDMA_FABRIC_H_
#define NCCL_GIN_ANVIL_SDMA_FABRIC_H_

#include "comm.h"
#include "nccl.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// ncclDdaUseFabricPath analog for GIN-SDMA. gfx1250 + MNNVL + cuMem VMM.
static inline bool ncclGinAnvilUseFabricPath(struct ncclComm* comm) {
#ifdef HIP_FABRIC_API
  return comm != nullptr && comm->MNNVL == 1 &&
         IsArchMatch(comm->archName, "gfx1250") && ncclCuMemEnable();
#else
  (void)comm;
  return false;
#endif
}

#ifdef HIP_FABRIC_API

#include <cuda.h>  // CUmemGenericAllocationHandle

class ncclFabricMemHandler;  // projects/rccl/src/include/fabric_mem_handler.h

// Owns the cross-node fabric exchange for ONE registered GIN buffer. On success,
// getPeerVa(pe) returns each clique rank's imported VA for this buffer; those go
// straight into ncclGinAnvilSdmaMemHandle::remote_vas (with vmmStride = 0) so the
// device tier-2 resolver (resolveRemotePeerVa) targets fabric memory.
class GinAnvilFabricPeers {
 public:
  GinAnvilFabricPeers(struct ncclComm* comm);
  ~GinAnvilFabricPeers();

  GinAnvilFabricPeers(const GinAnvilFabricPeers&) = delete;
  GinAnvilFabricPeers& operator=(const GinAnvilFabricPeers&) = delete;

  // selfPtr/selfHandle/size describe this rank's registered buffer (handle from
  // the new ncclDevrGetMemHandle hook). Performs a clique-wide allgather + import
  // and fills the peer-VA table. peerVas has length comm->nRanks; peerVas[rank]
  // == selfPtr.
  ncclResult_t exchange(void* selfPtr, CUmemGenericAllocationHandle selfHandle, size_t size);

  // Imported (or self) VA for a clique rank; nullptr before exchange().
  void* getPeerVa(int peer) const;

  const std::vector<uintptr_t>& peerVaTable() const { return peerVas_; }

 private:
  struct ncclComm* comm_;
  int rank_;
  int nranks_;
  ncclFabricMemHandler* handler_;  // reuse projects/rccl/src/fabric_mem_handler.cc
  std::vector<uintptr_t> peerVas_;
};

#endif  // HIP_FABRIC_API

#endif  // NCCL_GIN_ANVIL_SDMA_FABRIC_H_
