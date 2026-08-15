/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// STAGED / NOT-YET-WIRED prototype -- see gin_anvil_sdma_fabric.h header note.
// Destination when integrated: projects/rccl/src/gin/gin_anvil_sdma_fabric.cc,
// added to src/CMakeLists.txt only behind HIP_FABRIC_API. NOT compiled here.

#include "gin_anvil_sdma_fabric.h"

#ifdef HIP_FABRIC_API

#include "fabric_mem_handler.h"

#include <new>

GinAnvilFabricPeers::GinAnvilFabricPeers(struct ncclComm* comm)
    : comm_(comm),
      rank_(comm ? comm->rank : 0),
      nranks_(comm ? comm->nRanks : 0),
      handler_(nullptr) {}

GinAnvilFabricPeers::~GinAnvilFabricPeers() {
  delete handler_;  // ncclFabricMemHandler dtor unmaps/releases imported peers
}

ncclResult_t GinAnvilFabricPeers::exchange(void* selfPtr, CUmemGenericAllocationHandle selfHandle, size_t size) {
  if (comm_ == nullptr || nranks_ <= 0) return ncclInvalidArgument;

  // ncclFabricMemHandler drives a full-comm bootstrapAllGather of the shareable
  // (fabric) descriptor and imports+maps each peer -- the same mechanism the DDA
  // fabric path uses (fabric_mem_handler.cc), which works across the whole MNNVL
  // clique (unlike the intra-node LSA heap exchange in dev_runtime.cc).
  handler_ = new (std::nothrow) ncclFabricMemHandler(comm_->bootstrap, rank_, nranks_, comm_->memManager);
  if (handler_ == nullptr) return ncclSystemError;

  NCCLCHECK(handler_->addSelfDeviceMem(selfPtr, selfHandle, size));
  NCCLCHECK(handler_->exchangeMemPtrs());

  peerVas_.assign(static_cast<size_t>(nranks_), 0);
  for (int pe = 0; pe < nranks_; ++pe) {
    void* p = nullptr;
    if (pe == rank_) {
      p = selfPtr;
    } else {
      NCCLCHECK(handler_->getPeerDeviceMemPtr(pe, &p));
    }
    peerVas_[static_cast<size_t>(pe)] = reinterpret_cast<uintptr_t>(p);
  }
  return ncclSuccess;
}

void* GinAnvilFabricPeers::getPeerVa(int peer) const {
  if (peer < 0 || peer >= nranks_ || peerVas_.empty()) return nullptr;
  return reinterpret_cast<void*>(peerVas_[static_cast<size_t>(peer)]);
}

#endif  // HIP_FABRIC_API
