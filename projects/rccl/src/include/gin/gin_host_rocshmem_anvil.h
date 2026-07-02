/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef GIN_HOST_ROCSHMEM_ANVIL_H_
#define GIN_HOST_ROCSHMEM_ANVIL_H_

#include "plugin/nccl_net.h"

extern ncclGin_t ncclGinRocshmemAnvilPlugin;

struct ncclComm;
ncclResult_t ncclGinAnvilBindResourceWindowSignals(struct ncclComm* comm, void* resourceUserPtr,
                                                   size_t arenaByteOffset, int nContexts,
                                                   int nSignalsPerContext);

#endif
