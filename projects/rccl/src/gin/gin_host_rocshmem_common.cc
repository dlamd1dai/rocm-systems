/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#include "gin/gin_host_rocshmem_common.h"
#include "comm.h"

void ncclGinRocshmemSetInitContext(void *initCtx, struct ncclComm *comm) {
  struct ginRocshmemInitCtx *ctx = (struct ginRocshmemInitCtx *)initCtx;
  ctx->comm = comm;
}

#endif  // ENABLE_ROCSHMEM_GIN
