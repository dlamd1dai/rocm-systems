/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

/**
 * Default device OSS7 toggle for shared anvil_device.hpp templates (put/putSignal).
 *
 * RCCL runtime control of OSS7 MI4 fusion uses ncclGinAnvilSdmaGPUContext::sdmaOss7
 * (see useSdmaFusedSignal in gin_anvil_sdma.h). This global defaults to enabled on
 * gfx950; NCCL_GIN_ANVIL_SDMA_OSS7=0 is honored via the context field, not host
 * hipMemcpyToSymbol (which would leave an unresolved host symbol in librccl.so).
 *
 * rocSHMEM keeps a weak definition in ipc_policy.cpp for standalone binaries.
 */

#include <gin_anvil/sdma_factory.h>
#include <hip/hip_runtime.h>

#include "sdma/sdma_opcodes.h"

namespace gin_anvil {
namespace sdma {

#if SDMA_IS_OSS7
__device__ int gin_anvil_sdma_oss7_enabled = 1;
#endif

}  // namespace sdma
}  // namespace gin_anvil

#endif  // ENABLE_ROCSHMEM_GIN
