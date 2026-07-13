/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

/**
 * Device OSS7 runtime toggle for GIN Anvil SDMA (NCCL_GIN_TYPE=5).
 *
 * ENABLE_ROCSHMEM_GIN builds link only host GIN plugins into librccl.so; rocSHMEM
 * device globals live in librocshmem.a linked by the test binary. hipMemcpyToSymbol
 * from gin_plugin_anvil_sdma.cc still needs this symbol inside librccl.so or
 * dlopen fails before any test runs (undefined gin_anvil_sdma_oss7_enabled).
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
