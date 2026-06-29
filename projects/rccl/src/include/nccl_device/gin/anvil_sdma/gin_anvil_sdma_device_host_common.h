/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_DEVICE_HOST_COMMON_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_DEVICE_HOST_COMMON_H_

#include <stdint.h>

#define NCCL_GIN_ANVIL_SDMA_NET_VERSION 109

/** Default SDMA threshold (bytes). Transfers of at most this size use rocshmem_putmem;
 *  larger transfers use direct Anvil SDMA. Tuned for MI355: putmem wins below ~1 KiB
 *  per message; Anvil SDMA plateaus ~24.5 us for 2 KiB–64 KiB. */
// #define NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT 1024u
#define NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT 128u

/** Signaled/counter SDMA puts of at most this size drain inline (anvil::quiet) before
 *  the rocSHMEM fence/signal; larger SDMA ops defer quiet to gin.flush(). */
#define NCCL_GIN_ANVIL_SDMA_BLOCKING_MAX_DEFAULT 4096u

struct ncclGinAnvilSdmaGPUContext {
  void** queueHandles;
  uint64_t* sdmaDirty;
  uint64_t* signals;
  uint64_t* counters;
  uint32_t nSignals;
  uint32_t nCounters;
  uint32_t sdmaThreshold;
  uint32_t sdmaBlockingMax;
  int nRanks;
  int rank;
  int numChannels;
};

struct ncclGinAnvilSdmaMemHandle {
  uintptr_t baseAddr;  // Symmetric LSA flat VA; remote resolved via rocshmem_ptr
};

#endif
