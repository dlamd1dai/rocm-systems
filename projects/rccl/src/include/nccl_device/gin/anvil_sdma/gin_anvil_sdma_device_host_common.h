/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_DEVICE_HOST_COMMON_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_DEVICE_HOST_COMMON_H_

#include <stdint.h>

#define NCCL_GIN_ANVIL_SDMA_NET_VERSION 111

/** Must match host plugin and device kernel build; checked on device. */
#define NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC 0xA6E17111u

/** Default SDMA threshold (bytes). Transfers of at most this size use IPC flat stores
 *  (gin_anvil_ipc_copy.h); larger transfers use direct Anvil SDMA. Tuned for MI355:
 *  IPC put wins below ~128 B per message; Anvil SDMA plateaus ~24.5 us for 2 KiB–64 KiB. */
// #define NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT 1024u
#define NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT 128u

/** Default off: fused OSS7 copy+signal needs remote GPU signal VA; opt-in via env on MI355. */
#define NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL_DEFAULT 0u

struct ncclGinAnvilSdmaGPUContext {
  uint32_t layoutMagic;  // NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC
  void** queueHandles;   // [local_pe * numChannels + ch] SdmaQueueDeviceHandle*
  uint64_t* sdmaDirty;   // GIN-owned dirty bitmask (separate from rocSHMEM IPC SDMA)
  uint64_t* signals;
  uintptr_t* signalPeerAddrs;  // per-rank remote signal buffer base (allgather at createContext)
  uint64_t* counters;
  uint32_t nSignals;
  uint32_t nCounters;
  uint32_t sdmaThreshold;
  uint32_t fusedSdmaSignal;  // use COPY_LINEAR_WAIT_SIGNAL_MI4 for SignalInc SDMA puts
  int nRanks;
  int rank;
  int numChannels;
  int sdmaChannel;
  int sdmaChannelStride;
};

struct ncclGinAnvilSdmaMemHandle {
  // Rank-0 slot base in local LSA flat layout (SDMA path via add4G).
  uintptr_t lsaFlatBase;
  uint32_t stride4G;  // devr->bigSize >> 32; peer stride for ncclGetLsaPointer-style lookup
  // Per-rank registration VAs (allgather at regMrSym). IPC flat stores use remoteVas[peer]+off.
  uintptr_t* remoteVas;  // device pointer, length nRanks
  int nRanks;
};

#endif
