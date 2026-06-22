# GIN Anvil SDMA backend — design and rationale

This document describes the **GIN Anvil SDMA** backend added to RCCL (`NCCL_NET_DEVICE_GIN_ANVIL_SDMA` / `NCCL_GIN_TYPE=6`) and the supporting **rocSHMEM GIN Anvil factory** C API. It records the main **design choices** and why they were made.

## Goal

Provide a GIN path that **replaces** “GIN + rocSHMEM API + SDMA policy” (where device work goes through `rocshmem_putmem` / fence / quiet and SDMA is selected inside rocSHMEM) with a backend that:

1. Issues **SDMA through Anvil** (`rocshmem::anvil::put`, `quiet`, `signal`) **directly** from the GIN device templates.
2. Mirrors the **host-side lifecycle** of **GIN–GDA** as closely as practical: `connect()` establishes transport resources, `regMrSym()` exchanges per-rank addressing metadata, `createContext()` builds the device-visible context (signals, counters, handles).

## Selection and versioning

- **`NCCL_GIN_TYPE=6`** selects this plugin, matching the numeric pattern used for types 4 (API) and 5 (GDA).
- **`netDeviceVersion`** uses `NCCL_GIN_ANVIL_SDMA_NET_VERSION` (101) in the device/host common header so it is distinct from the GDA/API GIN version constant if operators need to disambiguate logs or compatibility checks.

**Reasoning:** Keeping `NCCL_GIN_TYPE` aligned with `ncclNetDeviceType` avoids special cases in host code that maps `getProperties()` to `ncclGinType_t`.

## Why a rocSHMEM-hosted C factory (`rocshmem_gin_anvil_*`)

Anvil’s host implementation (`SdmaQueue`, `AnvilLib`, HSA/KFD) already lives in **rocSHMEM** and is linked into `librocshmem.a`. Duplicating `anvil.cpp` inside `librccl.so` would risk **duplicate globals** (singleton `AnvilLib`) and ODR violations when an application links both RCCL and rocSHMEM.

**Choice:** Add a small **C-linkage factory** in rocSHMEM (`include/rocshmem/gin_anvil_factory.h`, `src/sdma/gin_anvil_factory.cpp`) that:

- Runs the same **device discovery + `anvil.connect()` + handle table** pattern as `SdmaImpl::sdmaHostInit`, but keyed by **NCCL world rank → HIP ordinal** via bootstrap allgather (see below).
- Is compiled into rocSHMEM for every build; when **`USE_SDMA` is off**, the implementation stubs return `probe()==0` / `create()==-1` so RCCL can still link.

**Reasoning:** Single copy of Anvil host state in rocSHMEM matches how GIN–GDA keeps heavy NIC/QP logic out of duplicate translation units while still letting RCCL’s GIN-only link mode resolve symbols from the final executable.

## Rank ↔ GPU mapping

rocSHMEM’s IPC SDMA path indexes queues by **local PE / device index** on a symmetric single-node team. RCCL’s **`peer`** in `gin.put` is a **communicator rank**.

**Choice:** During `connect()`, all ranks **`bootstrapAllGather`** an `int` HIP device ordinal per rank. SDMA queues are created with `connect(myDev, devs[peer], numChannels)`, and the device table is laid out as **`peer * numChannels + ch`** (peer = rank index).

**Reasoning:** This preserves the same **“one column of queues per peer index”** mental model as the IPC policy, but swaps PE indices for **NCCL ranks**, which is what GIN kernels already use.

## Memory registration (no IB MR)

GDA registers GPU memory with the **GDA PD** and exchanges **rkeys + VAs**. Anvil SDMA over XGMI uses **GPU VAs** visible to peer SDMA engines once **P2P is enabled**; there is no lkey/rkey surface in this path.

**Choice:** `regMrSym()` only **allgathers base VAs** per rank and packs them into a small device-side `ncclGinAnvilSdmaMemHandle` (local VA + device array of remote VAs), analogous to the VA half of the GDA mem handle without keys.

**Reasoning:** Same **exchange step** as GDA for addressing; drops IB-specific fields that do not exist on the SDMA path.

## Signals and counters

- **Signals:** Fine-grained GPU memory, **`bootstrapAllGather`** of each rank’s signal base pointer, device array **`signal_peer_addrs[peer]`** so a sender can compute the remote signal cell address (mirrors GDA’s `signal_raddrs`, without rkeys).
- **Counters:** Local-only `uint64_t` array on device (same idea as API/GDA plugins).
- **Ordering:** After `anvil::put`, use **`quiet`** when a **counter** is involved (must observe completion of the copy before bumping the local counter), else a **release fence** before `anvil::signal` when only a signal is used—aligned with the rocSHMEM API GIN template’s quiet/fence split.

**Reasoning:** Matches observable ordering from the API backend while using Anvil primitives instead of `rocshmem_fence` / `rocshmem_quiet`.

## `anvil::signal` and `SignalAdd`

The current Anvil helper **`signal()`** submits a fixed **64-bit atomic add of 1** (see `CreateAtomicIncPacket` in `anvil_device.hpp`). Arbitrary **`SignalAdd`** values are not implemented in this first revision.

**Reasoning:** Documented limitation to avoid inventing new SDMA packet shapes without hardware review; **SignalInc** / default increment path matches the hardware packet already used by Anvil.

## Dirty bitmask and channel selection

The device path sets a **per (peer, channel) dirty bit** after `put`, matching the IPC SDMA policy’s bitmask shape. **Channel index** is fixed to **0** in the device templates for simplicity; host-side **`NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS`** still creates multiple queues per peer for future wavefront spreading.

**Reasoning:** Keeps the first implementation correct and small; channel striping can follow `sdma_policy.hpp` later without changing the host factory API.

## `rocshmem_gin_anvil_destroy` and `anvil.disconnect()`

**Choice:** Destroy frees HIP allocations owned by the handle but **does not** call `anvil.disconnect()`, because disconnect tears down **all** SDMA queues in the process and could break concurrent rocSHMEM users sharing `AnvilLib`.

**Reasoning:** Conservative coexistence with rocSHMEM runtime in the same process; acceptable queue lifetime trade-off until a refcounted design exists.

## RCCL integration summary

| Area | Change |
|------|--------|
| `net_device.h` | `NCCL_NET_DEVICE_GIN_ANVIL_SDMA = 6` |
| `gin_device_common.h` | `NCCL_GIN_ROCSHMEM_ANVIL_ENABLE`, backend mask, dispatch `switch` case |
| `gin_device_api.h` | Include `anvil_sdma/gin_anvil_sdma.h` when enabled |
| `gin_host.cc` | Recognize new `netDeviceType` for `ginType` |
| `plugin/gin.cc` | Fourth internal plugin; `ncclGinRocshmemSetInitContext` for Anvil |
| `gin/CMakeLists.txt` | Build `gin_plugin_rocshmem_anvil.cc` |
| New headers | `gin_host_rocshmem_anvil.h`, `gin_anvil_sdma*.h` |
| rocSHMEM | `gin_anvil_factory.cpp` in `src/` + public header under `include/rocshmem/` |

## Build requirements

- rocSHMEM must be configured with **`USE_SDMA=ON`** for non-stub factory behavior (Anvil queues and `hsakmt`).
- RCCL tests or apps that execute device GIN kernels must link **device-capable rocSHMEM** the same way as for GDA/API (`ENABLE_ROCSHMEM` / `-fgpu-rdc --hip-link` as already documented for the tree).

## Usage sketch

```text
export NCCL_GIN_ENABLE=1
export NCCL_GIN_TYPE=6
export NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1   # optional, default 1
# Optional: same-node, one rank per GPU, hip ordinal == rank layout works best.
```

No `rocshmem_init()` is required for this backend’s host setup (mirroring the GDA “standalone factory” idea), but the final link must still provide **rocSHMEM symbols** (`--allow-shlib-undefined` for GIN-only `librccl.so` builds, resolved at app link).

---

*File: `docs/gin-anvil-sdma-backend-design.md` — generated as part of the GIN Anvil SDMA backend implementation.*
