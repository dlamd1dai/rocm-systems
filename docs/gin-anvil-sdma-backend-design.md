# GIN Anvil SDMA backend — design and rationale

This document describes the **GIN Anvil SDMA** backend added to RCCL (`NCCL_NET_DEVICE_GIN_ANVIL_SDMA` / `NCCL_GIN_TYPE=6`) and the supporting **GIN Anvil SDMA factory** C API (`gin_anvil_sdma_*`). It records the main **design choices** and why they were made.

## Goal

Provide a GIN path that **replaces** “GIN + rocSHMEM API + SDMA policy” (where device work goes through `rocshmem_putmem` / fence / quiet and SDMA is selected inside rocSHMEM) with a backend that:

1. Issues **SDMA through Anvil** (`gin_anvil::sdma::put`, `quiet`, `signal`) **directly** from the GIN device templates.
2. Uses **IPC flat stores** (`gin_anvil_ipc_copy.h`) for small messages — same datapath as rocSHMEM’s `memcpy_lane` Put policy, without calling the rocSHMEM runtime API.
3. Mirrors the **host-side lifecycle** of **GIN–GDA** as closely as practical: `connect()` establishes transport resources, `regMrSym()` exchanges per-rank addressing metadata, `createContext()` builds the device-visible context (signals, counters, handles).

## Selection and versioning

- **`NCCL_GIN_TYPE=6`** selects this plugin, matching the numeric pattern used for type 5 (GDA). Type 4 (rocSHMEM API GIN) was removed; use type 6 for intra-node Anvil SDMA.
- **`netDeviceVersion`** uses `NCCL_GIN_ANVIL_SDMA_NET_VERSION` (101) in the device/host common header so it is distinct from the GDA/API GIN version constant if operators need to disambiguate logs or compatibility checks.

**Reasoning:** Keeping `NCCL_GIN_TYPE` aligned with `ncclNetDeviceType` avoids special cases in host code that maps `getProperties()` to `ncclGinType_t`.

## Why a standalone Anvil SDMA factory (`gin_anvil_sdma_*`)

Anvil’s host implementation (`SdmaQueue`, `AnvilLib`, HSA/KFD) is compiled into **`librocshmem.a`** alongside the rest of the rocSHMEM tree. Duplicating `anvil.cpp` inside `librccl.so` would risk **duplicate globals** (singleton `AnvilLib`) and ODR violations when an application links both RCCL and rocSHMEM.

**Choice:** Add a small **C-linkage factory** in the rocSHMEM tree (`include/gin_anvil/sdma_factory.h`, `src/sdma/gin_anvil_sdma_factory.cpp`) that:

- Runs the same **device discovery + `anvil.connect()` + handle table** pattern as `SdmaImpl::sdmaHostInit`, but keyed by **NCCL world rank → HIP ordinal** via bootstrap allgather (see below).
- Is compiled into rocSHMEM for every build; when **`USE_SDMA` is off**, the implementation stubs return `probe()==0` / `create()==-1` so RCCL can still link.

**Reasoning:** Single copy of Anvil host state in rocSHMEM matches how GIN–GDA keeps heavy NIC/QP logic out of duplicate translation units while still letting RCCL’s GIN-only link mode resolve symbols from the final executable.

## Rank ↔ GPU mapping

rocSHMEM’s IPC SDMA path indexes queues by **local PE / device index** on a symmetric single-node team. RCCL’s **`peer`** in `gin.put` is a **communicator rank**.

**Choice:** During `connect()`, all ranks **`bootstrapAllGather`** an `int` HIP device ordinal per rank. SDMA queues are created with `connect(myDev, devs[peer], numChannels)`, and the device table is laid out as **`peer * numChannels + ch`** (peer = rank index).

**Reasoning:** This preserves the same **“one column of queues per peer index”** mental model as the IPC policy, but swaps PE indices for **NCCL ranks**, which is what GIN kernels already use.

## Memory registration (no IB MR, no rocSHMEM constant-memory table)

GDA registers GPU memory with the **GDA PD** and exchanges **rkeys + VAs**. Anvil SDMA over XGMI uses **GPU VAs** visible to peer SDMA engines once **P2P is enabled**; there is no lkey/rkey surface in this path.

**Choice:** `regMrSym()` resolves each buffer’s **LSA flat self VA** via `ncclDevrGetLsaSelfAddr` and registers it in the **GIN-owned IPC table** (`ncclGinAnvilIpcTableRegisterVmm` with VMM stride across ranks). The device mem handle stores the symmetric **`baseAddr`**; **`ginAnvilResolvePeerVa`** maps local VA + peer to the remote GPU VA — without IB keys and without `rocshmem_buffer_register_vmm` / `rocshmem_ptr`.

**Reasoning:** Constant-memory user-buffer lookup (legacy PGAS IPC) adds indirection and ties the backend to `rocshmem_init` semantics. A small host-side IPC table with device-visible entries keeps peer resolution to one table scan per put.

## Signals and counters

- **Signals:** Carved from the communicator **LSA symmetric resource window** after `ncclDevComm` setup (`ncclGinAnvilBindResourceWindowSignals`). Each GIN context gets a slot in the window arena; **`ncclGinAnvilIpcTableRegisterVmm`** maps peer signal VAs using the same VMM stride as data buffers. Remote updates use **`ipcFlatAtomicAddSys64`** on the resolved peer VA.
- **Counters:** Local-only `uint64_t` array on device via **`hipExtMallocWithFlags`** (same idea as GDA plugin).
- **Ordering:** After `gin_anvil::sdma::put`, use **`gin_anvil::sdma::quiet`** when a **counter** is involved; else **`__builtin_amdgcn_fence(release, agent)`** before SDMA-fused signal, or **`__threadfence_system()`** after IPC flat stores before a separate signal atomic.

**Reasoning:** Signals must be peer-accessible through the same LSA flat / IPC table path as data. Counters stay local and do not need symmetric mapping.

## `gin_anvil::sdma::signal` and `SignalAdd`

The current Anvil helper **`signal()`** submits a fixed **64-bit atomic add of 1** (see `CreateAtomicIncPacket` in `anvil_device.hpp`). Arbitrary **`SignalAdd`** values are not implemented in this first revision.

**Reasoning:** Documented limitation to avoid inventing new SDMA packet shapes without hardware review; **SignalInc** / default increment path matches the hardware packet already used by Anvil.

## Small-message path (IPC flat stores)

Transfers of at most **`NCCL_GIN_ANVIL_SDMA_THRESHOLD`** bytes (default 128 B, tunable via env) use **`ipcPut` / `ipcPutScalar`** from `gin_anvil_ipc_copy.h`: cached local loads plus **system-scope flat stores** to the peer GPU VA. This is the same mechanism rocSHMEM’s IPC `memcpy_lane` Put policy uses internally, inlined without `#include <rocshmem/rocshmem.hpp>` on the device.

**Reasoning:** Benchmarks on MI355 show IPC flat stores win below ~128 B–1 KiB per message; Anvil SDMA has ~24.5 µs setup overhead and wins above the threshold. Keeping both paths avoids paying SDMA doorbell cost on tiny AlltoAll slices.

## rocSHMEM API removal summary

| Former rocSHMEM API | Replacement | Performance note |
|---------------------|-------------|------------------|
| `rocshmem_buffer_register_vmm` + `rocshmem_ptr` | `ncclGinAnvilIpcTableRegisterVmm` + `ginAnvilResolvePeerVa` | Host table + device lookup |
| `rocshmem_putmem` / `*_p` | `ipcPut` / `ipcPutScalar` to resolved peer VA | Identical flat-store instructions |
| `rocshmem_uint64_atomic_add` | `ipcFlatAtomicAddSys64` on IPC-resolved peer signal VA | Same system-scope flat atomic |
| `rocshmem_quiet` / `rocshmem_fence` | `gin_anvil::sdma::quiet` (SDMA dirty queues) + `__threadfence_system` (IPC) | Avoids nocall into uninitialised PGAS runtime context |
| `rocshmem_malloc` / `rocshmem_free` (signals) | LSA resource-window arena + IPC table | Peer-accessible symmetric signal VAs |

**Still linked from `librocshmem.a` (by design):** `gin_anvil_sdma_*` C factory and `gin_anvil::sdma::*` SDMA device helpers live in the rocSHMEM build artifact to avoid duplicating `AnvilLib` / KFD state inside `librccl.so`. Naming is **gin-anvil**-specific; the backend does **not** call `rocshmem_init()` or other PGAS APIs.

## Dirty bitmask and channel selection

The device path sets a **per (peer, channel) dirty bit** after `put`, matching the IPC SDMA policy’s bitmask shape. **Channel index** is fixed to **0** in the device templates for simplicity; host-side **`NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS`** still creates multiple queues per peer for future wavefront spreading.

**Reasoning:** Keeps the first implementation correct and small; channel striping can follow `sdma_policy.hpp` later without changing the host factory API.

## `gin_anvil_sdma_destroy` and `anvil.disconnect()`

**Choice:** Destroy frees HIP allocations owned by the handle but **does not** call `anvil.disconnect()`, because disconnect tears down **all** SDMA queues in the process and could break concurrent rocSHMEM users sharing `AnvilLib`.

**Reasoning:** Conservative coexistence with rocSHMEM runtime in the same process; acceptable queue lifetime trade-off until a refcounted design exists.

## RCCL integration summary

| Area | Change |
|------|--------|
| `net_device.h` | `NCCL_NET_DEVICE_GIN_ANVIL_SDMA = 6` |
| `gin_device_common.h` | `NCCL_GIN_ANVIL_SDMA_ENABLE`, backend mask, dispatch `switch` case |
| `gin_device_api.h` | Include `anvil_sdma/gin_anvil_sdma.h` when enabled |
| `gin_host.cc` | Recognize new `netDeviceType` for `ginType` |
| `plugin/gin.cc` | Third internal plugin (GDA + Anvil); `ncclGinRocshmemSetInitContext` (GDA), `ncclGinAnvilSetInitContext` (Anvil) |
| `gin/CMakeLists.txt` | Build `gin_plugin_anvil_sdma.cc` |
| New headers | `gin_host_anvil_sdma.h`, `gin_anvil_sdma*.h` |
| rocSHMEM tree | `gin_anvil_sdma_factory.cpp` in `src/sdma/` + public header `include/gin_anvil/sdma_factory.h` |

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

## Test plan

This section is the **authoritative test plan** for GIN Anvil SDMA (`NCCL_GIN_TYPE=6`). Docker harness details and Test#5 wiring live in [`docker-gin-gda-ruby-gin-backends-and-tests.md`](docker-gin-gda-ruby-gin-backends-and-tests.md).

### 1. Objectives

Verify that the Anvil SDMA GIN backend:

1. **Initialises** on supported single-node, multi-GPU xGMI topologies without `rocshmem_init()`.
2. **Moves data correctly** via both datapaths: IPC flat stores (≤ threshold) and Anvil SDMA (> threshold).
3. **Orders** puts, signals, counters, and flush/quiet consistently with collective kernels (`GinAlltoAllKernel`).
4. **Performs** at or above the prior GIN–rocSHMEM–API path on intra-node workloads (type 4 removed; compare against Test#1 baseline and optional Test#4 GDA).
5. **Coexists** with rocSHMEM in-process (factory in `librocshmem`, no duplicate `AnvilLib` in `librccl.so`).

### 2. Scope

| In scope | Out of scope (this revision) |
|----------|------------------------------|
| Single-node, 1 rank per GPU, HIP ordinal == rank (default layout) | Multi-node / IB Anvil |
| `alltoall_perf -D 3` (`GinAlltoAllKernel`) | Arbitrary `SignalAdd` values (Anvil fixed +1 packet only) |
| Message sizes 128 B – 128 MiB (perf sweep) | Full RCCL collective suite beyond AlltoAll |
| MI300 / MI355 class (gfx942 / gfx950) | Non-AMD GPUs |
| Env tuning: channels, threshold, fused signal | `anvil.disconnect()` on destroy (queues persist by design) |

### 3. Test environment matrix

| ID | Hardware | GPUs | Image / build | Notes |
|----|----------|------|---------------|-------|
| **E1** | MI355X single node | 8 | `docker-gin-gda-sdma-build.bash`, `GPU_TARGETS=gfx950`, `ROCSHMEM_USE_SDMA=1` | Primary perf / regression (e.g. `smci355-*`) |
| **E2** | MI300X single node | 8 | Same, `GPU_TARGETS=gfx942` | xGMI Anvil validation |
| **E3** | MI355X, bare metal | 8 | RCCL + rocSHMEM installed from same tree | Optional; bypass Docker |
| **E4** | Any supported node | 1–8 | Same image | Scale-down (`NP=1`, `NP=2`, …) |

**Hard prerequisites (all environments):**

- rocSHMEM built with **`USE_SDMA=ON`** (`ROCSHMEM_USE_SDMA=1` in Docker build).
- RCCL built with **`ENABLE_ROCSHMEM_GIN=ON`**.
- Image **`libmlx5`** must export **`mlx5dv_reg_dmabuf_mr`** for RCCL NET init (Test#5 preflight). Install **`extra-rdma-debs/*.deb`** at image build or set **`TEST5_HOST_MLX5_LIB_DIR`** at run time (see [`extra-rdma-debs/README.md`](../extra-rdma-debs/README.md)).
- **`HSA_FORCE_FINE_GRAIN_PCIE=1`** for GIN runs (set by test scripts).
- **`ROCSHMEM_SDMA_ENABLED=0`** for Test#5 (does **not** select rocSHMEM API SDMA tunneling; Anvil is direct).

### 4. Build and install verification

| Test ID | Step | Pass criteria |
|---------|------|---------------|
| **B1** | `./docker-gin-gda-sdma-preflight.bash` | Required headers/sources present; no stale type-4 files; single `markSdmaDirty` in `gin_anvil_sdma.h` |
| **B2** | `RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=1 ./docker-gin-gda-sdma-build.bash 1` | Image build completes; log contains `OK: MLX5 DMA-BUF/sysfs symbols found` |
| **B3** | `docker run --rm $IMAGE rocshmem/bin/rocshmem_info` | Reports SDMA / Anvil enabled when `USE_SDMA=ON` |
| **B4** | `objdump -T $(readlink -f /usr/lib/x86_64-linux-gnu/libmlx5.so.1) \| grep mlx5dv_reg_dmabuf_mr` inside image | Symbol present (Test#5 preflight green) |
| **B5** | `grep -r gin_host_rocshmem_common projects/rccl/src/CMakeLists.txt` + files on disk | CMake hipify list matches filesystem (avoids `ddai-gin-build.log` CMake failure) |

### 5. Host / plugin lifecycle

Run with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` and `NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=6`.

| Test ID | Action | Pass criteria |
|---------|--------|---------------|
| **H1** | `gin_anvil_sdma_probe()` via tiny host binary or first RCCL init | Returns 1 on SDMA-capable HIP node; 0 when `USE_SDMA=OFF` stub build |
| **H2** | RCCL comm init, 8 ranks | Log: `GIN anvil-sdma: standalone SDMA queues (8 ranks, N ch, spread=…)` |
| **H3** | Wrong `NCCL_GIN_TYPE` (e.g. 5) with Anvil plugin forced | Anvil `init()` returns error; no silent fallback |
| **H4** | `connect()` rank ↔ GPU mapping | `bootstrapAllGather` of HIP ordinals; queue table `peer * numChannels + ch` |
| **H5** | `regMrSym()` on symmetric LSA window | IPC table entry via `ncclGinAnvilIpcTableRegisterVmm`; no `rocshmem_buffer_register_vmm` |
| **H6** | `createContext()` + resource window bind | Device context `layoutMagic == NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC`; LSA signal slots bound via `ncclGinAnvilBindResourceWindowSignals` |
| **H7** | Comm destroy | No crash; `gin_anvil_sdma_destroy` frees HIP-owned state (queues remain — by design) |

### 6. Device datapath (functional)

Use `alltoall_perf -D 3 -V 1` (validation on). Sweep `-b` / `-e` / `-f 2`.

| Test ID | Configuration | Exercises | Pass criteria |
|---------|---------------|-----------|---------------|
| **D1** | Default threshold (128 B) | IPC path for tiny messages | `#wrong == 0` at 128 B – 128 B |
| **D2** | `-b 256 -e 4K` | IPC vs SDMA boundary | `#wrong == 0`; latency knee near threshold on MI355 |
| **D3** | `-b 4K -e 128M` | Anvil SDMA bulk path | `#wrong == 0` all sizes |
| **D4** | `NCCL_GIN_ANVIL_SDMA_THRESHOLD=0` or `1` | Force SDMA for small msgs | Correctness maintained (may be slower) |
| **D5** | `NCCL_GIN_ANVIL_SDMA_THRESHOLD=65536` | Force IPC for medium msgs | Correctness maintained |
| **D6** | `NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL=1` | OSS7 copy+signal SDMA packet | Correctness on MI355; compare vs default `0` |
| **D7** | `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1,2,4,8` | Multi-queue per peer | Init succeeds; `#wrong == 0` at 1 MiB |
| **D8** | `Flush` after puts (`GinAlltoAllKernel` implicit quiet) | Dirty bitmask + `gin_anvil::sdma::quiet` | No hang; validation pass |

**Signal / counter spot checks (if exposed via future micro-test or debugger):**

- **D9** `SignalInc` / default increment only (no arbitrary `SignalAdd`).
- **D10** Counter increment visible after fence/quiet path.

### 7. Collective integration (primary harness)

Default automation: **`docker-gin-gda-sdma-test.bash`** / **`docker-gin-gda-sdma-ruby-test.bash`**.

| Test ID | Harness slot | Env summary | Pass criteria |
|---------|--------------|-------------|---------------|
| **C1** | Test#1 | `NCCL_GIN_ENABLE=0`, `-D 0` | Baseline; `#wrong == 0`; records busbw for comparison |
| **C2** | Test#5 | `NCCL_GIN_TYPE=6`, `-D 3`, `ROCSHMEM_SDMA_ENABLED=0` | `#wrong == 0`; `ginType != NONE` in init logs |
| **C3** | Test#5, `NP=1` | Same as C2 | Single-GPU smoke (may skip peer traffic) |
| **C4** | Test#5, `NP=2,4,8` | Same as C2 | Correctness at each scale |
| **C5** | Optional Test#4 | `NCCL_GIN_TYPE=5`, bnxt + firmware gate | GDA reference on supported NICs only |

**Example (Test#5, full matrix):**

```bash
RCCL_GIN_RUN_TESTS=5 \
  TEST5_MLX5_PREFLIGHT=1 \
  ./docker-gin-gda-sdma-test.bash 8 128M 2>&1 | tee ddai-gin-perf.log
```

Ensure Test#5 is not skipped: image must pass MLX5 preflight (see §3) or set `TEST5_HOST_MLX5_LIB_DIR`.

### 8. Performance regression

| Test ID | Metric | Method | Pass / track |
|---------|--------|--------|---------------|
| **P1** | AlltoAll busbw @ 128M | C1 vs C2 | Test#5 ≥ Test#1 on xGMI (intra-node) |
| **P2** | Small-message latency | C2 with `-b 128 -e 4K -f 2` | IPC path ≤ prior API-backend targets on MI355 |
| **P3** | Medium-message plateau | C2 with `-b 4K -e 64K` | Anvil SDMA ~24.5 µs/msg region (MI355 tuning note in header) |
| **P4** | Large-message bw | C2 with `-e 128M` | Stable vs baseline; no regression >5% vs best prior Anvil commit |
| **P5** | Channels scaling | `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1,4,8` | Document bw/latency; no correctness loss |

Record: hostname, GPU model, ROCm version, RCCL commit, `NCCL_GIN_ANVIL_SDMA_*` env, log file name.

### 9. Negative and robustness

| Test ID | Scenario | Expected |
|---------|----------|----------|
| **N1** | `NCCL_GIN_TYPE=6` without `USE_SDMA` rocSHMEM | Probe/create fails; clear init error |
| **N2** | Mismatched rank/GPU layout (ordinal ≠ rank) | Document failure or wrong results; file bug if silent corruption |
| **N3** | Missing IPC table entry / bad LSA registration | Init or run fails with WARN, not segfault |
| **N4** | Invalid `layoutMagic` on device | Kernels no-op (`anvilCtxValid` false); no GPU fault |
| **N5** | Concurrent rocSHMEM `rocshmem_init` + Anvil GIN | No duplicate `AnvilLib` crash (coexistence) |
| **N6** | `TEST5_MLX5_PREFLIGHT=0` without MLX5 symbols | Test runs but may fail NET init — documents infra gap only |

### 10. Comparison matrix (sanity)

Run same size sweep for each backend on **one** fixed environment (E1):

| Backend | `NCCL_GIN_TYPE` | `-D` | Role |
|---------|-----------------|------|------|
| Host baseline | 0 (GIN off) | 0 | CPU / CE reference |
| GIN Ib proxy | 2 | 3 | Host-progress GIN (needs verbs + GDR) |
| GIN GDA | 5 | 3 | NIC QP path (Test#4; bnxt gate) |
| **GIN Anvil SDMA** | **6** | **3** | **Primary subject** |

Anvil should lead intra-node xGMI for mid/large messages vs Test#1; vs GDA depends on NIC and message size.

### 11. CI / release checklist

Before merge or image publish:

- [ ] **B1–B5** pass on builder
- [ ] **C1 + C2** pass on at least one MI300/MI355 8-GPU node
- [ ] **D1–D3** validation clean on Test#5
- [ ] **P1** no regression vs last green perf log
- [ ] Docs: `gin-anvil-sdma-backend-design.md` + docker test doc updated
- [ ] Type 4 removed: no `gin_plugin_rocshmem_api` in tree or install list

### 12. Debugging playbook

| Symptom | Check |
|---------|--------|
| Test#5 skipped at start | MLX5 preflight; `extra-rdma-debs` or `TEST5_HOST_MLX5_LIB_DIR` |
| `ginType NONE` / `-D 3` error | `NCCL_GIN_ENABLE=1`, `NCCL_GIN_TYPE=6`, `NCCL_DEBUG=INFO` NET lines |
| Hang in AlltoAll | SDMA dirty / quiet; try `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1` |
| Wrong results small sizes | Threshold / IPC path; `NCCL_GIN_ANVIL_SDMA_THRESHOLD` |
| Wrong results large sizes | IPC table + LSA flat base resolution |
| `gin_anvil_sdma_create failed` | `USE_SDMA`, HIP devices visible, xGMI peer access |

### 13. Key environment variables (test tuning)

| Variable | Default | Test use |
|----------|---------|----------|
| `NCCL_GIN_ENABLE` | off | Must be `1` |
| `NCCL_GIN_TYPE` | — | Must be `6` |
| `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS` | `1` | D7, P5 |
| `NCCL_GIN_ANVIL_SDMA_THRESHOLD` | `128` | D4, D5 |
| `NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL` | `0` | D6 |
| `ROCSHMEM_SDMA_ENABLED` | — | Must be `0` for Test#5 |
| `HSA_FORCE_FINE_GRAIN_PCIE` | — | `1` (scripts) |
| `RCCL_GIN_RUN_TESTS` | `1,5` | Harness test selection |
| `TEST5_MLX5_PREFLIGHT` | `1` | Set `0` to force run without symbols |

---

*File: `docs/gin-anvil-sdma-backend-design.md` / `docs/gin-anvil-sdma-backend-design.rst` — GIN Anvil SDMA backend design and test plan.*
