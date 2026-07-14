GIN Anvil SDMA backend — design and rationale
=============================================


This document describes the GIN Anvil SDMA backend added to RCCL (`NCCL_NET_DEVICE_GIN_ANVIL_SDMA` / `NCCL_GIN_TYPE=5`) and the supporting GIN Anvil SDMA factory C API (`gin_anvil_sdma_*`). It records the main design choices and why they were made.


Goal
----


Provide an **intra-node GIN backend** for xGMI topologies that:

1. Issues SDMA through Anvil (`gin_anvil::sdma::put`, `quiet`, `signal`) directly from the GIN device templates.
2. Uses IPC flat stores (`gin_anvil_ipc_copy.h`) for small messages — the same flat-store instruction pattern as Anvil's IPC `memcpy_lane` Put path, without a PGAS runtime on the device datapath.
3. Mirrors the host-side lifecycle of GIN–GDA as closely as practical: `connect()` establishes transport resources, `regMrSym()` exchanges per-rank addressing metadata, `createContext()` builds the device-visible context (signals, counters, handles).


Selection and versioning
------------------------


- `NCCL_GIN_TYPE=5` selects this plugin. `NCCL_GIN_TYPE` must equal `ncclNetDeviceType` from `net_device.h`; `gin_host.cc` casts directly to `ncclGinType_t`.
- AMD GIN net-device values in `net_device.h`: **2** proxy, **3** GDAKI, **4** GDA (`NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA`), **5** Anvil SDMA (`NCCL_NET_DEVICE_GIN_ANVIL_SDMA`, this backend).
- `netDeviceVersion` uses `NCCL_GIN_ANVIL_SDMA_NET_VERSION` (**115**) in `gin_anvil_sdma_device_host_common.h` so it is distinct from the GDA GIN version constant if operators need to disambiguate logs or compatibility checks.

Reasoning: One numeric value for env selection and net-device typing.


Architecture overview
---------------------


The backend is a **hybrid**: RCCL owns the GIN plugin, peer VA resolution, small-message IPC path, and signal binding; the **Anvil SDMA engine** lives in the rocSHMEM tree and is linked via `librocshmem.a`.

.. code-block:: text

   alltoall_perf / GinAlltoAllKernel
            │
            ▼
   gin_anvil_sdma.h (device templates)
      ├─ bytes ≤ threshold ──► ipcPut / ipcPutScalar (gin_anvil_ipc_copy.h)
      │                              │
      │                              ▼
      │                    ginAnvilResolvePeerVa (GIN IPC table)
      │
      └─ bytes > threshold ──► gin_anvil::sdma::put / putSignal / quiet
                                     │
                                     ▼
                           queueHandles[peer * numChannels + ch]
                                     │
                                     ▼
                           gin_anvil_sdma_factory (librocshmem.a)

   Host plugin (gin_plugin_anvil_sdma.cc)
      connect()  ──► gin_anvil_sdma_create (bootstrap HIP ordinals, anvil.connect)
      regMrSym() ──► ncclDevrGetLsaSelfAddr + ncclGinAnvilIpcTableRegisterVmm
      createContext() ──► GPU context; signals deferred until resource window exists
      dev_runtime ──► ncclGinAnvilBindResourceWindowSignals (LSA arena slots)


**Datapath selection** happens per put on device: compare `bytes` to `sdmaThreshold` (from `NCCL_GIN_ANVIL_SDMA_THRESHOLD`, default 128). If SDMA queue lookup fails, the kernel falls back to IPC flat stores. If SDMA is selected but peer VA resolution fails, templates fall back to `ipcPut` / `ipcPutScalar`.


Why a standalone Anvil SDMA factory (`gin_anvil_sdma_*`)
------------------------------------------------------


Anvil’s host implementation (`SdmaQueue`, `AnvilLib`, HSA/KFD) is compiled into `librocshmem.a` alongside the rest of the rocSHMEM tree. Duplicating `anvil.cpp` inside `librccl.so` would risk duplicate globals (singleton `AnvilLib`) and ODR violations when an application links both RCCL and rocSHMEM.

Choice: Add a small C-linkage factory in the rocSHMEM tree (`include/gin_anvil/sdma_factory.h`, `src/sdma/gin_anvil_sdma_factory.cpp`) that:

- Runs the same device discovery + `anvil.connect()` + handle table pattern as `SdmaImpl::sdmaHostInit`, but keyed by NCCL world rank → HIP ordinal via bootstrap allgather (see below).
- Is compiled into rocSHMEM for every build; when `USE_SDMA` is off, the implementation stubs return `probe()==0` / `create()==-1` so RCCL can still link.

Reasoning: Single copy of Anvil host state in rocSHMEM matches how GIN–GDA keeps heavy NIC/QP logic out of duplicate translation units while still letting RCCL’s GIN-only link mode resolve symbols from the final executable.


Rank ↔ GPU mapping
------------------


Anvil SDMA queue tables index peers by local device / PE index on a symmetric single-node team. RCCL’s `peer` in `gin.put` is a communicator rank.

Choice: During `connect()`, all ranks `bootstrapAllGather` an `int` HIP device ordinal per rank. SDMA queues are created with `connect(myDev, devs[peer], numChannels)`, and the device table is laid out as `peer * numChannels + ch` (peer = rank index).

Reasoning: This preserves the same “one column of queues per peer index” mental model as the IPC policy, but swaps PE indices for NCCL ranks, which is what GIN kernels already use.


Memory registration (no IB MR, GIN-owned peer table)
-----------------------------------------------------------------


GDA registers GPU memory with the GDA PD and exchanges rkeys + VAs. Anvil SDMA over XGMI uses GPU VAs visible to peer SDMA engines once P2P is enabled; there is no lkey/rkey surface in this path.

Choice: `regMrSym()` resolves each buffer’s LSA flat self VA via `ncclDevrGetLsaSelfAddr` and registers it in the GIN-owned IPC table (`ncclGinAnvilIpcTableRegisterVmm` with VMM stride across ranks). The device mem handle stores the symmetric `baseAddr`; `ginAnvilResolvePeerVa` maps local VA + peer to the remote GPU VA at put time.

Reasoning: A small host-side IPC table with device-visible entries keeps peer resolution to one table scan per put and does not require PGAS heap registration or constant-memory buffer lookup.


GIN-owned IPC peer table
------------------------


Peer VA resolution is **RCCL-owned**:

=======================  =================================================================
Piece                    Role
=======================  =================================================================
`gin_anvil_ipc_table_host.cc`  Host master table; one-time `hipMalloc` for device copy
`gin_anvil_ipc_table_device.h`  `ginAnvilResolvePeerVa` on device
`ncclGinAnvilIpcTableRegisterVmm`  LSA flat self VA + VMM stride → `remote_bases[pe]`
`ncclGinAnvilIpcTableTrackContext`  Registers live GPU contexts for refresh
`refreshAllLiveContexts()`  Updates `ipcTable` / `ipcTableCount` on all contexts after sync
=======================  =================================================================

**VMM stride model:** `remote_bases[pe] = local_base + (pe - myRank) * strideBytes`. Data buffers and signal slots both register through the same table.

**Limits:** `NCCL_GIN_ANVIL_IPC_MAX_BUFS` (16), `NCCL_GIN_ANVIL_IPC_MAX_RANKS` (16).

Reasoning: A stable device table pointer plus context refresh avoids stale `ipcTable` when buffers register mid-comm. This replaced earlier `hipExtMalloc` signal paths that faulted at 128 B on MI355X.


Signals and counters
--------------------


Signal lifecycle
~~~~~~~~~~~~~~~~


1. **`createContext()`** allocates the device context, sets `signals = nullptr`, and if `nSignals > 0` adds the context to a **pending list** (no `hipExtMalloc` for signals).
2. **`dev_runtime.cc`** creates the symmetric **resource window**, calls `ncclGinDevCommSetup`, then **`ncclGinAnvilBindResourceWindowSignals`**.
3. **`ncclGinAnvilBindResourceWindowSignals`** assigns each context a **signal slot**, resolves LSA flat VA, registers in the IPC table, and copies `signals` into the GPU context.

Counters and ordering
~~~~~~~~~~~~~~~~~~~~~


- **Counters:** Local-only `uint64_t` array via `hipExtMallocWithFlags` (same as GDA).
- **Ordering:** `gin_anvil::sdma::quiet` when counters involved; else agent fence before fused SDMA signal, or `__threadfence_system()` before IPC signal atomics (`ipcFlatAtomicAddSys64`).

Reasoning: Signals must be peer-accessible through LSA flat addressing and the IPC table. Counters stay local.


`gin_anvil::sdma::signal` and `SignalAdd`
-----------------------------------------


**SDMA fused signal** (`NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL=1`, OSS7): Anvil `putSignal` / `signal()` submit a fixed +1 atomic (`CreateAtomicIncPacket`). Only `SignalInc` on the fused SDMA path.

**IPC / decoupled signals:** `signalPeer()` uses `ipcFlatAtomicAddSys64` with `signalOpArg`, so arbitrary `SignalAdd` values work when not using fused SDMA signaling.

Reasoning: Hardware-reviewed SDMA packet for fusion; flexible atomics on the IPC signal path.


Small-message path (IPC flat stores)
------------------------------------


Transfers of at most `NCCL_GIN_ANVIL_SDMA_THRESHOLD` bytes (default 128 B, tunable via env) use `ipcPut` / `ipcPutScalar` from `gin_anvil_ipc_copy.h`: cached local loads plus system-scope flat stores to the peer GPU VA. This matches the flat-store pattern used by Anvil's IPC `memcpy_lane` Put implementation, inlined in GIN device templates.

Reasoning: Benchmarks on MI355 show IPC flat stores win below ~128 B–1 KiB per message; Anvil SDMA has ~24.5 µs setup overhead and wins above the threshold. Keeping both paths avoids paying SDMA doorbell cost on tiny AlltoAll slices.


Device primitives (standalone GIN path)
---------------------------------------


The backend implements GIN puts with RCCL-owned tables and direct Anvil SDMA — **no `rocshmem_init()`** on the host setup path:

===============================================  =================================================================  =========================================
Operation                                        Implementation                                                     Notes
===============================================  =================================================================  =========================================
Peer VA lookup                                   `ncclGinAnvilIpcTableRegisterVmm` + `ginAnvilResolvePeerVa`        Host table; device scan per put
Small puts                                       `ipcPut` / `ipcPutScalar`                                          System-scope flat stores
Large puts                                       `gin_anvil::sdma::put` / `putSignal`                               Above threshold
Signal atomics                                   `ipcFlatAtomicAddSys64`                                            IPC-resolved peer signal VA
SDMA ordering                                    `gin_anvil::sdma::quiet` on dirty queues                           Flush walks peer×channel bitmask
IPC ordering                                     `__threadfence_system`                                             After flat stores before separate signal
Signals                                          LSA resource-window arena + IPC table                              Peer-accessible symmetric VAs
Counters                                         Local `hipExtMallocWithFlags` array                                No symmetric mapping
===============================================  =================================================================  =========================================

Still linked from `librocshmem.a` (by design): `gin_anvil_sdma_*` C factory and `gin_anvil::sdma::*` device helpers live in the rocSHMEM build artifact to avoid duplicating `AnvilLib` / KFD state inside `librccl.so`. The GIN plugin calls the factory and device helpers directly; it does not enter the PGAS runtime or `sdma_policy.hpp` dispatch layer.


What still comes from rocSHMEM / `librocshmem.a`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


=====  ===========================================  ==========================================
Layer  API / symbol                               Purpose
=====  ===========================================  ==========================================
Host factory  `gin_anvil_sdma_probe/create/destroy`  SDMA queues; no `rocshmem_init()`
Host  `gin_anvil::sdma::initEndpoint`, `anvil.connect`  AnvilLib / HSA/KFD
Device  `gin_anvil::sdma::put`, `putSignal`, `quiet`  Large puts (> threshold)
Headers  `anvil_device.hpp`, `sdma_opcodes.h`  Packet layouts
Link  `librocshmem.a` device symbols  App link resolves SDMA device code
=====  ===========================================  ==========================================

GIN Anvil SDMA calls the same underlying `gin_anvil::sdma::*` device code as Anvil's internal SDMA paths, but routes from `gin_anvil_sdma.h` templates rather than through rocSHMEM's `sdma_policy.hpp` layer.


Dirty bitmask and channel selection
-----------------------------------


The device path sets a per (peer, channel) dirty bit after `put`. Factory `sdmaChannelStride` is **1** when `NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS` is on (default), else **0**. **`effectiveChannel()`** uses `(sdmaChannel + sdmaChannelStride * (blockId / 64)) % numChannels` (base channel **0**). Host `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS` creates up to 8 queues per peer.

Reasoning: Spread reduces queue contention; dirty tracking ensures `Flush` only `quiet`s active queues.


`gin_anvil_sdma_destroy` and `anvil.disconnect()`
-------------------------------------------------


Choice: Destroy frees HIP allocations owned by the handle but does not call `anvil.disconnect()`, because disconnect tears down all SDMA queues in the process and could break concurrent rocSHMEM users sharing `AnvilLib`.

Reasoning: Conservative coexistence with rocSHMEM runtime in the same process; acceptable queue lifetime trade-off until a refcounted design exists.


RCCL integration summary
------------------------


=======================  =============================================  ==============================
Area                     File / symbol                                  Role
=======================  =============================================  ==============================
`net_device.h`           `NCCL_NET_DEVICE_GIN_ANVIL_SDMA = 5`           Net device type
`gin_device_common.h`    `NCCL_GIN_ANVIL_SDMA_ENABLE`                   Device API routing
`gin_plugin_anvil_sdma.cc`  Plugin vtable                               Host connect / reg / context
`gin_anvil_ipc_table_host.cc`  IPC table                                  Peer VA registration
`dev_runtime.cc`         `ncclGinAnvilBindResourceWindowSignals`        Signal arena bind
`gin_anvil_sdma.h`       Device templates                               IPC + SDMA put path
rocSHMEM tree            `gin_anvil_sdma_factory.cpp`                   Standalone SDMA factory
=======================  =============================================  ==============================

Plugin: **`ncclGinAnvilSdmaPlugin`**. Init hook: **`ncclGinAnvilSetInitContext`**.


Key source files
----------------


See `docs/gin-anvil-sdma-backend-design.md` for the full path table (`gin_plugin_anvil_sdma.cc`, `gin_anvil_ipc_table_host.cc`, `gin_anvil_sdma.h`, `sdma_factory.h`, `anvil_device.hpp`, etc.).


Build requirements
------------------


- rocSHMEM must be configured with `USE_SDMA=ON` for non-stub factory behavior (Anvil queues and `hsakmt`).
- RCCL tests or apps that execute device GIN kernels must link device-capable rocSHMEM (`ENABLE_ROCSHMEM` / `-fgpu-rdc --hip-link`). Device templates require `NCCL_GIN_ANVIL_SDMA_ENABLE`, which needs `ENABLE_ROCSHMEM` (not `ENABLE_ROCSHMEM_GIN` alone).


Usage sketch
------------


.. code-block:: text

   export NCCL_GIN_ENABLE=1
   export NCCL_GIN_TYPE=5
   export NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1   # optional, default 1
   # Optional: same-node, one rank per GPU, hip ordinal == rank layout works best.


No `rocshmem_init()` is required for this backend’s host setup (mirroring the GDA “standalone factory” idea), but the final link must still provide rocSHMEM symbols (`--allow-shlib-undefined` for GIN-only `librccl.so` builds, resolved at app link).


----


Test plan
---------


This section is the authoritative test plan for GIN Anvil SDMA (`NCCL_GIN_TYPE=5`). Docker harness details and Test#5 wiring live in `gin-anvil-sdma-backend-tests.md`. GTest unit suites A–H + G are documented in `gin-anvil-sdma-unit-test-plan.md`. MI355 orchestration: `gin-anvil-smci355-test.bash`.


1. Objectives
~~~~~~~~~~~~~


Verify that the Anvil SDMA GIN backend:

1. Initialises on supported single-node, multi-GPU xGMI topologies without `rocshmem_init()`.
2. Moves data correctly via both datapaths: IPC flat stores (≤ threshold) and Anvil SDMA (> threshold).
3. Orders puts, signals, counters, and flush/quiet consistently with collective kernels (`GinAlltoAllKernel`).
4. Performs at or above the Test#1 host baseline on intra-node workloads (optional comparison vs Test#4 GDA).
5. Coexists with rocSHMEM in-process (factory in `librocshmem`, no duplicate `AnvilLib` in `librccl.so`).


2. Scope
~~~~~~~~


=================================================================  ==========================================================
In scope                                                           Out of scope (this revision)
=================================================================  ==========================================================
Single-node, 1 rank per GPU, HIP ordinal == rank (default layout)  Multi-node / IB Anvil
`alltoall_perf -D 3` (`GinAlltoAllKernel`)                         Arbitrary `SignalAdd` on fused SDMA path (OSS7 fixed +1 only)
Message sizes 128 B – 128 MiB (perf sweep)                         Full RCCL collective suite beyond AlltoAll
MI300 / MI355 class (gfx942 / gfx950)                              Non-AMD GPUs
Env tuning: channels, threshold, fused signal                      `anvil.disconnect()` on destroy (queues persist by design)
=================================================================  ==========================================================


3. Test environment matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~


==  ==================  ====  =============================================================================  ============================================
ID  Hardware            GPUs  Image / build                                                                  Notes
==  ==================  ====  =============================================================================  ============================================
E1  MI355X single node  8     `docker-gin-gda-sdma-build.bash`, `GPU_TARGETS=gfx950`, `ROCSHMEM_USE_SDMA=1`  Primary perf / regression (e.g. `smci355-*`)
E2  MI300X single node  8     Same, `GPU_TARGETS=gfx942`                                                     xGMI Anvil validation
E3  MI355X, bare metal  8     RCCL + rocSHMEM installed from same tree                                       Optional; bypass Docker
E4  Any supported node  1–8   Same image                                                                     Scale-down (`NP=1`, `NP=2`, …)
==  ==================  ====  =============================================================================  ============================================


Hard prerequisites (all environments):

- rocSHMEM built with `USE_SDMA=ON` (`ROCSHMEM_USE_SDMA=1` in Docker build).
- RCCL built with `ENABLE_ROCSHMEM_GIN=ON`.
- Image `libmlx5` must export `mlx5dv_reg_dmabuf_mr` for RCCL NET init (Test#5 preflight). Install `extra-rdma-debs/*.deb` at image build or set `TEST5_HOST_MLX5_LIB_DIR` at run time (see [`extra-rdma-debs/README.md`](../extra-rdma-debs/README.md)).
- `HSA_FORCE_FINE_GRAIN_PCIE=1` for GIN runs (set by test scripts).
- `ROCSHMEM_SDMA_ENABLED=0` for Test#5 (Anvil uses the standalone factory queues directly; do not enable rocSHMEM runtime SDMA policy alongside Test#5).


4. Build and install verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


=======  ===================================================================================  =====================================================================================================  =======================================
Test ID  Step                                                                                 Pass criteria
=======  ===================================================================================  =====================================================================================================  =======================================
B1       `source docker-gin-gda-sdma-preflight.bash` (manual before MI355 build; Ruby build sources it)  Required GIN shared sources; no removed plugin files; single `markSdmaDirty` in `gin_anvil_sdma.h`
B2       `RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=1 ./docker-gin-gda-sdma-build.bash 1`        Image build completes; log contains `OK: MLX5 DMA-BUF/sysfs symbols found`
B3       `docker run --rm $IMAGE rocshmem/bin/rocshmem_info`                                  Reports SDMA / Anvil enabled when `USE_SDMA=ON`
B4       `objdump -T $(readlink -f /usr/lib/x86_64-linux-gnu/libmlx5.so.1) \                  grep mlx5dv_reg_dmabuf_mr` inside image                                                                Symbol present (Test#5 preflight green)
B5       `grep -r gin_host_rocshmem_common projects/rccl/src/CMakeLists.txt` + files on disk  CMake hipify list matches filesystem (avoids `ddai-gin-build.log` CMake failure)
=======  ===================================================================================  =====================================================================================================  =======================================


4.1 Unit tests (GTest suites A–H + G)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Automated unit tests complement the integration matrix below. They run on a single GPU (no MPI), use mocks/stubs where noted, and target ~96% weighted line / ~95% weighted branch across Anvil SDMA sources. Full test inventory: `gin-anvil-sdma-unit-test-plan.md`. **MI355:** 49/49 default tests pass via `./gin-anvil-smci355-test.bash unit` (30 fixtures + 19 plugin; suite F opt-in +12).


=======  ======  ============================  =====================================================  ==============================
Suite    Tests   Binary / target               CMake flags                                            Source focus
=======  ======  ============================  =====================================================  ==============================
A        9       `rccl-UnitTestsFixtures`      `ENABLE_ROCSHMEM_GIN=ON`                               `gin_anvil_ipc_table_host.cc`
B–E      9       `rccl-UnitTestsFixtures`      `+ GIN_ANVIL_UNIT_TESTS=ON` (or `ENABLE_ROCSHMEM=ON`)   Device IPC, detail, template IPC
H        12      `rccl-UnitTestsFixtures`      same as B–E                                            SDMA templates (memcpy stubs)
G        19      `rccl-UnitTestsGinAnvilPlugin`  `ENABLE_ROCSHMEM_GIN=ON`, ROCm 6.4+                  `gin_plugin_anvil_sdma.cc`
F        12      `rocshmem_unit_tests`         `USE_SDMA=ON`, `BUILD_TESTS=ON` (opt-in)               `gin_anvil_sdma_factory.cpp`
=======  ======  ============================  =====================================================  ==============================


.. code-block:: bash

   ./gin-anvil-smci355-test.bash unit

   # Or manual:
   ./gin-anvil-bm/build/rccl-unit/test/rccl-UnitTestsFixtures --gtest_filter='GinAnvil*'
   ./gin-anvil-bm/build/rccl-unit/test/rccl-UnitTestsGinAnvilPlugin --gtest_filter='GinAnvilPluginTest.*'


Unit tests do not replace C1–C2 (multi-GPU `alltoall_perf`), D6 (OSS7 fused signal on hardware), or P1–P5 (performance regression). See `gin-anvil-sdma-unit-test-plan.md` for integration ↔ unit ID mapping.


5. Host / plugin lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~


Run with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` and `NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=5`.


=======  ====================================================================  =======================================================================================================
Test ID  Action                                                                Pass criteria
=======  ====================================================================  =======================================================================================================
H1       `gin_anvil_sdma_probe()` via tiny host binary or first RCCL init  Returns 1 on SDMA-capable HIP node; 0 when `USE_SDMA=OFF` stub build
H2       RCCL comm init, 8 ranks                                               Log: `GIN anvil-sdma: standalone SDMA queues (8 ranks, N ch, spread=…)`
H3       Wrong `NCCL_GIN_TYPE` (e.g. 4) with Anvil plugin forced               Anvil `init()` returns error; no silent fallback
H4       `connect()` rank ↔ GPU mapping                                        `bootstrapAllGather` of HIP ordinals; queue table `peer * numChannels + ch`
H5       `regMrSym()` on symmetric LSA window                                  IPC table entry via `ncclGinAnvilIpcTableRegisterVmm`
H6       `createContext()` + resource window bind                              Device context `layoutMagic == NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC`; pending cleared after bind
H7       Comm destroy                                                          No crash; `gin_anvil_sdma_destroy` frees HIP-owned state (queues remain — by design)
H8       Mid-comm `regMrSym()`                                                IPC table refresh updates all live GPU contexts
=======  ====================================================================  =======================================================================================================


6. Device datapath (functional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Use `alltoall_perf -D 3 -V 1` (validation on). Sweep `-b` / `-e` / `-f 2`.


=======  =======================================================  ==============================  ===================================================
Test ID  Configuration                                            Exercises                       Pass criteria
=======  =======================================================  ==============================  ===================================================
D1       Default threshold (128 B)                                IPC path for tiny messages      `#wrong == 0` at 128 B – 128 B
D2       `-b 256 -e 4K`                                           IPC vs SDMA boundary            `#wrong == 0`; latency knee near threshold on MI355
D3       `-b 4K -e 128M`                                          Anvil SDMA bulk path            `#wrong == 0` all sizes
D4       `NCCL_GIN_ANVIL_SDMA_THRESHOLD=0`                                      Force SDMA (isolates SDMA path)         Correctness; fault at 128 B → signals or SDMA peer VA
D5       `NCCL_GIN_ANVIL_SDMA_THRESHOLD=65536`                                  Force IPC for medium msgs               Correctness maintained
D6       `NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL=1`                                   OSS7 copy+signal SDMA packet            Correctness on MI355; compare vs default `0`
D7       `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1,2,4,8`                            Multi-queue per peer                    Init succeeds; `#wrong == 0` at 1 MiB
D8       `Flush` after puts (`GinAlltoAllKernel` implicit quiet)              Dirty bitmask + `gin_anvil::sdma::quiet`  No hang; validation pass
D11      Default threshold, 128 B only                                        IPC data + LSA signal atomics           `#wrong == 0`; no GPU fault on signal VA
=======  =======================================================  ==============================  ===================================================


Signal / counter spot checks (if exposed via future micro-test or debugger):

- D9 Fused SDMA `SignalInc` only; IPC signal path supports arbitrary add.
- D10 Counter increment visible after fence/quiet path.


7. Collective integration (primary harness)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Default automation: `docker-gin-gda-sdma-test.bash` / `docker-gin-gda-sdma-ruby-test.bash`. Default `RCCL_GIN_RUN_TESTS=1,5` (Test#2 and Test#4 opt-in). All tests use `-V 1`. See `gin-anvil-sdma-backend-tests.md`.


=======  ==================  ====================================================  =====================================================
Test ID  Harness slot        Env summary                                           Pass criteria
=======  ==================  ====================================================  =====================================================
C1       Test#1              `NCCL_GIN_ENABLE=0`, `-D 0`                           Baseline; `#wrong == 0`; records busbw for comparison
C2       Test#5              `NCCL_GIN_TYPE=5`, `-D 3`, `ROCSHMEM_SDMA_ENABLED=0` (harness default `NCCL_DEBUG=VERSION`)  `#wrong == 0` full sweep; `ginType != NONE` in init logs
C3       Test#5, `NP=1`      Same as C2                                            Single-GPU smoke (may skip peer traffic)
C4       Test#5, `NP=2,4,8`  Same as C2                                            Correctness at each scale
C5       Optional Test#4     `NCCL_GIN_TYPE=4`, bnxt + firmware gate               GDA reference on supported NICs only
=======  ==================  ====================================================  =====================================================


Example (Test#5, full matrix):


.. code-block:: bash

   RCCL_GIN_RUN_TESTS=5 \
     ./docker-gin-gda-sdma-test.bash 8 128M 2>&1 | tee ddai-gin-perf.log


Isolation runs (debug IPC vs SDMA vs signals):


.. code-block:: bash

   NCCL_GIN_ANVIL_SDMA_THRESHOLD=0 RCCL_GIN_RUN_TESTS=5 ./docker-gin-gda-sdma-test.bash 8
   NCCL_GIN_ANVIL_SDMA_THRESHOLD=65536 RCCL_GIN_RUN_TESTS=5 ./docker-gin-gda-sdma-test.bash 8


On Ruby nodes use `docker-gin-gda-sdma-ruby-test.bash` (same env vars).

Default harness: **`TEST5_MLX5_PREFLIGHT=0`**. Set `TEST5_MLX5_PREFLIGHT=1` to skip when image `libmlx5` lacks DMA-BUF symbols.


8. Performance regression
~~~~~~~~~~~~~~~~~~~~~~~~~


=======  ======================  ========================================  ================================================================
Test ID  Metric                  Method                                    Pass / track
=======  ======================  ========================================  ================================================================
P1       AlltoAll busbw @ 128M   C1 vs C2                                  Test#5 ≥ Test#1 on xGMI (intra-node)
P2       Small-message latency   C2 with `-b 128 -e 4K -f 2`               IPC path competitive with Test#1 at small sizes on MI355
P3       Medium-message plateau  C2 with `-b 4K -e 64K`                    Anvil SDMA ~24.5 µs/msg region (MI355 tuning note in header)
P4       Large-message bw        C2 with `-e 128M`                         MI355X 8× reference ~82 GB/s busbw, `#wrong == 0`
P5       Channels scaling        `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1,4,8`  Document bw/latency; no correctness loss
=======  ======================  ========================================  ================================================================


Record: hostname, GPU model, ROCm version, RCCL commit, `NCCL_GIN_ANVIL_SDMA_*` env, log file name.


9. Negative and robustness
~~~~~~~~~~~~~~~~~~~~~~~~~~


=======  ===============================================  ================================================================
Test ID  Scenario                                         Expected
=======  ===============================================  ================================================================
N1       `NCCL_GIN_TYPE=5` without `USE_SDMA` rocSHMEM    Probe/create fails; clear init error
N2       Mismatched rank/GPU layout (ordinal ≠ rank)      Document failure or wrong results; file bug if silent corruption
N3       Missing IPC table entry / bad LSA registration       Init or run fails with WARN, not segfault
N4       Invalid `layoutMagic` on device                  Kernels no-op (`anvilCtxValid` false); no GPU fault
N5       Concurrent rocSHMEM `rocshmem_init` + Anvil GIN  No duplicate `AnvilLib` crash (coexistence)
N6       `TEST5_MLX5_PREFLIGHT=1` without MLX5 symbols    Test skipped; set `TEST5_MLX5_PREFLIGHT=0` to force run
N7       Signals not in LSA resource window               GPU memory fault at 128 B; fix signal bind + IPC table
=======  ===============================================  ================================================================


10. Comparison matrix (sanity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Run same size sweep for each backend on one fixed environment (E1):


==============  ===============  ====  =====================================
Backend         `NCCL_GIN_TYPE`  `-D`  Role
==============  ===============  ====  =====================================
Host baseline   0 (GIN off)      0     CPU / CE reference
GIN Ib proxy    2                3     Host-progress GIN (needs verbs + GDR)
GIN GDA         4                3     NIC QP path (Test#4; bnxt gate)
GIN Anvil SDMA  5                3     Primary subject
==============  ===============  ====  =====================================


Anvil should lead intra-node xGMI for mid/large messages vs Test#1; vs GDA depends on NIC and message size.


11. CI / release checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~


Before merge or image publish:

- B1–B5 pass on builder
- Unit suites A–H pass on at least one GPU node (see §4.1)
- C1 + C2 pass on at least one MI300/MI355 8-GPU node
- D1–D3 validation clean on Test#5
- P1 no regression vs last green perf log
- Docs: `gin-anvil-sdma-backend-design.md` / `.rst` + `gin-anvil-sdma-backend-tests.md` + `gin-anvil-sdma-unit-test-plan.md` updated


12. Debugging playbook
~~~~~~~~~~~~~~~~~~~~~~


==================================  ===================================================================
Symptom                             Check
==================================  ===================================================================
Test#5 skipped at start             `TEST5_MLX5_PREFLIGHT=1` and image lacks MLX5 symbols
`ginType NONE` / `-D 3` error       `NCCL_GIN_ENABLE=1`, `NCCL_GIN_TYPE=5`, `NCCL_DEBUG=INFO` NET lines
GPU memory fault at 128 B           Signal VA not peer-mapped — LSA resource-window bind + IPC table
Hang in AlltoAll                    SDMA dirty / quiet; try `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1`
Wrong results small sizes           `NCCL_GIN_ANVIL_SDMA_THRESHOLD`; isolation with `THRESHOLD=0`
Wrong results large sizes           IPC table + LSA flat base; `THRESHOLD=65536` to force IPC
`gin_anvil_sdma_create failed`      `USE_SDMA`, HIP devices visible, xGMI peer access
==================================  ===================================================================


13. Key environment variables (test tuning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


==================================  =======  ====================================
Variable                            Default  Test use
==================================  =======  ====================================
`NCCL_GIN_ENABLE`                   off      Must be `1`
`NCCL_GIN_TYPE`                     —        Must be `5`
`NCCL_DEBUG`                        `VERSION` (MPI_BASE); override to `INFO`  Init / NET diagnostics
`NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS`  `1`      D7, P5
`NCCL_GIN_ANVIL_SDMA_THRESHOLD`     `128`    D4, D5, D11; isolation runs
`NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL`  `0`      D6
`NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS`  on (factory)  Channel striping
`ROCSHMEM_SDMA_ENABLED`             —        Must be `0` for Test#5
`HSA_FORCE_FINE_GRAIN_PCIE`         —        `1` (scripts)
`RCCL_GIN_RUN_TESTS`                `1,5`    Harness test selection
`TEST5_MODE`                        `run`    Set `skip` to skip Test#5
`TEST5_MLX5_PREFLIGHT`              `0`      Set `1` to skip without MLX5 DMA-BUF symbols
`TEST5_HOST_MLX5_LIB_DIR`           unset    Bind-mount newer host `libmlx5*`
`TEST5_NUM_CHANNELS`                `1`      Maps to `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS`
==================================  =======  ====================================


----


File: `docs/gin-anvil-sdma-backend-design.md` / `docs/gin-anvil-sdma-backend-design.rst` — GIN Anvil SDMA backend design and test plan.
