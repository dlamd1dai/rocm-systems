# GIN-SDMA AllToAll fabric-heap: concrete implementation design

Turns `gin-sdma-fabric-a2a-workplan.md` into file-level, paste-ready changes.
Base: `origin/develop` @ `67d45e7ccfe`; working branch
`users/dondai/gin-sdma-a2a-meta-455-wip`.

**Status:** authored from source reading; **not compiled** (no HIP toolchain /
MI455 here). The load-bearing runtime assumptions are gated by the Phase 0 spike
(`ddai-artifacts/spikes/phase0-fabric-sdma`). Prepared source that is
intentionally **not wired into the build** lives in
`ddai-artifacts/staged-gin-fabric/` so it cannot break the SUT build before
review.

---

## 0. The fork, resolved by reading `dev_runtime.cc`

The GIN-SDMA symmetric memory is the **LSA (intra-node) heap**, and its segment
handle exchange is **intra-node only**:

- Rank list is node-local: `devr->lsaRankList[i] = comm->localRankToRank[i]`
  (`dev_runtime.cc:224`).
- Handle exchange + barrier are intra-node:
  `bootstrapIntraNodeAllGather(...lsaRankList, lsaSelf, lsaSize...)`
  (`dev_runtime.cc:444`), `bootstrapIntraNodeBarrier` (`:460`).
- The flat VA is reserved once for the **local** team:
  `cuMemAddressReserve(&addr, devr->lsaSize * devr->bigSize, ...)`
  (`dev_runtime.cc:450`); peer `r` maps at `base + r*bigSize + bigOffset`
  (`:406-407`).
- **The fabric branch already exists**, keyed on `ncclCuMemHandleType`:

```362:388:projects/rccl/src/dev_runtime.cc
  if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    msg->memHandle = memHandle;                       // export: POSIX fd (default)
  } else {
    CUCHECKGOTO(cuMemExportToShareableHandle(&msg->fabricHandle, memHandle, ncclCuMemHandleType, 0), ret, fail);
  }
  ...
    if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      ... ncclProxyClientGetFdBlocking ... cuMemImportFromShareableHandle(POSIX fd) ...
    } else {
      CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle, (void*)&msg->fabricHandle, ncclCuMemHandleType), ret, fail);
    }
```

**Conclusion — two substrates, one device path:**

| Topology | Peer set | Substrate mechanism | GIN plugin change |
|---|---|---|---|
| **Single-node MI455 / same pod** | node-local LSA ranks | **fabric-backed LSA heap** — make the segment export/import use `CU_MEM_HANDLE_TYPE_FABRIC` on gfx1250; the existing stride math is unchanged | ~none (stride path works) |
| **Multi-node MNNVL clique** | full comm clique | **per-buffer cross-node fabric exchange** → `remote_vas[]`, `vmmStride=0` (LSA heap does not span nodes) | new fabric branch in `ginAnvilRegMrSym` |

The device `Put` already selects transport per peer with no change (see §3).

---

## 1. Phase 1 — Gating predicate

New helper (staged: `staged-gin-fabric/gin_anvil_sdma_fabric.h`):

```cpp
// True when this comm should use the fabric heap for GIN-SDMA on gfx1250.
static inline bool ncclGinAnvilUseFabricPath(struct ncclComm* comm) {
#ifdef HIP_FABRIC_API
  return comm != nullptr && comm->MNNVL == 1 &&
         IsArchMatch(comm->archName, "gfx1250") && ncclCuMemEnable();
#else
  (void)comm; return false;
#endif
}
```

Mirrors `ncclDdaUseFabricPath` (`fabric_init.cu:29-34`). `comm->MNNVL`,
`comm->archName`, `ncclCuMemEnable()` all already exist and are used by the DDA
fabric path.

Coexistence: gfx942/gfx950 keep the xGMI stride path untouched; all fabric code
sits behind `#ifdef HIP_FABRIC_API` (see the `HIP_FABRIC_API` build gate,
`projects/rccl/CMakeLists.txt:346-353`, `src/CMakeLists.txt:903-916`).

---

## 2. Phase 2 — Fabric memory substrate

### 2a. Single-node: fabric-backed LSA heap

The only change is the **handle type** used when the LSA heap exports/imports
its segments. Today `ncclCuMemHandleType` defaults to
`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` (`misc/cudawrap.cc:19`).

Options (pick one during Phase 7 bring-up):
- **Global switch (simplest):** run with the env/config that sets
  `ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_FABRIC` when
  `hipDeviceAttributeHandleTypeFabricSupported` is true. The allocator already
  falls back to POSIX-fd when fabric is unsupported (`allocator.cc`).
- **Per-comm handle type (cleaner, more work):** thread a per-comm handle type
  through `symMemoryExportSegmentHandle`/`symMemoryImportAndMapSegmentHandle`
  and `memProp.requestedHandleType` (`dev_runtime.cc:190,1803`) so only gfx1250
  comms use fabric while others stay POSIX-fd.

Requirements to verify on SUT (Phase 0 covers the first two):
1. `hipDeviceAttributeHandleTypeFabricSupported == 1` on the device.
2. Fabric export/import + map succeeds (spike `probe` + `singlenode`).
3. SDMA-put into the fabric-mapped LSA segment works single-node (spike
   `singlenode` copy-engine test; then the real Anvil queue).

If (3) holds, **nothing in the GIN plugin changes for single-node** — the stride
resolver (`resolveRemotePeerVa` tier 1) already targets the fabric-mapped VAs.

### 2b. Multi-node: per-buffer cross-node fabric exchange

The LSA heap cannot reach cross-node peers, so `ginAnvilRegMrSym` needs a fabric
branch that builds an explicit peer-VA table for the whole clique. Reuse
`ncclFabricMemHandler` (`fabric_mem_handler.{h,cc}`), whose `exchangeMemPtrs`
uses a **full-comm** `bootstrapAllGather` (not intra-node) — exactly what a
cross-node clique needs.

**New hook required (core):** the plugin must obtain the registered buffer's
cuMem allocation handle to export it. Today the handle is private to the
symmetric heap (`mem->memHandles[segment]`, `dev_runtime.cc:439`). Add:

```cpp
// dev_runtime.{h,cc}: return the CUmem allocation handle backing a registered
// symmetric-heap buffer (single-segment fast path; multi-segment => error for now).
ncclResult_t ncclDevrGetMemHandle(struct ncclDevrState* devr, void* data,
                                  CUmemGenericAllocationHandle* outHandle, size_t* outSize);
```

**Hybrid registration** in `ginAnvilRegMrSym` (staged patch in
`staged-gin-fabric/regmrsym.fabric.patch.md`):

```
if (ncclGinAnvilUseFabricPath(cctx->comm)) {
    // 1. export self buffer's cuMem handle (ncclDevrGetMemHandle)
    // 2. GinAnvilFabricPeers::exchange() -> per-peer imported VAs (clique-wide allgather)
    // 3. remote_vas_host[pe] = intra-node ? LSA-stride VA : fabric-imported VA
    // 4. hostMh.vmmStride = 0;  hostMh.remote_vas = remote_vas_dev;
    // 5. ncclGinAnvilIpcTableRegisterExplicit(lsaSelfAddr, remote_vas_host, nranks, size);
} else {
    ... existing stride path (unchanged) ...
}
```

Setting `vmmStride = 0` forces the device tier-2 resolver
(`gin_anvil_sdma.h:49-53`) to use `remote_vas[]`. Intra-node peers may keep their
LSA-stride VA (SDMA still works); cross-node peers use the fabric-imported VA
(shader-core path, §3).

Teardown: extend `ginAnvilDeregMrSym` (`gin_plugin_anvil_sdma.cc:368-388`) to
release the per-buffer `GinAnvilFabricPeers` mappings (already frees
`remote_vas_dev` + `devHandle`).

---

## 3. Phase 3 — Per-peer transport (already implicit; confirm + perf follow-up)

The device `Put` already selects transport per peer: it asks for an SDMA queue
handle and, when there is none, falls back to the inline `ipcPut` (memcpy):

```200:229:projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma.h
    bool useIpcPut = hasWins && bytes <= threshold;
    ::sdma_anvil::SdmaQueueDeviceHandle* handle = nullptr;
    if (hasWins && !useIpcPut) {
      handle = queueHandle(rsCtx, peer, blockId);
      if (handle == nullptr) useIpcPut = true;   // no queue (e.g. cross-node) -> shader-core
    }
    ...
      if (useIpcPut) {
        void* dstAddr = resolveRemotePeerVa(rsCtx, dstMh, peer, dstOff);  // fabric VA via remote_vas
        if (dstAddr && srcAddr) ipcPut(dstAddr, srcAddr, bytes);          // shader-core memcpy
      }
```

So once `remote_vas[]` holds fabric-imported VAs (§2b), cross-node peers (no
SDMA queue) automatically use shader-core writes into the fabric memory, and
intra-node peers keep SDMA. **No device change is required for correctness.**

Ensure cross-node peers get **no** SDMA queue: the SDMA factory only connects
node-local devices (`gin_anvil_sdma_factory.cpp` uses `EnablePeerAccess(myDev,
remoteDev)` on HIP device ids), so `queueHandle(...)` already returns `nullptr`
for cross-node peers — confirm `queueHandles[peer*nCh+ch]` is `nullptr` there.

**Perf follow-up (Phase 7, not correctness):** `Put` runs on `thread_rank()==0`
only (`gin_anvil_sdma.h:189`), so the `ipcPut` memcpy is single-threaded. For
large cross-node messages consider a cooperative (whole-Coop) uint4 copy variant
when `handle == nullptr && bytes > threshold`.

---

## 4. Phase 4 — Signals + system-scope fence

1. **Signals over fabric:** on the multi-node path populate
   `signal_remote_addrs[]` / the explicit IPC table from the same fabric
   exchange, so `signalPeer` / `remoteSignalAddr` (`gin_anvil_sdma.h:60-82`)
   resolve to fabric-imported signal VAs. gfx1250 has no OSS7 fused signal
   (`SDMA_IS_OSS7` is `__gfx950__`-only) so the shader atomic `signalPeer` path
   is used — correct.

2. **Fence:** this branch still has the **pre-PR#3** guard, which is dead on HIP:

```195:197:projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma.h
    if ((required == cuda::thread_scope_system) && (given > required)) {
      __threadfence_system();
    }
```

The cross-node shader-core path needs a real system-scope release before the
signal is observed. **Action:** bring the PR#3 fix here (or merge that branch):
change to `(given < required)` on HIP and route through
`NCCL_GIN_THREADFENCE_SYSTEM()`. Also note the existing
`__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` (`:236`) before SDMA is
**agent-scope** — fine for same-die SDMA, but the cross-node shader path must use
**system** scope (the guard above) so writes are visible across the fabric.

---

## 5. Phase 5 — Host dispatch / eligibility

Extend `rcclUseAlltoAllGda` (`rccl_wrap.cc:510-519`) to admit the gfx1250 clique
shape when `ncclGinAnvilUseFabricPath(comm)` is true: allow single-node pod and
cross-node clique (`2 <= nRanks <= max`), dtype {fp32,fp16,bf16}, 16B-aligned —
matching `ncclAllToAllDdaFabricEligible`. Leave the multi-node + 8-GPU/node
requirement for the non-fabric rocSHMEM path unchanged.

---

## 6. Phase 6 — Tests

The GDA device-template test (`GinRocshmemGdaTemplate_test.cpp`) is **not on this
branch** (it lives on `users/dondai/gin-gda-a2a-unittests`). When integrating,
add there a fabric-mode case: populate `remote_vas` from a mock exchange with
`vmmStride=0`, force `queueHandle==nullptr` for a peer to exercise the
shader-core `ipcPut`, and assert one system-scope fence via the
`NCCL_GIN_THREADFENCE_SYSTEM()` counter seam (PR#3). Keep the CMake gate
`ENABLE_ROCSHMEM_GIN AND (ENABLE_ROCSHMEM OR GIN_ANVIL_UNIT_TESTS)` (PR#2).

---

## 7. Integration order (on the SUT)

1. Run Phase 0 spike → record go/no-go (fabric supported? SDMA-over-fabric
   single-node? shader-core cross-node?).
2. Wire Phase 1 predicate.
3. Single-node substrate (§2a) — validate stride path over fabric first.
4. Fence fix (§4.2).
5. Multi-node hook + `ncclDevrGetMemHandle` + hybrid `ginAnvilRegMrSym` (§2b).
6. Eligibility (§5) + signals (§4.1).
7. Tests (§6); bring-up single-node then multi-node.

## 8. New/changed files summary

| File | Change | Risk |
|---|---|---|
| `gin/gin_plugin_anvil_sdma.cc` | fabric branch in `ginAnvilRegMrSym` + dereg | med |
| `dev_runtime.{h,cc}` | `ncclDevrGetMemHandle`; (opt) per-comm handle type | med-high (core) |
| `include/nccl_device/gin/anvil_sdma/gin_anvil_sdma.h` | fence guard fix (PR#3) | low |
| `rccl_wrap.cc` | fabric eligibility in `rcclUseAlltoAllGda` | low |
| new `gin/gin_anvil_sdma_fabric.{h,cc}` | gating + `GinAnvilFabricPeers` | low (new, guarded) |
| `misc/cudawrap.cc` / `allocator.cc` | fabric handle-type selection on gfx1250 | med |

Staged prototypes for the two low-risk new pieces are in
`ddai-artifacts/staged-gin-fabric/`.
