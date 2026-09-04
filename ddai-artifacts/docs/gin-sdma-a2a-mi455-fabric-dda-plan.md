# GIN-SDMA A2A on MI455: Fabric DDA Adaptation Plan

Branch: `users/dondai/gin-sdma-a2a-mi455-fabric-wip`
Last updated: 2026-08-26

## Summary

Adapt the gin-sdma GIN device API AllToAll path for **MI455 (gfx1250 + MNNVL)** by replacing the
**LSA/xGMI stride + IPC flat-store** peer-memory model with the **fabric DDA VMM export/import**
model already used by host-side DDA collectives.

**Goal:** Make `gin.put` / `waitSignal` / `flush`-based AllToAll device kernels work on MI455.

**Non-goals (near term):**

- Unifying or replacing host `ncclAllToAllDdaFabric*` launchers
- Merging the devtime branch (`users/dondai/gin-sdma-a2a-devtime`)
- Multi-clique hybrid (deferred)

---

## Terminology

| Term | Meaning |
|------|---------|
| **xGMI** | Infinity Fabric GPU interconnect; current design assumes LSA memory is cache-coherent over xGMI |
| **Fabric DDA** | Direct Data Access fabric path: `ncclFabricMemHandler`, VMM shareable handles, `ddaPeerPtrsDev` |
| **GIN device API AllToAll** | Device kernel using `gin.put` + `waitSignal` + `flush` (not host `ncclAllToAll`) |
| **DDA LL lane** | Small-message fabric AllToAll using 16B LL packets, scratch staging, epoch flags (no GPU barrier) |

---

## Design Decisions (Confirmed)

| Question | Decision |
|----------|----------|
| Peer memory model | Fabric DDA (`ncclFabricMemHandler`) on gfx1250; keep LSA/xGMI path elsewhere |
| Platform gate | Auto-gated like DDA: `MNNVL=1 && gfx1250 && single clique && NCCL_CUMEM_ENABLE=1` |
| Entry point | GIN device API AllToAll only |
| Devtime | Not a prerequisite |
| Scale target | **72 ranks** (match `kDdaMaxNranks`) |
| Small messages | Proactive adaptation for MNNVL; mimic DDA LL A2A design |
| Multi-clique | **Refuse** near term; hybrid later |

---

## Current vs Target Architecture

### Today (MI355-style gin-sdma)

```
Host: ginAnvilRegMrSym → LSA flat VA + remote_vas via stride + IPC table
Device: gin.put
  ├─ bytes <= 128B: IPC flat store (__builtin_memcpy to peer LSA VA)
  └─ bytes > 128B:  Anvil SDMA put (segmented <= 128 MiB)
Signals: LSA stride + system-scope atomic add
Limits: IPC table 16 ranks × 16 buffers; channels forced to 1
```

**Key files:**

| Layer | Files |
|-------|-------|
| Device Put/Flush | `projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma.h` |
| Host plugin | `projects/rccl/src/gin/gin_plugin_anvil_sdma.cc` |
| SDMA factory | `projects/rocshmem/src/sdma/gin_anvil_sdma_factory.cpp` |
| Tests | `projects/rccl/test/gin/gin_sdma_policy_test.cpp`, `GinAnvilPlugin_test.cpp`, `GinDeviceMPITests.cpp` |

### Target (MI455 fabric path)

```
Comm init: ncclDdaFabricCommInit → ddaScratch, ddaPeerPtrsDev, ddaLLEpochDev
Small msgs: DDA LL kernel body (scratch staging, epoch flags) — NOT gin.put flat stores
Large msgs: gin.put + fabric-imported user-buffer peer VAs + SDMA (follow-on PR)
Multi-clique: skip fabric path, clear log, fall back
```

---

## Overall Phased Plan

### Phase 0 — Platform gate and eligibility (host)

Add `ginAnvilUseFabricMem(comm)` mirroring `ncclDdaUseFabricPath`:

```cpp
bool ginAnvilUseFabricMem(ncclComm* comm) {
  if (!ncclDdaUseFabricPath(comm)) return false;
  if (comm->clique.size != comm->nRanks) return false;  // refuse multi-clique
  if (!ncclCuMemEnable()) return false;
  return true;
}
```

- **gfx1250 + MNNVL + single clique + VMM:** fabric mem path
- **Everything else:** existing LSA/xGMI path unchanged
- **Multi-clique:** log + fall back

### Phase 1 — Fabric peer registration (large messages, follow-on)

Replace stride-based `ginAnvilRegMrSym` when `ginAnvilUseFabricMem` is true:

1. Export caller VMM allocation via `ncclFabricMemHandler`
2. `exchangeMemPtrs()` → `remote_vas[pe]` from fabric import (not stride math)
3. Set `vmmStride = 0`; skip IPC table for fabric path
4. Scale to 72 ranks via dynamic `remote_vas[nRanks]` (no 16-rank IPC table cap)

### Phase 2 — Fabric signal registration (follow-on)

- Allocate/export signal arena via fabric handler
- Set `signal_remote_addrs[peer]` from imported bases
- Keep `NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL=0` until MI455 validation

### Phase 3 — Large-message SDMA (follow-on)

- Reuse existing SDMA segmentation; re-measure `kGinPutSegBytes` on MI455
- Peer VA from fabric-imported `remote_vas`

### Phase 4 — Scale to 72 ranks

- Use `ddaPeerPtrsDev` pattern with runtime `nRanks`
- `ddaAllToAllFabricLL<T, 0>` fallback for non-specialized rank counts

### Phase 5 — Multi-clique hybrid (deferred)

- Intra-clique fabric puts + inter-clique network GIN
- Reference hybrid pattern from devtime `HybridAlltoAllKernel` conceptually

---

## Starter PR: 1p4g Small Messages (Mimic DDA LL A2A)

**Scope:** 1 process, 4 GPUs, small messages only. This resolves open design items Q1–Q3 by
following the existing DDA fabric LL AllToAll design rather than adapting `gin.put` flat stores.

### Why mimic DDA LL (not gin.put IPC)

DDA fabric LL AllToAll does **not** direct-write peer user buffers. It uses:

1. **Comm-level fabric scratch** (`comm->ddaScratch`) via `comm->ddaPeerPtrsDev`
2. **LL 16B packet protocol** — scatter to peer scratch slots, poll-gather from own scratch
3. **Epoch flags** (`comm->ddaLLEpochDev`) — no GPU barrier, no GIN signals
4. **2D grid** — `grid.x = nRanks`, `grid.y = blocksPerPeer`

Reference implementation:

- Kernel: `projects/rccl/src/algorithms/dda/alltoall/alltoall_dda_fabric_ll.h`
- Host launcher: `projects/rccl/src/algorithms/dda/alltoall/dda_alltoall_fabric_ll.cu`
- Eligibility: `ncclAllToAllDdaFabricLLEligible`

### Q1–Q3 resolved for starter

| Item | Answer |
|------|--------|
| **Q1 Handler sharing** | Reuse comm-level DDA fabric resources from `ncclDdaFabricCommInit`. No per-window GIN fabric handler for LL starter. |
| **Q2 Buffer source** | Stage through `comm->ddaScratch` / `ddaPeerPtrsDev`. User `sendbuff`/`recvbuff` are kernel inputs; scratch is fabric-shared like host DDA LL. |
| **Q3 Device table** | Use existing `ddaPeerPtrsDev` (4 entries for 1p4g). Design devComm fields as `nRanks`-sized; specialize `NRANKS_CT=4` now, `NRANKS_CT=0` later for 72 ranks. |

### Step 1: Expose DDA fabric fields on `ncclDevComm`

Add device-visible pointers (populated in `ncclGinDevCommSetup` when `ncclDdaUseFabricPath(comm)`):

```cpp
// ncclDevComm extension (fabric small-msg lane)
void** ginFabricPeerScratch;    // = comm->ddaPeerPtrsDev
uint32_t* ginFabricLLEpoch;     // = comm->ddaLLEpochDev
int ginFabricLLEpochLen;        // = comm->ddaLLEpochLen
bool ginFabricSmallMsgEnabled;  // gate for MI455
```

Verify at setup: `comm->clique.size == comm->nRanks`.

### Step 2: Device-side eligibility (mirror DDA LL)

Same rules as `ncclAllToAllDdaFabricLLEligible`:

| Check | Rule |
|-------|------|
| Fabric resources | peer scratch, epoch, scratch non-null |
| Ranks | `nRanks == 4` (starter); relax to `2..72` later |
| Datatype | fp32 / fp16 / bf16 |
| Alignment | `perChunkBytes % 16 == 0` |
| Staging cap | `perChunkBytes * 2 <= kDdaLLMaxBytes` (16 MiB) |
| Scratch fit | `ddaLLA2AScratchSize(nRanks) <= comm->ddaScratchBytes` |
| Total size | `nRanks * perChunkBytes <= RCCL_DDA_LL_THRESHOLD` (default 32 KiB) |

### Step 3: GIN AllToAll kernel — DDA LL branch

```cpp
if (devComm.ginFabricSmallMsgEnabled && ginAlltoAllFabricLLEligible(...)) {
  dda::common::ddaAllToAllFabricLL<T, 4>(
      reinterpret_cast<T**>(devComm.ginFabricPeerScratch),
      recvbuff, sendbuff,
      count * typeSize,
      devComm.rank, devComm.nRanks,
      devComm.ginFabricLLEpoch, devComm.ginFabricLLEpochLen);
} else {
  // existing gin.put + waitSignal + flush (large-msg path, later on MI455)
}
```

**Reuse** `alltoall_dda_fabric_ll.h` directly — do not reimplement scatter/poll/epoch logic.

**Grid launch:** DDA LL expects 2D grid `(nRanks, blocksPerPeer, 1)` with 256 threads.
For 1p4g starter: `grid=(4,1,1)`, `block=(256,1,1)`.

### Step 4: Host setup wiring

In `ncclGinDevCommSetup` (Anvil SDMA + gfx1250):

1. Assert `comm->ddaFabricMemHandler != nullptr`
2. Copy `ddaPeerPtrsDev`, `ddaLLEpochDev`, `ddaLLEpochLen` into `devComm`
3. Set `ginFabricSmallMsgEnabled = true`
4. Log: `"GIN A2A: fabric LL small-msg lane enabled (nRanks=4)"`

### Step 5: Anvil SDMA plugin (minimal for starter)

For small messages on MI455, **gin.put is not used**. Plugin changes limited to connect-time
check that fabric comm resources exist. Large-message `regMrSym` fabric rework is a follow-on PR.

### Starter file touch list

| File | Change |
|------|--------|
| `projects/rccl/src/include/nccl_device/impl/comm__types.h` | Add fabric scratch/epoch fields to `ncclDevComm` |
| `projects/rccl/src/gin/gin_host.cc` | Populate fields in `ncclGinDevCommSetup` |
| GIN AllToAll kernel site | Branch to `ddaAllToAllFabricLL<T,4>` |
| `projects/rccl/test/transport/GinDeviceMPITests.cpp` | Adjust grid for small-msg; 1p4g cases |
| `projects/rccl/src/gin/README.md` | Document LL lane, grid shape, env requirements |

**Reuse as-is:**

- `projects/rccl/src/algorithms/dda/fabric/fabric_init.cu`
- `projects/rccl/src/algorithms/dda/alltoall/dda_alltoall_fabric_ll.cu` (eligibility reference)
- `projects/rccl/src/algorithms/dda/fabric/fabric_mem_handler.cc`

---

## Test Plan

### 1p4g starter matrix

| Test | Config | Sizes (per-rank count, fp32) |
|------|--------|------------------------------|
| `GinDeviceMPITests.Alltoall_PureReference` | `mpirun -n 4`, 1 node, `NCCL_GIN_TYPE=6`, `NCCL_MNNVL_ENABLE=1`, `NCCL_CUMEM_ENABLE=1` | 1, 16, 256, 1024 |
| Sanity vs host DDA | Same comm, `ncclAllToAll` host API | Bit-identical results |
| Boundary | Total = 32 KiB (`DDA_LL_THRESHOLD`) | Pass |
| Over boundary | Total > 32 KiB | Falls back to gin.put (until large-msg PR) |
| Multi-clique negative | `clique.size < nRanks` | Fabric path skipped, no hang |

### Full integration (after starter)

| Ranks | Message sizes |
|-------|---------------|
| 2, 4, 8, 16, 32, 72 | 1B, 128B, 129B, 1MB, 128MB, 256MB, 1GB |

### Regression

- MI355/MI300: LSA/xGMI path unchanged; existing gin-sdma tests pass
- Add GIN Anvil-SDMA fabric cases to `mi455_ainic_roce.json` when ready

---

## Environment Requirements (MI455)

| Variable | Required |
|----------|----------|
| `NCCL_MNNVL_ENABLE` | `1` |
| `NCCL_CUMEM_ENABLE` | `1` |
| `NCCL_GIN_TYPE` | `6` (Anvil SDMA) |
| Single UALink clique | `comm->clique.size == comm->nRanks` |

Optional tuning (inherited from DDA):

| Variable | Default | Purpose |
|----------|---------|---------|
| `RCCL_DDA_LL_THRESHOLD` | 32 KiB | Small-message LL lane cutoff |
| `NCCL_GIN_ANVIL_SDMA_THRESHOLD` | 128 B | SDMA vs inline (large-msg path only) |

---

## Success Criteria

### Starter (1p4g)

1. GIN device API AllToAll correct at 4 ranks for small sizes (≤ 32 KiB total)
2. Results match host `ncclAllToAllDdaFabricLL` on same buffers
3. Uses comm scratch + LL protocol — no LSA/xGMI flat stores on this path
4. MI355/MI300 unchanged
5. Clear fallback when fabric resources missing

### Full adaptation

1. `GinDeviceMPITests.Alltoall_PureReference` passes at 2–72 ranks on MI455 MNNVL
2. No 16-rank IPC table limit on fabric path
3. Multi-clique graceful skip with clear log
4. Large-message SDMA path validated with fabric-imported peer VAs

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| ABI break (`LAYOUT_MAGIC`) | Bump magic if `ncclDevComm` layout changes |
| Fabric + LSA coexistence | Runtime gate; separate code paths |
| SDMA limits differ on MI455 | Re-measure before changing `kGinPutSegBytes` |
| Grid shape mismatch | Document DDA 2D grid requirement for LL branch |
| DDA fabric init skipped | Fail setup with clear message if resources null |

---

## Implementation Order

| Step | Work | Est. LOC |
|------|------|----------|
| 1 | **Starter:** devComm fabric fields + DDA LL kernel branch + 1p4g tests | ~400–600 |
| 2 | Platform gate + multi-clique refusal in gin plugin | ~80 |
| 3 | Fabric `regMrSym` / large-message SDMA | ~300 |
| 4 | Fabric signal registration | ~150 |
| 5 | Scale to 72 ranks | ~100 |
| 6 | MI455 integration matrix + test-runner configs | — |
| 7 | Multi-clique hybrid (deferred) | — |

---

## Architecture Diagram (Starter)

```
┌─────────────────────────────────────────────────────────┐
│ Host (already at comm init)                              │
│  ncclDdaFabricCommInit                                   │
│    → ddaScratch, ddaPeerPtrsDev, ddaLLEpochDev           │
│  ncclGinDevCommSetup                                     │
│    → copy pointers into ncclDevComm                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ GIN device AllToAll kernel                               │
│                                                          │
│  if (MI455 && small msg && fabric enabled)               │
│    ddaAllToAllFabricLL<T,4>(peerScratch, ...)            │
│  else                                                    │
│    gin.put loop + waitSignal + flush  [later: fabric SDMA]│
└─────────────────────────────────────────────────────────┘
```

---

## References

- GIN overview: `projects/rccl/src/gin/README.md`
- **Docker/build/test harness:** `ddai-artifacts/docs/gin-sdma-a2a-harness.md`
- DDA fabric init: `projects/rccl/src/algorithms/dda/fabric/fabric_init.cu`
- DDA LL A2A kernel: `projects/rccl/src/algorithms/dda/alltoall/alltoall_dda_fabric_ll.h`
- DDA LL128 A2A (mid-size follow-on): `projects/rccl/src/algorithms/dda/alltoall/alltoall_dda_fabric_ll128.h`
- GIN AllToAll reference test: `projects/rccl/test/transport/GinDeviceMPITests.cpp` (`Alltoall_PureReference`)
- Devtime branch (optional, not prerequisite): `users/dondai/gin-sdma-a2a-devtime`
