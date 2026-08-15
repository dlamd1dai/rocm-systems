# Staged GIN-SDMA fabric-heap source (NOT in the build)

Ready-to-integrate prototypes prepared in advance of MI455 SUT validation. These
files are **deliberately not referenced by any CMake target**, so they cannot
break the RCCL build before review. **None of this has been compiled** (no HIP
toolchain / MI455 on the authoring host).

Read `ddai-artifacts/docs/gin-sdma-fabric-a2a-impl-design.md` first — it resolves
the single-node vs multi-node substrate fork and lists every touch point.

## Files and their destinations

| Staged file | Destination on integration | Purpose |
|---|---|---|
| `gin_anvil_sdma_fabric.h` | `projects/rccl/src/include/gin/gin_anvil_sdma_fabric.h` | Phase 1 gating predicate + `GinAnvilFabricPeers` (cross-node fabric peer-VA exchange). |
| `gin_anvil_sdma_fabric.cc` | `projects/rccl/src/gin/gin_anvil_sdma_fabric.cc` (add to `src/CMakeLists.txt` under `if(HIP_FABRIC_API)`) | Definition; wraps `ncclFabricMemHandler`. |
| `regmrsym.fabric.patch.md` | edit `projects/rccl/src/gin/gin_plugin_anvil_sdma.cc` | Hybrid fabric branch in `ginAnvilRegMrSym` + teardown + the new `ncclDevrGetMemHandle` core hook. |

## What is prepared vs. what still needs the SUT

Prepared (this folder):
- Gating predicate (mirrors `ncclDdaUseFabricPath`).
- Cross-node fabric peer exchange, reusing the proven `ncclFabricMemHandler`.
- Exact `ginAnvilRegMrSym` / `ginAnvilDeregMrSym` integration diff.

Still required at integration time (documented, not staged, because they are core
edits best done + compiled on the SUT):
1. `ncclDevrGetMemHandle` in `dev_runtime.{h,cc}` (expose the buffer's cuMem
   allocation handle; snippet in `regmrsym.fabric.patch.md`).
2. Single-node fabric-backed LSA heap: fabric handle-type selection on gfx1250
   (`misc/cudawrap.cc` / `allocator.cc` / per-comm handle type in `dev_runtime`).
   Design §2a.
3. Fence fix (PR#3 guard `given < required`) in `gin_anvil_sdma.h` — this branch
   still has the dead `given > required` guard. Design §4.2.
4. Eligibility in `rcclUseAlltoAllGda` (`rccl_wrap.cc`). Design §5.
5. Fabric-mode unit test on the `gin-gda-a2a-unittests` branch. Design §6.

## Why the device kernel is not staged

The device `Put` already selects transport per peer: peers without an SDMA queue
(e.g. cross-node) fall back to the inline `ipcPut` shader-core copy into the
resolved peer VA (`gin_anvil_sdma.h:200-229`). Once `remote_vas[]` holds
fabric-imported VAs, cross-node peers work with **no device change**. The only
device follow-up is a *performance* one (cooperative large-copy), noted in
design §3.

## Validation path

Follow `gin-sdma-fabric-a2a-impl-design.md` §7: run the Phase 0 spike, then wire
Phase 1 → single-node substrate → fence → multi-node hook → eligibility → tests,
compiling and testing each step on the MI455 SUT.
