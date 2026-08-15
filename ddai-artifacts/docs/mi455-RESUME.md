# MI455 GIN-SDMA fabric AllToAll — resume here

Branch: `users/dondai/gin-sdma-a2a-meta-455-wip` (all work committed + pushed).
Base: `origin/develop` @ `67d45e7ccfe`.

## Goal
Adapt GIN-SDMA AllToAll from IPC/xGMI onto the fabric heap for **both**
single-node MI455 and multi-node MNNVL clique.

## Where things stand
- **Investigations done** (all with file:line evidence, captured in the docs):
  - DDA fabric path uses shader-core for cross-fabric, copy engine only for local staging.
  - rocSHMEM AllReduce: SDMA reserved for node-local xGMI peers; fabric peers use shader-core.
  - rocSHMEM `vmm_fabric` on gfx1250 single-node **does** SDMA-put into fabric-imported VAs.
  - `dev_runtime.cc` LSA heap is intra-node only, with an existing fabric-handle branch.
- **Design decided:** single-node = fabric-backed LSA heap (stride path unchanged);
  multi-node = per-buffer cross-node fabric exchange -> `remote_vas`, `vmmStride=0`.
  Device `Put` already selects transport per peer (no device change for correctness).
- **Code prepared (uncompiled; no SUT/toolchain here):** Phase 0 spike + staged
  production source, deliberately out of the build.

## Read order tomorrow
1. `gin-sdma-fabric-a2a-workplan.md` — goal, evidence (E1-E5), phased plan.
2. `gin-sdma-fabric-a2a-impl-design.md` — file-level changes; fork resolution; integration order (§7).
3. `staged-gin-fabric/README.md` + `regmrsym.fabric.patch.md` — ready-to-integrate source + diffs.
4. `spikes/phase0-fabric-sdma/README.md` — the de-risking spike to run first on the SUT.

Supporting: `alltoall-dda-trace-develop.md`, `mi450-mi350-coexistence-develop.md`,
`fabric-heap-support-mi300-mi350-mi355.md`.

## Next actions (need MI455 SUT)
1. Run the Phase 0 spike (`probe` -> `singlenode` -> `multinode`); record go/no-go.
2. Wire Phase 1 gating; then single-node fabric LSA heap; validate stride-over-fabric.
3. Apply PR#3 fence-guard fix on this branch (still has the dead `given > required`).
4. Multi-node: add `ncclDevrGetMemHandle` hook + hybrid `ginAnvilRegMrSym`.
5. Eligibility in `rcclUseAlltoAllGda`; fabric-mode unit test (on `gin-gda-a2a-unittests`).

## Open decision left for you
Whether to apply the core production edits (dev_runtime hook, handle-type swap,
fence fix) directly into `projects/rccl/src` now (uncompiled until SUT) vs. keep
them staged. Default so far: staged, to protect the SUT build.
