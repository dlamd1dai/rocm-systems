# Phase 0 spike: SDMA / copy-engine + shader-core over the fabric heap

De-risking spike for the GIN-SDMA AllToAll → fabric-heap adaptation
(`ddai-artifacts/docs/gin-sdma-fabric-a2a-workplan.md`, Phase 0). It validates on
real MI455 hardware the load-bearing assumption that lets us keep the SDMA
transport on single-node and fall back to shader-core cross-node.

**Status: authored but NOT compiled/run** — this workstation has no HIP
toolchain and no MI455. Build and run on the MI455 SUT (ROCm 7.14+).

## What each program answers

| Program | Work-plan item | Question |
|---|---|---|
| `probe_fabric_caps` | Phase 0 (1) | Is `HIP_FABRIC_API` present? Per-device fabric support + local peer-access matrix. |
| `sdma_over_fabric_singlenode` | Phase 0 (2) | Single-node: do copy-engine (`hipMemcpyAsync`) AND shader-core writes into a **fabric-imported peer VA** land correctly? |
| `multinode_fabric_mpi` | Phase 0 (3) | Cross-node: does shader-core store into a fabric-imported **remote-node** peer VA land (with `__threadfence_system`)? Is the copy engine viable cross-node (`--try-ce`)? |

`hipMemcpyAsync(D2D)` drives the same DMA copy-engine hardware the Anvil SDMA
queue uses; a PASS here is strong evidence the production Anvil-SDMA-put into the
imported VA will also work. A follow-up can wire the real
`projects/rocshmem/src/sdma` Anvil queue for an exact match.

## Files

- `fabric_vmm.hpp` — shared helper: fabric VMM alloc / export / import / map,
  mirroring `projects/rocshmem/src/memory/hip_allocator_vmm_{common,fabric}` on
  `origin/develop`.
- `probe_fabric_caps.cpp`, `sdma_over_fabric_singlenode.cpp`,
  `multinode_fabric_mpi.cpp`.

## Build

CMake (recommended):

```bash
cd ddai-artifacts/spikes/phase0-fabric-sdma
cmake -S . -B build -DCMAKE_CXX_COMPILER=hipcc \
      -DCMAKE_CXX_FLAGS="--offload-arch=gfx1250" \
      -DSPIKE_ENABLE_MPI=ON        # omit if you only want the single-node tests
cmake --build build -j
```

Or directly with hipcc (single-node only):

```bash
hipcc --offload-arch=gfx1250 -std=c++17 -DSPIKE_HAS_FABRIC \
      probe_fabric_caps.cpp -o probe_fabric_caps
hipcc --offload-arch=gfx1250 -std=c++17 -DSPIKE_HAS_FABRIC \
      sdma_over_fabric_singlenode.cpp -o sdma_over_fabric_singlenode
```

If the toolchain lacks the fabric API, drop `-DSPIKE_HAS_FABRIC`; the programs
then compile and report a clean "skip" (exit 77).

## Run

```bash
# 1) capabilities
./probe_fabric_caps

# 2) single-node (all GPU pairs on the node); optional element count (uint32)
./sdma_over_fabric_singlenode            # default 4 MiB
./sdma_over_fabric_singlenode 4194304    # 16 MiB

# 3) multi-node ring across >= 2 nodes, one rank per GPU
srun -N2 --ntasks-per-node=<gpus> ./build/multinode_fabric_mpi
#   add --try-ce to also probe the (expected non-viable) cross-node copy engine
srun -N2 --ntasks-per-node=<gpus> ./build/multinode_fabric_mpi --try-ce
```

## Interpreting results (GO/NO-GO)

- **`probe`**: `fabricSupported=1` on gfx1250 ⇒ proceed. `0` (or
  `SPIKE_HAS_FABRIC=0`) ⇒ the production fabric path must stay behind `#ifdef
  HIP_FABRIC_API` with a fallback (work-plan Phase 1/6).
- **single-node**: copy-engine PASS on all pairs ⇒ **keep the SDMA transport for
  single-node** (work-plan Phase 3). If only shader-core PASSes ⇒ single-node
  also uses shader-core and Phases 3/9 simplify.
- **multi-node**: shader-core PASS ⇒ multi-node transport confirmed (Phase 3
  fallback). If `--try-ce` reports the copy engine is *not viable*, that
  confirms evidence E4 (SDMA is node-local; cross-node needs shader-core).

Record the outcomes back into the work plan's Phase-0 go/no-go note.
