# Work plan: GIN-SDMA AllToAll over the fabric heap (single-node MI455 + multi-node MNNVL clique)

Snapshot base: `origin/develop` @ `67d45e7ccfe`. Working branch:
`users/dondai/gin-sdma-a2a-meta-455-wip`. Companion to
`alltoall-dda-trace-develop.md`, `mi450-mi350-coexistence-develop.md`, and
`fabric-heap-support-mi300-mi350-mi355.md`.

## Goal

Adapt the GIN-SDMA AllToAll design so it runs on the **fabric heap** (UALoE /
`hipMemHandleTypeFabric` memory) on gfx1250 / MI450 / MI455, for **both**:

- **Single-node MI455** (same UALoE pod): keep the Anvil SDMA copy engine as the
  bulk transport — this already works in rocSHMEM's `vmm_fabric` path.
- **Multi-node MNNVL clique** (cross-node UALoE): use shader-core
  gather/scatter for peers that are not SDMA-reachable, because no cross-node
  SDMA queue is ever created.

The unifying design is **one fabric-backed memory substrate for all peers, plus
per-peer transport selection** (SDMA when the peer has an SDMA queue,
shader-core otherwise) — rocSHMEM's `ipcCopy` pattern generalized to fabric
memory.

---

## Background: the evidence behind this design

### E1. GIN-SDMA today shares memory via the symmetric LSA heap + stride, not fabric handles

`ginAnvilRegMrSym` computes each peer VA by a fixed `bigSize` stride and sets
`vmmStride`, forcing the device's tier-1 (stride) peer resolution:

```332:356:projects/rccl/src/gin/gin_plugin_anvil_sdma.cc
  const ptrdiff_t stride = (ptrdiff_t)devr->bigSize;
  ...
  for (int pe = 0; pe < cctx->nranks; pe++) {
    remote_vas_host[pe] = (uintptr_t)lsaSelfAddr + static_cast<ptrdiff_t>(pe - devr->lsaSelf) * stride;
  }
  ...
  hostMh.vmmStride = stride;   // tier-1 stride path
```

The device already has a 3-tier peer resolver; tiers 2/3 are exactly what a
fabric adaptation needs:

```43:57:projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma.h
  ptrdiff_t stride = loadConst(&mh->vmmStride);
  if (stride != 0) { ... base + (peer-rank)*stride + off ... }   // tier 1: stride
  uintptr_t* remoteVas = loadConst(&mh->remote_vas);
  if (remoteVas != nullptr) { ... remoteVas[peer] + off ... }    // tier 2: explicit table
  ... return ginAnvilResolvePeerVa(sym, peer, table, count);     // tier 3: IPC table
```

The underlying symmetric heap already exports/imports with either POSIX-fd
(default) or a fabric handle, in `dev_runtime.cc`
(`symMemoryExportSegmentHandle` / `symMemoryImportAndMapSegmentHandle`), keyed
on `ncclCuMemHandleType` (default `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`,
`misc/cudawrap.cc:19`).

**Injection point:** on the fabric path, populate `remote_vas[]` from a
fabric-handle exchange and set `vmmStride = 0`; the device kernel is unchanged.

### E2. Host-initiated DDA fabric AllToAll never uses the copy engine across the fabric

Every DDA fabric collective uses the copy engine only for **local** D2D staging;
all cross-fabric movement is shader-core.

```57:60:projects/rccl/src/dda_alltoall_fabric.cu
  // Stage sendbuff into this rank's scratch before the peer exchange. A single
  // host-launched cudaMemcpyAsync avoids the per-block in-kernel copy race on
  // the fabric path.
  CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, totalCount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
```

```46:54:projects/rccl/src/include/algorithms/alltoall/alltoall_dda_fabric.h
    for (int r = 0; r < nRanksEff; ++r) {
      ...
      *reinterpret_cast<uint4*>(&recvbuff[destIdx]) = reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
```

No `cudaMemcpyPeer` / `hipMemcpyPeerAsync` anywhere in `projects/rccl/src`; the
LL/LL128 fabric variants contain zero memcpy calls. This is a **design-choice**
signal (shader-core is the sanctioned fabric transport), not a hardware proof.

### E3. rocSHMEM proves SDMA-put into fabric-imported memory works — but only node-local/pod

rocSHMEM's ring AllReduce moves the share phase with `putmem_nbi_*` →
`ipcCopy`, which selects SDMA purely by size (>= 256 B), not address type:

```411:423:projects/rocshmem/src/ipc_policy.hpp
  __device__ void ipcCopy_wg(void *dst, void *src, size_t size, int local_pe) {
    if (sdmaImpl_.sdmaEnabled && size >= sdmaImpl_.sdmaThreshold) {
      ...
      handle = sdmaImpl_.sdmaCopy<Kind>(dst, src, size, local_pe);
      ...
    }
    memcpy_wg<Kind>(dst, src, size);   // shader-core fallback
  }
```

Under the `vmm_fabric` allocator (gfx1250/MI455-only), the IPC peer set is the
fabric pod's IPC-capable ranks, and those peers' `ipc_bases[pe]` are
**fabric-imported VAs**:

```192:199:projects/rocshmem/src/ipc_policy.cpp
  bool use_pod_detection = (allocator->get_type() == AllocatorTypeVMMFabric);
  auto shm_ranks = use_pod_detection ? bootstr->getIpcCapableRanks() : bootstr->getLocalRanks();
  shm_size = shm_ranks.size();
```

SDMA connects to exactly that set (peer-access enable + one queue per peer):

```67:73:projects/rocshmem/src/sdma_policy.cpp
  for (int i = 0; i < shm_size; i++) {
    if (i != deviceId) {
      sdma_anvil::EnablePeerAccess(deviceId, i);   // hipDeviceCanAccessPeer + hipDeviceEnablePeerAccess
    }
    sdma_anvil::anvil.connect(deviceId, i, numChannels);
  }
```

So on single-node MI455 with `vmm_fabric`, **SDMA-put targets fabric-imported
peer memory today** — a supported, shipping path
(`CHANGELOG.md`: "Added single node support for gfx1250 / MI455X";
`docs/build.rst`: `vmm_fabric` "only supported on gfx1250 (MI455)").

### E4. But SDMA is node-local by construction; cross-node needs shader-core

- The connect loop uses the peer index `i` as a **local HIP device id**
  (`EnablePeerAccess(deviceId, i)`, `connect(deviceId, i, ...)`), and engine
  selection uses HSA preferred-copy-engine / `xgmi_physical_id` between local
  agents (`sdma/anvil.cpp`). This only resolves for node-local devices.
- Coherence rationale is explicitly same-die:

```84:89:projects/rocshmem/src/sdma_policy.hpp
      // ... Agent scope is sufficient because SDMA probes GL2 via the coherence
      // protocol on the same die.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
```

- No cross-node SDMA queue construct exists; a genuinely remote peer gets no
  queue handle, so `ipcCopy` falls through to shader-core.
- Arch caveat: gfx1250 uses the legacy MI300X-class `COPY_LINEAR`(+`ATOMIC`)
  packets; the fused OSS7/MI4 path is gated on `__gfx950__` only
  (`sdma/sdma_opcodes.h`), so it is compiled out for gfx1250.

### E5. Conclusion that shapes the plan

| Target topology | SDMA-put to fabric memory? | Transport |
|---|---|---|
| Single-node / same pod, gfx1250 + fabric | **Works (rocSHMEM `vmm_fabric` precedent)** | Anvil SDMA copy engine |
| Cross-node UALoE (multi-node clique) | Not wired (no cross-node SDMA queue) | Shader-core gather/scatter + system-scope fence |

Both topologies share the same **fabric-handle memory substrate**; only the
**transport** is chosen per peer.

---

## Design

### D1. Fabric memory substrate (topology-agnostic)

On the fabric path, `ginAnvilRegMrSym` exports the buffer's VMM allocation with
`CU_MEM_HANDLE_TYPE_FABRIC`, allgathers descriptors across the whole clique,
imports + maps each peer (`ncclCuMemAllocAddr`), writes peer VAs into
`remote_vas[]`, sets `vmmStride = 0`, and calls
`ncclGinAnvilIpcTableRegisterExplicit`. Reuses the proven `ncclFabricMemHandler`
mechanism (`fabric_mem_handler.cc`).

### D2. Per-peer transport classification (the new mechanism)

At connect, classify each peer: SDMA-reachable (node-local/pod, HIP peer-access
+ SDMA queue created) vs fabric-remote (cross-node). Persist a per-peer table in
`ncclGinAnvilSdmaGPUContext`. The device `Put` already gates SDMA on
`handle != nullptr`:

```198:205:projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma.h
    size_t threshold = loadConst(&rsCtx->sdmaThreshold);
    bool useIpcPut = hasWins && bytes <= threshold;
    ...
    bool sdmaDataPath = hasWins && !useIpcPut && handle != nullptr;
```

**Gap to fill:** when `handle == nullptr` and `bytes > threshold` (cross-node
fabric peer), add a **shader-core bulk copy** fallback (uint4 loop into the
fabric VA). Today the large-transfer path assumes an SDMA queue.

### D3. Signals + ordering over fabric

Share the signal buffers over fabric too; on gfx1250 use the shader-side atomic
signal (`ipcFlatAtomicAddSys64`, no OSS7 fused signal). The fabric-remote
shader-core path must issue a **system-scope release fence** before signaling —
the corrected guard from PR #3 (`required==system && given<required` on HIP,
routed through `NCCL_GIN_THREADFENCE_SYSTEM()`).

---

## Phased work plan

### Phase 0 — De-risking spike (this deliverable; no SUT dependency to author)

1. Confirm `HIP_FABRIC_API` on the MI455 SUT (`hipMemFabricHandle_t` +
   `hipMemImportFromShareableHandle`; ROCm 7.14+ / ROCm-systems #2170).
2. Microbench: copy-engine (`hipMemcpyAsync`) and Anvil-SDMA-put into a
   **fabric-imported peer VA**, single-node — validate E3 on the actual SUT.
3. Microbench: cross-node fabric peer — confirm (a) no SDMA queue cross-node,
   (b) shader-core store lands, (c) a system-scope fence is required.
   - Deliverable: go/no-go note. If single-node SDMA-over-fabric fails, single
     node also falls back to shader-core and Phases 3/9 simplify.
   - **Code:** `ddai-artifacts/spikes/phase0-fabric-sdma/` (this commit).

### Phase 1 — Capability gating & coexistence
- `ginSdmaUseFabricPath(comm) = MNNVL==1 && gfx1250 && HIP_FABRIC_API && ncclCuMemEnable()`.
- Leave gfx942/gfx950 xGMI stride path untouched; new code behind
  `#ifdef HIP_FABRIC_API` with graceful fallback.

### Phase 2 — Fabric memory substrate for GIN registration
- Fabric-handle branch in `ginAnvilRegMrSym`; fill `remote_vas[]`,
  `vmmStride = 0`, explicit IPC-table register; teardown in `ginAnvilDeregMrSym`.

### Phase 3 — Per-peer transport classification (core)
- Record SDMA-reachability per peer at connect; persist to GPU context.
- Device `Put`: keep small inline store; SDMA for reachable peers (retain
  128 MiB segmentation); **new shader-core bulk fallback** for fabric-remote.

### Phase 4 — Signals / ordering over fabric
- Fabric-share signals; shader atomic signal on gfx1250; system-scope fence on
  the shader-core path (PR #3 guard).

### Phase 5 — Host dispatch & eligibility
- Extend `rcclUseAlltoAllGda` to admit the gfx1250 clique shape
  (`2<=nRanks<=max`, dtype/16B), single- and multi-node.

### Phase 6 — Tests
- GDA device-template fabric-mode seam (`remote_vas` mock, `vmmStride=0`);
  two transport cases (SDMA vs forced-null shader-core); fence-count assertions.

### Phase 7 — Bring-up & validation on SUT
- Single-node MI455 first, then multi-node MNNVL clique; verify per-peer
  selection and correctness/perf.

### Phase 8 — Documentation
- Design doc + Phase-0 findings.

---

## Risks / open questions

- **Phase 0 is a real gate** — single-node SDMA-over-fabric is inferred from
  rocSHMEM's wiring; the spike must confirm it on the SUT.
- **Mixed-transport within one collective** (some peers SDMA, some shader-core
  in a multi-node clique) needs a unified completion/ordering model.
- **Cross-node fabric-handle exchange scale** and peer-VA lifetime/teardown.
- **Barrier vs signals:** keep GIN signal/counter model; confirm sufficient for
  A2A completion cross-node rather than importing DDA's `FabricGpuBarrier`.

---

## Code references

- `projects/rccl/src/gin/gin_plugin_anvil_sdma.cc` — `ginAnvilRegMrSym` (memory injection point).
- `projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma.h` — `resolveRemotePeerVa`, `Put` transport selection.
- `projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h` — `ncclGinAnvilSdmaGPUContext`, `ncclGinAnvilSdmaMemHandle`.
- `projects/rccl/src/fabric_mem_handler.cc` / `include/fabric_mem_handler.h` — fabric export/allgather/import template.
- `projects/rccl/src/fabric_init.cu` — `ncclDdaUseFabricPath`, `ncclDdaFabricCommInit`.
- `projects/rocshmem/src/ipc_policy.hpp` / `ipc_policy.cpp` — `ipcCopy` transport selection, `vmm_fabric` peer set.
- `projects/rocshmem/src/sdma_policy.cpp` / `sdma_policy.hpp` — SDMA connect loop, same-die coherence fence.
- `projects/rocshmem/src/memory/hip_allocator_vmm_fabric.cpp` / `hip_allocator_vmm_common.hpp` — fabric VMM create/export/import/map.
- `projects/rocshmem/src/sdma/sdma_opcodes.h` — `SDMA_IS_OSS7` (`__gfx950__`) packet gate.
- `projects/rccl/src/collectives.cc` / `rccl_wrap.cc` — `ncclAlltoAll_impl`, `rcclUseAlltoAllGda`.
