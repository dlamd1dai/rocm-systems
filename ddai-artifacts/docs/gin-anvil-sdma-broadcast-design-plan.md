# GIN-SDMA Broadcast on xGMI: Design & Implementation

**Document type:** Internal AMD design/implementation record
**Status:** **Implemented & validated** in `broadcast_perf -D 3` on 8× MI355X (`NCCL_GIN_TYPE=6`).
A four-tier device-initiated broadcast (LL → LSA → flat GIN fan-out → scatter+allgather) is live
in `projects/rccl-tests/src/broadcast.cu`; `#wrong == 0` across the full root sweep (0..N−1),
in-place/OOP, float/int8/double, and non-divisible counts. See **"Current design (as implemented)"**
below for the at-a-glance summary; §3–§4 give the reasoning and per-tier detail. The tier logic is
shared with the device kernels via `gin_sdma_collective_policy.h` and unit-tested to 100 % branch
coverage (§6.1). The final build — plus the host policy unit tests, which gate the image at build
time — is baked into `rccl-gin-gda-sdma-713:latest` (§6). Sections 2 (host ring baseline) and §4.7
(inter-node tree) remain background/deferred.
**Date:** 2026-07-22 (rev. 2026-07-24, 2026-07-27)
**Location:** `work-plans/gin-sdma-backend/broadcast/` (not committed to git)
**Branch:** `users/dondai/gin-stage2c-sdma-bcast-wip`
**Related:** PR #7826 (Anvil SDMA backend), AllGather plan (`allgather/gin-anvil-sdma-allgather-design-plan.md`)

> **Backend selector update (2026-07-24):** the Anvil SDMA GIN backend is now
> **`NCCL_GIN_TYPE=6`** (`NCCL_GIN_TYPE_ANVIL_SDMA`, `core.h`). It was `5` in the
> PR-7826 era; the NCCL GIN **v14 API sync** (`[AICOMRCCL-1281] adapt GIN-SDMA/GIN-GDA
> to v14`) reassigned type **5 → rocSHMEM-GDA** and moved **Anvil SDMA to 6**. All
> broadcast configs below use `=6`. Verified this session on the current ROCm 7.13
> image: A2A (`-D 3`, type 6) passes at ~81.9 GB/s busbw; the broadcast gate
> (BC-C1 host + BC-C2 GIN) and the AllGather gate (AG-C1 host + AG-C2 GIN) both
> pass across the full size range with `#wrong == 0`.
>
> **Kernel set (2026-07-24):** the image builds RCCL device kernels with
> `ONLY_FUNCS="SendRecv|AlltoAllPivot|AlltoAllGda|AlltoAllvGda|Broadcast|AllGather"`
> (Dockerfile ARG + `docker-gin-gda-sdma-build.bash`). `Broadcast` and `AllGather`
> were added so the host baselines (`-D 0`) run the full range — without them the
> host paths faulted above 64M (`ncclDevFuncId not found for coll:0`/`coll:2`).
>
> **Shared policy header + test suite (2026-07-27):** all tier-selection / threshold /
> chunk-math / device-requirement decisions now live in one pure `__host__ __device__`
> header, `projects/rccl-tests/src/gin_sdma_collective_policy.h` (`namespace gin_sdma`), so
> the host launch path and the device kernels cannot drift on tier boundaries. A GoogleTest
> host suite (`gin_sdma_policy_test`, **100 % line + branch coverage**), an opt-in GPU
> functional CTest matrix, and a gcov coverage target back it; the host unit tests are baked
> into `rccl-gin-gda-sdma-713:latest` as a `docker build`-time gate and run as a GPU-free
> preflight in the `gin-sdma-*-test.bash` gates. This is a behavior-preserving refactor — same
> thresholds, chunking and env knobs. See §6.1.

---

## Current design (as implemented)

Device-initiated, **single-node xGMI** broadcast exposed as `broadcast_perf -D 3` with
`NCCL_GIN_TYPE=6` (Anvil SDMA GIN backend). No backend/GIN-ABI changes — the design lives entirely
in two rccl-tests kernels in `projects/rccl-tests/src/broadcast.cu`:

- **`GinHybridBroadcastKernel`** — the small/medium tiers (LL, LSA, flat GIN fan-out). All ranks
  launch; only the root moves data; non-roots hit barriers and a receiver-side `waitSignal`.
- **`GinScatterAllgatherBroadcastKernel`** — the large tier: a root scatter followed by an in-place
  allgather, so egress is spread across **all** ranks instead of the root alone.

Host code (`BroadcastRunColl` case 3) picks the tier by full message size (`msgBytes`) and dispatches
the matching kernel; `BroadcastGetDevCommRequirements` case 3 reserves the resources.

### Tier ladder (by full `msgBytes`, `N` = nRanks)

| Tier | Size range | Path | Mechanism | Default cutoff | Env override |
|---|---|---|---|---|---|
| **LL** | ≤ 2 KiB | `BroadcastLLImpl` (single CTA) | packed 8 B data + epoch flag in LL scratch; no LSA barrier | `2048` (cap 64 KiB) | `NCCL_GIN_ANVIL_BCAST_LL_MAX_BYTES` (0 = off) |
| **LSA** | ≤ 256 KiB | `BroadcastLsaDirect` (root only) | root SM stores the source to every peer's `recvbuff` over LSA/IPC | `262144` | `NCCL_GIN_ANVIL_SDMA_THRESHOLD_BROADCAST` (or shared `…_THRESHOLD`) |
| **Flat / star** | 256 KiB … 2 MiB | root `gin.put × (N−1)` | one-round direct fan-out; Anvil picks SDMA (> 128 B/put); receiver-side `waitSignal(+1)` | — (between LSA and SAG cutoffs) | — |
| **Scatter + allgather (SAG)** | ≥ 2 MiB (and `count ≥ N`) | `GinScatterAllgatherBroadcastKernel` | root scatters chunk `r`→rank `r`, then all ranks in-place allgather; two signal indices (scatter=0, gather=1) | `2097152` (2 MiB) | `NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES` (0 = disable, keep flat) |

Thresholds resolve host-side via the shared policy header (`gin_sdma::resolveThreshold` /
`gin_sdma::parseSize` in `gin_sdma_collective_policy.h`; `common.h`'s `testResolveSdmaThreshold` /
`testParseSdmaThresholdEnv` are thin `getenv()` wrappers over them): a collective-specific env var
wins, else the shared `NCCL_GIN_ANVIL_SDMA_THRESHOLD` (if explicitly set, so global force-knobs
still work), else the data-driven default above. Values accept a `K`/`M`/`G` suffix. This resolution
(and the SAG-eligibility / chunk-split logic below) is unit-tested to 100 % branch coverage (§6.1).

### Sync / completion model

- **Entry barrier only** (world barrier for GIN tiers, LSA barrier for the LSA tier): guarantees every
  rank's `recvbuff` is past the rccl-tests `initData` memset before any peer writes it.
- **Receiver-side signal completion, no exit barrier.** `ncclGin_SignalInc` is a *remote* action that
  increments the **receiver's** signal. Flat: each non-root receives exactly one put and waits
  `base+1`; the root receives none and only `flush`es. SAG: scatter completion is a non-root
  `waitSignal(0, base0+1)`; gather completion is `waitSignal(1, base1+(N−1))` on every rank; a single
  `flush` drains the queues. The two distinct signal indices let the phases run without an inter-phase
  barrier.

### DevComm requirements (`-D 3`, case 3)

`barrierCount = lsaBarrierCount = deviceCtaCount`; `ginSignalCount = max(deviceCtaCount, 2)` (≥ 2 for
the SAG two-signal scheme); `ginConnectionType = NCCL_GIN_CONNECTION_FULL`; plus an LL scratch
resource (`ncclLLA2ACreateRequirement`, `nBlocks=1`) sized to the LL cap for the tiny tier.

### Measured performance (8× MI355X, `NCCL_GIN_TYPE=6`, in-place)

- **Small (LL tier):** 128 B–1 KiB ≈ **6.1–6.3 µs** (~13–16 % below the LSA path).
- **Large:** flat fan-out plateaus at the root-egress ceiling (~60 GB/s @128 MiB); **SAG lifts
  128 MiB to ~224 GB/s (≈3.7×)**. SAG sits at the AllGather roofline near the elbow (≤128 MiB) and
  plateaus at ~229 GB/s for ≥ 256 MiB (serial scatter no longer hidden — see §4.8.5).
- `#wrong == 0` across the full root sweep, in-place/OOP, float/int8/double, and non-divisible counts.
- Not helped: extra SDMA channels; robust across `-V 8..32`.

The remaining sections give the rationale (§3–§4.3), the flat-path detail (§3.2–§4.4), the LL fast
path (§4.3.4), the large-message SAG tier and its measurements (§4.8), the gate/CI plan (§6), and the
deferred inter-node tree (§4.7).

---

## 0. Confirmed decisions (2026-07-23 design review)

The following were reviewed against the RCCL source (`gin_anvil_sdma.h`, `all_gather_gin.cuh`,
`ce_coll.cc`) and confirmed for v1:

| Decision | Choice | Notes |
|----------|--------|-------|
| Initiation model | **Device-initiated GIN kernel** | Root drives `gin.put`; no host CE/RMA orchestration |
| Topology scope | **Single-node, intra-node LSA fan-out only** | Inter-node rail stage deferred (see §4.7) |
| Fan-out mechanic | **Loop `gin.put` per peer** | No Anvil-SDMA backend / GIN ABI changes |
| Algorithm | **Flat / star (single-round direct) fan-out** | Neither ring nor multi-level tree — see §3.1.1 |
| Integration | **Standalone `broadcast_perf -D 3` + gate first** | Production `ncclBroadcast` wiring deferred |
| Data mover | **SDMA for payload, IPC store for ≤ threshold tails** | Backend picks per put by size — see §3.4.1 |

These decisions mirror the "non-GIN host-initiated broadcast design" (`ncclHierCeAllGather`
Phase 4 self-broadcast in `ce_coll.cc`): a one-source → N-peer flat fan-out with per-peer
signaling. We reuse that *shape* but swap the transport (CE `hipMemcpyBatchAsync` → device
`gin.put`) and the sync primitive (`ncclMemOpSync` → GIN barrier/signals).

---

## 1. Scope note

No standalone **“RCCL host-initiated broadcast design”** document was found in this workspace (unlike the AllGather plan at `allgather/gin-anvil-sdma-allgather-design-plan.md`). Section 2 synthesizes the current RCCL host path from source (`broadcast.cu`, `broadcast.h`, `collectives.cc`, `tuning.cc`) plus GIN-SDMA backend docs. If a Confluence or branch-specific doc exists elsewhere, it should be linked here.

---

## 2. RCCL host-initiated broadcast (current design)

**Host-initiated** means `broadcast_perf -D 0`: the CPU enqueues work via `ncclBroadcast`, and GPU kernels run under RCCL’s normal planner — no device GIN API.

### 2.1 API and test path

In `projects/rccl-tests/src/broadcast.cu`, `deviceImpl == 0` calls `ncclBroadcast`; any other `deviceImpl` returns `testNotImplemented`. A GIN broadcast would be `-D 3`, mirroring AlltoAll/AllGather.

### 2.2 Host enqueue pipeline

1. **`ncclBroadcast`** → `ncclEnqueueCheck` with `ncclFuncBroadcast`, chunk/slice steps, root rank.
2. **Tuning** selects algorithm/protocol/channels. For broadcast, **only ring is considered** (tree is explicitly excluded in `tuning.cc`: `(coll == ncclFuncBroadcast || coll == ncclFuncReduce) && a != NCCL_ALGO_RING`).
3. **Chunking pattern**: ring uses `ncclPatternPipelineFrom`; latency model uses **`nRanks - 1` steps** per chunk.
4. **Device kernel** (`broadcast.h`): classic ring broadcast:
   - **Root**: `directSend` or `directCopySend` (OOP)
   - **Rank before root in ring**: `directRecv`
   - **Everyone else**: `directRecvCopyDirectSend` (relay)
5. **Multi-channel**: CBD splits the message across ring channels like other collectives.
6. **Protocols**: SIMPLE, LL, LL128 (gfx942/gfx950 single-node LL gets a 1.65× busBw boost in tuning).
7. **No direct fast path**: unlike AllGather, there is no `rcclDirectBroadcast` equivalent to `rcclDirectAllGather`. Broadcast always goes through the ring planner.

### 2.3 Semantic model

| Rank | sendbuff | recvbuff |
|------|----------|----------|
| root | source (read) | destination (written) |
| non-root | ignored | destination (written) |

Message size is **full broadcast size** (not `count/nRanks` like AllGather). Bandwidth reporting in `broadcast.cu` uses factor **1.0** (not AllGather’s `(nRanks-1)/nRanks`).

### 2.4 Limitations motivating GIN

- **Sequential ring**: 7 hops on 8 GPUs even when xGMI is a full mesh.
- **Host planner overhead**: chunking, proxy ops, channel scheduling — unnecessary for single-node P2P.
- **No device-initiated path**: `-D 3` returns `testNotImplemented`.

AllGather’s AG-C1 gate noted the same split: host path (`-D 0`) uses RCCL ring/direct selection; GIN path (`-D 3`) is separate.

---

## 3. GIN-SDMA broadcast: `GinHybridBroadcastKernel` (implemented)

Follows the **settled AllGather hybrid** (`GinHybridAllGatherKernel`): no transport/plugin changes,
only collective kernels in `broadcast.cu`. This section covers the LL/LSA/flat tiers of
`GinHybridBroadcastKernel`; the large **scatter+allgather** tier (`GinScatterAllgatherBroadcastKernel`)
is §4.8. See "Current design (as implemented)" above for the full tier ladder at a glance.

### 3.1 High-level algorithm

On single-node xGMI with `NCCL_GIN_TYPE=6` (by full `msgBytes`):

| Message size | Active rank(s) | Path |
|--------------|----------------|------|
| ≤ LL cap (**2 KiB default**, §4.3.4) | **root + all** | LL packed data+flag fast path, single CTA (no LSA barrier) |
| ≤ LSA↔GIN threshold (**256 KiB default**, §4.3.1) | **root only** | LSA direct stores to every peer’s `recvbuff` |
| threshold … scatter-AG min (**2 MiB**, §4.8) | **root only** | One-round **`gin.put` to all non-self peers**; Anvil picks IPC (≤ 128 B/put) or SDMA (>) |
| ≥ scatter-AG min (**2 MiB default**, §4.8) | **all ranks** | **scatter + in-place allgather** (`GinScatterAllgatherBroadcastKernel`) — distributes egress |
| all ranks | all | Entry world/LSA barrier; **non-roots complete via receiver-side `waitSignal`** (see §4.4) |

> **Validated (8× MI355X, `NCCL_GIN_TYPE=6`):** `#wrong == 0` for every root 0..N−1, 128 B–1 GiB,
> in-place and OOP, float/int8/double and non-divisible counts, at `THRESHOLD=128` (hybrid),
> `THRESHOLD=0` (all-SDMA, BC-D4), and with SAG on (default) and off. Perf @128 MiB in-place:
> flat ~60 GB/s, SAG ~224 GB/s (≈3.7×); OOP flat ~41 GB/s.

**Reject ring/tree** for the same reasons as AllGather: full xGMI mesh, one-round direct is native; ring adds steps and signal complexity with no topology benefit.

### 3.1.1 Algorithm classification: flat / star fan-out

The chosen algorithm is a **flat (a.k.a. star, or depth-1 tree) fan-out**, and it is worth
stating precisely because it is neither of the two algorithms people usually mean by
"broadcast":

| Algorithm | Hops (N GPUs) | Root egress | When it wins | Used here? |
|-----------|---------------|-------------|--------------|------------|
| **Flat / star (chosen)** | 1 | N−1 concurrent puts | Single-node full mesh (xGMI/IPC), independent SDMA queues per peer | **Yes** |
| Ring | N−1 (serialized) | 1 segment at a time | Sender can't saturate all links; balance egress across large multi-node msgs | No |
| K-ary / binomial tree | ⌈log_k N⌉ | ≤ k | Limited NICs per node; receivers re-broadcast (mainly *inter-node*) | No (deferred) |

Why flat is correct for the current (intra-node) scope:

- All GPUs are directly reachable over xGMI/IPC, so no relay/forwarding hop is needed.
- Each `gin.put` to a distinct peer lands on its **own per-(peer, channel) SDMA queue**
  (`queueHandle(rsCtx, peer, blockId)`), so the N−1 copies run **concurrently on independent
  SDMA engines**. Latency is ~1 hop; the limit is root outbound bandwidth / SDMA-queue count,
  not hop count.
- This is the exact shape of the non-GIN host-initiated design's intra-node "self-broadcast"
  (`ce_coll.cc` Phase 4), which loops `hipMemcpyBatchAsync` over LSA peers with no relay.

Ring/tree only become attractive when adding the **inter-node** rail stage (limited NICs per
node) — at which point a tree across nodes with a flat fan-out *within* each node is the
natural extension (see §4.7). For single-node v1 they add serialized hops and signal
complexity with no topology benefit.

### 3.2 Pseudocode

```text
msgBytes = count * sizeof(T)
threshold = BroadcastGetSdmaThreshold(devComm)   // same helper pattern as AllGather

// --- Phase A: small messages, LSA only ---
if msgBytes <= threshold:
  lsaBar.sync(relaxed)
  if rank == root:
    BroadcastLsaDirect<T>(sendwin, sendoffset, recvwin, recvoffset,
                          count, lsa.nRanks, tid, nthreads)   // writes every peer incl. self
  lsaBar.sync(release)
  return

// --- Phase B: large messages, GIN ---
gin = ncclGin(devComm, context=0)
signalIndex = 0
signalValue = gin.readSignal(signalIndex)   // every rank reads ITS OWN signal base

worldBar.sync(relaxed)   // all ranks: recv buffers quiescent before root writes

if rank == root:
  // out-of-place self copy (no-op in-place); 16B-vectorized when aligned
  if localSrc != localDst:
    BroadcastLocalCopy(localDst, localSrc, count, tid, nthreads)
  for r = tid; r < nRanks; r += nthreads:
    if r == root: continue
    gin.put(world, r,
            recvwin, recvoffset,     // same offset on every peer
            sendwin, sendoffset,     // ncclGin_SignalInc is a REMOTE action:
            msgBytes, SignalInc{signalIndex})  //   it increments PEER r's signal
  gin.flush(Cta)                     // source buffers safe to reuse
else:
  // each non-root receives exactly ONE put from root -> wait for it to settle
  gin.waitSignal(Cta, signalIndex, signalValue + 1)
```

> **Critical (corrected 2026-07-24):** `ncclGin_SignalInc` is the `put` **`RemoteAction`**
> ("action to take *on peer* when put completes", `gin.h`) — it increments the **receiver's**
> signal, not the sender's. The root receives no puts, so it must **not** `waitSignal` on its
> own signal (an earlier draft's `signalValue + (nRanks-1)` on the root would deadlock). The
> completion is the mirror image of AllGather: every non-root receives exactly one put and
> waits `signalValue + 1`; the root only `flush`es. No exit barrier is used — see §4.4.

### 3.3 `BroadcastLsaDirect` (root-only)

Analogous to `AllGatherLsaDirect`, but root reads **one** send slice and writes the **same offset** on every LSA peer:

```cpp
// root only
T* src = (T*)ncclGetLocalPointer(sendwin, sendoffset);
for (size_t i = tid; i < count; i += nthreads) {
  T v = src[i];
  for (int lp = 0; lp < nRanks; lp++) {
    T* dst = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp) + i;
    dst[0] = v;
  }
}
```

Non-root ranks only participate in LSA barriers — they do not issue GIN ops.

### 3.4 Put addressing (large path)

Same simplicity as AllGather large path, but **only root executes puts**:

```cpp
gin.put(world, peer,
    recvwin, recvoffset,    // identical on all peers
    sendwin, sendoffset,    // root’s source
    msgBytes, ncclGin_SignalInc{0});
```

Self-copy: skip `peer == root` in the loop; the root performs an explicit local
`send → recv` copy (`BroadcastLocalCopy`, 16-byte-vectorized when aligned, no-op in-place)
instead of a self-`gin.put`. Completion signalling is **receiver-side**: each non-root waits
`signalValue + 1` (it receives one put); the root does **not** `waitSignal` (it receives none)
and only `flush`es. See §4.4.

### 3.4.1 When SDMA is actually used (per-put size decision)

"GIN-SDMA broadcast" drives the GPU SDMA copy engines, but the Anvil-SDMA backend chooses the
data mover **per `gin.put`, by byte count** — so the broadcast only uses SDMA when each
per-peer put exceeds the threshold. The relevant backend logic (`gin_anvil_sdma.h`):

```cpp
size_t threshold = loadConst(&rsCtx->sdmaThreshold);   // NCCL_GIN_ANVIL_SDMA_THRESHOLD, default 128 B
bool useIpcPut = hasWins && bytes <= threshold;
if (hasWins && !useIpcPut) {
  handle = queueHandle(rsCtx, peer, blockId);          // per-(peer,channel) SDMA queue
  if (handle == nullptr) useIpcPut = true;             // no queue -> IPC fallback
}
// bytes > threshold and handle != nullptr:
sdmaFusedSignal ? ::sdma_anvil::putSignal(*handle, dst, src, bytes, remoteSig)
                : ::sdma_anvil::put(*handle, dst, src, bytes);   // SDMA_OP_COPY / SUBOP_COPY_LINEAR
markSdmaDirty(...);                                      // gin.flush() -> sdma_anvil::quiet() on dirty queues
```

Consequences for the broadcast:

- **> `sdmaThreshold` (default 128 B) per put → SDMA.** Each per-peer chunk copies through that
  peer's SDMA queue; completion is fused into the signal (`putSignal`) when eligible, and
  `gin.flush` quiesces the dirty queues via `sdma_anvil::quiet`. This is the true gin-sdma path.
- **≤ `sdmaThreshold` per put → NOT SDMA.** Falls back to `ipcPut` (direct peer-VA stores by the
  SM); also falls back to IPC if no SDMA queue handle exists for that peer/channel.

To make the payload path *provably* SDMA in tests: keep each `gin.put` size above the threshold
(the large path always does), sweep sizes across 128 B to cover both branches, and use
`NCCL_GIN_ANVIL_SDMA_THRESHOLD=0` (gate **BC-D4**) to force SDMA even for small transfers.

### 3.5 DevComm requirements (`-D 3`)

Mirror AllGather case 3:

```text
barrierCount      = deviceCtaCount
lsaBarrierCount   = deviceCtaCount
ginSignalCount    = deviceCtaCount
ginConnectionType = NCCL_GIN_CONNECTION_FULL
```

Multi-CTA (`-V 8`): all CTAs on **root** cooperative-striping the put loop; **shared signal index 0**; non-root CTAs only hit barriers.

### 3.6 What stays unchanged

| Layer | Change |
|-------|--------|
| `gin_plugin_anvil_sdma.cc` | **None** |
| IPC table / signals / factory | **None** (reuse as-is) |
| `gin_anvil_sdma.h` device templates | **None** |
| CMake (beyond linking `broadcast_perf` like `all_gather_perf`) | Minimal |

---

## 4. Reasoning in detail

### 4.1 Why root-put direct (flat fan-out), not ring

See §3.1.1 for the full flat-vs-ring-vs-tree classification. In short: ring broadcast on 8 GPUs
needs **7 sequential steps** per chunk; bandwidth is bounded by one link at a time. On xGMI, the
root can issue **(N−1) parallel SDMA puts** onto independent per-peer SDMA queues — the same
primitive proven at 128 MiB for AlltoAll (~300 µs) and AllGather (~394 GB/s busbw).

AllGather explicitly rejected ring after a pipelined prototype hung under multi-CTA. Broadcast is strictly simpler (one sender, no per-step relay logic).

### 4.2 Why root-put, not non-root pull (rocSHMEM get-broadcast)

rocSHMEM IPC broadcast switches to **get from root** when `num_pes >= 4` to avoid root write amplification. On GIN Anvil SDMA:

- The validated API surface is **`put` + `waitSignal` + `flush`** (AlltoAll, AllGather).
- Root-put with parallel SDMA engines matches the 128 MiB perf invariant (one OSS7 submission per peer).
- Non-root pull would need a **`gin.get`** path through Anvil SDMA (not exercised in PR-7826 gates) and a different completion model.

Root-put is the conservative, proven choice. Pull-broadcast is a future optimization if profiling shows root SDMA engine saturation.

### 4.3 Why hybrid LSA + GIN (not always GIN)

AllGather MI355X data shows a **~7 µs → ~22 µs cliff** when crossing from LSA to GIN setup. Broadcast small/medium messages are latency-bound; LSA peer stores avoid GIN session/signal/flush overhead, and LSA stays ahead until the message is large enough that root-outbound SDMA bandwidth beats SM-store bandwidth.

The shared `NCCL_GIN_ANVIL_SDMA_THRESHOLD` still controls the backend **`gin.put` IPC vs SDMA split** (per put, in `gin_anvil_sdma.h`) for every collective. The **kernel LSA↔GIN branch** in the rccl-tests kernels, however, is now tunable **per collective** so each can sit at its own measured crossover.

#### 4.3.1 Per-collective LSA↔GIN threshold and defaults (rccl-tests)

`broadcast.cu`, `all_gather.cu`, and `alltoall.cu` resolve their kernel branch threshold host-side with the chain (`gin_sdma::resolveThreshold` in `gin_sdma_collective_policy.h`, via `common.h`'s thin `testResolveSdmaThreshold` wrapper):

1. the collective-specific env var, if set;
2. the shared `NCCL_GIN_ANVIL_SDMA_THRESHOLD`, if explicitly set (keeps the global force knob — e.g. **BC-D4** `THRESHOLD=0` → all-GIN — working);
3. the collective's data-driven **default**.

| Collective | Env var | Compared against | Default | Basis (8× MI355X, `NCCL_GIN_TYPE=6`) |
|---|---|---|---|---|
| Broadcast | `NCCL_GIN_ANVIL_SDMA_THRESHOLD_BROADCAST` | full `msgBytes` | **262144 (256 KiB)** | (2026-07-24) LSA wins ≤256K (256K: 22.8 µs vs GIN 29.9); GIN wins ≥512K (512K: 34.0 µs vs LSA 39.2; 128M: 60.6 vs 17.0 GB/s) |
| AllGather (`-D 3`) | `NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER` | per-rank `chunkBytes` (= total/nRanks) | **262144 (256 KiB/rank)** | (2026-07-24, reconfirmed 2026-07-26 at `-V 8`) LSA wins ≤256K/rank i.e. ≤2M total (2M busbw 82.5 vs SDMA 57.1 GB/s); GIN/SDMA wins ≥512K/rank i.e. ≥4M total (4M: 101.8 vs 98.4; 128M: 390 vs 128 GB/s). Like AlltoAll, the LSA branch needs `-V 8` (2M 82.5 @V8 vs 15.4 @V1); SDMA is CTA-insensitive |
| AlltoAll (`-D 3`) | `NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALL` | per-peer `chunkBytes` (= total/nRanks) | **262144 (256 KiB/peer)** | (2026-07-26, `-V 8`) LSA wins ≤256K/peer i.e. ≤2M total (small-msg 11 µs vs GIN 24.5; 2M busbw 61.7 vs 57.8 GB/s); GIN/SDMA wins ≥512K/peer i.e. ≥4M total (4M: 102.9 vs 44.3; 128M: 389.6 vs 56.4 GB/s) |

Values are bytes with an optional `K`/`M`/`G` suffix. The Anvil backend is unchanged (no GIN ABI change): the per-collective value is read in `RunColl` and passed to the kernel as an extra launch argument (`testLaunchDeviceKernelThreshold`).

**AlltoAll `-D 3` is a size-hybrid** (`GinHybridAlltoAllKernel`): a per-peer chunk ≤ threshold uses a direct all-peers **LSA** copy across all CTAs (latency-optimal for small, ~2× mid-range busbw); above the threshold it uses all-peers **GIN puts** over SDMA copy engines (bandwidth-optimal, scaling to ~390 GB/s @128M). The LSA branch needs enough CTAs, so Test#5 runs `-V 8` (SDMA is CTA-insensitive: `-V 1` ≈ `-V 8` at ~390 GB/s). This supersedes the earlier topology-only note and the shell-level `-D 4`/`-D 3` two-phase split; `-D 4` (`HybridAlltoAllKernel`, topology-based intra-node LSA vs inter-node GIN) is retained for debugging. An optional LL tiny-message tier below the LSA copy is available but off by default — see §4.3.3.

#### 4.3.2 AllGather small-message LSA branch: single-CTA + LL fast path

The AllGather LSA branch (`GinHybridAllGatherKernel`, `all_gather.cu`) is tiered for latency-bound sizes (all measured on 8× MI355X `NCCL_GIN_TYPE=6`, `-V 8`):

- **Fix A — vectorized LSA copy** (`AllGatherLsaVectorized`): the single source chunk is loaded once (wide vector + unroll) and broadcast to every peer's `recvbuff[rank*count]` slot, hoisting the load out of the peer loop. Falls back to scalar for unaligned buffers.
- **Fix B — single-CTA collapse** (`chunkBytes ≤ ALLGATHER_LSA_SINGLE_CTA_MAX = 8 KiB/rank`): tiny AllGather is barrier-bound, not bandwidth-bound. Using all `deviceCtaCount` CTAs multiplies the per-CTA cross-rank LSA barrier traffic for no copy-throughput gain, so the copy collapses to one CTA (CTAs > 0 return before touching a barrier).
- **Fix C — LL packed data+flag fast path** (`chunkBytes ≤ ALLGATHER_LL_MAX_BYTES = 4 KiB/rank`, single CTA, `AllGatherLLImpl` on `ncclLLA2ASession`): each rank `bcast`s its chunk (as 8-byte units) into an epoch-tagged LL scratch buffer and `recv`s every rank's chunk out of its **own** scratch into its **local** `recvbuff`. This removes **both** LSA barriers: cross-rank traffic lives entirely in the LL scratch and is ordered by the per-slot epoch tag, and each rank writes only its own `recvbuff`.
  - **Why the naive "drop a barrier" fix C is wrong.** The two LSA barriers are *not* both removable. rccl-tests' datacheck loop `Barrier`s (host/MPI only, no device sync) *before* `initData` but not between `initData` (which `cudaMemset`s `recvbuff`) and `startColl`. The in-kernel **entry** barrier is what guarantees all ranks are past their memset before any peer writes their `recvbuff`; dropping it lets a fast rank's direct-LSA write get clobbered by a slow rank's memset (reproduced on MI355 as intermittent ~3.5K wrong elts). LL sidesteps this entirely because cross-rank writes never touch `recvbuff`.
  - **Requirement wiring.** An LL scratch resource is requested in `AllGatherGetDevCommRequirements` via `ncclLLA2ACreateRequirement(nBlocks=1, nSlots = nRanks·ALLGATHER_LL_MAX_BYTES/8)`; the returned `ncclLLA2AHandle` (a `{bufHandle, nSlots}` POD, `bufHandle` assigned by `ncclDevCommCreate`, buffer zero-initialized) is passed to the kernel via `testLaunchDeviceKernelThresholdLL`. `nSlots==0` (LL not configured) makes the kernel fall back to the vectorized LSA path.
  - **Cutoff.** LL doubles wire volume (16-byte line = 8 B data + 2× epoch words). Tuned 2026-07-26: LL beats vectorized single-CTA LSA up to **4 KiB/rank** (32 KiB total: 12.2 vs 13.2 µs) but loses at 8 KiB/rank (64 KiB total: 19.0 vs 13.5 µs).
  - **Result (small-message latency, out-of-place):** 128 B 13.3→**10.2 µs**, 2 KiB 13.7→**10.6**, 8 KiB ~13→**10.6**, 16 KiB ~13→**11.4**, 32 KiB 13.2→**12.2** (≈7–23 % lower); ≥64 KiB unchanged. `#wrong=0` across repeated runs.

#### 4.3.3 AllToAll LL fast path: implemented, measured marginal, default-off

The same LL idea was ported to the AllToAll LSA branch (`GinHybridAlltoAllKernel`, `alltoall.cu`) to see whether it could cut small-message latency the way it did for AllGather. It cannot on 8× MI355X, and the path is therefore **off by default**.

- **Mechanism** (`AlltoAllLLImpl` on `ncclLLA2ASession`, single CTA). AllToAll delivers a *distinct* chunk per peer, so — unlike AllGather's `bcast` — it uses the LL A2A session's point-to-point **`send(peer, elt, data)`** (a scatter). Rank *r* scatters its per-peer chunk into peer *p*'s epoch-tagged scratch keyed by the **source** rank (slot region `[r·chunkU64 …]`), then polls all `nRanks` source regions out of its **own** scratch into its **local** `recvbuff`. As with AllGather, cross-rank traffic never touches `recvbuff`, so it is immune to the `initData` recvbuff-memset race and needs **no LSA barrier**.
- **Why it does not help (the key difference from AllGather).** AllGather's LSA path had a barrier-dominated latency floor that LL removed. AllToAll's direct-LSA all-CTA scatter is already at a **~11 µs fixed-overhead floor** (kernel launch + comm setup) that LL cannot beat: LL still needs every rank to poll all `nRanks` source slots — an implicit all-to-all sync — and moves **2× the wire volume**.
- **Measurement (careful same-build A/B, 50 iters × 3 reps, 2026-07-27).** LL is at best a **marginal, within-noise ~2–3 % faster at the very smallest sizes (≤ 32 B/peer)** and neutral-to-slightly-worse above that; run-to-run variance is itself ~2–3 % (a control row using the *same* path in both arms differed by 2.3 %). An earlier single-shot A/B that looked like a 5–10 % win was an inflated (~12 µs) baseline moment, corrected by the repeats. `#wrong=0` in every arm.
- **Decision + wiring.** The LL path defaults **OFF** (no regression to the proven direct-LSA copy) and is **opt-in / runtime-tunable** via `NCCL_GIN_ANVIL_A2A_LL_MAX_BYTES=<per-peer bytes>` (0/unset = disabled, no recompile needed for future tuning). When enabled, `AlltoAllGetDevCommRequirements` requests the LL scratch (`ncclLLA2ACreateRequirement`, sized to `nRanks · cap/8` slots, capped at a compile-time `ALLTOALL_LL_MAX_BYTES = 64 KiB/peer` ceiling); when disabled the requirement is skipped so `nSlots==0` and the kernel takes the direct-LSA path. The handle rides the shared `testLaunchDeviceKernelThresholdLL` launcher.

#### 4.3.4 Broadcast LL small-message fast path: default-on, ~16 % faster

Broadcast is the case where the LL idea pays off cleanly, because its small-message LSA path (`BroadcastLsaDirect`) is genuinely two-barrier-bound: only the root writes, so almost all of the ~7 µs floor is the entry+exit LSA barriers, not copy work. LL replaces one of them.

- **Mechanism** (`BroadcastLLImpl` on `ncclLLA2ASession`, single CTA). Broadcast is one-to-all, so it maps straight onto the session's `bcast()` primitive: only the root `bcast`s its message (as 8-byte units) into every peer's epoch-tagged scratch, and every rank (root included) `recv`s it out of its **own** scratch into its **local** `recvbuff`. Only `msgBytes/8` slots are needed (one message), versus `nRanks·chunk/8` for AllGather/AllToAll.
- **Why one barrier stays (the deadlock that AllGather LL doesn't have).** The session double-buffers by epoch parity (`(epoch&1)·nSlots`), tolerating only a **< 2-epoch** writer/reader skew. AllGather's mutual `recv` self-throttles every rank to within one epoch, so it needs **no** barrier. Broadcast has **no backpressure** — only the root writes — so across the perf loop's back-to-back kernel launches a fast root can run ≥ 2 epochs ahead of a lagging non-root and overwrite a slot region that rank is still polling with a newer tag → the tag never matches → **hang** (reproduced on MI355). Keeping a single **entry** LSA barrier bounds the skew to < 1 epoch and makes the double-buffer safe; the **exit** barrier stays removed (the `recv` epoch-tag match *is* the completion signal). Net: two barriers → one.
- **Result (small-message latency, out-of-place, 8× MI355X, 50 iters × 3 reps, 2026-07-27):** 128 B 7.27→**6.28 µs**, 256 B 7.23→**6.11**, 512 B 7.23→**6.07**, 1 KiB 7.22→**6.09** (≈ **13–16 % lower**), 2 KiB 7.25→**6.64** (≈ 8 %); crossover at 4 KiB (+3 %). Consistent across all reps; `#wrong=0` across the full root sweep (0..N−1) and for int8/double.
- **Decision + wiring.** Because the win is robust, the LL path is **on by default up to `BROADCAST_LL_DEFAULT_MAX_BYTES = 2 KiB`** and tunable via `NCCL_GIN_ANVIL_BCAST_LL_MAX_BYTES=<bytes>` (`0` = disable, unset = 2 KiB), clamped to a compile-time `BROADCAST_LL_MAX_BYTES = 64 KiB` ceiling. `BroadcastGetDevCommRequirements` requests `cap/8` slots via `ncclLLA2ACreateRequirement`; the handle rides the shared `testLaunchDeviceKernelThresholdLL` launcher. The pre-sized slot count is what enforces the effective cutoff (`chunkU64 ≤ nSlots`).

**LL across the three collectives.** AllGather: robust 7–23 % win, no barrier (default-on). Broadcast: robust 13–16 % win, one entry barrier retained for backpressure (default-on ≤ 2 KiB). AllToAll: within-noise, direct-LSA already at its floor (default-off, opt-in). The differentiator is how barrier-bound the baseline small path is and whether the collective supplies its own backpressure.

### 4.4 Synchronization model

**Signal semantics (the key fact).** `ncclGin::put`'s `ncclGin_SignalInc` is the template's
**`RemoteAction`** parameter — documented in `gin.h` as *"Action to take **on peer** when put
completes"*. It increments the **receiving** peer's signal, **not** the sender's. (`put` has a
separate `LocalAction` parameter, e.g. `ncclGin_WeakCounterInc`, for a *local* completion
counter — a different mechanism entirely.) This is confirmed by AllGather/AlltoAll, where each
rank issues N puts, **receives** N puts, and waits `base + N` on its own signal.

**Challenge.** In AllGather every rank both sends and receives N puts, so the "wait `base + N`"
is symmetric. In broadcast only the root sends, so the flow is asymmetric:

- The **root** issues `N-1` puts but **receives 0** puts → its own signal is never incremented.
- Each **non-root** receives **exactly 1** put → its signal goes `base → base + 1`.

**Solution (receiver-side completion):**

- **Root:** issue the `N-1` puts, then `flush()`. `flush` guarantees the root's *source buffers
  are safe to reuse*; the root does **not** `waitSignal` (waiting on its own never-incremented
  signal would **deadlock** — this was the bug in the earlier draft that told the root to wait
  `base + nRanks - 1`).
- **Non-roots:** `waitSignal(0, base + 1)`. Because `SignalInc` visibility "implies all preceding
  puts are settled," this is exactly what makes the payload visible on the receiver.

**Why no exit barrier.** `gin.h` is explicit that `flush` *"does not guarantee that data has
settled in remote memory."* So a post-root world barrier — even with `memory_order_release` —
would **not** be sufficient to guarantee non-roots see the data; a non-root could pass the
barrier before the SDMA write to its HBM lands. The receiver-side `waitSignal(+1)` is the
correct and necessary completion primitive, and it makes an exit barrier redundant. Only the
**entry** barrier is kept (recv buffers quiescent before the root starts writing, mirroring
AllGather's entry barrier).

### 4.5 Root rotation and in-place

`broadcast_perf` sweeps **root = 0..N−1**. The kernel uses `devComm.rank == root` (parameter `root` passed to kernel). Every rank runs the same kernel; only the root rank executes LSA/GIN sends.

**In-place** (`sendbuff == recvbuff`): root skips self-put; LSA path writes to all peers including self via local pointer. **OOP**: root may need local send→recv copy; `gin.put` self path handles this if included, but skipping self-put + explicit local copy is clearer.

### 4.6 Expected performance

| Regime | Expectation | Measured (8× MI355X, 2026-07-24) |
|--------|-------------|-----------------------------------|
| ≤128 B | ~7–8 µs (LSA); ~6 µs with the LL fast path (§4.3.4) | ~7.2 µs @128 B (LSA), **~6.2 µs with LL (default ≤ 2 KiB, −16 %)**; ~23 µs when forced to GIN (`THRESHOLD=0`) |
| Medium | GIN setup floor ~21–22 µs | ~23 µs @1–8 KB |
| 128 MiB | Root fan-out (N−1) puts; busbw ≈ `(N-1)/N × link_BW` | **in-place ~60 GB/s**, **OOP ~41 GB/s** (OOP bounded by the root's local send→recv copy; 16B-vectorized) |

> The OOP < in-place gap is the extra root-local `send → recv` copy that in-place skips.
> Vectorizing that copy (16-byte stores) lifted OOP from ~24 → ~41 GB/s @128 MB with `-V 8`.
> Further OOP gains would come from overlapping the local copy with the remote puts.
>
> **Large-message ceiling:** the ~60 GB/s in-place figure is the root-egress cap
> (`busBw = B_root_egress / N`, §4.8.1), not an SDMA-engine limit. The **scatter + allgather**
> tier (§4.8, default ≥ 2 MiB) lifts 128 MiB in-place to **~224 GB/s (≈3.7×)** — measured,
> at the AllGather roofline for the `M/N` chunk size — by distributing egress across all N ranks.

Bus bandwidth formula for the test harness:

```cpp
*busBw = algBw * (double)(nranks - 1) / (double)nranks;
```

(one copy of the message delivered to N−1 receivers)

### 4.7 Scope boundaries

- **Single-node only** — GIN Anvil SDMA is intra-node xGMI (backend enforces `nNodes == 1`).
- **rccl-tests first** — same staging as AllGather: `-D 3` in `broadcast.cu`, gate script, then optional production `ncclBroadcast` direct path later.
- **No ring/tree variants** — consistent with AllGather v1 decision table.

### 4.8 Large-message optimization: scatter + allgather

**Status:** **implemented & validated on 8× MI355X** (`GinScatterAllgatherBroadcastKernel` in
`broadcast.cu`, 2026-07-27): `#wrong == 0` across the full root sweep (in-place/OOP,
float/int8/double, non-divisible counts), and **128 MiB in-place 60.5 → 223.9 GB/s (≈3.7×)** vs
the flat path (same-build A/B). See §4.8.5 for the measured sweep. This adds a **third
large-message tier** above the flat fan-out; it does not change the LL/LSA small tiers or the
flat path.

#### 4.8.1 Why the flat fan-out plateaus (root-egress ceiling)

The flat/star path (§3.1.1, §4.1) is latency-optimal (1 hop) but its large-message bandwidth is
**structurally capped by root egress**: only the root sends, so it pushes **N−1 full copies** of
the message out over its own outbound xGMI/SDMA links. For the harness busbw definition
(`busBw = algBw · (N−1)/N`, one copy delivered to N−1 receivers):

```text
flat broadcast:  T = (N-1)·M / B_root_egress
                 algBw = M/T = B_root_egress / (N-1)
                 busBw = algBw·(N-1)/N = B_root_egress / N
```

So flat busbw is **root egress ÷ N**, independent of how many SDMA queues are added once the
root's links are saturated. Measured 128 MiB in-place ~60 GB/s on 8× MI355X ⇒ root egress
≈ **480 GB/s fully saturated**. This is exactly why the same SDMA engines yield **388 GB/s for
AllGather** (`allgather-host-sdma-vs-gin-sdma-benchmark.md`) but only ~60 for broadcast:
AllGather spreads egress across all N ranks; flat broadcast concentrates it on one.

> **Corollary — pull/`gin.get` does not fix this (revises §4.2's "future optimization" hope).**
> A non-root `gin.get`-from-root variant still funnels N−1 copies through the *root's* HBM read
> port and outbound links. Get-vs-put changes *who initiates*, not the total data leaving the
> root, so the `B_root_egress / N` ceiling is unchanged. Only algorithms where **non-root ranks
> also forward data** (ring, tree, or scatter+allgather) break the ceiling.

#### 4.8.2 Algorithm: scatter → allgather (van de Geijn)

Reformulate a large broadcast as the bandwidth-optimal two-phase pipeline, reusing the already
validated **`GinHybridAllGatherKernel`** (388 GB/s @128 MiB):

| Phase | Action | Data off the root | Egress distribution |
|-------|--------|-------------------|---------------------|
| **1. Scatter** | root partitions M into N chunks; sends chunk *i* to rank *i* (`gin.put` of M/N per peer) | **M total** (not (N−1)·M) | root only, but 1/(N−1) the volume of flat |
| **2. AllGather** | all ranks run the existing hybrid AllGather over the M/N-per-rank chunks | (N−1)/N · M **per rank** | **across all N ranks** |

After phase 2 every rank holds all N chunks = the full message. This is the standard
scatter+allgather (a.k.a. "split") broadcast; MPI/NCCL use it (or a scatter + ring-allgather)
for exactly this reason on large messages.

#### 4.8.3 Placement in the tier ladder

A new host-resolved threshold `NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES` selects the tier;
below it, the proven flat path is retained (lower latency, no scatter setup):

| Message size | Path | Rationale |
|---|---|---|
| ≤ LL cap (§4.3.4) | LL packed data+flag | latency floor |
| ≤ 256 KiB | `BroadcastLsaDirect` (LSA) | latency-bound |
| 256 KiB … scatter-AG min | **flat fan-out** (`gin.put × (N−1)`) | 1-hop latency still beats 2-phase setup |
| ≥ scatter-AG min (**2 MiB**, measured) | **scatter + allgather** | root-egress ceiling dominates; distribute egress |

The crossover is data-driven and was measured (§4.8.5): SAG becomes a win at **2 MiB** and
decisive ≥ 4 MiB, so the default `SCATTER_AG_MIN` is **2 MiB**.

#### 4.8.4 Pseudocode (as implemented)

The prototype uses **two distinct signal indices** (scatter=0, gather=1) instead of a single
signal split by an inter-phase barrier. This removes the extra global barrier entirely: the
per-phase completion counts can never interleave, so the only synchronization is the entry
barrier plus receiver-side `waitSignal`s.

```text
msgBytes = count * sizeof(T)
scatterAgMin = env NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES (default 2 MiB, host-resolved)

// Host gate (BroadcastRunColl): fall through to the flat/LSA/LL tiers unless
if scatterAgMin != 0 and nRanks >= 2 and msgBytes >= scatterAgMin and count >= nRanks:
  launch GinScatterAllgatherBroadcastKernel   // else launch GinHybridBroadcastKernel

// --- GinScatterAllgatherBroadcastKernel ---
N          = devComm.nRanks
baseCount  = count / N                  // remainder folded into rank N-1's slice
myCount    = (rank == N-1) ? count - baseCount*(N-1) : baseCount
myByteOff  = rank * baseCount * sizeof(T)
gin        = ncclGin(devComm, 0)
base0      = gin.readSignal(0)           // scatter signal
base1      = gin.readSignal(1)           // gather signal

worldBar.sync(relaxed)                   // entry: recv buffers quiescent before root writes

// Phase 1: scatter chunk r -> rank r (root only)
if rank == root:
  if send != recv:                       // own slice stays local (no-op in-place)
    BroadcastLocalCopy(recv+myByteOff, send+myByteOff, myCount)
  for r = tid; r < N; r += nthreads:
    if r == root: continue
    gin.put(world, r, recvwin, recvoffset + r*baseCount*sizeof(T),
                      sendwin, sendoffset + r*baseCount*sizeof(T),
                      chunkBytes(r), SignalInc{0})       // increments peer r's signal 0
  // no intermediate flush: scatter+allgather both read the read-only sendwin
else:
  gin.waitSignal(Cta, 0, base0 + 1)      // my scatter slice landed (CTA-wide => visible)

// Phase 2: in-place allgather of the N slices (every rank forwards its own slice)
srcWin = (rank == root) ? sendwin : recvwin   // root reads stable sendwin; no copy-vs-put order
for r = tid; r < N; r += nthreads:
  if r == rank: continue
  gin.put(world, r, recvwin, recvoffset + myByteOff,
                    srcWin, srcOff + myByteOff,
                    myCount*sizeof(T), SignalInc{1})     // increments peer r's signal 1
gin.waitSignal(Cta, 1, base1 + (N-1))    // every rank receives exactly N-1 gather puts
gin.flush(Cta)
```

Notes on the implementation choices:

- **Two signals, no inter-phase barrier.** Scatter puts hit signal 0, gather puts hit signal 1,
  so a laggard's gather put can never be mistaken for its scatter put. A non-root forwards only
  after `waitSignal(0, base0+1)` (a CTA-wide op that also makes the scatter data visible); the
  root reads its own slice from the stable `sendwin` (never the just-written `recvwin`), so its
  local copy needs no ordering versus its gather puts. Requires **`ginSignalCount >= 2`** (set in
  `BroadcastGetDevCommRequirements` case 3; the flat/LSA paths still use only signal 0).
- **Uniform completion.** Every rank (root included) receives exactly `N-1` gather puts, so the
  final wait is the same on all ranks — `base1 + (N-1)`.
- **Composition, not a call.** Phase 2 reimplements the AllGather large-path put shape inline
  (in-place on `recvbuff`) rather than calling `GinHybridAllGatherKernel`; §8 Q8 tracks factoring
  the shared body into a device helper.

#### 4.8.5 Performance — measured (8× MI355X, `NCCL_GIN_TYPE=6`, `-V 32`, 2026-07-27)

Validated in the dev container (incremental `broadcast_perf` rebuild on the `rccl-gin-gda-sdma-713`
image). `#wrong == 0` across the full root sweep (0..7), in-place and OOP, for float/int8/double
and for a deliberately non-8-divisible `int8` count (tail-chunk path). A/B is a same-build toggle
via `NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES=0` (forces the flat path).

| Total size | Flat busbw (GB/s) | Scatter+AG busbw (GB/s) | Speedup |
|---|---:|---:|---:|
| 4 MiB   | 44.0  | 60.9  | 1.4× |
| 8 MiB   | 51.6  | 97.2  | 1.9× |
| 16 MiB  | 56.0  | 140.3 | 2.5× |
| 32 MiB  | 58.4  | 177.6 | 3.0× |
| 64 MiB  | 59.6  | 206.4 | 3.5× |
| **128 MiB** | **60.5** | **223.9** | **3.7×** |

(in-place; OOP tracks within ~2–3 %, e.g. 128 MiB 216 GB/s.) The measured 128 MiB result
(**~224 GB/s**) beats the §4.8.5 projection (~210) and matches the model:

```text
T_scatter = (N-1)/N · M / B_root_egress ≈ (7/8)·M/480 = M/548
T_ag      = M / algBw_ag                ≈ M/443
busBw     = (M/(T_scatter+T_ag))·(N-1)/N ≈ 214 GB/s   (measured 224)
```

Root sweep is flat across ranks (128 MiB in-place: 215.4–224.8 GB/s over roots 0..7).

##### Crossover sweep and default (2026-07-27, `-V 32`, in-place, root 0)

Same-build A/B (`NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES=0` = flat) across the small-large
transition:

| Total size | Flat busbw | Scatter+AG busbw | Winner |
|---|---:|---:|---|
| 256 KiB | 20.8 (LSA) | 5.4 | flat/LSA |
| 512 KiB | 14.8 | 10.4 | flat |
| 1 MiB   | 23.7 | 19.4 | flat |
| **2 MiB** | **33.4** | **35.9** | SAG (slight) |
| 4 MiB   | 42.2 | 62.1 | **SAG** |
| 8 MiB   | 48.5 | 97.8 | **SAG** |

SAG loses below 2 MiB (its unconditional SDMA `gin.put` allgather is inefficient for the tiny
`M/N` sub-chunks there, where the flat path's LSA/direct stores win), is a wash at 2 MiB, and
wins decisively ≥ 4 MiB. **Default set to `SCATTER_AG_MIN = 2 MiB`** (was 4 MiB) — captures the
2 MiB crossover with no regression below it.

##### Ceiling analysis: SAG is at the AllGather roofline near the elbow, but plateaus at large M

A broadcast of `M` bytes runs an allgather with **per-rank chunk `M/N`**, so SAG sits on the
AllGather bandwidth curve *at that chunk size*, not at the 1 GiB operating point where AllGather
peaks. Measured side-by-side (`-D 3`, 8× MI355X):

| Broadcast M | per-rank `M/8` | AllGather roofline @ `M/8` | SAG broadcast busbw |
|---|---:|---:|---:|
| 128 MiB | 16 MiB | 204 | **212** (at roofline) |
| 256 MiB | 32 MiB | 279 | 221 |
| 512 MiB | 64 MiB | 335 | 226 |
| 1 GiB   | 128 MiB | 376 | 229 |

Near the elbow (≤ ~128 MiB) the per-rank chunk is small, so the allgather is relatively slow and
`T_scatter` is proportionally hidden — SAG **matches (even slightly beats) the AllGather roofline**
(212 vs 204) and is essentially optimal. At larger M the allgather roofline keeps climbing
(→ 376 at `M/8 = 128 MiB`, the source of the "~388" figure) but SAG **plateaus at ~229 GB/s**:
the serial `T_scatter` (root egress ≈ `(N-1)/N·M/B`) stops being hidden and dominates. So the
"~388 ceiling" is a *different operating point* (1 GiB per-rank allgather), not headroom the
current kernel leaves on the table at 128 MiB.

##### Knobs that don't help, and the remaining headroom

- **SDMA channels** (`NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS` 1/2/4) — no effect (128 MiB: 216.7 /
  217.1 / 216.4). SAG is not SDMA-queue bound.
- **CTA count `-V`** (8/16/24/32) — robust and correct at all counts; ~flat perf (8 MiB:
  101.8 / 103.8 / 98.9 / 97.2), a slight edge to fewer CTAs. (An early full-matrix sweep hit a
  transient hang at `-V 16` that did **not** reproduce in isolation — orphaned MPI/GPU state
  between back-to-back launches, not a kernel deadlock.)
- **Remaining headroom is only at large M (≥ 512 MiB)** and requires a genuinely more complex
  **chunked software pipeline**: split `M/N` into `K` sub-chunks and overlap scatter of sub-chunk
  `k+1` with the allgather of sub-chunk `k`, so `T_scatter` is hidden behind the allgather rather
  than serialized ahead of it. Projected upside ~1.6× at 1 GiB (229 → ~376). This is a scoped
  future rewrite (per-sub-chunk signal accounting across both phases); the current kernel is
  optimal at the elbow and is the right default now.

#### 4.8.6 Correctness / edge cases

- **In-place** (`sendbuff == recvbuff`): the root's own slice is already in place (skip its
  self-copy); non-root scatter targets are distinct `recvbuff` slots, and the in-place AllGather
  is the same one the AllGather gate already validates.
- **Non-divisible M**: give the remainder bytes to the last rank's chunk (standard uneven
  AllGather handling); the AllGather kernel already supports per-rank counts.
- **Root sweep 0..N−1**: symmetric — chunk *r* always goes to rank *r*; only the source offset
  on the root differs by root.
- **Completion**: scatter completion is the receiver-side `waitSignal(0, base0+1)` (non-roots
  only); gather completion is `waitSignal(1, base1+(N-1))` on every rank. `flush` runs once at
  the end. No exit barrier (the gather `waitSignal` is the completion primitive, as in the flat
  path and AllGather).
- **DevComm requirement delta**: the two-signal scheme needs `ginSignalCount >= 2`; case 3 now
  sets `ginSignalCount = max(deviceCtaCount, 2)`. Everything else (barrier, LSA barrier, LL
  scratch) is unchanged, so no new resource *types* are requested.

#### 4.8.7 Relationship to the deferred inter-node tree (§4.7)

Scatter+allgather is the natural **intra-node** building block for the deferred multi-node
design: a cross-node k-ary/binomial **tree** carrying M/N-sized chunks, with a **flat scatter +
allgather within each node**, generalizes cleanly. Landing intra-node scatter+allgather now de-
risks that extension.

---

## 5. Architecture sketch

```text
  broadcast_perf (-D 3, NCCL_GIN_TYPE=6)
           │
  BroadcastRunColl case 3 — tier select by msgBytes
           │
   ┌───────┬────────────────┬────────────────────┬───────────────────────────┐
   │ ≤2 KiB│ ≤256 KiB       │ 256 KiB … 2 MiB    │ ≥2 MiB (count ≥ N)        │
   ▼       ▼                ▼                    ▼
 LL fast   LSA direct       flat GIN fan-out     scatter + allgather
 path      (root SM stores  (root gin.put×(N−1)) (GinScatterAllgatherBroadcastKernel)
 (1 CTA,   to every peer)   IPC/SDMA per put     ├─ root scatter: chunk r → rank r (sig 0)
 no LSA    │                │                    └─ all ranks in-place allgather (sig 1)
 barrier)  │                │                          egress spread across all N ranks
   └───────┴──────┬─────────┴────────────────────┴───────────────────────────┘
                  ▼
           xGMI P2P mesh (LSA/IPC + SDMA copy engines)
                  │
   completion: entry barrier + receiver-side waitSignal (no exit barrier); single flush
   non-root ranks in flat/LSA tiers: barriers + waitSignal only
```

First three tiers are `GinHybridBroadcastKernel`; the ≥ 2 MiB tier is
`GinScatterAllgatherBroadcastKernel`. See "Current design (as implemented)" for cutoffs/env vars.

---

## 6. Proposed test gate (`gin-anvil-broadcast-test.bash`)

The implemented gate is `ddai-artifacts/scripts/gin-sdma-bcast-test.bash` (the `gin-anvil-broadcast-test.bash`
name below is the original design placeholder). It now opens with a GPU-free unit-test preflight
(**BC-UT**) before the GPU sections.

| ID | Test | Config |
|----|------|--------|
| **BC-UT** | Host policy unit tests | `gin_sdma_policy_test` (no GPU) run as a **hard preflight** in `gin-sdma-bcast-test.bash`; validates the shared tier / threshold / chunk / requirement logic to 100 % line+branch. `RUN_POLICY_UT=0` to skip. Also enforced at `docker build` time (`ctest -L unit`). |
| **BC-C1** | Host `ncclBroadcast` | `-D 0`, 128 B–`MAX_BYTES` (full range, no 64M cap — host broadcast validated to 256M on MI355X), `NCCL_CUMEM_ENABLE=0` — perf reference, **on by default**. Requires the broadcast host ring kernels, which the image now ships via `ONLY_FUNCS="…|Broadcast|AllGather"` (Dockerfile ARG + `docker-gin-gda-sdma-build.bash`). Earlier images omitted them (`ncclDevFuncId … not found for coll:0`); unlike AllGather, broadcast has no direct/copy fast path so it always needs the ring device func. The sibling AllGather gate (`gin-sdma-ag-test.bash`, AG-C1) had the same 64M host cap for the same reason and has likewise been lifted now that `AllGather` is in `ONLY_FUNCS`. |
| **BC-C2** | GIN hybrid | `-D 3 -V 8`, 128 B–128 M, `NCCL_GIN_TYPE=6` |
| **BC-C3** | rocSHMEM `team_broadcast` | optional cross-check |
| **BC-D4** | All-SDMA gate | `THRESHOLD=0` |
| **BC-C5** | Scatter+AG large tier (§4.8) | `-D 3 -V 8`, ≥ `SCATTER_AG_MIN`–128 M, `NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES` swept; correctness parity with BC-C1 |
| **BC-P1** | Perf vs host @ 128 M, all roots | |
| **BC-P2** | Flat vs scatter+AG crossover sweep (§4.8.3) | `-D 3 -V 8`, 1 M–128 M; locate data-driven `SCATTER_AG_MIN` and confirm ≈3.5× @128 M |

Pass: `#wrong == 0`, sweep roots 0..N−1, in-place and OOP.

### Example commands

```bash
# Full gate (8 GPU, MI355X)
./gin-anvil-broadcast-test.bash 8 128M

# GIN-only (skip host + rocSHMEM)
RUN_HOST_BASELINE=0 RUN_ROCSHMEM_BCAST=0 ./gin-anvil-broadcast-test.bash 8 128M

# All-SDMA gate (BC-D4)
NCCL_GIN_ANVIL_SDMA_THRESHOLD=0 RUN_HOST_BASELINE=0 ./gin-anvil-broadcast-test.bash 8 128M
```

Default gate env (BC-C2): `NCCL_GIN_TYPE=6`, `THRESHOLD=128`, `NUM_CHANNELS=1`, `FUSED_SIGNAL=0`, `-V 8`.

### 6.1 Implemented test suite: host policy unit tests + GPU functional + coverage (2026-07-27)

The tier-selection / threshold / chunk-math / device-requirement logic that both the host launch
path and the device kernels depend on was extracted into a single pure header,
`projects/rccl-tests/src/gin_sdma_collective_policy.h` (`namespace gin_sdma`, `__host__ __device__`).
`common.h` delegates to it (`testResolveSdmaThreshold` / `testParseSdmaThresholdEnv` are thin
`getenv()` wrappers over `gin_sdma::resolveThreshold` / `gin_sdma::parseSize`), so host and device
agree on every tier boundary by construction. No behavior change.

Tests live under `projects/rccl-tests/test/` and are wired into CTest (opt-in `-DBUILD_TESTS=ON`;
falls back to FetchContent when system GoogleTest is absent):

| Target | Kind | Coverage |
|---|---|---|
| `gin_sdma_policy_test` (ctest label `unit`) | GoogleTest, host-only, no GPU | Every branch of the shared header — `parseSize`, `resolveThreshold`, `resolveLLCap`, `alignChunkCount`, and the per-collective tier / eligibility predicates incl. `bcastUseScatterAllgather`, `bcastLLEligible` and the Broadcast tier ladder. **100 % line and branch** (29 cases). |
| `gin_sdma_gpu_{broadcast,all_gather,alltoall}` (label `gpu_functional`, `-DBUILD_GPU_FUNCTIONAL_TESTS=ON`) | CTest + MPI, needs GPUs | Drives the real device kernels through the `*_perf` binaries across every tier, root sweep, in-place/OOP, and tail/alignment, using the built-in `#wrong` correctness check. |

- **Coverage target.** `-DENABLE_COVERAGE=ON` adds a `gin_sdma_coverage` make target
  (`test/gin_sdma_coverage.sh`) that runs the unit tests and reports line+branch coverage of the
  policy header, gating on `GIN_SDMA_COVERAGE_MIN_PCT` (default 95 %). It auto-selects
  `llvm-cov gcov` for amdclang/hipcc builds and GNU `gcov` otherwise (so it works locally and in
  the ROCm image).
- **Image bake-in (build-time gate).** The canonical image installs `libgtest-dev` (dedicated late
  layer, so it does not bust the RCCL/rocSHMEM build cache) and builds rccl-tests with
  `-DBUILD_TESTS=ON -DBUILD_GPU_FUNCTIONAL_TESTS=ON`, then runs `ctest -L unit` during
  `docker build` — a regression in the host policy fails the image build instead of surfacing on
  the GPU gate.
- **Gate-script preflight (BC-UT).** `gin-sdma-{bcast,ag,a2a}-test.bash` run the baked
  `gin_sdma_policy_test` (GPU-free) as a hard preflight before the GPU sections
  (`RUN_POLICY_UT=0` to skip).
- **Validated end-to-end (8× MI355X, gfx950, 2026-07-27):** full `docker-gin-gda-sdma-build.bash`
  rebuild passed the in-build unit gate (`100% tests passed, 0 failed out of 1`), the post-build
  GIN Anvil-SDMA smoke assert (`NCCL_GIN_TYPE=6`, ~8.1 GB/s @1 MiB), and the baked binary running
  29/29 green via the BC-UT preflight path.

> **Build-hardening tie-in (2026-07-24):** the image build now runs a post-build GIN
> Anvil-SDMA smoke assert (`RCCL_IMAGE_GIN_SMOKE` in `docker-gin-gda-sdma-build.bash`,
> currently exercising the A2A Test#5 with `NCCL_GIN_TYPE=6`). When
> `gin-anvil-broadcast-test.bash` lands, add a BC-C2 bring-up case to that assert so a
> broken image (communicator `ginType=NONE`) fails the build instead of surfacing later
> as a broadcast test error.
>
> **Canonical image bake-in (2026-07-27):** the final `broadcast.cu` (SCATTER_AG_MIN
> default 2 MiB) **plus the GIN-SDMA host policy unit tests** are baked into
> `rccl-gin-gda-sdma-713:latest`. The Dockerfile now installs `libgtest-dev` (dedicated late
> layer, preserving the RCCL/rocSHMEM build cache), builds rccl-tests with
> `-DBUILD_TESTS=ON -DBUILD_GPU_FUNCTIONAL_TESTS=ON`, and runs `ctest -L unit` at `docker build`
> time as a hard gate. A full `docker-gin-gda-sdma-build.bash` rebuild on 8× MI355X (gfx950)
> passed end-to-end: the in-build unit gate (`100% tests passed, 0 failed`), the post-build GIN
> Anvil-SDMA smoke assert (`NCCL_GIN_TYPE=6`, ~8.1 GB/s @1 MiB), and the baked `gin_sdma_policy_test`
> running 29/29 green via the gate-script preflight. Earlier fresh-container default tier selection
> (no env override) was also confirmed: 1 MiB 23.8 (flat) → 2 MiB 35.6 (SAG) → 4 MiB 57.2 →
> 128 MiB 216.6, `#wrong==0`. Note: a **no-cache** from-scratch build can still hit Docker Hub's
> anonymous pull limit on `FROM ubuntu:24.04`; use cached base layers or `docker login` if it recurs.

---

## 7. Implementation checklist (when coding)

| File | Change |
|------|--------|
| `projects/rccl-tests/src/broadcast.cu` | `GinHybridBroadcastKernel` (-D 3), `BroadcastGetSdmaThreshold`, `BroadcastLsaDirect`, `BroadcastGetDevCommRequirements`, `BroadcastRunColl` case 3 |
| `projects/rccl-tests/src/broadcast.cu` (§4.8) | **DONE & validated:** `GinScatterAllgatherBroadcastKernel` (scatter signal 0 + in-place allgather signal 1, entry barrier only), host gate in `BroadcastRunColl` case 3 via `NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES` (**default 2 MiB**, 0=disable), `ginSignalCount>=2` in `BroadcastGetDevCommRequirements`. Crossover measured (§4.8.5); robust across `-V 8..32` and SDMA channels. **Optional future:** chunked scatter/allgather pipeline for ≥512 MiB (§4.8.5), factor AllGather body into a shared device helper (§8 Q8) |
| `ddai-artifacts/scripts/gin-sdma-bcast-test.bash` (§4.8) | **DONE (prototype):** `SCATTER_AG_MIN` env → `NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES` passthrough on BC-C2 (0 forces flat path for A/B) |
| `projects/rccl-tests/src/CMakeLists.txt` | `broadcast_perf` links rocSHMEM (if needed, mirror `all_gather_perf`); adds `gin_sdma_collective_policy.h` to the hipify `COMMON_FILES` list |
| `projects/rccl-tests/src/gin_sdma_collective_policy.h` | **DONE (2026-07-27):** new shared `__host__ __device__` policy header — the single source of truth for tier / threshold / chunk / device-requirement logic used by all three `.cu` kernels and `common.h` (which now delegates to it) |
| `projects/rccl-tests/test/{gin_sdma_policy_test.cpp, gin_sdma_gpu_functional.sh, gin_sdma_coverage.sh, CMakeLists.txt}` + top-level `CMakeLists.txt` `BUILD_TESTS` | **DONE (2026-07-27):** host GoogleTest suite (100 % line/branch, ctest label `unit`), opt-in GPU functional ctest matrix (label `gpu_functional`), and the `gin_sdma_coverage` gcov/llvm-cov target. See §6.1 |
| `ddai-artifacts/docker/Dockerfile-rccl-gin-gda-sdma` | **DONE (2026-07-27):** `libgtest-dev` (late layer), `-DBUILD_TESTS=ON -DBUILD_GPU_FUNCTIONAL_TESTS=ON`, build-time `ctest -L unit` gate |
| `ddai-artifacts/scripts/gin-sdma-{bcast,ag,a2a}-test.bash` | **DONE (2026-07-27):** GPU-free `gin_sdma_policy_test` preflight (BC-UT, hard gate, `RUN_POLICY_UT=0` to skip) |
| `gin-anvil-broadcast-test.bash` | Gate harness (BC-C1/C2/C3). **Implemented as `gin-sdma-bcast-test.bash`** |
| `ddai-artifacts/scripts/docker-gin-gda-sdma-build.bash` | Extend the post-build GIN smoke assert to cover BC-C2 (see §6 build-hardening tie-in) |

No changes to `gin_plugin_anvil_sdma.cc` or device templates unless a broadcast-specific bug is found.

---

## 8. Open questions

**Resolved in the 2026-07-23 review (see §0):**

- ~~**Integration target**~~ — v1 stays **`broadcast_perf -D 3` only** (standalone kernel +
  gate first); production `ncclBroadcast` / `rcclDirectBroadcast` wiring is deferred.
- ~~**Algorithm**~~ — **flat / star fan-out** confirmed (not ring, not tree); see §3.1.1.
- ~~**Initiation / scope / fan-out mechanic**~~ — device-initiated GIN kernel, single-node
  intra-node LSA fan-out, loop `gin.put`, no backend changes.

**Still open:**

1. **Design doc location** — Is there a Confluence/JIRA doc titled “RCCL host initiated broadcast design” that should be linked or merged into this plan? (The non-GIN host-initiated reference we are leveraging is `ncclHierCeAllGather` Phase 4 in `ce_coll.cc`.)

2. **Root sweep** — Must the gate pass for **every root rank** (0..N−1) on MI355X, or is root=0 sufficient for v1?

3. **Pull-broadcast** — Any interest in a follow-on **non-root `gin.get` from root** variant (rocSHMEM-style) for comparison, or root-put only?

4. **Branch placement** — Same branch as `users/dondai/gin-stage2b-sdma-ag-wip`, or wait until PR-7826 merges?

5. **Cross-check** — Include **rocSHMEM `team_broadcast`** in BC-C3 (like AllGather’s fcollect), knowing it requires `rocshmem_init` and can contaminate GPU state?

6. **Inter-node extension** — When the deferred rail stage is added (§4.7), confirm the
  cross-node schedule is a **k-ary/binomial tree** with a flat fan-out within each node.

7. ~~**Scatter+AG crossover (§4.8)**~~ — **Resolved (2026-07-27, §4.8.5):** crossover measured;
  `SCATTER_AG_MIN` default set to **2 MiB**. Plain scatter + linear allgather is at the AllGather
  roofline near the elbow (≤128 MiB: 212 vs 204) and is the right v1. The "~388 ceiling" is a
  *different operating point* (1 GiB per-rank allgather); SAG plateaus at ~229 for ≥256 MiB.
  **Still open (deferred):** a chunked/pipelined scatter⇄allgather kernel to recover ~1.6× at
  ≥512 MiB — worth it only if very-large broadcasts matter. Awaiting go/no-go.

8. **Reuse mechanics** — Can `GinHybridAllGatherKernel`'s body be factored into a
  `__device__` helper callable from both `all_gather.cu` and `broadcast.cu` (shared header),
  or should broadcast inline an in-place AllGather to avoid a cross-TU device dependency?

---

## 9. References

- GIN Anvil SDMA backend design: `pr7826-cleanup/docs/gin-anvil-sdma-backend-design.md`
- GIN Anvil SDMA detailed design: `pr7826-cleanup/docs/gin-anvil-sdma-detailed-design.md`
- AllGather design (pattern to follow): `allgather/gin-anvil-sdma-allgather-design-plan.md`
- AlltoAll design notes: `design-notes/gin-anvil-alltoall-design-notes.md`
- RCCL ring broadcast kernel: `projects/rccl/src/device/broadcast.h`
- RCCL host API: `projects/rccl/src/collectives.cc` (`ncclBroadcast`)
- PR: [ROCm/rocm-systems #7826](https://github.com/ROCm/rocm-systems/pull/7826)

---

*Internal AMD planning document. Not committed to the repository.*
