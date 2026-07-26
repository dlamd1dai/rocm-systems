# GIN-SDMA Broadcast on xGMI: Design Proposal

**Document type:** Internal AMD planning
**Status:** Implemented & validated in `broadcast_perf -D 3` (rev. 2026-07-24) — see §3.2/§4.4 sync correction. Host baseline BC-C1 (`-D 0`) now runs the full size range (64M cap removed) since the image ships the Broadcast **and** AllGather host ring kernels.
**Date:** 2026-07-22 (rev. 2026-07-24)
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

## 3. Proposed GIN-SDMA broadcast: `GinHybridBroadcastKernel`

Follow the **settled AllGather hybrid** (`GinHybridAllGatherKernel`): no transport/plugin changes, only a new collective kernel in `broadcast.cu`.

### 3.1 High-level algorithm

On single-node xGMI with `NCCL_GIN_TYPE=6`:

| Message size | Active rank(s) | Path |
|--------------|----------------|------|
| ≤ broadcast LSA↔GIN threshold (**256 KiB default**, see §4.3.1) | **root only** | LSA direct stores to every peer’s `recvbuff` |
| > threshold | **root only** | One-round **`gin.put` to all non-self peers**; Anvil picks IPC (≤ threshold) or SDMA (> threshold) |
| all ranks | all | Entry world/LSA barrier; **non-roots complete via receiver-side `waitSignal(+1)`** (see §4.4) |

> **Validated 2026-07-24 (8× MI355X, `NCCL_GIN_TYPE=6`, `-V 8`):** `#wrong == 0` for
> every root 0..7, 128 B–128 MB, in-place and OOP, at both `THRESHOLD=128` (hybrid) and
> `THRESHOLD=0` (all-SDMA, BC-D4). Perf @128 MB: in-place ~60 GB/s, OOP ~41 GB/s.

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

`broadcast.cu`, `all_gather.cu`, and `alltoall.cu` resolve their kernel branch threshold host-side with the chain (`testResolveSdmaThreshold`, `common.h`):

1. the collective-specific env var, if set;
2. the shared `NCCL_GIN_ANVIL_SDMA_THRESHOLD`, if explicitly set (keeps the global force knob — e.g. **BC-D4** `THRESHOLD=0` → all-GIN — working);
3. the collective's data-driven **default**.

| Collective | Env var | Compared against | Default | Basis (8× MI355X, `NCCL_GIN_TYPE=6`) |
|---|---|---|---|---|
| Broadcast | `NCCL_GIN_ANVIL_SDMA_THRESHOLD_BROADCAST` | full `msgBytes` | **262144 (256 KiB)** | (2026-07-24) LSA wins ≤256K (256K: 22.8 µs vs GIN 29.9); GIN wins ≥512K (512K: 34.0 µs vs LSA 39.2; 128M: 60.6 vs 17.0 GB/s) |
| AllGather | `NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER` | per-rank `chunkBytes` (= total/nRanks) | **262144 (256 KiB/rank)** | (2026-07-24) LSA wins ≤256K/rank i.e. ≤2M total (2M: 22.6 µs vs GIN 32.3); GIN wins ≥512K/rank i.e. ≥4M total (4M: 36.5 µs vs LSA 38.0; 128M: 388 vs 129 GB/s) |
| AlltoAll (`-D 3`) | `NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALL` | per-peer `chunkBytes` (= total/nRanks) | **262144 (256 KiB/peer)** | (2026-07-26, `-V 8`) LSA wins ≤256K/peer i.e. ≤2M total (small-msg 11 µs vs GIN 24.5; 2M busbw 61.7 vs 57.8 GB/s); GIN/SDMA wins ≥512K/peer i.e. ≥4M total (4M: 102.9 vs 44.3; 128M: 389.6 vs 56.4 GB/s) |

Values are bytes with an optional `K`/`M`/`G` suffix. The Anvil backend is unchanged (no GIN ABI change): the per-collective value is read in `RunColl` and passed to the kernel as an extra launch argument (`testLaunchDeviceKernelThreshold`).

**AlltoAll `-D 3` is a size-hybrid** (`GinHybridAlltoAllKernel`): a per-peer chunk ≤ threshold uses a direct all-peers **LSA** copy across all CTAs (latency-optimal for small, ~2× mid-range busbw); above the threshold it uses all-peers **GIN puts** over SDMA copy engines (bandwidth-optimal, scaling to ~390 GB/s @128M). The LSA branch needs enough CTAs, so Test#5 runs `-V 8` (SDMA is CTA-insensitive: `-V 1` ≈ `-V 8` at ~390 GB/s). This supersedes the earlier topology-only note and the shell-level `-D 4`/`-D 3` two-phase split; `-D 4` (`HybridAlltoAllKernel`, topology-based intra-node LSA vs inter-node GIN) is retained for debugging.

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
| ≤128 B | ~7–8 µs (LSA), similar to AllGather LSA | ~7.2 µs @128 B (LSA); ~23 µs when forced to GIN (`THRESHOLD=0`) |
| Medium | GIN setup floor ~21–22 µs | ~23 µs @1–8 KB |
| 128 MiB | Root fan-out (N−1) puts; busbw ≈ `(N-1)/N × link_BW` | **in-place ~60 GB/s**, **OOP ~41 GB/s** (OOP bounded by the root's local send→recv copy; 16B-vectorized) |

> The OOP < in-place gap is the extra root-local `send → recv` copy that in-place skips.
> Vectorizing that copy (16-byte stores) lifted OOP from ~24 → ~41 GB/s @128 MB with `-V 8`.
> Further OOP gains would come from overlapping the local copy with the remote puts.

Bus bandwidth formula for the test harness:

```cpp
*busBw = algBw * (double)(nranks - 1) / (double)nranks;
```

(one copy of the message delivered to N−1 receivers)

### 4.7 Scope boundaries

- **Single-node only** — GIN Anvil SDMA is intra-node xGMI (backend enforces `nNodes == 1`).
- **rccl-tests first** — same staging as AllGather: `-D 3` in `broadcast.cu`, gate script, then optional production `ncclBroadcast` direct path later.
- **No ring/tree variants** — consistent with AllGather v1 decision table.

---

## 5. Architecture sketch

```text
  broadcast_perf (-D 3, NCCL_GIN_TYPE=6, -V 8)
           │
  GinHybridBroadcastKernel (all ranks launch)
           │
     rank == root?
           │
     ┌─────┴──────────────────────────────┐
     │ msgBytes ≤ threshold               │ msgBytes > threshold
     ▼                                    ▼
  BroadcastLsaDirect                   gin.put × (N−1)
  (LSA peer stores)                         │
                                       IPC or SDMA
                                            │
                                     xGMI P2P mesh
           │
     non-root ranks: barriers only
```

---

## 6. Proposed test gate (`gin-anvil-broadcast-test.bash`)

| ID | Test | Config |
|----|------|--------|
| **BC-C1** | Host `ncclBroadcast` | `-D 0`, 128 B–`MAX_BYTES` (full range, no 64M cap — host broadcast validated to 256M on MI355X), `NCCL_CUMEM_ENABLE=0` — perf reference, **on by default**. Requires the broadcast host ring kernels, which the image now ships via `ONLY_FUNCS="…|Broadcast|AllGather"` (Dockerfile ARG + `docker-gin-gda-sdma-build.bash`). Earlier images omitted them (`ncclDevFuncId … not found for coll:0`); unlike AllGather, broadcast has no direct/copy fast path so it always needs the ring device func. The sibling AllGather gate (`gin-sdma-ag-test.bash`, AG-C1) had the same 64M host cap for the same reason and has likewise been lifted now that `AllGather` is in `ONLY_FUNCS`. |
| **BC-C2** | GIN hybrid | `-D 3 -V 8`, 128 B–128 M, `NCCL_GIN_TYPE=6` |
| **BC-C3** | rocSHMEM `team_broadcast` | optional cross-check |
| **BC-D4** | All-SDMA gate | `THRESHOLD=0` |
| **BC-P1** | Perf vs host @ 128 M, all roots | |

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

> **Build-hardening tie-in (2026-07-24):** the image build now runs a post-build GIN
> Anvil-SDMA smoke assert (`RCCL_IMAGE_GIN_SMOKE` in `docker-gin-gda-sdma-build.bash`,
> currently exercising the A2A Test#5 with `NCCL_GIN_TYPE=6`). When
> `gin-anvil-broadcast-test.bash` lands, add a BC-C2 bring-up case to that assert so a
> broken image (communicator `ginType=NONE`) fails the build instead of surfacing later
> as a broadcast test error.

---

## 7. Implementation checklist (when coding)

| File | Change |
|------|--------|
| `projects/rccl-tests/src/broadcast.cu` | `GinHybridBroadcastKernel` (-D 3), `BroadcastGetSdmaThreshold`, `BroadcastLsaDirect`, `BroadcastGetDevCommRequirements`, `BroadcastRunColl` case 3 |
| `projects/rccl-tests/src/CMakeLists.txt` | `broadcast_perf` links rocSHMEM (if needed, mirror `all_gather_perf`) |
| `gin-anvil-broadcast-test.bash` | Gate harness (BC-C1/C2/C3) |
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
