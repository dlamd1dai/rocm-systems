# GIN-SDMA Collectives Expansion: Design & Implementation Plan

**Document type:** Internal AMD design/implementation plan
**Status:** **In progress** — extends the implemented GIN-SDMA collectives (AllToAll, AllGather,
Broadcast) to the remaining single-node collectives. No RCCL backend / GIN-ABI changes.
**P1 (SendRecv, Scatter, Gather `-D 3`), P2 (ReduceScatter `-D 3`), P3 (AllReduce `-D 5`/`-D 6`)
and P4 (Reduce `-D 3`) have all landed** (see §3.1–3.3, §3.5–3.7, marked IMPLEMENTED).
**Only P5 (AllToAllv) remains** — currently WIP on `users/dondai/gin-stage2j-sdma-A1457-a2av-wip`;
the design (§3.4, §4.5) is resolved and it is the last `-D 3` collective in the expansion.
**As-built note:** the AllReduce and Reduce large tiers deviated from the original §3.6/§3.7
proposals during the perf campaign (AllReduce keeps a one-shot small tier but in a hang-safe
GIN-barrier'd form, not the originally-planned no-GIN version; Reduce's large tier became an
edge-disjoint multi-ring, not RS+Gather). The detailed campaigns live in
[gin-sdma-perf-optimization-plan.md](gin-sdma-perf-optimization-plan.md) (§9.x); the as-built notes
below are the authoritative summary.
**Date:** 2026-07-27 (updated 2026-08-05)
**Scope:** `projects/rccl-tests/` device kernels (`-D 3`, `NCCL_GIN_TYPE=6`), the shared
`gin_sdma_collective_policy.h`, host policy unit tests, and per-collective gate scripts.
**Related:**
- `ddai-artifacts/docs/gin-anvil-sdma-broadcast-design-plan.md` (the master pattern this follows)
- `ddai-artifacts/docs/allgather-host-sdma-vs-gin-sdma-benchmark.md`
- PR #7826 (Anvil SDMA backend)

---

## 1. Background: the "GIN-SDMA approach" as a reusable pattern

The three implemented collectives (AllToAll, AllGather, Broadcast) are **device-initiated `-D 3`
kernels in rccl-tests** that drive the **GIN Anvil-SDMA backend** (`NCCL_GIN_TYPE=6`) over
intra-node xGMI. They are not in the RCCL library; they share one pure policy header and a small,
uniform primitive toolkit. This plan reuses that toolkit verbatim.

### 1.1 The primitive toolkit (unchanged, reused by every new collective)

| Primitive | Role | Notes |
|---|---|---|
| `gin.put(team, peer, dstWin, dstOff, srcWin, srcOff, bytes, SignalInc{idx})` | peer→peer copy | Backend picks **SDMA copy engine** when `bytes > sdmaThreshold` (default 128 B; per-`(peer,channel)` queue ⇒ concurrent), else an **SM peer-VA store** (`ipcPut`). `SignalInc` is a **receiver-side** action (increments the *receiver's* signal). |
| `gin.waitSignal(cta, idx, base + k)` | completion | Receiver-side: wait for `k` puts to land. |
| `gin.flush(cta)` | source reuse | Quiesces local SDMA queues. Does **not** imply remote landing. |
| `ncclGetLsaPointer` / `ncclGetLocalPointer` | peer VA | SM-driven LSA copies **and reductions** (small/medium tiers). |
| `ncclLsaBarrierSession` / `ncclBarrierSession(World)` | entry barrier | Ensures recvbuf is past `initData`'s memset before peers write it. |
| `ncclLLA2ASession` (`bcast`/`send`/`recv`) | tiny-message LL | Epoch-tagged scratch; barrier-light. |

### 1.2 The structural pattern (identical across all three, and adopted here)

1. **Tier ladder by message size:** LL (tiny) → LSA (small/med) → GIN/SDMA (large).
2. **Entry barrier only** (world barrier for GIN tiers, LSA barrier for LSA/LL tiers).
3. **Receiver-side signal completion, no exit barrier**; a single `flush`.
4. **Symmetric vs asymmetric completion:** AllGather/AllToAll are symmetric (each rank puts N,
   receives N ⇒ `waitSignal(base+N)`); Broadcast is asymmetric (root puts N−1; non-roots
   `waitSignal(+1)`); the large Broadcast tier composes **scatter + allgather** with **two signal
   indices** to break the root-egress ceiling.
5. **Single source of truth:** all tier/threshold/chunk/requirement logic lives in
   `gin_sdma_collective_policy.h` (`__host__ __device__`, unit-tested to 100 % branch coverage).
6. **No backend/ABI change.** All work is in `rccl-tests` + policy header + gate scripts.

### 1.3 The one hard constraint: SDMA moves bytes, it does not compute

The SDMA copy engines and LSA peer-VA stores **only move data**. Reductions must be performed by
the **SM** — either as a *read-reduce over LSA peer VA* (small/medium) or as a *`gin.put` of
partials into scratch followed by an SM reduce* (large). This is the single new ingredient the
reduction collectives add; the movement machinery is otherwise identical. (Confirmed acceptable
in the 2026-07-27 scoping review.)

---

## 2. Collective inventory and fit

| Collective | Class | gin-sdma today | Fit | Core shape |
|---|---|---|---|---|
| AllToAll | movement | ✅ | — | per-peer put loop, symmetric |
| AllGather | movement | ✅ | — | own-slot put to all peers, symmetric |
| Broadcast | movement | ✅ | — | root fan-out; large = scatter+allgather |
| **Scatter** | movement | ✅ (P1, `-D 3`) | **excellent** | root puts distinct chunk to each rank; LL→LSA(interleaved)→GIN |
| **Gather** | movement | ✅ (P1, `-D 3`) | **excellent** | each rank puts its chunk to root's slot |
| **SendRecv** | movement | ✅ (P1, `-D 3`) | **excellent** | single put + waitSignal |
| **AllToAllv** | movement | ❌ (P5, WIP) | **good** | AllToAll put loop with per-peer displacements/counts |
| **ReduceScatter** | reduction | ✅ (P2, `-D 3`) | **excellent** | single-tier direct LSA read-reduce; adaptive load schedule (peer-unroll grid-stride / pack+peer-unroll+prefetch). No scratch. See [reducescatter-gin-sdma-phase2.md](reducescatter-gin-sdma-phase2.md) |
| **AllReduce** | reduction | ✅ (P3, `-D 5`/`-D 6`) | — | RS (LSA read-reduce) + in-place AllGather; single-launch (`-D 5`, grid barrier) or two-launch (`-D 6`). Threshold selects the AG tier (LSA store-gather / SDMA put). Tiny-OOP one-shot tier kept (GIN-barrier'd, hang-safe). See §3.6 |
| **Reduce** | reduction | ✅ (P4, `-D 3`) | — | small = reduce-scatter-to-root LSA read-reduce (no scratch); large OOP = edge-disjoint multi-ring (not RS+Gather). See §3.7 |
| Barrier | control | (have sessions) | trivial | reuse barrier session |

**Out of scope / poor fit:** multi-node scaling (backend is single-node xGMI only); custom
non-standard reduction ops.

---

## 3. Design per collective

Every kernel is `GinHybrid<Coll>Kernel` in the matching `.cu`, dispatched from `<Coll>RunColl`
case 3, with a `<Coll>GetDevCommRequirements` case 3, and a threshold resolved host-side via
`gin_sdma::resolveThreshold(...)` (env `NCCL_GIN_ANVIL_SDMA_THRESHOLD_<COLL>` → shared
`…_THRESHOLD` → default 256 KiB). Small tiers reuse `ncclGetLsaPointer`; large tiers use the
`gin.put`/`waitSignal`/`flush` triple. All keep the **entry-barrier-only** sync model.

### 3.1 Scatter (`scatter.cu`) — Tier A — **IMPLEMENTED (`GinScatterKernel`, 2026-07-29)**

Root sends a distinct chunk `r` to rank `r` (elements `[r*count .. (r+1)*count)` of its send
buffer). The **as-built** kernel is a three-tier ladder keyed on the **per-rank chunk** bytes
(`chunk = total/N`), not a two-tier small/large split. All tiers keep the entry-barrier-only model.

- **Tiny — LL (≤ `NCCL_GIN_ANVIL_SCATTER_LL_MAX_BYTES`, default 2 KiB/chunk; 64 KiB compile
  ceiling):** single-CTA packed data+flag path. The root writes each peer's distinct chunk into
  that peer's epoch-tagged LL scratch — **including its own** so every rank uniformly polls its own
  scratch into `recvbuff`. The `(peer, slot)` fan-out is flattened **peer-major** (`r = w % N`) so
  consecutive threads target distinct peers and all xGMI egress links fire concurrently in one burst
  (the old peer-outer nesting serialized the root's ~N stores and dominated LL latency). A single
  **entry** LSA barrier bounds root-vs-laggard epoch skew (only the root writes, so the ring lacks
  the mutual-recv backpressure AllGather has); the recv's epoch tag replaces the exit barrier, and
  each rank writes only its own `recvbuff` ⇒ immune to the `initData` recvbuff-memset race. Each
  receiver takes one chunk, so it needs only `chunk/8` u64 slots (Broadcast/SendRecv sizing, not the
  `N*…` AllGather form). `NCCL_GIN_ANVIL_SCATTER_LL_MAX_BYTES=0` disables it.
- **Small/med — LSA (chunk ≤ threshold):** root SM-stores each peer's chunk over xGMI, entry+exit
  LSA barrier. Because **only the root writes**, its fan-out *layout* decides link utilization, so
  the store loop is **peer-interleaved by default** (`ScatterLsaFanout`,
  `NCCL_GIN_ANVIL_SCATTER_LSA_INTERLEAVE`, default on): map each CTA to a peer-slot
  (`blockIdx % min(gridDim, N)`) so all peers' links run concurrently, with extra CTAs splitting a
  peer's chunk for more threads/link. `INTERLEAVE=0` selects the historical sequential loop (all SMs
  drive one link at a time) for A/B.
- **Large — GIN/SDMA (chunk > threshold):** root issues one `gin.put` **per non-self peer**
  (`SignalInc{0}`) and does its own slice with a local copy; non-roots `waitSignal(base+1)`; root
  `flush`es. Asymmetric completion (same as flat Broadcast, distinct source offsets). Each put is
  **chunked to ≤1 GiB** segments (the signal rides the final segment) to avoid the 30-bit SDMA
  copy-count overflow on >1 GiB per-rank chunks.
  - **Settled fan-out (2026-07-29, 8× MI355X):** the flat one-put-per-peer shape is optimal as-is.
    The backend routes each peer's put to its own per-peer queue (`handles[r*numChannels+ch]`), so
    the N−1 copies already run concurrently on independent SDMA engines. **No interleave knob on this
    tier:** neither extra SDMA channels (`NUM_CHANNELS`=1/2/4 measured flat, ≥2 also deadlocks) nor
    slicing each peer's chunk into 2/4 sub-puts helps — a single put already saturates its per-peer
    xGMI link, and segmentation regressed 15–40 % from added SDMA descriptor overhead.
- **Threshold (`kScatterSdmaThresholdDefault` = 128 KiB/chunk):** LSA is **root-egress-bound** (the
  root alone stores all N−1 peer chunks, ~64 GB/s ceiling). Measured crossover on 8× MI355X
  (2026-07-27) sits between a 128 KiB chunk (LSA still wins) and 256 KiB (GIN wins; 512 MiB total:
  390 vs 64 GB/s). The **128 KiB default is the threshold, not the crossover** — the largest chunk
  LSA still wins — so ≤128 KiB routes to LSA and ≥256 KiB to GIN. Override with
  `NCCL_GIN_ANVIL_SDMA_THRESHOLD_SCATTER` or the shared `…_THRESHOLD`.
- **In-place:** detected by `sendwin == recvwin`; rank `r`'s recv slot is `base + r*chunk` and the
  root's own chunk is already in place. The root reconstructs the shared base as
  `recvoffset − rank*chunk`.
- **DevComm:** `barrier = lsaBarrier = ginSignal = deviceCtaCount`, `needsGin`, plus a single-CTA LL
  scratch requirement sized from the LL cap (bypassed when `nSlots==0`).

### 3.2 Gather (`gather.cu`) — Tier A — **IMPLEMENTED (`GinGatherKernel`, P1)**

Inverse of Scatter: each rank `r` puts its chunk into root's `recvbuff[r*chunk]`.

- **Small:** LSA — each rank SM-stores its chunk into `ncclGetLsaPointer(recvwin, …, root)`
  at slot `r*chunk`; entry+exit LSA barrier.
- **Large:** each non-root `gin.put`s to root (`SignalInc{0}`); root does its own slice locally;
  **root** `waitSignal(base + N−1)` (root is the sole receiver); non-roots `flush`.
- **DevComm:** as Scatter.

### 3.3 SendRecv (`sendrecv.cu`) — Tier A — **IMPLEMENTED (`GinSendRecvKernel`, P1; removed the old `testNotImplemented`)**

Point-to-point rank pairing (the rccl-tests SendRecv pairs rank `r` with `r ± nRanks/2`).

- **All sizes:** sender `gin.put`s to its peer (`SignalInc{0}`) then `flush`es; receiver
  `waitSignal(base+1)`. Small tier optionally uses a direct LSA store. No reduction, foundational.
- **DevComm:** `barrier = ginSignal = deviceCtaCount`, `needsGin`.

### 3.4 AllToAllv (`alltoallv.cu`) — Tier A (metadata plumbing RESOLVED, see §4.5)

Variable-count AllToAll: rank `r` sends `M[r][p]` elements to peer `p`. Same put loop as
`GinHybridAlltoAllKernel`, but per-peer sizes and offsets vary.

**Key enabler:** the test's size matrix `M` is generated **deterministically and identically on
every rank** (`AlltoAllvGenSizeMatrix`, seeded by env; `alltoallv.cu:32`). So each rank already
knows the *entire* matrix and can compute, with no exchange, everything the kernel needs — in
particular the **receiver-side landing offset** on each peer (the column prefix sum), which the
current host path does not build.

**Per-rank device metadata (3 arrays of length N, byte units, + 1 scalar):**
- `sendBytes[p]  = M[rank][p] * eltSize`
- `srcByteOff[p] = eltSize · Σ_{d<p} M[rank][d]`   (sender-side; the existing `sdispls`)
- `dstByteOff[p] = eltSize · Σ_{s<rank} M[s][p]`    (**receiver-side, column-p prefix — NEW**)
- `nIncoming     = #{ s : M[s][rank] ≠ 0 }`         (expected non-zero puts received)

**Kernel (large tier):**
```cpp
for (int p = tid; p < N; p += nthreads) {
  if (sendBytes[p] == 0) continue;                 // skip empty pairs
  gin.put(world, p, recvwin, recvoffset + dstByteOff[p],
                    sendwin, sendoffset + srcByteOff[p],
                    sendBytes[p], ncclGin_SignalInc{0});
}
gin.waitSignal(cta, 0, base + nIncoming);           // only non-zero senders signal us
gin.flush(cta);
```
Small tier: same offsets via `ncclGetLsaPointer` LSA copies (entry+exit LSA barrier). Byte offsets
are passed directly (no `sizeof(T)` multiply in-kernel), matching how the other kernels treat
`sendoffset`/`recvoffset` as byte offsets.

- **DevComm:** as AllToAll case 3 (`barrier = lsaBarrier = ginSignal = deviceCtaCount`, `needsGin`).
- **In-place:** unsupported by the A2Av test (`reportErrors = 0` in-place), same as AllToAll.

#### 3.4.1 As-built & measured (2026-08-05, 8× MI355X, gfx950, float, `-V 32`)

Two post-bring-up refinements landed after the P5 kernel passed its functional gate (commits
`e9eb2929b3` src, `87ca8f1006` test on `…-a2av-wip`):

1. **Grid-wide LSA scatter (was one-CTA-per-peer).** The first cut mapped one CTA to each peer
   (`for (p = blockIdx.x; p < N; p += gridDim.x)`), which left `gridDim − N` CTAs idle and pinned
   each peer chunk to a single 512-thread block. That capped the mid-band (128 K–512 K/peer nominal)
   at ~4 GB/s regardless of `-G` (a *parallelism* limit, not launch overhead). The LSA tier was
   rewritten to a **grid-wide cooperative copy** — the full grid drives every peer chunk via a
   grid-global `(tid, nthreads)`, visiting peers in a `(rank + blockIdx.x)`-rotated order (F2+F3) so
   distinct CTAs hit distinct peers and spread xGMI egress. This mirrors the host RING's
   multi-channel parallelism and the sibling A2A `AlltoAllVectorizedImpl`. Result: the 128 K–512 K
   band jumped ~2.7–4.8× and GIN-LSA now **beats** host RING through 256 KiB/peer.

2. **Measured LSA↔GIN threshold = 256 KiB/peer** (`kAllToAllvSdmaThresholdDefault = 262144`, now
   *measured*, previously an unmeasured placeholder — which happened to be correct). Nominal
   per-peer crossover sweep (all `#wrong==0`): LSA wins at 256 KiB/peer (39.6 vs 36.3 GB/s), GIN/SDMA
   takes over from 384 KiB/peer (47.5 vs 41.4; 512 KiB 50.7 vs 43.4; 768 KiB 61.2 vs 56.9). Convention
   is "largest per-peer chunk LSA still wins" (`bytes <= threshold → LSA`), i.e. 256 KiB.

**Host-vs-GIN envelope (out-of-place busbw, GB/s; GIN on `-G 4` graph-replay = device throughput,
HOST launch-inclusive; `best_host` was host-RING at every size, host-CE never won):**

| total size | best host | best GIN (tier) | gin/host | regime |
|---:|---:|---:|---:|:--|
| ≤ 256 K | 0.01–10.6 | 1.0–1.84× higher (LSA) | **1.4–1.84×** | GIN-LSA wins (grid-wide scatter + graph replay) |
| 512 K | 25.1 | 28.2 (LSA) | 1.12× | GIN edge |
| 1 M–2 M | 41.4–55.6 | 34.5–39.5 (LSA) | 0.71–0.83× | host RING ramps faster; GIN mid-band soft spot |
| 4 M–64 M | 67.8–87.5 | 52.8–84.2 (GIN) | 0.78→0.96× | GIN/SDMA closing |
| 128 M–1 G | ~87 | ~86–87.5 (GIN) | 0.99–1.01× | **parity at xGMI ceiling** |
| 2 G | 87.1 | 65.6 (GIN) | 0.76× | copy-engine dip (see below) |

**2 GB copy-engine dip (investigated, not a regression).** GIN busbw drops to 65.6 at 2 GB while
host-RING holds ~87. Reproducible across 3 runs (65.5/65.5/65.6), `#wrong==0`, no truncation/overflow
warnings. The put chunk is already the compile-time 1 GiB (`kGinPutMaxBytes`; `ginPutChunked`), so the
1 GiB SDMA descriptor boundary is *not* newly triggered. Key corroboration: the **host copy-engine
path (host-CE) also collapses at 2 GB (60.3 vs RING 87.1)**, and GIN-SDMA (65.6) actually *beats*
host-CE there — so both DMA/copy-engine paths sag at 2 GB while the SM-copy ring stays flat. This is a
**copy-engine throughput characteristic of this system at 2 GB transfers**, not a GIN-kernel or
chunk-size issue; the only levers are the SM-copy (LSA) tier or more SDMA channels
(`NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS`), not the put chunk size.

**Remaining opportunity:** the 512 K–2 M/rank mid-band (0.71–0.83× host) — GIN-LSA plateaus (~40 GB/s)
while the GIN/SDMA tier hasn't fully ramped. Candidate follow-ups: multi-channel SDMA in that band, or
a size-adaptive LSA CTA ladder like A2A's `a2aLsaCtaCount`.

### 3.5 ReduceScatter (`reduce_scatter.cu`) — Tier B (introduces SM reduce)

Rank `r` owns output slice `r` (of `M/N` elements) = sum over all ranks of their slice `r`.

- **Small/medium — LSA read-reduce (no scratch, no signals):** each rank reads its owned slice
  `[rank*chunk]` from **every** peer's `sendbuff` via `ncclGetLsaPointer`, accumulates in
  registers with the op, writes `recvbuff`. Egress is balanced (each rank reads N−1 remote slices
  of `M/N`). Only an entry+exit LSA barrier. This is the latency-optimal small path and needs no
  scratch window.
- **Large — put-partials + SM reduce (bandwidth-optimal):** each rank `gin.put`s the partial
  destined for rank `p` (its slice `p`, `M/N` bytes) into rank `p`'s **scratch window** at
  slot `[srcRank*chunk]` (`SignalInc{0}`); after `waitSignal(base + N−1)`, each rank SM-reduces
  the N contributions in its scratch (its own slice + N−1 received) into `recvbuff`; `flush`.
  This is the standard direct reduce-scatter, egress balanced across all N ranks.
- **Reduce op:** templated on `ncclRedOp_t` using rccl-tests' existing host reduction helpers'
  device analogue (add a small `Apply<op,T>` device functor; sum/prod/min/max/avg).
- **DevComm:** `barrier = lsaBarrier = ginSignal = deviceCtaCount`, `needsGin`, **plus a scratch
  window requirement** (~`M` bytes = `N * chunk`) for the large path. See §4.2.

### 3.6 AllReduce (`all_reduce.cu`) — Tier B — **IMPLEMENTED (`GinAllReduceKernel`, `-D 5`/`-D 6`, 2026-08-02)**

> **As-built (authoritative).** AllReduce ships as **deviceImpl 5 (single-launch, default) / 6
> (two-launch)**, not `-D 3` (`-D 1..4` are the upstream LSA/multimem demo kernels). **Every size**
> runs one composition: **RS (LSA read-reduce of the owned slice) → in-place AllGather**, sharing
> factored phase bodies (`arReduceScatterOwnedSlice` + `arAllGatherInPlace`). The two realizations
> differ only in the RS→AG sync:
> - **`-D 5` single-launch (default):** one kernel; a hand-rolled in-kernel device-wide barrier
>   (`arGridBarrier`) separates RS and AG. Halves host launches but busy-spins, so it **requires all
>   CTAs co-resident** (fine for these small grids). `cg::this_grid()` SIGSEGVs on this ROCm build
>   (issue #2805), hence the hand-rolled barrier.
> - **`-D 6` two-launch (fallback):** `GinAllReduceRSKernel` then `GinAllReduceAGKernel` back-to-back
>   on the same stream — the launch boundary is the global sync, no co-residency requirement.
>
> **Threshold selects the AllGather TIER (total bytes), default 16 MiB (`NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLREDUCE`, measured 2026-08-02):**
> small = **LSA store-gather** (`arAllGatherInPlaceLsa`, each rank stores its reduced slice into
> peers' recvbuf); large = **SDMA AllGather** (put + `waitSignal`). Both keep **exactly one GIN world
> barrier per launch** so the GIN completion cadence stays uniform.
>
> **The small one-shot tier (Q2) ships — but in a HANG-SAFE form.** The originally-planned *no-GIN*
> one-shot was first prototyped and dropped (interleaving no-GIN LSA-only launches between GIN
> launches deterministically re-triggered a cumulative GPU hang — a backend GIN signal-completion
> pipeline disturbance). It was then **re-introduced** as `arAllReduceOneShotLsa` for **tiny
> out-of-place** messages (< `NCCL_GIN_ANVIL_ONESHOT_THRESHOLD_ALLREDUCE`, default 256 KiB): it now
> performs **exactly one GIN world barrier** itself, so the per-launch GIN cadence stays uniform (the
> property the no-GIN version violated), and it collapses RS + grid barrier + AllGather into that one
> barrier — the latency floor for tiny messages. In-place tiny messages stay on the RS+AG LSA tier (a
> one-shot would race sendbuf reads vs recvbuf writes). Two-shot grid capped to
> `kAllReduceTwoShotMaxCtas = 16` to avoid deadlock at high `-V`. Op/type via `SPECIALIZE_REDUCE_KERNEL`;
> fp8 `{prod, avg, mulsum}` excluded → `testNotImplemented`.

*Original proposal (superseded in part by the as-built note above), kept for rationale:*

Two-tier hybrid (the standard **one-shot vs two-shot** all-reduce split): a single-phase direct
LSA reduce for small, and the bandwidth-optimal **ReduceScatter + AllGather** composition for
large (mirroring the SAG broadcast composition with two signal indices).

**Q2 — small-message tier decision (RESOLVED): add the direct one-shot tier, default on.**
Unlike the AllToAll LL result (marginal, off — broadcast doc §4.3.3), the alternative here is a
*structural* penalty: RS+AG is two phases (two barrier/signal rounds + a scratch round-trip), so
for latency/barrier-bound small messages (the ~7→~22 µs LSA→GIN setup cliff, broadcast doc §4.3)
running it unconditionally roughly **doubles** the fixed overhead. A one-shot direct path is
single-phase, so it is ~half the floor at small sizes and needs no scratch or signals. This matches
the well-established one-shot(small)/two-shot(large) all-reduce split used by NCCL/RCCL.

- **Small — one-shot direct LSA reduce (single phase, default on):** every rank reads the full `M`
  from each peer's `sendbuf` via `ncclGetLsaPointer`, reduces in registers with `Apply<op,T>`,
  writes `recvbuf`. One entry+exit LSA barrier; no scratch, no signals. Cost `(N−1)·M` reads/rank —
  cheap while `M` is small; the LSA↔GIN threshold (measured, §5) is where two-shot takes over.
- **Tiny — LL sub-tier (OPTIONAL, default off):** an LL reduce-on-recv variant (each rank `bcast`s
  its contribution, then `recv`s all N contributions and reduces them). More code than AllGather-LL
  (adds the register reduce) and its win is unproven, so implement only if the small-message sweep
  shows a barrier-bound floor LL can remove — same disposition as AllToAll LL.
- **Large — two-phase, two signals:**
  1. **ReduceScatter phase (sig 0):** as §3.5 large — each rank ends holding the fully-reduced
     slice `rank` in `recvbuff[rank*chunk]` (or scratch).
  2. **AllGather phase (sig 1):** in-place allgather of the N reduced slices — reuse the
     `GinHybridAllGatherKernel` large-path put shape inline (`SignalInc{1}`,
     `waitSignal(base1 + N−1)`).
  Entry barrier only; the two distinct signals remove the inter-phase barrier (exactly as SAG
  broadcast). `flush` once at the end.
- **DevComm:** `barrier = lsaBarrier = deviceCtaCount`, `ginSignal = max(deviceCtaCount, 2)`
  (two-signal scheme), `needsGin`, scratch window (~`M` bytes).

### 3.7 Reduce (`reduce.cu`) — Tier B — **IMPLEMENTED (`GinReduceKernel`, `-D 3`, 2026-08)**

> **As-built (authoritative).**
> - **Small / default — reduce-scatter-to-root LSA read-reduce (`GinReduceKernel`, no scratch/
>   signals/GIN puts):** folds bit-for-bit identically to `GinReduceScatterKernel`; the only
>   difference is the write target (root's `recvbuff`, offset by the slice). Entry+exit LSA barrier
>   only, in-place safe (root is the sole reader and writer).
> - **Large OOP — edge-disjoint MULTI-RING (`GinReduceMultiRingKernel`), NOT the planned RS+Gather.**
>   The perf campaign found the RS+Gather large tier plateaued at the serial gather; the fix (same as
>   the broadcast large tier) is an edge-disjoint multi-ring: CTA `b` runs ring `b % nRings` on buffer
>   stripe `b`, so every GPU drives all its links every cycle. Default **ON for OOP totals ≥ 64 MiB**
>   (`reduceRingMinBytes()`, env-gated); CTA count self-selects (`reduceRingCtas()`, default 128,
>   decoupled from `-V` like the broadcast ring); pipeline depth via `reduceRingChunks()` /
>   `reduceRingAutoChunks`. A `GinReducePipelinedKernel` (SM-reduce ∥ SDMA-put) variant also exists.
>   Full campaign in [gin-sdma-perf-optimization-plan.md](gin-sdma-perf-optimization-plan.md) §9.5.
>
> Reduce's tier/threshold logic lives in `reduce.cu` (the `reduceRing*()` helpers), **not** in the
> shared `gin_sdma_collective_policy.h`. Op/type coverage as AllReduce (fp8 `{prod, avg, mulsum}`
> excluded).

*Original proposal (large tier superseded by the multi-ring as-built note above), kept for rationale:*

Reduce is the **exact dual of Broadcast** (root-*ingress* bound instead of root-egress). The tier
ladder mirrors the broadcast ladder.

**Q5 — large tier decision (RESOLVED): use ReduceScatter + Gather, not direct root-ingress.**
- A *direct root-ingress* reduce (non-roots push full `M` to root scratch, root sums `N` buffers)
  is capped at `busbw = B_root_ingress / N` — the dual of flat broadcast's `B_root_egress / N`
  ceiling (~60 GB/s @128 MiB) — and needs `(N−1)·M` of root scratch plus all-on-root reduce work.
- **RS + Gather** distributes the reduce across all ranks (RS egress balanced, ~ReduceScatter
  roofline) and then Gathers only the *reduced* result (`(N−1)/N·M` root ingress vs `(N−1)·M`), an
  ≈`(N−1)×` ingress cut — the same win SAG gives broadcast. It reuses the RS (§3.5) and Gather
  (§3.2) kernels already in the plan, so it is minimal marginal code. It plateaus at very large `M`
  where the serial gather dominates (mirror of SAG's plateau); a chunked/pipelined variant is
  possible future work, exactly as noted for SAG broadcast.
- The direct-*push* path is dominated at both ends, so it is **not** shipped.

- **Small — LSA pull-reduce to root (single phase, no scratch):** only the root reads the full `M`
  from every rank's `sendbuf` via `ncclGetLsaPointer`, reduces with `Apply<op,T>` into `recvbuff`;
  entry+exit LSA barrier. Latency-optimal; no scratch (pull avoids the root-scratch that a push
  reduce would need). Root-ingress bound but that is irrelevant while `M` is small.
- **Large — ReduceScatter + Gather-to-root:** RS phase (§3.5, sig 0) leaves each rank holding its
  fully-reduced slice; Gather phase (§3.2, sig 1) collects the `N` reduced slices to root. Two
  signal indices, entry barrier only, single `flush` (same structure as SAG broadcast / AllReduce).
- **DevComm:** `barrier = lsaBarrier = deviceCtaCount`, `ginSignal = max(deviceCtaCount, 2)`
  (two-phase), `needsGin`, scratch window (~`M` bytes) for the RS phase; the small pull tier uses
  no scratch/signals.

---

## 4. Cross-cutting work

### 4.1 Shared policy header (`gin_sdma_collective_policy.h`)

Add, mirroring the existing structure:
- Thresholds: `kScatter/kGather/kSendRecv/kAllToAllv/kReduceScatter/kAllReduce/kReduceSdmaThresholdDefault` (start at 256 KiB; retune per §5). `kAllToAllv` is now **measured** at 256 KiB/peer (§3.4.1).
- Tier predicates + enums: `scatterKernelTier`, `gatherKernelTier`, `reduceScatterKernelTier`,
  `allReduceKernelTier` (two-phase gate like `bcastUseScatterAllgather`), `reduceKernelTier`.
- `DevReqs`-style helpers per collective (extend the `a2aDevReqs` pattern), including the
  `ginSignalCount = max(deviceCtaCount, 2)` rule for AllReduce, and a `needsScratch`/scratch-size
  field for the reduction collectives.
- Chunk/offset math helpers (reuse `alignChunkCount`, `sagChunk`).

### 4.2 Scratch window for reductions (RESOLVED — the one new resource)

Large reduction paths need a registered scratch window of ~`M` bytes/rank (`N * chunk`). The
devComm API already provides exactly this via the **generic resource-buffer requirement**;
`ncclLLA2ACreateRequirement` is just a wrapper over it. **No new backend API is needed.**

**Mechanism.** `ncclDevResourceRequirements` (`projects/rccl/src/include/nccl_device/core.h:144`)
exposes `bufferSize`, `bufferAlign`, and `outBufferHandle`. Fill those directly, link the node into
`reqs->resourceRequirementsList`, and `ncclDevCommCreate` concatenates all requested bytes into a
single **symmetric window** `comm.resourceWindow` (created via `symWindowCreate` in
`dev_runtime.cc`), assigning `*outBufferHandle = byteOffset / 128` (`dev_runtime.cc:1680-1689`).
Effective granularity/min-alignment is **128 bytes** (request `bufferAlign >= 128`).

**It is GIN-put-addressable.** `comm.resourceWindow` is a first-class `ncclWindow_t`
(`impl/comm__types.h`), and the device accessors turn a handle into a window+offset / pointers:
- `ncclGetResourceBufferOffset(h)` → `h * 128` (`impl/core__funcs.h:189`)
- `ncclGetResourceBuffer(comm, h)` → `ncclSymPtr<char>{comm.resourceWindow, h*128}` (`core__funcs.h:240`)
- `ncclGetResourceBufferLocalPointer / …LsaPointer / …PeerPointer` for direct reads (`core__funcs.h:194-220`)

`gin.put` accepts both `(window, offset)` and `ncclSymPtr` dst forms, so a kernel writes a peer's
scratch with either. **In-tree proof:** RCCL's own GIN reduce-scatter stages partials through this
exact resource-buffer-backed scratch (`device/symmetric/reduce_scatter_gin.cuh`,
`gin_scratch__funcs.h` `ncclGinInboxA2ASession::postSends` puts into `ncclGetResourceBuffer`-derived
`ncclSymPtr`). Do **not** reuse the LL handle's buffer as generic scratch — request your own node.

**Wiring (mirrors the existing LL wiring in `all_gather.cu:99-108`):**

```cpp
static ncclDevResourceRequirements g_scratchReq = {};
static ncclDevResourceHandle       g_scratchHandle = 0;
// in <Coll>GetDevCommRequirements(), case 3:
memset(&g_scratchReq, 0, sizeof(g_scratchReq));
g_scratchReq.bufferSize      = (size_t)N * chunkBytes;   // ~M bytes/rank
g_scratchReq.bufferAlign     = 128;
g_scratchReq.outBufferHandle = &g_scratchHandle;
g_scratchReq.next            = reqs->resourceRequirementsList;
reqs->resourceRequirementsList = &g_scratchReq;
```

Device side (put to a peer's scratch, then read my own back):

```cpp
size_t base = ncclGetResourceBufferOffset(scratchHandle);       // = handle*128
gin.put(ncclTeamWorld(devComm), peer, devComm.resourceWindow, base + srcRank*chunk,
        sendwin, sendoffset + peer*chunk, chunk, ncclGin_SignalInc{sig});
// after waitSignal(base + N-1): SM-reduce the N contributions
char* mine = (char*)ncclGetResourceBufferLocalPointer(devComm, scratchHandle);
```

Pass `g_scratchHandle` via a new `testLaunchDeviceKernelThresholdScratch` launcher (§4.4). The
small LSA read-reduce path needs **no** scratch, so the requirement is sized/added only for the
large tier (or requested at the worst-case `MAX_BYTES` and left unused below threshold, matching
how the LL scratch is always requested but bypassed when `nSlots==0`).

**Caveats:** (1) `comm.resourceWindow == nullptr` if *no* buffer bytes were requested — only rely
on it when this (or LL/barrier) code requested some. (2) `resourceWindow` is created with
`winFlags=0`; the in-tree GIN reduce-scatter puts into it prove it is GIN-capable on the current
path, but if a specific backend ever needs an explicit GIN-registerable flag, verify against
`symWindowCreate`. There is **no** `ncclWindowCreateRequirement`/`ncclScratchCreateRequirement` —
the direct `ncclDevResourceRequirements` fill is the sanctioned mechanism.

### 4.3 Op-aware device dispatch and op/type coverage (Q4 RESOLVED)

`op` is already threaded through `RunColl`/launchers but is unused by the movement kernels, and the
current `SPECIALIZE_KERNEL` macro (`common.h:484`) is **sum-only over 9 base types** (no bf16/fp8).
Reduction collectives need op-aware *and* extended-type dispatch plus an on-device reduce.

**Ops in rccl-tests** (`common.cu:88`, `:1533-1540`): `test_ops = {sum, prod, max, min, avg,
mulsum}` where `mulsum` = `ncclPreMulSum`; `avg`/`mulsum` are version-gated. **Types** (`:1534-`):
9 base (`int8/uint8/int32/uint32/int64/uint64/half/float/double`) + `bf16` (`RCCL_BFLOAT16`) +
`fp8 e4m3/e5m2` (`RCCL_FLOAT8`).

**Reference is `ncclVerifiable`, not a host loop.** Correctness is checked by
`ncclVerifiablePrepareExpected` / `…Verify` (`common.cu:472-491`), which model NCCL's *actual*
reduction semantics (accumulation type, avg scaling, premul scalar). The `-D 3` kernel's arithmetic
must match these to get `#wrong == 0` — this is the real gate, not "does it sum."

**Decision — coverage for v1:**

| Op | v1? | Device handling |
|---|---|---|
| sum, prod, min, max | **yes** | `Apply<op,T>` functor (a small new rccl-tests device header). |
| avg | **yes** | sum then scale by `1/nRanks` **once** on the fully-reduced value — in the RS phase for two-shot, after the local sum for one-shot. |
| mulsum (`ncclPreMulSum`) | **DEFER** | needs the premul **scalar** (via `ncclRedOpCreatePreMulSum`) matched to `ncclVerifiable`'s convention and plumbed to device; return `testNotImplemented` in case 3 for `PreMulSum` in v1. |

- **Types:** cover everything the build enables — extend to a `SPECIALIZE_REDUCE_KERNEL(kernel,
  type, op)` that dispatches `(type × op)` including `bf16` and `fp8 e4m3/e5m2` (the ML-relevant
  types), leaving `SPECIALIZE_KERNEL` (movement, sum-only) untouched.
- **fp8 exclusions — mirror the existing per-collective host gates exactly:**
  - ReduceScatter excludes fp8 `{prod, mulsum}` (fp8 **avg is supported**) — `reduce_scatter.cu:114`.
  - AllReduce / Reduce exclude fp8 `{prod, avg, mulsum}` — `all_reduce.cu:574`, `reduce.cu:114`.
- **Skip mechanism (reuse existing):** any unsupported `(op,type)` maps to a `nullptr` kernel →
  `testLaunchDeviceKernel*` returns `testNotImplemented` → the harness skips it (exactly how
  `op != sum` is skipped today). So deferring mulsum / honoring fp8 exclusions needs no new harness
  logic — just return `nullptr`/`testNotImplemented` for those combinations in case 3.

**Accumulation-type caveat (implementation risk).** For low-precision inputs (fp8/half/bf16),
`ncclVerifiable` expects NCCL's specific accumulation behavior. `Apply<op,T>` must accumulate in the
matching type/promotion (likely promote fp8/half/bf16 to `float` for the running reduction, then
narrow) to hit `#wrong == 0`. Mitigation: sweep types against the verifier early and adjust the
promotion per type; keep it isolated in the `Apply` functor so the kernels don't change.

### 4.4 Launchers (`common.h`)

Add `testLaunchDeviceKernelThresholdScratch(...)` (threshold + scratch handle) and, if needed,
`…ThresholdScratchLL(...)`. Movement collectives reuse the existing
`testLaunchDeviceKernelThreshold` / `…ThresholdLL`.

### 4.5 AllToAllv metadata plumbing (RESOLVED)

The kernel needs the §3.4 metadata arrays on device. Decision: **pass small device arrays via a
bespoke launcher.** Rationale for the choice among the options considered:

| Option | Verdict |
|---|---|
| **Device arrays (`cudaMalloc` + async H2D, cached by N)** | **Chosen.** Lowest friction, standard, reuses the host matrix the test already builds. |
| Recompute `M` on device | Rejected — `AlltoAllvGenSizeMatrix` uses `std::mt19937_64` + FP `pow/floor`; painful and risky to port to device. |
| Pack metadata into the `resourceWindow` scratch | Rejected — unnecessary; metadata is host-local input, not peer-shared. |
| Pass fixed-size arrays by value | Rejected — `N` is unbounded; won't fit a fixed kernel signature. |

**Why this is easy:** `AlltoAllvRunColl` already has `comm` (which `testLaunch*` casts straight to
`ncclDevComm*`, `common.h:447`) and `sendbuff`/`recvbuff` (already `ncclWindow_t` handles,
`common.h:449`). So a bespoke `testLaunchDeviceKernelA2Av` mirrors the existing launchers, adding
three `const size_t*` device pointers and an `int nIncoming`:

```cpp
template <typename F>
testResult_t testLaunchDeviceKernelA2Av(F kernel, void* sendbuff, size_t sendoffset,
    void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op,
    int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride,
    const size_t* dSendBytes, const size_t* dSrcOff, const size_t* dDstOff, int nIncoming) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff, recvwin = (ncclWindow_t)recvbuff;
  kernel<<<deviceCtaCount, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count,
      root, *devComm, sdmaThresholdOverride, dSendBytes, dSrcOff, dDstOff, nIncoming);
  return testSuccess;
}
```

`AlltoAllvRunColl` extends its existing per-peer loop (`alltoallv.cu:208-219`) to also compute
`dstByteOff` (column prefix) and `nIncoming`, uploads the three arrays with `cudaMemcpyAsync` on
`stream` into a per-`(nranks)` cached device allocation, then calls the launcher. `op` stays
`ncclSum` (movement only), so `SPECIALIZE_KERNEL` (`common.h:484`, sum-only) is unchanged.

**Note:** this bespoke launcher is the *only* genuinely new plumbing in Tier A; Scatter/Gather/
SendRecv reuse the existing `testLaunchDeviceKernelThreshold`.

### 4.6 Tests, gates, image

- Extend `gin_sdma_policy_test.cpp` with cases for every new predicate/threshold/DevReqs — keep
  **100 % line+branch** (it is a `docker build`-time gate).
- Add `gin_sdma_gpu_{scatter,gather,sendrecv,alltoallv,reduce_scatter,all_reduce,reduce}` GPU
  functional CTest targets (label `gpu_functional`).
- Add gate scripts `gin-sdma-{scatter,gather,sendrecv,a2av,rs,ar,reduce}-test.bash` mirroring
  `gin-sdma-bcast-test.bash` (UT preflight, host baseline `-D 0`, GIN sweep `-D 3 -V 8`, all-SDMA
  `THRESHOLD=0`, `#wrong==0`, root sweep where applicable).
- Ensure host baselines build: extend `ONLY_FUNCS` in `Dockerfile-rccl-gin-gda-sdma` /
  `docker-gin-gda-sdma-build.bash` with the needed device funcs (Reduce/ReduceScatter/AllReduce/
  Scatter/Gather/SendRecv) so `-D 0` runs the full range; add smoke asserts.

---

## 5. Methodology for thresholds (per broadcast-plan precedent)

Each LSA↔GIN default is **measured**, not guessed, on 8× MI355X (`NCCL_GIN_TYPE=6`, `-V 8`):
same-build A/B via the env threshold knob, sweep sizes across the crossover, pick where GIN/SDMA
overtakes LSA. Reduction collectives additionally sweep the small (LSA read-reduce) vs large
(put-partials) crossover, and AllReduce/Reduce sweep the direct-vs-composed crossover. Record the
basis table exactly like §4.3.1 of the broadcast plan.

---

## 6. Phasing

| Phase | Deliverable | Risk | Notes |
|---|---|---|---|
| **P1** ✅ | SendRecv, Scatter, Gather (`-D 3`) | low | **Landed** (2026-07-29). Pure movement; each adds an LL tiny tier. Scatter also gained a peer-interleaved LSA fan-out; thresholds tuned (Scatter 128 KiB; Gather/SendRecv LSA-always). |
| **P2** ✅ | ReduceScatter (`-D 3`) | med | **Landed** (2026-07-30). Final design is single-tier **direct LSA read-reduce** (no scratch/put-partials): each rank reads its output slice from every peer and folds locally, adaptive load schedule ported from the host symmetric LD kernel. 97–100% of host at ≥64 MiB (beats host 64–256 MiB). SDMA-scatter (>host ceiling) investigated + de-risked, not implemented — see [reducescatter-gin-sdma-phase2.md](reducescatter-gin-sdma-phase2.md). |
| **P3** ✅ | **AllReduce** (`-D 5`/`-D 6`) | med | **Landed** (2026-08-02). Shipped as deviceImpl 5 (single-launch, grid barrier) / 6 (two-launch), not `-D 3`. RS+AG all sizes; threshold picks the AG tier (LSA store-gather / SDMA put). Tiny-OOP one-shot kept in a hang-safe GIN-barrier'd form. See §3.6. **Highest ML value.** |
| **P4** ✅ | Reduce (`-D 3`) | med | **Landed** (2026-08). Small = reduce-scatter-to-root LSA read-reduce; large OOP = **edge-disjoint multi-ring** (not RS+Gather — the planned composition plateaued). Tier logic in `reduce.cu`. See §3.7. |
| **P5** | AllToAllv (`-D 3`) | med | **NEXT / WIP** (`…-a2av-wip`). Last remaining collective. Needs device-visible displ/count arrays (Q3, RESOLVED §4.5). `AlltoAllvRunColl` has no `case 3` yet. |
| **Xcut** | policy header + unit tests + gates + image | continuous | one PR per phase, each gated green |

Recommended order by value/effort was **P1 → P2 → P3 (AllReduce)** first; **P1–P4 are done**, P5
(AllToAllv) is the only remaining item.

---

## 7. Open questions

1. ~~**Scratch window mechanism**~~ — **RESOLVED (§4.2):** use the generic
   `ncclDevResourceRequirements{bufferSize, bufferAlign, outBufferHandle}` → `comm.resourceWindow`
   + `ncclGetResourceBuffer(Offset)` accessors; GIN-put-addressable, proven by RCCL's own GIN
   reduce-scatter. No new backend API.
2. ~~**AllReduce small path**~~ — **RESOLVED, REVISED IN IMPLEMENTATION (§3.6):** the *no-GIN*
   one-shot was first prototyped and dropped (its LSA-only launches, interleaved among GIN launches,
   deterministically re-triggered a cumulative GPU hang — GIN signal-completion pipeline disturbance
   from the irregular cadence), then **re-introduced hang-safe** as `arAllReduceOneShotLsa` for tiny
   out-of-place messages (< 256 KiB), now performing exactly one GIN world barrier so the cadence
   stays uniform. Separately, a **size-adaptive AllGather tier** (LSA store-gather < 16 MiB, SDMA put
   ≥) wins the small/mid RS+AG range. AllReduce ships as `-D 5`/`-D 6`, not `-D 3`.
3. ~~**AllToAllv arrays**~~ — **RESOLVED (§3.4, §4.5):** pass three `size_t[N]` metadata arrays
   (`sendBytes`, `srcByteOff`, `dstByteOff`) + `nIncoming` as cached device buffers via a bespoke
   `testLaunchDeviceKernelA2Av` launcher. The deterministic, all-ranks-identical size matrix lets
   each rank compute the receiver-side column-prefix offset locally — no exchange needed. No packed
   resource; device recompute rejected (host RNG/FP).
4. ~~**Op coverage**~~ — **RESOLVED (§4.3):** v1 = sum/prod/min/max/avg over all build-enabled
   types (base + bf16 + fp8) via a new `SPECIALIZE_REDUCE_KERNEL` + `Apply<op,T>`; **defer
   mulsum/PreMulSum** (needs the premul scalar matched to `ncclVerifiable`). Mirror per-collective
   fp8 exclusions exactly (RS excludes fp8 prod/mulsum; AR/Reduce also exclude fp8 avg). Unsupported
   `(op,type)` reuse the existing `nullptr → testNotImplemented` skip. Key risk: matching
   `ncclVerifiable`'s accumulation/promotion for fp8/half/bf16 to reach `#wrong==0`.
5. ~~**Reduce large tier**~~ — **RESOLVED, then REVISED IN IMPLEMENTATION (§3.7):** RS+Gather was
   planned, but the perf campaign found it plateaus at the serial gather, so the shipped large OOP
   tier is an **edge-disjoint multi-ring** (`GinReduceMultiRingKernel`, default ON ≥ 64 MiB) — the
   same cure the broadcast large tier used. Direct root-ingress push remains rejected. Small tier =
   reduce-scatter-to-root LSA read-reduce (no scratch), as planned. See perf-opt-plan §9.5.
6. ~~**Integration target**~~ — **RESOLVED (recommendation, §8):** v1 = all collectives at `-D 3`
   in rccl-tests (matches the broadcast doc §0/§8 precedent: standalone kernel + gate first,
   production wiring deferred). Promote to production later, per-collective, once the gate is green
   and the threshold is measured — **AllReduce first**. This is a scope call; confirm the policy.

---

## 8. Integration & promotion policy (Q6)

**Recommendation: keep everything at `-D 3` in rccl-tests for v1**, exactly as the three
implemented collectives and as the broadcast design doc already decided (§0/§8 there: "v1 stays
`broadcast_perf -D 3` only … production `ncclBroadcast` / `rcclDirectBroadcast` wiring is
deferred"). Rationale: no backend/ABI risk, fast gate-driven iteration, and thresholds must be
*measured* (§5) before any production tuning is meaningful.

**Promotion criteria (per collective, before touching production):**
1. GIN gate green — `#wrong == 0` across the full size sweep, root sweep (where applicable),
   in-place/OOP, and all covered `(op, type)` combos;
2. LSA↔GIN (and two-phase) thresholds measured and defaulted (§5);
3. measured perf beats the current production path on 8× MI355X at the target sizes;
4. host policy unit tests + GPU functional CTest green and baked into the image.

**Where each would land in production (single-node xGMI only; guard `nNodes == 1`, else fall back):**

| Collective | Natural production home | Notes |
|---|---|---|
| **AllReduce** | RCCL **symmetric GIN** kernels (`device/symmetric/`) | **Promote first** (hottest ML collective). RS half already exists as `reduce_scatter_gin.cuh`; AG half as `all_gather_gin.cuh` — AllReduce = compose them. |
| ReduceScatter | `device/symmetric/reduce_scatter_gin.cuh` | already a production GIN kernel; our `-D 3` validates the rccl-tests-side design/thresholds. |
| AllGather | `rcclDirectAllGather` / `all_gather_gin.cuh` | direct fast path already exists — lowest-friction promotion. |
| Reduce | RS+Gather over the symmetric framework | dual of broadcast; promote after AllReduce/RS. |
| Broadcast | needs a new `rcclDirectBroadcast` | none exists today (broadcast doc §2.2); larger lift. |
| Scatter / Gather / SendRecv / AllToAllv | CE / symmetric or point-to-point paths | lower priority; promote on demand. |

**Sequencing:** promote **AllReduce** first (highest value, both halves already have production GIN
kernels to build on), then ReduceScatter / AllGather, then Reduce; leave Scatter/Gather/SendRecv/
AllToAllv at `-D 3` until there is demand.

**Boundary:** the Anvil-SDMA backend is intra-node only, so any production wiring must gate on
`nNodes == 1` and defer to the existing ring/CE/tree paths otherwise (same constraint the backend
already enforces).

---

*Internal AMD planning document.*
