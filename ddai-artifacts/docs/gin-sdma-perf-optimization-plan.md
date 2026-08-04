# GIN-SDMA Collectives: Performance Optimization Plan

**Document type:** Internal AMD performance/optimization plan
**Status:** **In progress** — the 9 GIN-SDMA collectives (`-D 3`, `NCCL_GIN_TYPE=6`) are
functionally landed and gate-green; this plan drives systematic per-collective performance
tuning toward production-promotion (§8 of the expansion plan).
**Date:** 2026-08-03
**Scope:** `projects/rccl-tests/` device kernels, the shared `gin_sdma_collective_policy.h`
tunables (thresholds / CTA ladders / barrier model), and the `ddai-artifacts/scripts/`
measurement harness. **No RCCL backend / GIN-ABI changes.**
**Related:**
- `ddai-artifacts/docs/gin-sdma-collectives-expansion-plan.md` (§5 methodology, §8 promotion)
- `ddai-artifacts/scripts/gin-rs-hostvsgin-sweep.bash` (host-vs-GIN A/B template)
- `ddai-artifacts/scripts/gin-fabric-ceiling.bash` (xGMI roofline)

---

## 1. Goal & success criteria

For each collective, at every size tier, maximize `-D 3` busbw **subject to zero correctness
regressions**. Per-collective "done" = one of:

1. **Beats the host `-D 0` path** at the target sizes (the promotion bar, expansion-plan §8.3), **and**
2. **≥ 90–100 % of the applicable roofline** — xGMI read/write BW for movement collectives,
   host-symmetric-kernel busbw for reductions — **or** a documented structural reason it cannot
   (e.g. root-egress / root-ingress bound).

Out of scope for this pass (assessed as already in good shape): **AllToAll** (pull redesign,
barrier-minimal, ≥ correct-push busbw) and **AllReduce**.

---

## 2. Transferable lessons from the AllToAll campaign (these drive where to look)

These are the empirically-established levers/pitfalls from the A2A pull redesign. They set
priors for every collective below.

1. **Barrier count/cost dominates small–mid sizes** — each LSA barrier costs ~20–25 % of busbw at
   1 MiB/peer. **Dropping the exit barrier is the single biggest lever** wherever the tier is
   "own-writes-local" (pull/gather-shaped).
2. **Pull (read peers → write own) is self-completing with one entry barrier** and is immune to
   the recvbuf-`initData` memset race. **Push (write peers)** needs two barriers to be correct.
3. **Push writes are ~12 % faster than pull reads on xGMI** — for *root-egress-bound* collectives,
   push layout still matters; for symmetric collectives, barrier count wins.
4. **When BW-bound, CTA-count / MLP / peer-unroll tuning does nothing.** Confirm the bind before
   investing there (the A2A CTA sweep + PEER_UNROLL=4 experiments produced no gain).
5. **The LL tiny-tier is usually marginal** — keep it off-by-default unless a sweep proves a
   barrier-bound latency floor it removes.
6. **Beware cross-iteration overlap artifacts in perf numbers** — a barrier-free variant inflates
   busbw unrealistically (the push mode-1 "111 GB/s" ceiling was not a real, correct number). Only
   compare *correct* variants.

---

## 3. Shared methodology (build once, reuse for all)

- **Harness:** same-build A/B via per-collective env knobs
  (`NCCL_GIN_ANVIL_SDMA_THRESHOLD_<COLL>`, `..._LSA_CTAS`, `..._LL_MAX_BYTES`, interleave flags).
  Generalize `gin-rs-hostvsgin-sweep.bash` into a board across all collectives (see §7 M0).
- **Sweep:** `-b 8 -e <max> -f 2`, out-of-place (`-z 0`) for the headline number; `-c 0` for perf,
  `-c 1` for the correctness gate. `-g 1 -R 2 -V <CTA>`, `NCCL_GIN_TYPE=6`, 8× MI355X.
- **Stability:** `-n 20 -w 5` (match the existing sweep scripts); treat ~2–3 % run-to-run as noise;
  require ≥ 1.5× noise to call a win.
- **Cold-start (verified 2026-08-03):** the FIRST mpirun in a fresh container runs ~2–3× slow (host
  scatter @2M: ~40 cold vs ~114 warm), and per-run `-w` warmup iterations do NOT cure it — the ramp is
  at the container/clock level, across mpirun launches. **Always discard ≥1 warmup invocation** before
  recording a host `-D 0` (or any) baseline; never compare a cold host leg against a warm GIN leg. This
  is measurement order, not env (gin-env warm ≈ clean warm). **`gin-sdma-hostvsgin-board.bash` now
  enforces this via `run_warm` (1 discarded + 1 measured mpirun per collective/mode);** the warm
  re-baseline reproduced the M0 board exactly (§9.1), confirming the artifact was confined to the §9.2
  scatter re-check.
- **Baselines per size:** capture (host `-D 0`, current `-D 3`, roofline) → gap table.
- **Guardrails:** after **every** change, re-run the collective's correctness gate
  (`gin_sdma_gpu_functional.sh <coll>` → `#wrong == 0`, full sweep + root/op/type where
  applicable) before trusting a perf delta.
- **Isolation invariant (verified 2026-08-03, all 9 collectives):** GIN-SDMA device code is never
  reachable from the host-initiated collective. Each `RunColl` splits `deviceImpl == 0` (a pure RCCL
  library `ncclXxx(...)` call — the "production"/host path) from `deviceImpl != 0` (the GIN device
  kernels, launched only via `testLaunchDeviceKernel*` / gated `*DeviceTime`). Even the reduction math
  is not shared: GIN kernels use `gin_sdma_reduce.h`; the host-path verifier uses a separate
  `verifiable.h` implementation (deliberately bit-matching, not shared). **Consequence: optimizing any
  `-D 3` device kernel cannot affect the host-initiated collectives — edit GIN device code in place;
  no fork needed.** Preserve this split (never launch a GIN kernel from the `deviceImpl == 0` branch,
  never call `gin_sdma_*` device helpers from the host/verify path).

---

## 4. Cross-cutting levers, ranked

| # | Lever | Applies to | Expected impact |
|---|---|---|---|
| 1 | **Exit-barrier elimination audit** (pull-shape ⇒ entry-only) | AllGather, ReduceScatter (small), Reduce (small), any own-writes-local LSA tier | **High** at small–mid (the A2A win: race immunity + ~1 barrier saved) |
| 2 | **Threshold re-measurement** (LSA↔GIN, one-shot↔two-shot, flat↔SAG) | all reduction + SAG collectives | **High** — a wrong crossover strands whole size-bands on the slow tier |
| 3 | **CTA ladder retune per bind** (read- vs write-bound) | all LSA tiers | Med — only where latency/occupancy-bound |
| 4 | **Fan-out / interleave** (root-bound collectives) | Broadcast, Scatter, Gather | Med–High for the root-bound tier |
| 5 | **LL disposition** (prove with data or delete) | Broadcast, Scatter, SendRecv, AllGather | Low, cleanup |

---

## 5. Per-collective plan (AllToAll & AllReduce excluded)

| Collective | Bind | Top hypotheses / actions | Priority |
|---|---|---|---|
| **ReduceScatter** | egress-balanced | Foundational for AR/Reduce. (a) exit-barrier audit on the LSA read-reduce; (b) implement the **de-risked SDMA put-partials tier** for sizes *above* the host-copy ceiling (known headroom, expansion-plan P2 notes); (c) re-measure LSA↔GIN threshold. | **1** |
| **Broadcast** | root-egress | (a) SAG (scatter+allgather) chunk/pipeline to attack the large-M serial plateau; (b) LSA peer-interleave fan-out (mirror Scatter); (c) re-measure flat↔SAG crossover. | **2** |
| **AllGather** | symmetric pull | Prime **entry-only** candidate (it is pull-shaped like A2A: reads peers, writes own recvbuf); CTA ladder retune; LL prove/delete. Likely quick win. | **3** |
| **Reduce** | root-ingress | (a) RS+Gather gather-phase plateau (chunk/pipeline, mirrors Broadcast SAG); (b) small LSA pull-reduce exit-barrier audit; (c) threshold. | **4** |
| **Gather** | root-ingress | Verify large put-based fan-in; is LSA-at-all-sizes still optimal? re-measure a GIN crossover. | **5** |
| **Scatter** | root-egress | Mostly tuned (2026-07-29, 128 KiB threshold + peer-interleave). Re-verify optimum holds after any shared-code changes. | **6** |
| **SendRecv** | p2p latency | Latency-floor focus: LL default cap sweep; confirm barrier-light path. Low headroom. | **7** |

---

## 6. Sequencing & milestones

- **M0 — Baseline & roofline board (do first).** One sweep per collective capturing host `-D 0`,
  current `-D 3`, and the fabric roofline; publish a gap table. This localizes real headroom and
  prevents chasing BW-bound walls. Harness: §7.
- **M1 — Cross-cutting quick wins.** Exit-barrier audit (AllGather + reduction small tiers) +
  threshold re-measurement. Highest ROI, lowest risk, reuses the A2A pattern.
- **M2 — ReduceScatter deep-dive** (incl. put-partials above-ceiling tier) → unblocks AR/Reduce.
- **M3 — Broadcast & Reduce** (SAG / gather-phase pipelining).
- **M4 — Gather / Scatter / SendRecv verification + LL cleanup.**
- **Cross-cutting:** correctness gate after each change; commit retuned defaults + a per-collective
  perf-basis table; keep the expansion-plan §6 phasing table in sync.

---

## 7. M0 harness

Generalize `gin-rs-hostvsgin-sweep.bash` (host `-D 0` vs GIN `-D 3`, same binary, `-c 0`) across
all collectives, and reuse `gin-fabric-ceiling.bash` for the xGMI roofline. Output: a per-size
`(host, gin, gin/host %, gin/roofline %)` table per collective, so M1 targeting is data-driven.

Script: `ddai-artifacts/scripts/gin-sdma-hostvsgin-board.bash` (added by this plan).

---

## 8. Deliverables & tracking

Per collective: committed retuned thresholds / CTA ladders / barrier model, a measured perf-basis
table (like the broadcast basis table in the expansion plan), a green correctness gate, and a
short results entry appended to this document.

---

## 9. Results

### 9.1 M0 — baseline board (2026-08-03, 8× MI355X, `-D 0` host vs `-D 3` GIN, OOP, `-c 0`, `-n 20 -w 5`, float)

Same fresh `/rt-build` binaries (branch `gin-stage2i` @ `c114d50`) drive both modes.
`gin/host%` < 100 % = GIN slower than host (closable headroom); > 100 % = GIN already beats host.
Full boards captured via `gin-sdma-hostvsgin-board.bash`.

> ✅ **Warm re-baseline (2026-08-03, corrected script).** Re-ran the full board with
> `gin-sdma-hostvsgin-board.bash` after adding `run_warm` (one discarded + one measured mpirun per
> collective/mode, per the cold-start guardrail). The M0 numbers **reproduce exactly** on the unoptimized
> collectives — scatter **57.6 %** @2 MiB, reduce **75.0 %** @2 GiB, broadcast **71.2 %** @2 GiB — so the
> original board was already warm-consistent; the cold-start artifact was isolated to the *separate*
> scatter re-check in §9.2, not the board. The two optimized collectives now show their applied wins on
> the warm board: **all_gather** mid-band lifted (128 KiB 68→**82 %**, 512 KiB→**113 %**, ≥8 MiB 125–194 %)
> and **reduce_scatter** mid-band lifted (2 MiB 78→**94 %**, 8 MiB→**98 %**, small 100–132 %). The table
> rows below are updated to this warm re-baseline (post-AG-pull, post-RS-exit-drop).

| Collective | Small (≤256 KiB) | Mid (1–33 MiB) | Large (≥64 MiB) | Worst point | Verdict |
|---|---|---|---|---|---|
| **broadcast** | strong (150–210 %) | fading (85→82 %) | **71–79 %** | 71 % @ 1–2 GiB | Large SAG plateau ~229 GB/s (host 320) |
| **all_gather** _(post pull-fix)_ | 82–125 % (min **76 %** @64 KiB) | 85–194 % | excellent (122–194 %) | 76 % @ 64 KiB | Pull redesign closed the 68 % dip (was §9.2) |
| **scatter** | strong (135–250 %) | **cliff 57.6 % @2 MiB** | 79–91 % | **57.6 % @ 2 MiB** | LSA↔GIN threshold cliff + ~10 % large gap |
| **gather** | strong (150–214 %) | 108–121 % | 100–107 % | 99.6 % @ 2 GiB | Healthy — no action |
| **sendrecv** | strong (142–200 %) | 102–113 % | 99–102 % | 98.8 % @ 2 GiB | Healthy — ~62 GB/s ring ceiling |
| **reduce_scatter** _(post exit-drop)_ | 100–132 % | 94 % @2 MiB, **82–98 %** 2–33 MiB | parity (98–101 %) | **81.6 % @ 33 MiB** | Exit-barrier drop lifted mid-band (was §9.4) |
| **reduce** | excellent (up to 278 %) | 103–262 % | **75–84 %** | 75.4 % @ 2 GiB | Large RS+Gather plateau ~214 GB/s (host 284) |

**Key synthesis:**
- **Biggest, most coherent gap = the large-size composition plateau shared by Broadcast (≥8 MiB, →71 %)
  and Reduce (≥64 MiB, →75 %).** Both plateau ~215–230 GB/s while their building-block collectives
  reach the fabric ceiling (AllGather 424 GB/s, Gather 437 GB/s at large). The scatter/gather phase of
  the SAG / RS+Gather composition is serialized against the egress/ingress root, not pipelined.
- **Two clean quick wins:** Scatter's **2 MiB threshold cliff** (57.6 % — a misplaced LSA↔GIN crossover)
  and AllGather's **mid LSA-band dip** (68 % @128 KiB — the entry-only/exit-barrier candidate, mirrors
  the A2A pull win).
- **ReduceScatter** is at host parity for large (good) but ~80 % across 1–33 MiB — the mid tier and the
  de-risked above-ceiling put-partials tier are the levers.
- **Gather and SendRecv are already ≥ host everywhere** — drop from the active optimization list.

**Data-driven attack order (supersedes §5 priority):**
1. **Quick wins:** Scatter 2 MiB threshold cliff; AllGather mid-band exit-barrier audit.
2. **Biggest headroom:** shared large-tier composition plateau — Broadcast (SAG) + Reduce (RS+Gather)
   phase pipelining; confirm the scatter-phase egress bound vs AllGather's achieved 424 GB/s.
3. **ReduceScatter** mid-band (1–33 MiB) + above-ceiling put-partials tier.
4. Gather / SendRecv: no action (already ≥ host).

### 9.2 Quick-win investigations

**Scatter 2 MiB cliff — NOT a quick win (threshold already correct).** Tier diagnostic
(`gin-scatter-tier-diag.bash`, `-D 3`, 1–32 MiB total, default vs all-LSA vs all-GIN):

| total (rank chunk) | default | all-LSA | all-GIN | host (M0) |
|---|---|---|---|---|
| 1 MiB (128 KiB) | 49.7 (LSA) | 49.8 | 34.4 | 46.9 |
| 2 MiB (256 KiB) | 61.3 (GIN) | 64.5 | 63.2 | 101.7 |
| 4 MiB (512 KiB) | 108.3 (GIN) | 75.6 | 108.4 | 133.2 |
| 8 MiB (1 MiB) | 167.3 (GIN) | 82.0 | 171.1 | 224.0 |
| 32 MiB (4 MiB) | 295.7 (GIN) | 88.4 | 298.3 | 345.5 |

LSA clearly wins ≤128 KiB/rank (1 MiB: 49.7 vs GIN 34.4) and saturates at ~88 GB/s (single-GPU root
egress bound); GIN wins from 512 KiB/rank up. The 128 KiB/rank threshold is well placed — no threshold
move helps. The residual 2–8 MiB gap vs host is a **GIN root fan-out inefficiency** (same root-egress
class as Broadcast's SAG plateau). **Reclassified from "quick win" into the root-egress family (attack
order #2).**

**Re-check (2026-08-03) — cliff CONFIRMED real; structural; threshold near-optimal.** Re-swept 1–8 MiB
with a host anchor + forced-tier + a candidate `thr=256K/rank` bump. GIN device numbers reproduce the
above exactly (2M: default 61.1, all-LSA 64.4, all-GIN 63.6). Warm host @2M ≈ **114 GB/s** (see the
cold-start note below), so gin/host ≈ **53 %** — the cliff is real, matching M0's 57.6 %. Both device
tiers plateau ~61–65 at 2M (root-egress bound) while host does ~114, so **no device-side tier choice
closes it**. Bumping the threshold to 256 KiB/rank routes the 2M point to LSA for a **marginal +6 %**
(61→65, no regression at 1/4/8 MiB) — real but small, and it does not close the cliff. **Verdict:
threshold left as-is (128 KiB/rank); the 2 MiB trough is the same structural root-egress bound as §9.3
(needs a multi-channel backend), NOT a threshold bug.**

> ⚠️ **Cold-start caveat (methodology).** Host `-D 0` scatter @2M measured **~40 GB/s cold** (first
> mpirun in a fresh container) vs **~112–117 GB/s warm** (after a discarded warmup) — a ~2–3× cold-start
> effect, largely env-independent (gin-env warm ≈ clean warm). An early version of this re-check read
> host=41.7 (cold) and wrongly concluded "gin beats host / no cliff." **Fix (applied): the board script
> now runs `run_warm` — one discarded + one measured mpirun per collective/mode — so every host and GIN
> leg is warm.** Re-checked warm, the scatter cliff is confirmed at **57.6 %** @2 MiB (identical to M0).
> The concern that this inflated the ReduceScatter A/B in §9.4 did **not** materialize: the warm
> re-baseline reproduces RS at 2 MiB 93.6 %, 8 MiB 97.6 % (§9.4 table below updated to warm), so the RS
> host leg was warm-enough in the original A/B. Net: the cold artifact was isolated to *this* scatter
> re-check only; the M0 board (§9.1) and RS A/B (§9.4) both hold up warm.

**AllGather mid-band dip — DONE, confirmed win (push→pull, drop exit barrier).** The LSA vectorized
tier was a PUSH copy (each rank reads its own send chunk once, writes to every peer's recvbuf +
entry+exit barriers) — the exact A2A push pattern behind the 68 % @128 KiB dip. Flipped to PULL (read
each peer's read-only chunk, write only own recvbuf) + single entry barrier. Unlike A2A, AllGather
supports **in-place**, where each rank's send slice is at a rank-dependent offset; the pull source is
therefore mode-aware (out-of-place: peer's sendbuf; in-place: peer's `recvbuf[p*count]`). Each rank
writes only its own recvbuf and reads idempotently-stable `[p*count]` slots, so a single entry barrier
(no exit) is correct in both modes.

- **Correctness:** GIN gate green, `#wrong == 0` for both in-place and out-of-place across the full
  sweep (the first pull draft failed in-place only — fixed by the mode-aware source).
- **Perf (gin/host%, mid LSA band):** 128 KiB 68→79, 256 KiB 70→82, 512 KiB 80→111, 1 MiB 94→111.
  The M0 dip is largely closed. GIN large tier unchanged (≈421 GB/s @1 GiB). Minor ~6 % dip at 64 KiB
  (single-CTA tier: pull's N remote reads cost slightly more than push's 1 local read; sub-µs).

Net: mid-band recovered, race-immune, one barrier saved. Ready to commit.

### 9.3 Broadcast SAG pipelining — NEGATIVE result (reverted), plateau is channel-bound

Hypothesis (attack order #2): the SAG large tier plateaus (~213 GB/s @128 MiB) because a non-root
must receive its **whole** scatter slice before it forwards it in the allgather, so scatter (root
egress) and allgather run serially — busBw collapses toward the harmonic mean of the two phases.
Proposed fix: split each slice into K sub-chunks (`NCCL_GIN_ANVIL_BCAST_SAG_CHUNKS`) so a rank forwards
sub-chunk k while the root scatters sub-chunk k+1, overlapping the root-egress scatter behind the
distributed allgather.

Implemented (pipelined kernel, two signal indices, per-sub-chunk `waitSignal`; K=1 = original) and
A/B-swept K∈{1,2,3,4,8}. **Correct** (gate green at K=4), but pipelining **monotonically hurt** at
every size:

| size | K=1 | K=2 | K=3 | K=4 | K=8 |
|---|---|---|---|---|---|
| 8 MiB (`-c 1`) | **93** | 70 | 55 | 46 | 26 |
| 128 MiB | **213** | 203 | — | 185 | — |
| 512 MiB | **226** | 223 | — | 217 | — |

**Root cause of the plateau ≠ phase serialization.** With `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1` there is
a single SDMA queue per peer, so the scatter and allgather puts to a given peer **cannot overlap on the
wire** — they are already serialized by the one channel. Chunking therefore buys no overlap and only
adds cost (K× more puts + smaller/less-efficient transfers + K sequential `waitSignal` round-trips),
which is why K=1 wins and larger K is progressively worse. The harmonic-mean model was wrong: the SAG
busBw is bounded by **single-GPU root egress through the available channels**, not by a serialization
the kernel can hide.

**Change reverted** (K=1/original restored, baseline rebuilt + gated). This confirms the whole
root-egress family (Broadcast SAG, Scatter GIN fan-out, Reduce RS+Gather) plateaus for the *same*
structural reason — per-peer egress on one SDMA channel — not a fixable software serialization. The
only lever that could raise this ceiling is **more SDMA channels per peer** (parallel queues to lift
per-GPU egress), a global backend change (fixed at 1 channel, with a known chan≥2 deadlock risk) that
is out of scope for a per-collective quick win. **Recommend closing the root-egress plateau line
(Broadcast §5(a), Scatter, Reduce §5(a)) as "structural / needs multi-channel backend"** and moving
remaining effort to the exit-barrier / threshold quick wins (attack order #1) instead.

**Multi-channel probe — TESTED, no gain (2026-08-03, `gin-mc-probe.bash`).** Directly swept the "more
SDMA channels" lever above on broadcast @256 MiB (`-D 3`, warm), `NUM_CHANNELS × -V`:

| -V \ ch | ch1 | ch2 | ch4 | ch8 |
|---|---|---|---|---|
| V8  | 202.0 | 201.2 | 201.8 | HANG |
| V16 | 217.4 | 217.2 | 217.1 | HANG |
| V32 | 224.1 | 224.4 | **2.6** | HANG |

**Adding channels does nothing:** ch1≈ch2≈ch4 at every CTA count (identical to 3 s.f.), because the extra
SDMA queues share the *same physical xGMI links* — no new wire bandwidth. Worse, ch≥4 at high CTA counts
collapses (ch4/V32 → 2.6 GB/s) and ch8 **deadlocks** at every -V (matches the two-shot AllReduce chan≥2
note in `gin_sdma_collective_policy.h`). The only positive slope is **CTA count** (V8→V16→V32 =
202→217→224), but with clear diminishing returns — a single GPU's egress is saturating by V32. **Verdict:
the root-egress plateau is a hard single-GPU-egress bound, not a queue-parallelism deficit; multi-channel
is a dead lever. Root-egress plateau line CLOSED as structural.** (A V>32 confirmation run was
contaminated by another tenant's job pinning the shared node's GPUs at 100 %, so it is deferred; the V8–32
scaling and the channel-invariance above were captured clean before the contention and are sufficient.)

**Real device-side pipelined ring — IMPLEMENTED + TESTED, NEGATIVE (2026-08-03, `smci350-rck-g03-d07-21`, NP=4).**
The last remaining hypothesis was that a *true* van-de-Geijn pipelined ring (root→rank+1→…→rank+(N-1),
every rank forwarding each chunk to its successor so all N-1 links carry different chunks in parallel)
could distribute egress across all GPUs and beat the single-root SAG plateau. Implemented as
`GinRingBroadcastKernel` in `broadcast.cu` (single CTA drives the pipeline; root never waits; linear
pos0→…→pos(N-1) wait graph — no deadlock; opt-in via `NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES`,
`NCCL_GIN_ANVIL_BCAST_RING_CHUNKS`, default **OFF**). Gate is **green**: `#wrong=0` at every size
256 KiB→256 MiB for **both** out-of-place and in-place. But the warm A/B (ring ON vs SAG, 128 MiB→2 GiB)
is a large **regression**:

| size | SAG GB/s | RING GB/s (auto 64 chunks) | ring/SAG |
|---|---|---|---|
| 128 MiB | 108.3 | 12.1 | 11.2% |
| 512 MiB | 111.5 | 12.9 | 11.6% |
| 2 GiB | 112.4 | 13.1 | 11.7% |

Chunk-count sweep @256 MiB shows the ring is chunk-sensitive but never competitive:
K=2→30.8, K=4→40.7, **K=16→52.1 (best)**, auto-64→12.6 GB/s. Even the best point (~52) is **<½** of
SAG (~110).

**8-GPU confirmation (2026-08-03, `smci355-ccs-aus-m03-17`, NP=8) — verdict airtight.** Re-ran the same
gate+A/B on the canonical 8-GPU node. Gate again **green** (`#wrong=0`, all sizes, OOP + in-place). A/B:

| size | SAG GB/s | RING GB/s | ring/SAG |
|---|---|---|---|
| 128 MiB | 182.7 | 12.3 | 6.7% |
| 512 MiB | 156.1 | 13.1 | 8.4% |
| 2 GiB | 230.1 | 13.3 | 5.8% |

The ring is pinned at **~13 GB/s independent of N** (identical to the NP=4 run) — it is serial-bound by the
single-CTA/single-channel forward and does **not** scale with rank count — while **SAG scales up with N**
(→230 GB/s at 8 GPUs), so on 8 GPUs the ring is only ~6 % of SAG (worse than the 4-GPU ratio).

**Same root cause as the SAG-chunking and multi-channel dead ends:** with `NUM_CHANNELS=1`
the single CTA issues each forward put from `tid==0` and blocks on `waitSignal` before forwarding, so
receive and forward **cannot overlap on the one SDMA queue** — the "pipeline" runs serially hop-by-hop.
More channels can't help (ch≥4 deadlocks, per the probe above). **Verdict: a correct real ring does not
break the root-egress plateau under single-channel SDMA; SAG remains the best GIN broadcast large tier.
Ring kept OFF by default (opt-in only). Root-egress plateau line remains CLOSED as structural / needs a
multi-channel (parallel-queue) SDMA backend.**

**Attack (a) — SM-driven peer-store ring — IMPLEMENTED + TESTED (2026-08-03, `smci355`, NP=8).**
Replaced the ring's single-queue SDMA forward (one `gin.put` from `tid==0`) with **all-threads direct
xGMI peer stores** into the successor's recvbuff via `ncclGetLsaPointer` (the `BroadcastLsaDirect`
mechanism), striped across CTAs (each CTA = an independent ring pipeline on its slice, per-block LSA
barrier line) and pipelined with a per-step release barrier. New kernel `GinRingSmBroadcastKernel`;
selected by default when the ring tier is on (`NCCL_GIN_ANVIL_BCAST_RING_SM=0` falls back to the SDMA
ring for A/B). Gate **green** (`#wrong=0`, all sizes, OOP + in-place). Result — a **4.5× jump over the
SDMA ring** but still well under SAG:

| size | SAG | SDMA-ring | **SM-ring (a)** | SM/SAG |
|---|---|---|---|---|
| 128 MiB | 212 | 12.3 | **58.0** | 27% |
| 512 MiB | 226 | 13.1 | **58.3** | 26% |
| 2 GiB | 230 | 13.3 | **58.3** | 25% |

Chunk sweep @256 MiB (single-ring SM): C=4→28, 16→47, 32→54, **64→58** GB/s — rises with chunk count
then asymptotes at **~58 GB/s ≈ one xGMI link's SM-store rate**. I initially (wrongly) read this as a
"ring uses one link per GPU" structural ceiling. **That was incorrect** — see the multi-ring result next.

**Root cause of the single-ring ceiling = single ORIENTATION, not the ring topology (2026-08-03).**
`NCCL_DEBUG=INFO` on the host `-D 0` broadcast shows RCCL builds **`nChannels = 28`** rings with
*rotated orderings* (e.g. rank 4: `Ring0 6→4→1`, `Ring1 7→4→6`, `Ring2 7→4→1`, `Ring3 3→4→0`, …): each
GPU's successor differs per channel, so across the 28 rings every GPU drives **all** its xGMI links at
once → ~320 GB/s (**confirmed**: warm default-tuner host `-D 0` measures ~294 GB/s @2 GiB here; the
tuner must run with no `NCCL_ALGO`/`PROTO`/channel pin — pinning Ring/Simple+32ch mis-measures it at
~165 GB/s, see the 7-ring table below). My single-ring SM kernel launched ~32 CTAs but **all with
successor `rank+1`**, so
every CTA piled onto the *same one* link → ~58 GB/s. The ring topology was never the limit; using one
orientation was.

**Multi-ring SM broadcast — IMPLEMENTED + TESTED, near-parity with SAG (2026-08-03, `smci355`, NP=8).**
New kernel `GinRingSmMultiBroadcastKernel` (default when SM ring on; `NCCL_GIN_ANVIL_BCAST_RING_MULTI=0`
for the single-orientation A/B). It uses every stride `s` coprime to `N` as an independent Hamiltonian
ring (order root, root+s, root+2s, …; successor `(rank+s)%N`) and assigns CTA `b` stride
`strides[b % nStrides]`, so different CTAs forward on different links (N=8 → strides {1,3,5,7} = 4
links; N=4 → {1,3}). Each CTA owns a buffer stripe and runs the same SM-store pipeline (per-block LSA
barrier). Gate **green** (`#wrong=0`, all sizes, OOP + in-place). Warm A/B (8 GPUs):

| size | SAG | SDMA-ring | SM 1-ring | **SM multi-ring** | multi/SAG |
|---|---|---|---|---|---|
| 128 MiB | 212 | 12 | 58 | **164** | 77% |
| 1 GiB | 229 | 13 | 58 | **193** | 84% |
| 2 GiB | 230 | 13 | 58 | **196–218** | 85–95% |

CTA/chunk tuning @2 GiB: **V=32 (default) is best at ~218 GB/s (≈95% of SAG)**; V=64 oversubscribes
(~205); chunk count (64 vs 128 vs 256) is ~flat. So attack (a) done right takes the device ring from
13 → ~200+ GB/s (**~15×**) and to **near-parity with SAG**. The residual gap is that the coprime-stride
set uses 4 of the 7 links/GPU (offsets 1,3,5,7), whereas SAG's allgather touches all peers; closing it
would need an edge-disjoint Hamiltonian decomposition that also uses the offset-2/4/6 links (as RCCL's
28-ring set does). **Verdict: a device-side SM ring CAN match a multi-link design once it rotates
orientations; SAG still edges it (~5–15%) so SAG stays the default, but the ring is no longer a dead
end and both remain opt-in.**

**Full edge-disjoint 7-ring SM broadcast — IMPLEMENTED + TESTED, overtakes SAG *and* host (2026-08-04,
`smci355`, NP=8).** New kernel `GinRingSmTableBroadcastKernel` + host builder `buildBcastRingDecomp`.
The coprime-stride set only reaches 4 of a GPU's 7 xGMI links because the even offsets (2,4,6) don't
form single cycles (they split the 8 ranks into evens/odds). To use **all 7 links** we decompose the
complete symmetric digraph K₈* (56 arcs) into **7 arc-disjoint directed Hamiltonian cycles** — which
exists for all N≠4,6 (Tillson's theorem). The host computes the decomposition once per N by verified
backtracking (each ring a single N-cycle; every arc covered exactly once — asserted), uploads the
`succ`/`pos` tables to `__constant__` memory, and CTA `b` runs ring `b%(N-1)` on buffer stripe `b`.
N∈{4,6} (no decomposition) and unsupported N fall back to the coprime-stride kernel. Gate **green**
(`#wrong=0`, all sizes, OOP + in-place). Warm A/B (8 GPUs, `-V 32`):

| size | host `-D 0` (warm, default tuner) | SAG | **7-ring** | 7-ring/SAG | 7-ring/host |
|---|---|---|---|---|---|
| 128 MiB | 223 | 211 | 185 | 87.6% | 83% |
| 256 MiB | 256 | 221 | 208 | 94.0% | 81% |
| 512 MiB | 271 | 226 | 221 | 98.0% | 82% |
| 1 GiB | 292 | 228 | **228** | 100.0% | 78% |
| 2 GiB | 294 | 230 | **232** | 101.2% | 79% |

**Result (corrected):** the 7-ring device broadcast **edges past SAG at ≥1 GiB** (232 vs 230 @2 GiB)
and takes the device ring 58 → 232 GB/s — but it **does NOT overtake host-initiated**. It reaches
**~79% of host** at large sizes. The earlier "host ~320 GB/s" figure **reproduces** (~294 GB/s @2 GiB
warm here); an initial mis-measurement of 165 GB/s came from pinning **Ring/Simple + 32 channels**
(the `BC_C1` hybrid-mode config in `gin-sdma-bcast-test.bash`), which is ~1.8× slower than letting the
**default NCCL tuner** pick the algo/proto/channel count for large broadcast. Always baseline host
broadcast with the **default tuner** (`-D 0`, no `NCCL_ALGO`/`PROTO`/channel pin, `CUMEM=0 -R 0`);
the pinned Ring/Simple path is a red herring at ≥512 MiB.

**Corrected ceiling analysis:** the ~230 GB/s SAG/7-ring plateau is **NOT** the per-GPU egress cap
(single-GPU root egress ≈ **480 GB/s** fully saturated — §4.8.5 of the broadcast design plan, 60 GB/s
flat × 8). Host reaches 294 GB/s on the same fabric, so there is real headroom above 230. Going
4-stride (218) → all-7-links (232) is only +6%, which means the wall is **not** the link count either —
it is **not** the per-step LSA barrier (see the P2P negative result next) — most likely the SM-copy
per-link efficiency vs the host generic ring's LL/LL128 + copy-engine path, and looser stripe pipelining
than the host's many channels. At <512 MiB the 7-ring trails SAG (pipeline fill over more rings), so any
default-on should engage only ≥~512 MiB. The ring tier stays opt-in via `..._BCAST_RING_MIN_BYTES`; knobs
`NCCL_GIN_ANVIL_BCAST_RING_SM=1` + `..._RING_MULTI=1` (defaults) select the table kernel.

**Point-to-point per-hop signaling — IMPLEMENTED + TESTED, NEGATIVE result (2026-08-04, `smci355`,
NP=8).** Hypothesis: the per-step grid-wide `bar.sync` across all 32 CTAs was the ~230 wall, so
`GinRingSmP2PBroadcastKernel` replaces it with lightweight per-hop successor signaling — each hop
SM-copies chunk c into the successor's recvbuff then bumps ONLY the successor's per-CTA GIN signal
(`signalPeer`, an IPC/LSA atomic-add, no SDMA queue), and the successor waits on its own signal
(`gin.waitSignal`, acquire) before forwarding. Per-CTA signal `sig=blockIdx.x` has exactly one writer, so
the atomic count is race-free; only the entry barrier is kept. Gate **green** (`#wrong=0`). But A/B @2 GiB:

| size | SAG | barrier-table 7-ring | **P2P 7-ring** |
|---|---|---|---|
| 512 MiB | 226 | 221 | 170 |
| 1 GiB | 228 | 228 | 185 |
| 2 GiB | 230 | **232** | **191** |

**P2P is ~18% SLOWER than the barrier kernel**, so the per-step barrier was NOT the bottleneck — it
efficiently bundles the cross-GPU release fence with the sync and gives tight lock-step pipelining,
whereas P2P pays a `__threadfence_system` per chunk/hop (needed so all CTA threads' peer stores are
visible before the signal) and lets rings drift out of step. **Verdict: the barrier table kernel stays
the ring default; P2P is opt-in (`NCCL_GIN_ANVIL_BCAST_RING_P2P=1`) and retained only as a documented
dead end.** The path to actually beat host (294) is therefore NOT the sync mechanism but the copy path
itself (e.g. LL128-style flagged stores / copy-engine assist), or simply keeping SAG which already sits
at the same ~230 with less complexity.

#### LL128-style store path — NEGATIVE result (kept as marginal default)

Following the "copy path itself" lead, tested whether the store *type* is the gap. The reusable RCCL
device primitive `ncclLLA2ASession` (`ll_a2a.h`) reveals the mechanism: its `amdLLA2aStoreLine` pushes
peer writes with `__builtin_amdgcn_global_store_b128(dst, v, RCCL_SYSTEM_SYNCSCOPE)` — a **nontemporal,
system-scope 128-bit store** that is cross-GPU-visible with no separate `__threadfence_system`. Our
`BroadcastLocalCopy` used plain **cached** `uint4` stores. Added `BroadcastPeerCopyStream` (streaming
b128 peer stores) behind `NCCL_GIN_ANVIL_BCAST_RING_STREAM` (default 1) in the 7-ring table kernel.

Warm A/B, 8× MI355X, `-D 3`, `-c 0`, `-n 20 -w 5`, OOP (busbw GB/s):

| size | SAG | ring, cached uint4 | ring, streaming b128 |
|------|-----|--------------------|----------------------|
| 512 MiB | 226 | 224 | 226 |
| 1 GiB | 229 | 232 | 234 |
| 2 GiB | 230 | 236 | **238** |

Streaming stores are only **+0.8%** over cached at 2 GiB (gate green, `#wrong=0` OOP + in-place). **The
store type is not the bottleneck** — the peer copies already saturate the same ~235 ceiling, so the
232→294 gap is not the SM-copy store path. This also deflates the full LL128 idea: the only reusable
device primitive is **2×-overhead LL** (8 B payload per 16 B line: `{data,epoch,data,epoch}`), which for
the large tier would *cost* bandwidth, and the one cheaply-testable half (store type) shows no headroom.
Kept streaming as the marginal default; **not pursuing full LL128 flag reformatting.** Remaining
candidates for the gap: CTAs-per-link / stripe-count & chunk-depth parallelism (host uses 28 rotated
channels vs our 7 rings), or copy-engine (SDMA) assist running concurrently with SM stores.

#### CTA count / stripe parallelism — WIN, ring beats host (350 vs 294)

The next candidate — parallelism — was the real bottleneck all along. The ring launches
`deviceCtaCount` CTAs (set by the `-V` flag; the table kernel splits the message into `gridDim.x`
stripes and runs ring `blockIdx.x % 7` on each). **The entire campaign measured at `-V 32`** = 32 CTAs ×
512 threads = 16K threads, which badly under-occupies a ~256-CU MI355X and cannot hide xGMI latency.
Sweeping `-V` (pure runtime knob, no rebuild), warm, `-D 3`, `-c 0`, `-n 20 -w 12`, OOP (busbw GB/s):

| `-V` (CTAs) | SAG 2 GiB | ring 2 GiB |
|-------------|-----------|------------|
| 32  | 229 | 238 |
| 64  | —   | **345** |
| 96  | —   | 141 |
| 128 | 237 | **350** |

**Ring @ `-V 128` = 350 GB/s @ 2 GiB, gate green (`#wrong=0`, OOP + in-place), reproducible (350 measured
4×).** That is **+47% over the `-V 32` plateau (238)** and **119% of host's 294** — the ring now decisively
beats host-initiated broadcast. The knee is between 32 and 64 CTAs (238→345); 64 and 128 are within ~2%.

Two caveats:
- **SAG does NOT scale with CTA count** (229→237): this is a ring-specific win. The pipelined ring finally
  has enough parallel CTAs per link (128/7 ≈ 18 CTAs/ring) to saturate all 7 xGMI links; SAG's
  scatter+allgather structure is already CTA-saturated at 32.
- **Only power-of-2 CTA counts are safe.** `-V` ∈ {56, 84, 96, 112} collapse to 86–162 GB/s — a
  wave-quantization / occupancy artifact (a partial tail wave nearly doubles kernel time). `-V` ∈
  {32, 64, 128} land cleanly. `-V 128` is the practical optimum (the flag caps at 128).

**Productization gap:** the win depends on the caller passing `-V 128`; the default `deviceCtaCount` is
16. To make the ring deliver 350 unconditionally, the ring launch should pick its own CTA count (a
power-of-2 near 128) decoupled from `-V`, and `BroadcastGetDevCommRequirements` must request that many
LSA barriers (the table kernel indexes `devComm.lsaBarrier` by `blockIdx.x`, so launching more CTAs than
allocated barriers would corrupt/hang). Mirror AllToAll's `kA2aLsaMaxCtas` pattern
(`testLaunchDeviceKernelThresholdLLCtas`).

#### Productization — DONE, ring self-selects CTA count (350 by default)

Implemented the decoupling. `bcastRingCtas()` (broadcast.cu) picks the ring's own count — preferred 128
(or the `NCCL_GIN_ANVIL_BCAST_RING_CTAS` override) — then **clamps it to `-V`/`deviceCtaCount` as the
max**, rounds DOWN to a power of 2, and caps at [1,128]. So `-V` is an upper bound the ring never
exceeds, while a non-power-of-2 `-V` (e.g. 96) rounds down to 64 and cannot hit the wave-quant cliff.
The SM ring kernels (table / multi-stride / single) launch via a new `testLaunchDeviceKernelThresholdCtas`
(common.h) with that grid instead of `deviceCtaCount`; the P2P variant stays on `deviceCtaCount` (its
per-CTA GIN signals are sized to it). Both `BroadcastGetDevCommRequirements` variants allocate
`max(deviceCtaCount, bcastRingCtas())` LSA barriers so the ring grid is always backed.

The **`-V` default was raised 16 → 128** (common.cu: init, out-of-range fallback, help text) so the ring
gets its full CTA budget out of the box. NOTE: `-V` is global, so this also raises the default CTA count
for every other collective's device path in this tool.

Validation, 8× MI355X, `-D 3`, warm (busbw GB/s @ 2 GiB):

| launch config | 2 GiB | gate |
|---------------|-------|------|
| **default (no `-V`, now 128)** | **350** | `#wrong=0` OOP + in-place |
| `-V 32` | 238 | ring capped at 32 |
| `-V 64` | 345 | ring picks 64 |
| `-V 96` | 344 | rounds down to 64 (was 140 before) |
| `-V 128` | 350 | ring picks 128 |

The ring self-selects up to the `-V` ceiling and delivers **~350 GB/s by default**, gate green OOP **and**
in-place. **Net: GIN broadcast large tier = 350 GB/s = 119% of host's 294**, up from the 238 the campaign
had plateaued at. Files: `broadcast.cu` (`bcastRingCtas`, barrier sizing, ring launches), `common.h`
(`testLaunchDeviceKernelThresholdCtas` + `#else` stub), `common.cu` (`-V` default 128).

#### Ring enabled by default — DONE (replaces SAG as the large tier)

With the CTA fix in place, a warm crossover A/B (8× MI355X, default CTAs, `-c 0`) shows the ring beats SAG
at **every** size >= 2 MiB, so `kBroadcastRingMinDefault` was flipped `0 -> 2 MiB` (the same crossover the
SAG tier uses; the ring gate is checked first, so the ring becomes the default large tier and SAG is the
opt-out fallback via `NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0`):

| size | SAG | ring |
|------|-----|------|
| 2 MiB   | 16.7 | 31.1 |
| 8 MiB   | 54.8 | 93.1 |
| 32 MiB  | 127.6 | 180.3 |
| 128 MiB | 198.4 | 238.3 |
| 512 MiB | 228.1 | 342.0 |
| 2 GiB   | 236.9 | 350.3 |

Verified with **no `NCCL_GIN_ANVIL_BCAST_*` knobs set** (compiled defaults only): gate `#wrong=0` OOP +
in-place across 2 MiB–2 GiB, default perf @2 GiB = **349 GB/s**. Below 2 MiB the flat/LSA/LL hybrid still
handles the message. File: `gin_sdma_collective_policy.h` (`kBroadcastRingMinDefault = 2 MiB`).

#### Can the SDMA ring match the SM/CTA ring? — NO (per-link SDMA copy is the wall)

`NCCL_GIN_ANVIL_BCAST_RING_SM=0` selects the original `GinRingBroadcastKernel`: a **single-CTA, single**
ring where `tid 0` enqueues an SDMA `gin.put` per chunk to its one successor. Measured @ 512 MiB / 2 GiB
(8× MI355X, `-c 0`) vs the SM/CTA ring:

| ring | 2 GiB |
|------|-------|
| SM/CTA ring (default) | **350** |
| SDMA ring, channels ∈ {1,2,4} | **~13** |

Channels do not move it (13.1 / 13.3 / 13.1). A chunk-count sweep confirms it is not merely a per-chunk
`waitSignal` latency artifact — busbw rises with pipeline depth then **plateaus at ~13 GB/s** (CHUNKS 1→
5.9, 2→8.3, 4→10.3, 8→11.7, 16→12.6, 32→13.1). So ~13 GB/s is the SDMA ring's true ceiling: a single GIN
Anvil-SDMA queue over IPC/xGMI delivers only ~13 GB/s per link, vs the SM single-ring's ~58 GB/s per link
and the SM 7-ring's 350 aggregate. **Analysis:** broadcast-ring busbw ≈ per-link forward rate, so even a
hypothetical SDMA *multi*-ring driving all 7 links would top out near 7 × 13 ≈ 90 GB/s (optimistically,
assuming 7 independent SDMA engines with no xGMI contention) — still ~4× below the SM ring, which already
saturates xGMI at 350 (= 119% of host). **Verdict: the SDMA ring cannot match the SM/CTA ring**; the SM
store path is fundamentally faster per link and scales across all 7. SM ring stays the default; the SDMA
ring is retained only as an `SM=0` A/B baseline. (`ch=8` failed to run — likely an invalid channel count
on this backend — but ch≤4 already bounds the answer.)

#### In-device (wall_clock64) timing — DONE (broadcast joins the shared scaffold)

Broadcast was the only GIN-SDMA collective without a `deviceTime` hook (so `NCCL_GIN_ANVIL_DEVICE_TIMING`
could not measure it). Added it via the shared `gin_sdma_devtime.h` scaffold, mirroring AllGather: the
table-ring body was extracted into a `__device__` `ginRingSmTableBroadcastBody` reused by both the normal
kernel and a new persistent `GinRingSmTableBroadcastTimedKernel` (runs skip+loop bodies under one launch,
brackets the timed region with `wall_clock64` per CTA); `BroadcastDeviceTime` drives it through
`gin_devtime::measure` at the ring's own CTA count and only when the ring is the active tier (gated on
`bcastUseRing` + built decomposition, `g_bcastBuiltNRings`). Registered as the 8th `broadcastTest` field.
loop/skip via `NCCL_GIN_ANVIL_BCAST_DEVTIME_LOOP/_SKIP` (default 10/10).

Device-timed vs host/graph-timed (8× MI355X, default ring, `-c 0`, busbw GB/s):

| size | host/graph | in-kernel wall_clock64 (mode 2) |
|------|-----------|----------------------------------|
| 512 MiB | 340 | 315 |
| 1 GiB   | 346 | 321 |
| 2 GiB   | 349 | **324** |

The device metric (min(start)..max(end) grid busy window, per iteration, MPI-MAX across ranks) is ~8%
below the host wall-clock — expected, since it excludes host launch/stream/teardown and includes CTA
start/finish skew. This is the stricter "pure device-function" number the other GIN-SDMA collectives
report; **evaluate broadcast with `NCCL_GIN_ANVIL_DEVICE_TIMING=2`** (mode 1 augments with a
`#[bcast-devtime]` line). Refactor verified behavior-preserving: gate `#wrong=0` OOP + in-place (mode 0)
and in the device-only path (mode 2). File: `broadcast.cu` (body extraction, timed kernel, hook,
`gin_sdma_devtime.h` include, ops-struct field).

> Environment note: GIN (`-D 3`) would not initialize on `smci350` with the pre-existing images
> (`rocshmem-api: buffer register failed size 2097152` on the 13-day image; `librccl.so.1: undefined
> symbol rocshmem::envvar::log_flags` on the 9-day image — a rocSHMEM/RCCL version skew). A **fresh
> in-place rebuild** of the image from local source (`docker-gin-gda-sdma-build.bash`, rocSHMEM+RCCL+
> rccl-tests rebuilt consistently) resolved it; GIN then initialized cleanly (all_gather `-D 3`
> 156 GB/s, 0 wrong). Host `-D 0` was healthy throughout. The ring is baked into the rebuilt image.

### 9.4 ReduceScatter exit-barrier drop — DONE, confirmed win (own-writes-local pull)

Exit-barrier audit of the two remaining LSA-tier reduction collectives:

- **ReduceScatter** — single-tier LSA read-reduce: each rank reads its owned slice `[rank*count]` from
  every peer's **read-only sendbuff** and writes **only its local recvbuff** (`ncclGetLocalPointer`).
  This is the AllGather-pull shape (own-writes-local), so the trailing
  `lsaBar.sync(memory_order_release)` was redundant: no cross-rank write to publish, no memset race to
  fence. The entry barrier already guarantees peers' sendbuffs are filled before any read; a rank that
  finishes early cannot corrupt a slow peer's reads (sendbuff is never kernel-written) and the next
  collective's entry barrier resynchronizes before sendbuff is re-read. In the looped **timed** kernel
  each iteration's entry barrier is itself a full inter-iteration sync, so entry-only stays lockstep.
  **Dropped the exit barrier in place** in `ginReduceScatterBody` (the `-D 3` device kernel; edited
  directly since it is NOT used by any host-initiated collective — the host path goes through the
  library `ncclReduceScatter`, which is untouched).
- **Reduce** — audited, **exit barrier KEPT (required, not a candidate)**. Reduce writes each rank's
  reduced slice into the **root's** recvbuff via `ncclGetLsaPointer(recvwin,..,root)[r*base]` — a
  cross-rank push (root-ingress). The exit barrier guards completion/visibility of all non-roots' writes
  into the root's recvbuff before the collective returns and the root's output is verified. Dropping it
  would let the root read an incomplete result.

**Correctness:** RS GIN gate green, `#wrong == 0` across both in-place and out-of-place over the full
sweep.

**Perf (8× MI355X, warm re-baseline 2026-08-03 via `run_warm`, `-D 0` vs `-D 3`, OOP, `-c 0`, `-n 20 -w 5`;
host and GIN legs both warm):**

| size | M0 gin/host% | warm gin/host% | host GB/s | GIN GB/s |
|---|---|---|---|---|
| 256 KiB | ~78–100 | **103** | 20.8 | 21.4 |
| 512 KiB | — | **132** | 32.3 | 42.8 |
| 1 MiB | — | **114** | 63.5 | 72.5 |
| 2 MiB | 77.9 | **93.6** | 114.2 | 106.9 |
| 4 MiB | ~80 | **91.4** | 163.3 | 149.2 |
| 8 MiB | ~80 | **97.6** | 190.2 | 185.6 |
| 16 MiB | ~80 | **86.3** | 246.8 | 213.0 |
| 33 MiB | ~80 | **81.6** | 288.0 | 235.0 |
| 64 MiB | parity | **101.0** | 248.9 | 251.3 |

Net: the mid-band dip is substantially closed (2 MiB 78→94, 8 MiB 80→98) and small sizes now beat host;
large returns to parity. The residual 16–33 MiB trough (~82–86 %) is the same root-egress class as the
other reduction collectives, not a barrier issue. One barrier saved, race-immune, gate green.
