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
