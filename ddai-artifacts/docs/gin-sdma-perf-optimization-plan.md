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
- **Baselines per size:** capture (host `-D 0`, current `-D 3`, roofline) → gap table.
- **Guardrails:** after **every** change, re-run the collective's correctness gate
  (`gin_sdma_gpu_functional.sh <coll>` → `#wrong == 0`, full sweep + root/op/type where
  applicable) before trusting a perf delta.

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

| Collective | Small (≤256 KiB) | Mid (1–33 MiB) | Large (≥64 MiB) | Worst point | Verdict |
|---|---|---|---|---|---|
| **broadcast** | strong (150–210 %) | fading (85→82 %) | **71–79 %** | 71 % @ 1–2 GiB | Large SAG plateau ~229 GB/s (host 320) |
| **all_gather** | dip (min **68 %** @128 KiB) | recovering (83–195 %) | excellent (122–195 %) | 68 % @ 128 KiB | Mid LSA-band dip |
| **scatter** | strong (135–250 %) | **cliff 57.6 % @2 MiB** | 79–91 % | **57.6 % @ 2 MiB** | LSA↔GIN threshold cliff + ~10 % large gap |
| **gather** | strong (150–214 %) | 108–121 % | 100–107 % | 99.6 % @ 2 GiB | Healthy — no action |
| **sendrecv** | strong (142–200 %) | 102–113 % | 99–102 % | 98.8 % @ 2 GiB | Healthy — ~62 GB/s ring ceiling |
| **reduce_scatter** | ~78–100 % | **77.9 % @2 MiB**, ~80 % 2–33 MiB | parity (98–101 %) | 77.9 % @ 2 MiB | Mid-band dip + above-host-ceiling opportunity |
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
