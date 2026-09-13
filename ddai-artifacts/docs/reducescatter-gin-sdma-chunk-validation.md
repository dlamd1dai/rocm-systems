# ReduceScatter GIN/SDMA tier — validation of the chunk-partitioned redesign

Validates `d65bb4bdda5` ("partition the RS SDMA tier by output chunk, not by peer")
on 8x MI355X (gfx950), `smci355-ccs-aus-m03-17`.

## Provenance

The image `rccl-gin-gda-sdma-rs-hybrid` was built from `d65bb4bdda5`; the
`reduce_scatter.cu` inside it is byte-identical to the worktree at that commit:

```
c23ad118ae7b703c1006ee3de1aad161  worktree projects/rccl-tests/src/reduce_scatter.cu
c23ad118ae7b703c1006ee3de1aad161  image /tmp/local/projects/rccl-tests/src/reduce_scatter.cu
c23ad118ae7b703c1006ee3de1aad161  image /workspace/rccl/src/projects/rccl-tests/src/reduce_scatter.cu
```

## Environment sanity

| Check | Result |
|---|---|
| `HIP_VMM_UNCACHED_MEMORY enabled` | yes, configure log line 265 (via the container's HIP 7.13.99004, not the host's 7.0.51831 — the other arm of the same CMake gate) |
| `Assigned GIN plugin gin-anvil-sdma to comm` | yes, on all 8 ranks |
| `NCCL_GIN_TYPE` | 5 (correct for this NCCL 2.30.4 base; 6 would silently load no plugin and pass vacuously) |

## Correctness re-derivation

**Signal count is exactly `nRanks`, self-put included.** The put loop is
`for (r = threadIdx.x; r < nRanks; r += blockDim.x)` with `blockDim.x == 512 > nRanks`,
so threads 0..`nRanks-1` each issue exactly one put and `r == devComm.rank` is
included — the self-put is issued and does increment. No put is issued twice.
Segmentation does not inflate the count: `gin_sdma::ginPutSegment` flags exactly
one segment final and "it alone carries the caller's SignalInc", so one put is
one increment regardless of size. Corroborated two ways: the AllToAll reference
kernel puts to all `r` including self and likewise waits for `nRanks`; and every
one of the ~136 completed runs here would have hung on its first launch if the
self-put did not increment.

**Empty trailing chunk does not wait.** `waitSignal`/`flush` sit inside
`if (chunkLen != 0)`, so a block with an empty chunk never waits for arrivals
that are never sent, and its fold range `[chunkOff/16, (chunkOff+0)/16)` is
empty. `chunkLen` is derived only from `sliceBytes` and `gridDim`, both uniform
across ranks, so an empty block is empty on every rank. Empty chunks are
reachable when `sliceBytes < 128 * gridDim`, i.e. total bytes `< 1024 * gridDim`
— at `RS_CTAS=15` that is total < ~15 KiB, *below* the 128 KiB probe, so
section E was added to actually hit it: at 4 KiB total the slice is 512 B and
the chunk is 128 B, leaving blocks 4..14 empty at `RS_CTAS=15`. All six E runs
completed, 0 wrong.

**Alignment and bounds hold for the final partial chunk.** `chunk` is rounded up
to 128 B, so `chunkOff = blockIdx*chunk` is a multiple of 128 and hence of
`sizeof(Pack)=16`. `sliceBytes = count*sizeof(T)` is a multiple of 16 because
`ReduceScatterGetCollByteCount` keeps `count` a multiple of `16/sizeof(T)`, so
the tail `chunkLen = sliceBytes - chunkOff` is also a multiple of 16 — `pkEnd` is
exact and no tail elements are dropped. The clamp
`(sliceBytes - chunkOff < chunk) ? sliceBytes - chunkOff : chunk` makes
`chunkOff + chunkLen <= sliceBytes` by construction, so
`pkEnd = (chunkOff+chunkLen)/16 <= nPacks` and every `Pack` load stays in bounds
and 16 B aligned.

**Pools cover the maximum reachable grid.** `reduceScatterDevReqs` is sized from
`max(reduceScatterMaxCtas()=48, deviceCtaCount, rsLaunchCtas)`, so
`barrierCount == lsaBarrierCount == ginSignalCount >= 48`. The SDMA tier is
`gridDim < kReduceScatterSdmaCtaCeil == 16`, so at most 15 signal indices are
live — covered with margin. This matters more than before the redesign, because
every block now uses its own signal index rather than only `min(gridDim,nRanks)`
of them, but the requirement is still `gridDim` and is satisfied. Since
`6e7d3abfb80` an explicit `NCCL_GIN_ANVIL_RS_CTAS` pin up to 128 is also covered.

## Sweep: 64 MiB total, busbw GB/s (out-of-place / in-place), `#wrong`

Every point is `#wrong = 0` in both placements.

| RS_CTAS | int32 oop | int32 ip | float oop | float ip | #wrong |
|---|---|---|---|---|---|
| 4  | 41.16 | 41.12 | (deadlock) | — | 0 |
| 7  | 65.60 | 65.64 | 61.24 | 61.35 | 0 |
| 8  | 73.20 | 73.25 | 68.61 | 68.60 | 0 |
| 9  | 74.50 | 74.68 | 74.77 | 69.89 | 0 |
| 10 | 81.38 | 84.04 | 76.16 | 78.49 | 0 |
| 12 | 90.32 | 90.85 | 87.63 | 85.31 | 0 |
| 14 | 110.18 | 103.59 | 94.57 | 94.91 | 0 |
| 15 | 97.68 | 108.93 | 102.35 | 99.52 | 0 |
| 16 | 277.28 | 275.97 | 272.05 | 272.96 | 0 |
| 32 | 378.44 | 378.35 | 377.95 | 379.81 | 0 |

The 15 -> 16 discontinuity still marks the tier switch at
`kReduceScatterSdmaCtaCeil = 16`: ~98–109 -> ~272–277 GB/s, a 2.5–2.8x jump.
The >= 16 rows are the untouched LSA read-reduce tier and are unchanged from
`6e7d3abfb80` (276.03/378 -> 277.28/378.4), confirming the redesign is confined
to the SDMA tier.

### Sub-16 cost of chunk partitioning vs `6e7d3abfb80` (int32, oop)

| RS_CTAS | 6e7d3abfb80 | d65bb4bdda5 | delta |
|---|---|---|---|
| 4  | 42.49 | 41.16 | -3% |
| 7  | 68.27 | 65.60 | -4% |
| 8  | 75.35 | 73.20 | -3% |
| 12 | 101.23 | 90.32 | -11% |
| 15 | 118.43 | 97.68–111.89 | -6% to -18% (run-to-run spread is wide here) |

Expected direction: chunk partitioning issues `nRanks * gridDim` puts per launch
instead of `nRanks`, so each SDMA copy is smaller and there are 4–15x more of
them. The cost grows with `gridDim`, which is consistent with per-descriptor
overhead rather than bandwidth loss.

## Other correctness points (all `#wrong = 0`)

| Case | Result |
|---|---|
| 1 MiB/rank (8 MiB total) and 8 MiB/rank (64 MiB total) at `RS_CTAS=4` | 0 wrong |
| 1 MiB and 8 MiB *total* at `RS_CTAS=4` | 0 wrong |
| 128 KiB, 256 KiB total at `RS_CTAS` 12 and 15 | 0 wrong |
| 4/8/16 KiB total at `RS_CTAS` 12 and 15 (empty-chunk path) | 0 wrong |
| float 1M..64M at `RS_CTAS=4` (the case that deadlocked `6e7d3abfb80`) | 3/5 completed, 0 wrong; 2/5 deadlocked |
| int32 1M..64M at `RS_CTAS=4` | 3/3 completed, 0 wrong |

## Residual: intermittent deadlock (NOT root-caused)

The corruption is gone — zero wrong elements in every completed run, across
~136 runs. What remains is a low-rate deadlock in `waitSignal`, which always
manifests as a hang with no wrong data, never as a miscompare.

Rate, from a balanced 50-run experiment (64 MiB, float, 10 reps per grid):

| RS_CTAS | pass | deadlock |
|---|---|---|
| 4  | 10 | 0 |
| 6  | 10 | 0 |
| 8  | 9  | 1 |
| 12 | 10 | 0 |
| 15 | 10 | 0 |

An earlier apparent "only `RS_CTAS=4` deadlocks" pattern (6 of ~21 cta4 runs, 0
of ~33 others) was **selection bias** — those rounds were heavily weighted toward
cta4. The balanced experiment puts cta4 at 10/10 clean and the single failure at
cta8, so the deadlock is not grid-specific. Overall rate is roughly 5% of
SDMA-tier runs. The LSA tier never deadlocked (0 of 7 multi-size runs).

Leading hypothesis, untested: `projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma_put_policy.h`
documents a hardware failure in exactly this shape — on the fused
`COPY_LINEAR_WAIT_SIGNAL_MI4` path "the fused copy never lands AND its SignalInc
never fires -> every rank spins forever in waitSignal (a HANG, not a data
miscompare)". That is the observed signature. The redesign issues 4–15x more
fused copy+signal descriptors per launch than the peer-partitioned version, so it
would sample any such flakiness proportionally more often, which would explain
why it surfaced now.

**Not established:** whether this deadlock predates the redesign. The pre-redesign
kernel was only run 3 times at `RS_CTAS=4`, and at a ~5% rate three passes is
unremarkable. The decisive test is to rebuild the pre-`6e7d3abfb80` kernel and run
the same 50-run protocol.

## Read on uncached staging

The consumer in this tier does plain 128-bit `Pack` loads straight out of the GIN
resource window, which is `hipMemAllocationTypeUncached` under
`HIP_VMM_UNCACHED_MEMORY`. `reducescatter-gin-sdma-phase2.md` warns that
`vmmUncached` is badly incoherent for bulk plain reads and that an agent-level
fence does not fix it.

This sweep is evidence **against** that warning applying to this pattern: zero
wrong elements across ~136 runs, int32 and float, in-place and out-of-place,
4 KiB to 64 MiB total, grids 4 to 15. The plausible reason it differs from the
Phase-2 microbenchmark is ordering, not memory type: here the consumer reads only
after `gin.waitSignal` on a signal the SDMA engine itself raises on copy
completion, preceded by `gin.flush` — the producer's completion is
engine-attested rather than fence-inferred.

Deliberately not overstated:

- One node, 8 ranks, one GPU model, one ROCm, sum only, <= 64 MiB. Green here is
  suggestive, not proof; it does not generalise to multi-node or other ranks.
- A coherence failure would be expected to be sporadic, and there *is* a sporadic
  residual here — but its signature is a hang with no wrong data, whereas
  incoherence would surface as a miscompare. So the residual is not obviously a
  coherence signal, and it should not be read as one without root-causing it.

On that evidence, pinned staging (Phase-2) looks like a **performance** option
for this tier rather than a correctness prerequisite. That conclusion is
conditional on root-causing the deadlock, which is a separate, non-coherence
defect.
