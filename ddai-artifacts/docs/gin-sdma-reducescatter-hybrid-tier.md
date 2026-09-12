# When a multi-tier hybrid ReduceScatter with SDMA is a good idea

Discussion captured from the `users/dondai/gin-stage3e-sdma-rs` reconciliation
(2026-09-12). Complements the measured Phase-2 investigation in
[`reducescatter-gin-sdma-phase2.md`](reducescatter-gin-sdma-phase2.md).

A hybrid SDMA ReduceScatter is a good idea when **SDMA moves bytes that LSA
SM-loads cannot**, or when **the extra hop is paid back by overlap or
hierarchy**. On today’s single-node full-LSA path it is usually a bad idea.

## Why the naive hybrid lost

AllGather and Broadcast are **movement** collectives: an SDMA `put` *is* the
algorithm. ReduceScatter is a **reduction**: someone must fold N values.

The earlier RS hybrid was:

1. Each rank **SDMA-puts** its partial into the owner’s GIN scratch window.
2. The owner **SM-reduces** all N partials from that **one local buffer**.

On 8× MI355X with uncached VMM that lost for two structural reasons (not “we
didn’t tune the threshold”):

- **Extra hop.** Data already sits in peer sendbuffs. Staging it locally, then
  reading it again, is strictly more traffic than `ncclGetLsaPointer` + fold.
- **Lost read parallelism.** Direct LSA RS spreads the reduce across **N peer
  memories / xGMI links**. Put-partials serializes those reads onto **one local
  buffer**, so you fight incast plus a second memory trip instead of N-way
  xGMI reads.

The shipped `GinReduceScatterKernel` therefore stays **single-tier LSA
read-reduce** for all sizes. The CTA ladder and mid vs large **load schedule**
still change with size; the **algorithm** does not. `RSTier` and the 256 KiB
`NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCESCATTER` cutover are reserved for
forward-compat / documentation.

Hybrid AllGather / Broadcast do not have that problem: there is no fold, so
SDMA replaces SM copies instead of adding a stage in front of them.

Phase-2 measurements on the same node show SDMA can drive the *same fabric
traffic* ~7–12% harder than CU vector loads (AllGather ~420 GB/s vs RS CU
~376 GB/s at 1 GiB). A **serial** scatter-then-reduce still regresses
(~356 GB/s at 1 GiB). Only a **pipelined** scatter overlapping local reduce
can approach the ~410 envelope. See
[`reducescatter-gin-sdma-phase2.md`](reducescatter-gin-sdma-phase2.md).

## When a multi-tier SDMA RS *would* be worth it

A second tier pays off only if at least one of these is true.

### 1. LSA is missing or incomplete

Not all ranks share an LSA team (multi-node, partitioned P2P, GIN-only peers).
Then you *must* GIN/SDMA the remote slices. The natural hybrid is **LSA
read-reduce inside the node, SDMA/GIN across nodes** — not a size split on a
full-mesh LSA node.

### 2. Hierarchy beats an N-way pull

If the topology is teams (node / rail / switch), reduce **inside** the LSA
team first, then **SDMA-put a smaller partial** to the owner. Bytes on the
slower link drop from `(N−1)×slice` toward `(teams−1)×slice`. That is the
case where “put-partials” is the right shape, because the partial is already
reduced.

### 3. SDMA and SM reduce overlap, not serialize

A pipelined ring or tree: SDMA delivers chunk `k+1` while SMs reduce chunk
`k`. The naive design was **stage all, then reduce**. Overlap can hide the
extra hop once the slice is large enough that copy time ≫ kernel launch/sync
(typically hundreds of MiB+, platform-dependent).

Phase-2 notes that this overlap is contention-free if reduce reads are **local
HBM** while SDMA uses **xGMI ingress**; they share only HBM bandwidth, which
has ample headroom. Staging must not use a `vmmUncached` window (incoherent
for bulk plain reads); it needs coarse-grained `PINNED` mapped into the LSA
aperture.

### 4. SM copies are the limiter, not xGMI read

If LSA loads steal CUs from the ALU fold, or SM load BW saturates before the
fabric, offloading **movement** to SDMA engines and keeping SMs on **combine**
can win. That needs:

- SDMA write BW ≥ SM remote-load BW on that part, **and**
- a reduce that does **not** re-read all N copies from one scratch buffer
  (streamed/tree reduce, or reduce-then-put).

On current MI355X uncached VMM, SM remote loads already fill much of xGMI;
naive SDMA staging did not add BW. A pipelined design targeting the AllGather
SDMA ceiling is the remaining ~9% large-message bet.

### 5. You need bounded concurrency, not N-way incast

At some sizes 48 CTAs already **hurt** RS (xGMI incast). A **ring RS**
(Broadcast-like): each hop is one SDMA put + one local combine, concurrency is
O(1) per rank. That can beat a full pull when N is large or the fabric hates
N-to-1 reads — even if a 2-rank LSA pull is faster.

### 6. Caching / memory type actually differs

The discarded hybrid assumed scratch might be “better” memory. Under
`HIP_VMM_UNCACHED_MEMORY`, sendbuff **and** the GIN window are uncached, so
there is no cache win from staging into the resource window. A hybrid becomes
interesting again if user buffers are **cached** (or host/fine-grain) and SDMA
can DMA from/to a path SMs cannot use well (for example uncached staging,
PCIe, or a future engine that bypasses CU load pipes). Phase-2 additionally
requires **PINNED** staging if SDMA writes are consumed by plain CU loads.

## Practical cut: size split vs topology split

| Split | Usually good? | Why |
| --- | --- | --- |
| **Size** (small LSA, large SDMA put-all-partials, no overlap) | No, on full LSA single node | Same algorithm plus a hop; LSA pull already wins through 2 GiB |
| **Size** (large pipelined SDMA-scatter + overlapped local reduce) | Maybe, ≥64 MiB | Physics de-risked; ~+9% ceiling vs current CU path; RCCL staging-window work |
| **Topology** (LSA local, SDMA remote / inter-node) | Yes | SDMA is the only path; hybrid is required |
| **Hierarchical reduce-then-put** | Yes, if teams exist | Fewer bytes on the slow link |
| **Pipelined ring + SDMA** | Maybe, large N or incast-bound sizes | Bounded hops, overlap copy and combine |
| **Schedule-only “tiers”** (CTA count, unroll) | Already doing this | No extra hop; this is the right “multi-tier” on MI355X today |

A size-based LSA↔SDMA switch like AllGather’s 256 KiB threshold is a good idea
for RS **only after** a large-tier algorithm that is **not** “put N raw
partials, then reduce locally.” Until that exists, keep RS single-tier LSA and
treat `NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCESCATTER` as documentation.

The experiment most likely to beat the current kernel is **intra-node LSA
reduce + inter-node SDMA of reduced partials**, or the Phase-2 **pipelined
SDMA-scatter + local reduce** on a coarse PINNED window — not a second size
tier of naive put-partials on 8 GPUs of one node.
