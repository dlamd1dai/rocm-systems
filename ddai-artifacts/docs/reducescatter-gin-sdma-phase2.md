# ReduceScatter (gin-sdma `-D 3`): Phase-2 performance findings

Work on the GIN-SDMA ReduceScatter device kernel (`rccl-tests/src/reduce_scatter.cu`)
and an investigation into whether SDMA copy engines can push large-message
ReduceScatter past the CU load-reduce ceiling. All numbers on **8× MI355X
(gfx950)**, single node xGMI, `NCCL_GIN_TYPE=5`.

- Image: `rccl-gin-gda-sdma-713-rs` (host symmetric RS device kernel present, for the `-D 0` baseline)
- Tool: `rccl-tests/reduce_scatter_perf`, `-g 1 -R 2 -V 32 -n 20 -w 5 -o sum -d float -z 0`
- Metric: **bus bandwidth (busbw, GB/s)**, higher is better. "Size" = total send buffer (`N·count·elt`).
- Scripts: [`gin-rs-hostvsgin-sweep.bash`](../scripts/gin-rs-hostvsgin-sweep.bash),
  [`gin-fabric-ceiling.bash`](../scripts/gin-fabric-ceiling.bash);
  microbench: [`microbench/coarse_xgmi_coherence.hip`](../microbench/coarse_xgmi_coherence.hip)

---

## Phase-2a (shipped): port the host LD load schedule into the large path

The host symmetric ReduceScatter LD kernel
(`projects/rccl/src/device/symmetric/reduce_scatter.cuh`, `reduceDeep`) uses the
**same algorithm** as our GIN kernel — direct LSA read-reduce from every peer's
symmetric buffer, local write — over the *same* uncached windows (both are
`hipMemAllocationTypeUncached` under `HIP_VMM_UNCACHED_MEMORY`). Its ~4% edge came
purely from two load-scheduling techniques our large path lacked:

1. **Peer-unroll (`UnrollPeers=2`)** on top of pack-unroll (`UnrollPacks=4`): issue
   `2·U = 8` outstanding 128-bit loads before any reduce, overlapping consecutive
   source ranks' xGMI read latency (we previously pack-unrolled but consumed one
   source rank at a time, leaving the per-source dependency chain exposed).
2. **Software-pipelined seed**: prefetch the next full tile's source-0 packs while
   the current tile reduces peers and writes output.

Both preserve the ascending source-rank fold (bit-for-bit identical to
`verifiable.cu`); the mid/small (`< 48 MiB`) and tail paths are unchanged. Full
functional gate and 128 B–2 GiB sweep are correctness clean (`#wrong=0`).

### Result — busbw (GB/s), same binary for both `-D 3` (GIN) and `-D 0` (host)

| Size | Phase-1b GIN | **Phase-2a GIN** | host `-D 0` | GIN vs host |
|---|---:|---:|---:|---:|
| 64 MiB | 249 | 249.3 | 244.6 | **+1.9%** |
| 128 MiB | 302 | 304.2 | 296.6 | **+2.6%** |
| 256 MiB | 338 | 341.7 | 338.1 | **+1.1%** |
| 512 MiB | — | 363.8 | 366.0 | −0.6% |
| 1 GiB | ~373 | 376.3 | 383.3 | −1.8% |
| 2 GiB | 377 | 384.6 | 392.0 | −1.9% |

GIN-SDMA ReduceScatter now **beats the host CU kernel at 64–256 MiB** and is
**98–99.4%** of it at 512 MiB–2 GiB (up from ~96%), at the shared CU-load xGMI
ceiling. The mid/small range is unchanged (adaptive peer-unroll-2 grid-stride).

---

## Phase-2b (investigated, not implemented): SDMA-scatter + local reduce

### 1. There is headroom above the CU ceiling

`all_gather` is the exact dual of `reduce_scatter` (identical fabric traffic), so
its busbw is the apples-to-apples fabric ceiling. Measured at 1 GiB via
`gin-fabric-ceiling.bash`:

| Collective (`-D 3`) | Engine | busbw @ 1 GiB |
|---|---|---:|
| `all_gather` | SDMA push | **~420** |
| `alltoall` | SDMA push | ~407–414 |
| `reduce_scatter` (our CU kernel) | CU LSA read | ~373 |

**SDMA drives the same traffic ~7–12% harder than CU vector loads.** So the CU
path is *not* xGMI-saturated; SDMA extracts more from the links.

### 2. The design and why pipelining is mandatory

ReduceScatter ≡ **scatter (one alltoall) + local reduce**: each rank SDMA-pushes
its slice-`r` to rank `r`'s per-source staging (runs at the ~410 alltoall ceiling),
then rank `r`'s CUs sum the N staged slices locally.

A naive *scatter-then-reduce* is a **regression**. At 1 GiB (per-rank slice
128 MiB, N=8): scatter ≈ 2.13 ms (410 busbw) + serial local reduce ≈ 0.33 ms
(read `N·count` from local HBM) ⇒ ~2.46 ms ⇒ **~356 busbw, worse than today's 376.**
The reduce must be **overlapped** with the scatter (segmented pipeline: reduce
segment *i* while SDMA scatters *i+1*). Only then does total ≈ scatter ≈ ~410.
The overlap is contention-free: reduce reads are local HBM (no xGMI), SDMA uses
xGMI ingress; they share only HBM bandwidth (~8 TB/s), which has ample headroom.

### 3. Backend feasibility (code investigation)

Anvil-SDMA source (`gin_anvil_sdma.h`, `gin_plugin_anvil_sdma.cc`, `dev_runtime.cc`):

- **Addressing — OK.** The SDMA `COPY_LINEAR` descriptor targets a plain peer VA
  from the LSA aperture (`resolveRemotePeerVa`). Cacheability is invisible to the
  DMA engine — a coarse-grained window is written correctly.
- **Registration — needs an RCCL change.** `ginAnvilRegMrSym → ncclDevrGetLsaSelfAddr`
  only accepts buffers mapped into the symmetric-VMM LSA aperture; a plain
  `hipMalloc` pointer is rejected. All window backings use the
  `HIP_VMM_UNCACHED_MEMORY`-gated `cuMemCreate`. A coarse-grained staging window
  must be allocated via the symmetric allocator forced to
  `CU_MEM_ALLOCATION_TYPE_PINNED` and mapped into the LSA aperture.
- **Coherence — no consumer-side L2 invalidate exists.** `waitSignal` is a
  system-scope acquire on the *flag*; `flush` uses `__threadfence_system()` +
  SDMA quiesce. The design explicitly relies on windows being fine-grained
  (HW-coherent). No `buffer_inv`/L2 flush anywhere in the GIN path.

### 4. Coherence + speed microbenchmark (`coarse_xgmi_coherence.hip`)

Standalone HIP, zero RCCL/GIN dependency, mirroring GIN's model (remote GPU writes
a buffer living on the consumer GPU, sets a flag with system-release; consumer
system-acquires the flag, then reads). Stale-element counts over 64 epochs.

**Speed (local CU read, 256 MiB):** every buffer type — `coarse` (hipMalloc),
`finegrain`, `vmmUncached` (the GIN window type), `vmmPinned` — reads at
**~5400 GB/s** plain, ~3600 nontemporal, but only **36 GB/s** for per-element
system-acquire atomic loads.

**Coherence with a faithful SDMA producer** (`hipMemcpyPeerAsync`, i.e. the copy
engine, completion stream-ordered before the flag = GIN's `put → flush → signal`):

| buffer | plain | agent-fence+plain | sys-atomic |
|---|---:|---:|---:|
| coarse | **0** | 0 | 0 |
| vmmPinned | **0** | 0 | 0 |

**0 stale even with plain reads.** (A CU-based producer showed non-deterministic
small stale counts — those were producer-side write/flag ordering races in the
microbench, not consumer L2 staleness; GIN's `flush`/quiesce provides exactly the
producer-side ordering that makes the SDMA producer clean.)

### 5. Verdict and corrections

- **Design is physically viable and coherence-safe.** SDMA-written coarse/pinned
  staging is read correctly by plain CU loads at ~5400 GB/s once the write
  completes before the signal — which GIN already guarantees. No exotic fence
  strictly required (a one-shot `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent")`
  is harmless insurance and independently gave 0 stale on coarse).
- **The "uncached = 144 GB/s" Phase-1 premise was a misdiagnosis.** Raw local reads
  of *every* type (incl. `vmmUncached`) are ~5400 GB/s. Whatever throttled the old
  put-partials path was the staging round-trip/pattern, not uncached read
  bandwidth. (Does not change the current kernel, already 97–100% of host.)
- **Must not stage into a `vmmUncached` window** — it is the one type that is badly
  incoherent for bulk plain reads (an agent-fence does not fix it). Staging must be
  coarse-grained (`PINNED`).

### 6. Remaining work if pursued (paused here)

1. RCCL: allocate a dedicated `CU_MEM_ALLOCATION_TYPE_PINNED` staging window mapped
   into the LSA aperture (alongside the uncached windows); rebuild `librccl` + image.
2. Kernel: segmented **pipeline** — SDMA-scatter segment *i* while CUs reduce
   segment *i−1* from local staging (per-segment completion signals).
3. Validate correctness across the gate; benchmark against the 376 (current) /
   ~410 (target) envelope.

**Best case ≈ +9% (410 vs 376), large messages (≥64 MiB) only** — a range where
Phase-2a already beats host at 64–256 MiB. Physics is de-risked; this is a
go/no-go on the RCCL-rebuild engineering investment.
