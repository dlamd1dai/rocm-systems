# AICOMRCCL-1966 work plan: `ncclGinFenceLevel` barrier semantics

- **JIRA:** [AICOMRCCL-1966](https://amd-hub.atlassian.net/browse/AICOMRCCL-1966)
- **Summary:** GIN Enhancements: Added proper `ncclGinFenceLevel` semantics for barriers (NCCL 2.30.7 release-note line item).
- **Branch:** `users/dondai/aicomrccl-1966-gin-fence-level` (from `upstream/develop`, pushed to `origin`)
- **Status of ticket:** Open; no comments or linked issues at planning time.

JIRA required workflow: research → tests → end-user RCCL docs → run tests to see if enabled → if not functional, enablement work (this ticket **owns** the backend drain fix; do not split to a child ticket).

---

## Locked decisions

| Question | Decision |
|---|---|
| Release backends | **Proxy** (`NCCL_GIN_TYPE=2`) **and Anvil SDMA** (`NCCL_GIN_TYPE=6`). GDA is out of scope. |
| Work order | **Three stages** (do not skip ahead): (1) single-node everything, (2) multi-node **proxy only** on MI300/MI350/MI355, (3) multi-node **proxy and SDMA** on MI450. |
| `gin.get` | **Required in Stage 1** on both backends (single-node). |
| AllContexts + `useWorldForFence` | AllContexts in **Stage 1**. `useWorldForFence` is **Stage 2** (proxy 2×2 on MI300/350/355) and **Stage 3** (SDMA 2×2 on MI450). |
| Drain / flush bugs | **Fix on this ticket**, but Stage 1 only needs intra-node drain. Cross-node proxy drain is Stage 2; cross-node SDMA drain is Stage 3. |
| Examples | **Leave as-is** (no example or README edits). Docs only in userguide + how-to. |
| MI450 | Generally **no allocation**. Stage 3 is last; skip-gate only until hardware exists. |

**Coverage target:** Standard on fence branches, accumulated across stages (Stage 1 does not need `useWorldForFence` true).

---

## Work stages (execution order)

Do **not** start Stage 2 until Stage 1 Put/Get/AllContexts pass on **one node** for **both** proxy and SDMA. Do **not** start Stage 3 until Stage 2 proxy multi-node is green on MI300/350/355.

### Stage 1 — Single-node, all backends (NOW)

**GPUs:** MI300 / MI350 / MI355 (MI450 single-node is nice-to-have, not required).  
**Layout:** 1 node × ≥2 GPUs. `RCCL_ENABLE_INTRANET=1`. SDMA payload above threshold.

**In scope**
- Spike + tests: Put, self-put, Get, default `Put|Get`, AllContexts Put/Get
- SDMA `ncclGinApi_Get` / `FlushAsync` / `Wait` so Get and AllContexts do not trap
- Docs for fence-level choice (can land with Stage 1)
- Existing Relaxed arrival tests still pass

**Out of scope**
- `useWorldForFence` / hybrid Put from a non-rail peer (world == LSA on one node)
- Cross-node proxy or SDMA

**Stage 1 merge bar:** visibility tests PASS with `NCCL_GIN_TYPE=2` and `=6` on one node. AllContexts does not trap.

### Stage 2 — Multi-node proxy only (MI300 / MI350 / MI355)

**After Stage 1 is green.** Same GPUs, ≥2 nodes, `NCCL_GIN_TYPE=2` only.

**In scope**
- Cross-node Put/Get drain over IB
- Proxy 2×2 `useWorldForFence` / hybrid Put from a non-rail peer
- Timeout + fence on the network path

**Out of scope**
- Multi-node SDMA on these GPUs — **SKIP** (unsupported). Do not treat that SKIP as a Stage 2 failure.

### Stage 3 — Multi-node proxy and SDMA (MI450)

**Last.** Needs MI450; **deferred** (no allocation for the most part).

**In scope when hardware exists**
- Repeat Stage 2 proxy on MI450
- Multi-node SDMA Put/Get, AllContexts, `useWorldForFence` 2×2
- Cross-node SDMA drain (not only local queue quiet)

Until then: GPU-family SKIP for multi-node SDMA on MI300/350/355; do not block Stage 1 or 2.

---

---

## What “proper fence semantics” means

Before 2.30, `ncclGinFenceLevel` was effectively a boolean (`Relaxed` vs drain). In current RCCL (NCCL 2.30.7) it is a **bitmask**:

| Value | Meaning after `sync` returns |
|---|---|
| `None` | Arrival only. No put/get drain. |
| `Put` | Puts **from other team members targeting this rank** (including **self-puts**) issued before the barrier are visible in this rank’s memory. |
| `Get` | Gets **issued by this rank** before the barrier have landed in **local** memory. |
| `Put \| Get` | **Default** if the caller omits `fence`. Strongest guarantee. |
| `Relaxed` | Deprecated alias for `None` (source compatibility). |

Implementation: `ncclGinBarrierSession::syncInternal` in `projects/rccl/src/include/nccl_device/impl/gin_barrier__funcs.h`.

- **Put** includes the calling rank in the signal/wait set (`nPeerSigs = nRanks` vs `nRanks - 1`) so self-puts get the same visibility as remote puts.
- **Get** runs a post-signal `flush` so peers do not wait on our outstanding gets.
- **`ncclGinAllContexts`** fences **every** GIN context (puts/signals on different QPs are unordered at the NIC).
- Hybrid `ncclBarrierSession` with **Put** and **non-railed** GIN contexts uses the **world** GIN barrier (`useWorldForFence` in `barrier__funcs.h`) so Put visibility is not limited to the rail team.

A world-team hybrid barrier with `Put` is **not** “LSA + rail GIN”; it is a **world GIN** barrier. That distinction is only observable when ranks span nodes (rail ⊂ world). On MI300/MI350/MI355, SDMA is single-node so world == LSA — do **not** use those GPUs to sign off `useWorldForFence`. On **MI450**, multi-node SDMA can exercise rail ⊂ world the same way proxy does.

### How users should pick a fence

- **`None`:** kernel already completes work with `waitSignal` / `waitCounter` / explicit `gin.flush`; barrier is only phase alignment (current AlltoAll examples).
- **`Put`:** next local/remote read of *this rank’s* buffer must see inbound puts without a separate remote signal wait.
- **`Get`:** after `gin.get`, next local use of the destination must see those gets without an extra flush.
- **Default `Put\|Get`:** mixed one-sided ops; extra self-signal (Put) and extra flush (Get).
- **`ncclGinAllContexts(comm)`:** puts/gets were issued on more than one context, or that cannot be proven. Single-context `ncclGin` only drains that QP.

---

## Current tree state (at planning)

| Area | Status |
|---|---|
| Header API + operators | Present (`gin_barrier.h`) |
| Device implementation | Present (single-ctx, all-ctx, hybrid world-fence) |
| User-guide API reference | Good — `docs/userguide/source/api/device_gin.rst` documents bits, default, `Relaxed` |
| How-to | Stale — still RCCL 2.30.4 / NCCL 2.30.3; timeout snippet uses `Relaxed` |
| Examples / kernels | Use `Relaxed`/`None`; **do not change** per locked decision |
| Tests | Gap — GIN barrier tests only pass `Relaxed`. They assert lockstep **completion**, not **memory visibility**. No `Put`, `Get`, `Put\|Get`, or `ncclGinAllContexts` tests. |

Existing arrival tests (`Barrier_TwoRanks`, `Barrier_FourRanks`, `Barrier_WorldTeamUsesWorldPool`, hybrid/LSA, timeout) stay. LSA-only hybrid already checks store visibility after the barrier; GIN Put/Get tests should copy that pattern.

This ticket is **validation + docs + AMD drain enablement**, not a re-port of the enum.

### SDMA drain gap (critical path)

Anvil SDMA (`projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma.h`):

- **Implemented:** Put, PutValue, signals/counters, `ncclGinApi_Flush` (`sdma_anvil::quiet` on dirty queues).
- **Stubs (`__builtin_trap()`):** `ncclGinApi_Get`, `FlushAsync`, `Wait`.

Barrier **Get** and AllContexts Put drain call `fenceFlush`:

- Single context → `ncclGin::flush` → SDMA Flush exists.
- AllContexts → per-(ctx, peer) `flushAsync` + `wait` → **SDMA traps**.

`gin.get` on SDMA also traps, so Get-fence tests cannot issue a get until Get is implemented.

**Spike expectation:** proxy mostly green; SDMA Get + AllContexts red until Phase 2.

SDMA falls back to proxy below `NCCL_GIN_ANVIL_SDMA_THRESHOLD` (default 128). Fence tests must use sizes **above** the threshold so SDMA is hit, plus at least one run **at/below** threshold for mixed proxy+SDMA fencing.

Do not paper over SDMA by routing AllContexts flush only through proxy.

---

## Phase 0 — PR test plan

Write the test plan in the PR body (RCCL feature-unit-testing practice: do not commit `TEST_PLAN.md`). State the **three stages** and that only Stage 1 is the current bar.

- Feature: bitmask fence on `ncclGinBarrierSession` / `ncclGinBarrier` / hybrid `ncclBarrierSession`.
- **Stage 1 hardware:** 1 node × ≥2 GPUs, MI300/350/355, `RCCL_ENABLE_INTRANET=1`, `NCCL_CUMEM_ENABLE=1`, `NCCL_DMABUF_ENABLE=1`. Run **both** `NCCL_GIN_TYPE=2` and `=6` (SDMA payload above threshold; `--rocshmem-gin` for SDMA).
- **Stage 2 hardware (later):** ≥2 nodes, same GPUs, **proxy only**.
- **Stage 3 hardware (last):** MI450 multi-node, proxy **and** SDMA. SKIP-gate now; no allocation required.
- **Stage 1 acceptance:** Put + Get + AllContexts PASS on single-node proxy and SDMA. Do not require hybrid world-fence.

---

## Topology split (GPU generation)

SDMA node support is **not** “always single-node.” It depends on the GPU:

| GPU | SDMA collectives | Notes |
|---|---|---|
| MI300, MI350, MI355 | **Single-node only** | Intra-node Anvil SDMA + LSA/IPC. Multi-node SDMA is unsupported — SKIP, do not schedule as a failure. |
| MI450 | **Single-node and multi-node** | Multi-node SDMA **not in this PR’s merge bar** (no MI450 allocation). Keep SKIP gates so Job C can run later without a test rewrite. |

| Backend + GPU | Layout | Stage | What it can prove |
|---|---|---|---|
| Proxy MI300/350/355 | 1 node × ≥2 GPUs | **1** | Intra-node Put/Get, AllContexts (intranet) |
| SDMA MI300/350/355 | 1 node × ≥2 GPUs | **1** | Same; SDMA drain |
| Proxy MI300/350/355 | 2 nodes × 1 GPU | **2** | Cross-node Put/Get over IB |
| Proxy MI300/350/355 | 2 nodes × 2 GPUs | **2** | `useWorldForFence`; hybrid Put from a non-rail peer |
| Proxy MI450 | 1 node / multi-node | **1** then **3** | Same as proxy above when hardware exists |
| SDMA MI450 | 1 node × ≥2 GPUs | **1** | Same intra-node as MI300 SDMA |
| SDMA MI450 | 2 nodes × 1 or 2×2 | **3** | Cross-node SDMA Put/Get and world-fence |

Gate tests by **backend + GPU family + node count**, not a single `crossNodeReason()` for all SDMA. Suggested helper: skip multi-node SDMA unless the device is MI450 (detect via HIP arch / `hipGetDeviceProperties`). Single-node GIN still needs `intranetReason()`.

---

## Spike (Stage 1 only — do this first)

**One node**, both backends. Do **not** allocate multi-node until Stage 1 is green.

```text
# Single-node proxy (intranet)
NCCL_GIN_TYPE=2 NCCL_CUMEM_ENABLE=1 NCCL_DMABUF_ENABLE=1 RCCL_ENABLE_INTRANET=1

# Single-node SDMA
NCCL_GIN_TYPE=6 NCCL_GIN_ANVIL_SDMA_THRESHOLD=128 NCCL_CUMEM_ENABLE=1 \
NCCL_DMABUF_ENABLE=1 RCCL_ENABLE_INTRANET=1
# Optional: NCCL_P2P_DISABLE=1 to keep traffic on GIN/SDMA rather than XGMI P2P.
```

Kernels (both backends):

1. **Put visibility:** inbound `put` (no `waitSignal`), then `sync(..., Put)`, peer reads window.
2. **Get visibility:** `gin.get`, then `sync(..., Get)`, local dest read (no extra `flush`).
3. **AllContexts Put:** `put` on context 1, session on `ncclGinAllContexts`, `Put`.

No hybrid world-fence kernel in Stage 1.

Record per backend: pass / stale / hang / trap. SDMA Get + AllContexts are expected red until intra-node Get/FlushAsync/Wait exist.

**Stage 2 spike (later):** same kernels on ≥2 nodes, proxy only, MI300/350/355; plus hybrid Put from a non-rail peer on 2×2.

**Stage 3 spike (last):** same on MI450 multi-node for proxy and SDMA. SKIP-gate in tests from Stage 1 so this does not FAIL on MI300/350/355.

---

## SDMA drain (Stage 1)

Implement real Anvil ops instead of `__builtin_trap()` so **single-node** Get and AllContexts work:

1. **`ncclGinApi_Get`** — SDMA/IPC copy in the reverse direction of Put (local dest, remote src). Must quiet or mark dirty so a later Flush sees it.
2. **`ncclGinApi_FlushAsync` + `Wait`** — per-peer quiet of dirty SDMA channels (Flush already loops peers; FlushAsync should quiet **one** peer). AllContexts fence depends on this.

Also verify **Put+signal ordering** (`fenceBeforeSignal` / quiet). Put-fence does **not** flush; it relies on the barrier **signal** not overtaking in-flight puts.

Prefer drain that can later serve cross-node MI450 (Stage 3), but **Stage 1 sign-off is intra-node only**.

---

## Tests (land by stage)

Follow `GinMPIDeviceTests`: MPI + GIN skip reasons, `EXPECT_` before barriers, broadcast SKIP, timeouts on waits. **One commit per test.** Keep old `Relaxed` arrival tests.

Write Stage 1 tests against the contract even while SDMA still traps. Gate Stage 2/3 layouts so they SKIP until those stages run — do not FAIL Stage 1 CI on missing multi-node.

### Stage 1 (single-node, proxy + SDMA)

| Test | Layout | Asserts |
|---|---|---|
| `BarrierFence_Put_MakesInboundPutVisible` | 1 node, TYPE=2 and TYPE=6 | Data visible with no `waitSignal` |
| `BarrierFence_Put_IncludesSelfPut` | Same | Self-put visible |
| `BarrierFence_Get_MakesLocalGetVisible` | Same | Local dest valid after Get fence |
| `BarrierFence_DefaultIsPutAndGet` | Same | Omitted fence argument |
| `BarrierFence_AllContexts_Put` | Same | Put on ctx 1, AllContexts |
| `BarrierFence_AllContexts_Get` | Same | Get on non-zero ctx + AllContexts |

Should-have in Stage 1: `Relaxed == None`; SDMA payload above and below threshold; four-rank Put **on one node**.

### Stage 2 (later — proxy multi-node, MI300/350/355)

Same visibility tests with `crossNodeReason()` and `NCCL_GIN_TYPE=2`. Add `BarrierSession_Hybrid_PutUsesWorldWhenNotRailed` (2×2). Timeout + fence on IB. Multi-node SDMA: SKIP.

### Stage 3 (last — MI450 multi-node proxy + SDMA)

Enable the Stage 2 tests for SDMA as well (GPU-family gate). Hybrid world-fence on SDMA 2×2.

**Do not assert** “None leaves data invisible.” Copy LSA visibility (`dErr` after barrier), not “kernel returned.”

---

## Documentation (examples frozen; can land with Stage 1)

- `projects/rccl/docs/userguide/source/api/device_gin.rst`: AllContexts constructors on the session; choosing None vs Put vs Get vs default; timeout + fence.
- `projects/rccl/docs/userguide/source/usage/deviceapi.rst`: one sentence that the AlltoAll sample uses `None` **because** it already `waitSignal`s + `flush`es.
- `projects/rccl/docs/how-to/device-api-gin.rst`: 2.30.7 fence semantics; timeout snippet may mention `Put\|Get` as the default.

Do not treat `docs/contrib/GIN/...` as user docs unless already published.

---

## Hardware sign-off

| Stage | Run | Required to proceed |
|---|---|---|
| **1** | `BarrierFence_*` TYPE=2 and TYPE=6, **1 node**, intranet, MI300/350/355 | **Yes — current bar** |
| **1** | `run-gin-ci.sh` proxy + SDMA AlltoAll on one node | Yes, no `None`-kernel regression |
| **2** | Same tests TYPE=2, ≥2 nodes; hybrid 2×2 | After Stage 1 |
| **3** | TYPE=2 and TYPE=6, ≥2 nodes, MI450 | Last; skip-gate until hardware |

JIRA comments per stage: backends, node layout, pass/fail, drain-fix summary.

---

## Sequencing

```text
NOW     Stage 1 spike: single-node proxy + single-node SDMA (MI300/350/355)
        Stage 1 drain: SDMA Get + FlushAsync/Wait
        Stage 1 tests + skip gates for Stage 2/3 layouts
        Docs can land here
NEXT    Stage 2: multi-node proxy only on MI300/350/355 (incl. useWorldForFence)
LAST    Stage 3: multi-node proxy + SDMA on MI450 (when allocation exists)
```

Do not wait on a full coverage campaign before the spike. If Put is a no-op on AMD flush, more tests will all fail the same way.

---

## Risks

- AllContexts on SDMA **must not** keep trapping; **Stage 1** merge blocker.
- Do not run or require multi-node (Stage 2/3) until Stage 1 is green.
- Do not treat a passing **single-node** hybrid barrier as `useWorldForFence` coverage (world == LSA). That is Stage 2 (proxy 2×2) / Stage 3 (MI450 SDMA).
- Hybrid Put on proxy 2×2 with **railed** contexts (`ginContextsRailed`) takes the rail path, not world. Tests should **observe** `useWorldForFence` (or equivalent handle/team) so a railed machine does not give a false pass.
- SDMA fused-signal (`NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL`) is experimental; default **off** for fence tests unless an explicit fused-signal case is added later.
- Default `Put\|Get` is a behavior change vs old `Relaxed` defaults. Callers who omit the argument now pay drain cost. Docs + default test must make that explicit. Examples stay on `None`/`Relaxed` by design.

---

## Key source files

- `projects/rccl/src/include/nccl_device/gin_barrier.h` — enum, AllContexts, session API
- `projects/rccl/src/include/nccl_device/impl/gin_barrier__funcs.h` — `syncInternal`, Put/Get/all-ctx
- `projects/rccl/src/include/nccl_device/impl/barrier__funcs.h` — `useWorldForFence`, hybrid `sync`
- `projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma.h` — SDMA Put/Flush; Get/FlushAsync/Wait stubs
- `projects/rccl/src/include/nccl_device/gin/proxy/gin_proxy.h` — proxy Get/Flush
- `projects/rccl/test/transport/GinDeviceMPITests.cpp` — existing barrier tests; home for new `BarrierFence_*`
- `projects/rccl/test/GinNcclTimeoutMPITests.cpp` — timeout + fence
- `projects/rccl/docs/userguide/source/api/device_gin.rst`
- `projects/rccl/docs/userguide/source/usage/deviceapi.rst`
- `projects/rccl/docs/how-to/device-api-gin.rst`
- `projects/rccl/tools/ci/lib/gin-tests.json` / `tools/ci/run-gin-ci.sh`
- `projects/rccl/.claude/skills/feature-unit-testing/SKILL.md` — test methodology

---

## Why this shape

AllContexts in Stage 1 proves multi-QP drain on one node. Hybrid world-fence waits for Stage 2 (proxy) and Stage 3 (SDMA on MI450).

**Right now:** Stage 1 spike — **one node**, `NCCL_GIN_TYPE=2` and `=6`, MI300/350/355. No multi-node jobs until that is green.
