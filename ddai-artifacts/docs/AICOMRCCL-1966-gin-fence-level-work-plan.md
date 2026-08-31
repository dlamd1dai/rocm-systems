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
| Topology | **Split by backend.** GIN-SDMA collectives are **single-node only**. Proxy Put/Get/hybrid world-fence stay **multi-node**. |
| `gin.get` | **Required.** Get-fence tests are not optional. |
| AllContexts + `useWorldForFence` | **In this ticket.** AllContexts on both backends. `useWorldForFence` is **proxy multi-node only** (SDMA cannot create a non-trivial rail vs world split). |
| Drain / flush bugs | **Fix on this ticket** (no child enablement JIRA). |
| Examples | **Leave as-is** (no example or README edits). Docs only in userguide + how-to. |

**Coverage target:** Standard on fence branches in `gin_barrier__funcs.h` / `barrier__funcs.h` (Put vs not, Get vs not, all-ctx vs single-ctx, `useWorldForFence` true/false — last one proxy-only).

**Merge bar:** Put + Get visibility tests **PASS** on proxy (multi-node) **and** SDMA (single-node, ≥2 GPUs). AllContexts must not trap on either. Hybrid Put uses world when not railed — **proxy 2×2 only**.

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

A world-team hybrid barrier with `Put` is **not** “LSA + rail GIN”; it is a **world GIN** barrier. That distinction is only observable when ranks span nodes (rail ⊂ world). On single-node SDMA, world and LSA coincide, so do not use SDMA to sign off `useWorldForFence`.

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

Write the test plan in the PR body (RCCL feature-unit-testing practice: do not commit `TEST_PLAN.md`).

- Feature: bitmask fence on `ncclGinBarrierSession` / `ncclGinBarrier` / hybrid `ncclBarrierSession`.
- Hardware:
  - **Proxy:** ≥2 nodes, IB, `NCCL_CUMEM_ENABLE=1`, `NCCL_DMABUF_ENABLE=1`.
  - **SDMA:** **one node**, ≥2 (prefer 4–8) Anvil GPUs, `--rocshmem-gin` build. Single-node GIN also needs `RCCL_ENABLE_INTRANET=1` (see `intranetReason()` in `GinDeviceMPITests.cpp`). Payload **>** `NCCL_GIN_ANVIL_SDMA_THRESHOLD` so the SDMA path is hit.
- Acceptance: Put + Get visibility PASS on proxy (multi-node) and SDMA (single-node). AllContexts does not trap. Hybrid world-fence PASS on proxy 2×2 only.

---

## Topology split (correction)

GIN-SDMA collectives are **intra-node** (Anvil SDMA + LSA/IPC). Do not schedule SDMA fence tests across nodes; they are unsupported, not a skip to paper over.

| Backend | Layout | What it can prove |
|---|---|---|
| Proxy `NCCL_GIN_TYPE=2` | 2 nodes × 1 GPU | Cross-node Put/Get drain over IB |
| Proxy | 2 nodes × 2 GPUs | `useWorldForFence` (rail ⊂ world); hybrid Put from a non-rail peer |
| SDMA `NCCL_GIN_TYPE=6` | 1 node × ≥2 GPUs | Intra-node Put/Get, AllContexts, self-put, threshold mix. World == LSA, so **not** a rail/world test. |

Existing tests that call `crossNodeReason()` stay proxy-only. New SDMA tests should **require** all ranks on one node (inverse of `crossNodeReason`) plus `intranetReason()`.

---

## Phase 1 — Spike (first)

Same three kernels, **two jobs** (do not mix backends on one allocation):

```text
# Job A — proxy, multi-node (2 nodes × 1 GPU, then 2×2 for hybrid)
NCCL_GIN_TYPE=2 NCCL_CUMEM_ENABLE=1 NCCL_DMABUF_ENABLE=1 NCCL_IB_MERGE_NICS=0

# Job B — SDMA, single node (≥2 GPUs; README example uses 8)
NCCL_GIN_TYPE=6 NCCL_GIN_ANVIL_SDMA_THRESHOLD=128 NCCL_CUMEM_ENABLE=1 \
NCCL_DMABUF_ENABLE=1 RCCL_ENABLE_INTRANET=1
# Optional: NCCL_P2P_DISABLE=1 if you need to keep traffic on the GIN/SDMA path
# rather than XGMI P2P (matches gin/README.md single-node AlltoAll).
```

Kernels:

1. **Put visibility:** inbound `put` (no `waitSignal`), then `sync(..., Put)`, peer reads window.
2. **Get visibility:** `gin.get`, then `sync(..., Get)`, local dest read (no extra `flush`).
3. **AllContexts Put:** `put` on context 1, session on `ncclGinAllContexts`, `Put`.

On proxy 2×2 only, add a fourth kernel: hybrid `ncclBarrierSession` + `Put` from a **non-rail** peer.

Record per backend: pass / stale data / hang / trap. Ranking drives Phase 2 vs Phase 3 order. SDMA Get + AllContexts are still expected red until FlushAsync/Wait/Get exist.

---

## Phase 2 — SDMA drain (same ticket)

Implement real Anvil ops instead of `__builtin_trap()`:

1. **`ncclGinApi_Get`** — SDMA/IPC copy in the reverse direction of Put (local dest, remote src). Must quiet or mark dirty so a later Flush sees it.
2. **`ncclGinApi_FlushAsync` + `Wait`** — per-peer quiet of dirty SDMA channels (Flush already loops peers; FlushAsync should quiet **one** peer). AllContexts fence depends on this.

Also verify **Put+signal ordering** (`fenceBeforeSignal` / quiet). Put-fence does **not** flush; it relies on the barrier **signal** not overtaking in-flight puts. If SDMA signals without quieting the queue, Put-fence is a data race even when Flush is correct.

Proxy: only touch if spike shows Get/Flush do not actually drain.

---

## Phase 3 — Tests

Follow `GinMPIDeviceTests`: MPI + GIN skip reasons, `EXPECT_` before barriers, broadcast SKIP, timeouts on waits. **One commit per test.** Keep old `Relaxed` arrival tests (`Relaxed` now means `None`).

Run matrix **proxy (multi-node) × SDMA (single-node)**. Gate with `crossNodeReason()` vs single-node + `intranetReason()`, not one layout for both. Skip a backend only if GIN cannot activate; never skip Get on a backend that compiled GIN.

Write tests against the **contract** even while SDMA still traps; they are the regression net for the drain fix. Do not merge tests that only pass on proxy if SDMA is a release backend for this ticket.

### Must-have

| Test | Layout | Asserts |
|---|---|---|
| `BarrierFence_Put_MakesInboundPutVisible` | Proxy: multi-node. SDMA: 1 node × ≥2 GPUs | Data visible with no `waitSignal` |
| `BarrierFence_Put_IncludesSelfPut` | Both | Self-put visible (`nPeerSigs = nRanks`) |
| `BarrierFence_Get_MakesLocalGetVisible` | Both | Local dest valid after Get fence, no extra flush |
| `BarrierFence_DefaultIsPutAndGet` | Both | `sync(coop, order)` with omitted fence |
| `BarrierFence_AllContexts_Put` | Both | Put on ctx 1, AllContexts session |
| `BarrierFence_AllContexts_Get` | Both | Get on non-zero ctx + AllContexts (forces FlushAsync/Wait) |
| `BarrierSession_Hybrid_PutUsesWorldWhenNotRailed` | **Proxy 2×2 only** | Inbound put from **non-rail** peer visible; world handle selected |

### Should-have

- Compile-time `Relaxed == None`.
- Timeout `sync(..., Put/Get, timeoutCycles)` so timeout is not Relaxed-only (`GinNcclTimeoutMPITests.cpp`).
- Four-rank Put (peer rotation).
- Payload **> threshold** and **< threshold** on SDMA.

**Do not assert** “None leaves data invisible” (that is a race, not a guarantee).

Pattern: copy LSA visibility (`dErr` after barrier), not “kernel returned.”

---

## Phase 4 — Documentation (examples frozen)

- `projects/rccl/docs/userguide/source/api/device_gin.rst`: AllContexts constructors on the session; choosing None vs Put vs Get vs default; timeout + fence.
- `projects/rccl/docs/userguide/source/usage/deviceapi.rst`: one sentence that the AlltoAll sample uses `None` **because** it already `waitSignal`s + `flush`es.
- `projects/rccl/docs/how-to/device-api-gin.rst`: 2.30.7 fence semantics; timeout snippet may mention `Put\|Get` as the default.

Do not treat `docs/contrib/GIN/...` as user docs unless already published.

---

## Phase 5 — Hardware sign-off and JIRA close

| Run | Why |
|---|---|
| `GinMPIDeviceTests.Barrier*` + `BarrierFence_*` with `NCCL_GIN_TYPE=2`, ≥2 nodes | Proxy contract |
| Same with `NCCL_GIN_TYPE=6`, **1 node**, payload above SDMA threshold, `RCCL_ENABLE_INTRANET=1` | SDMA contract |
| Timeout MPI with Put/Get | Composition |
| `run-gin-ci.sh` proxy AlltoAll + `NCCL_GIN_TYPE=6` AlltoAll | No regression on `None` kernels |
| Optional: `NCCL_GIN_NCONTEXTS>1` AllContexts | Multi-QP drain |

JIRA comment: backends, node layout, pass/fail, drain-fix summary.

---

## Sequencing

```text
Day 1–2   Spike: proxy multi-node job + SDMA single-node job
Day 2–N   SDMA Get + FlushAsync/Wait (+ Put/signal quiet if spike shows stale Put)
          in parallel with writing tests against the intended contract
Then      Tests green on both backends (correct layouts) → docs → CI gin jobs → close 1966
```

Do not wait on a full coverage campaign before the spike. If Put is a no-op on AMD flush, more tests will all fail the same way.

---

## Risks

- AllContexts on SDMA **must not** keep trapping; merge blocker.
- Do not treat a passing SDMA single-node hybrid barrier as `useWorldForFence` coverage; world and LSA are the same team.
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

The 2.30.7 change is **memory visibility**, not “barriers exist.” Arrival tests cannot catch a regression that turns Put into None. AllContexts and hybrid world-fence exist **only** to make Put/Get true across QPs and rails.

First concrete execution step: three-kernel spike as **two jobs** — proxy on ≥2 nodes (`NCCL_GIN_TYPE=2`), SDMA on **one node** (`NCCL_GIN_TYPE=6`).
