# GIN SDMA — Shared Policy Refactor Plan

**Scope:** `gin_sdma_*_policy.h` across AllToAll / AllGather / Broadcast, plus the rccl-tests pytest harnesses
**Status:** Planned — blocked on fork [#5](https://github.com/dlamd1dai/rocm-systems/pull/5) and the AICOMRCCL-2217..2221 stack landing first
**Authored:** 2026-09-02

---

## Context

All three GIN Anvil-SDMA collectives now exist, but they were developed as a stack and each invented its own copy of the same policy scaffolding. AllToAll and AllGather are already on `develop`; Broadcast is in review.

| Collective | Upstream PR | State | Policy code lives in |
|---|---|---|---|
| AllToAll (`-D 3`, `-D 4`) | [#7826](https://github.com/ROCm/rocm-systems/pull/7826), [#10420](https://github.com/ROCm/rocm-systems/pull/10420), [#10658](https://github.com/ROCm/rocm-systems/pull/10658), [#10672](https://github.com/ROCm/rocm-systems/pull/10672), [#10675](https://github.com/ROCm/rocm-systems/pull/10675) | Merged | inline in `alltoall.cu` — **no policy header, no policy unit test** |
| AllGather (`-D 3`) | [#10930](https://github.com/ROCm/rocm-systems/pull/10930) | Merged 2026-09-01 | `gin_sdma_allgather_policy.h`, `namespace gin_sdma_allgather` |
| Broadcast (`-D 3`) | fork [#5](https://github.com/dlamd1dai/rocm-systems/pull/5) | Open, rebased onto `develop` | `gin_sdma_broadcast_policy.h`, `namespace gin_sdma` |
| SDMA put segmentation | — | Merged | `gin_anvil_sdma_put_policy.h`, `namespace gin_sdma` |

Fork PR #5 was rebased off the (now-merged) AllGather branch onto `develop` on 2026-09-02: 27 commits / 19 files / +3752 reduced to 7 commits / 7 files / +2240, touching zero AllGather files. Pre-rebase head preserved as `users/dondai/gin-stage3c-sdma-bc-prerebase`.

---

## Duplication inventory

Measured against merged `gin_sdma_allgather_policy.h`, the Broadcast header in fork #5, and the inline logic in `alltoall.cu`.

| # | Duplicated thing | AllGather | Broadcast | Notes |
|---|---|---|---|---|
| 1 | Host/device attribute macro | `GIN_SDMA_AG_HD` | `GIN_SDMA_HD` | Third copy in `gin_anvil_sdma_put_policy.h`. See hazard below. |
| 2 | Namespace | `gin_sdma_allgather` | `gin_sdma` | Backend put policy is also `gin_sdma`. |
| 3 | Threshold precedence ladder | `pickSdmaThreshold(bool perCollSet, unsigned long long perCollVal, ...)` | `resolveThreshold(size_t collVal, size_t sharedVal, size_t dflt)` | Same per-collective → global → default semantics, **incompatible signatures**: bool+value pair vs sentinel. |
| 4 | 16-byte chunk alignment | `chunkBaseCount(count, eltSize, nranks)` | `alignChunkCount(count, nRanks, eltSize)` | Identical math, **transposed argument order** — a real footgun once both are in scope. |
| 5 | LSA/SDMA tier predicate | `chunkUsesLsaTier(bytes, threshold)` | `bcastChunkUsesLsaTier(msgBytes, threshold)` | Both are the same `<=` compare. |
| 6 | CTA ladder + pool clamp | `allGatherMaxCtas` / `allGatherPoolCtas` / `allGatherCtas` | `bcastHybridMaxCtas` / `bcastHybridPoolCtas` / `bcastPoolCtas` / `bcastHybridCtas` | Same 16-LSA / 4-SDMA constants; same "clamp the pin to the launched pool so `blockIdx.x` cannot index past the barrier array" invariant. |
| 7 | Env parsing | `parseEnvU64`, `parseAllGatherCtasEnv`, `resolveSdmaThresholdFromEnv` — all in header, all unit-tested | `parseSize` in header (tested); `BroadcastParseCtasEnv`, `bcastRingCtas` are **statics in `broadcast.cu`, untested** | Five implementations across two testability tiers. |
| 8 | pytest launch + connectivity-gate retry | `_launch_ag_gin_sdma` / `_run_ag_gin_sdma` | `_launch_bcast_gin_sdma` / `_run_bcast_gin_sdma` | AllToAll has its own copy. Being deduped for AG/A2A by fork [#10](https://github.com/dlamd1dai/rocm-systems/pull/10). |

### Why item 7 matters concretely

The rebase of fork #5 surfaced two defects in Broadcast that AllGather's review had already caught and fixed, precisely because AllGather's parsers sit in a unit-tested header and Broadcast's sit as statics in the `.cu`:

- `BroadcastParseCtasEnv` only checked that `strtoull` consumed one digit, so `NCCL_GIN_ANVIL_BCAST_CTAS=-1` wrapped to a ~16 EiB CTA count and `=8foo` silently parsed as `8`. Upstream `parseAllGatherCtasEnv` rejects the leading sign, trailing junk, and a literal sentinel collision.
- `.getDevCommRequirements` was wired into `ncclTestEngine` only under `ENABLE_DEVICE_API && >= 2.28`, while the function is *defined* (and `common.h` declares the slot) for NCCL >= 2.29 regardless. On a 2.29+ build without the device API the slot was left null, so `broadcast_perf -D 3` requested no devComm resources.

Both fixed in `1439a299b35` on the PR branch. Moving these parsers into a shared, tested header is what stops the next one.

### Standalone hazard worth fixing regardless

`gin_anvil_sdma_put_policy.h` defines `GIN_SDMA_HD` **unconditionally**:

```c
#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(GIN_SDMA_HOST_ONLY)
#define GIN_SDMA_HD __host__ __device__
#else
#define GIN_SDMA_HD
#endif
```

`gin_sdma_broadcast_policy.h` wraps its own define in `#ifndef` to avoid a clash — but in `broadcast.cu` the Broadcast header is included first (line 10) and `put_policy.h` arrives transitively at line 14, so the `#ifndef` never fires and `put_policy.h` redefines the macro. It is legal today only because both expansions are token-identical. If either condition ever drifts it silently becomes an ill-formed redefinition. **The `#ifndef` belongs on `put_policy.h`**, which is the canonical owner.

---

## Target layout

New `projects/rccl-tests/src/gin_sdma_policy_common.h` in `namespace gin_sdma`, owning the substrate:

| Concern | Symbol |
|---|---|
| Attribute macro | single `GIN_SDMA_HD` definition (`#ifndef`-guarded) |
| Sentinel | `kThresholdUnset` |
| Env parsing | `parseSize`, `parseEnvU64`, `parseCtasEnv` (hardened form) |
| Threshold precedence | `resolveThreshold` |
| Chunk math | `alignChunkCount`, `chunkBytes` |
| Tier predicate | `usesLsaTier` |
| CTA pool | pool-clamp helper |
| Reporting | `bandwidthGBps` |

Each collective header keeps only what is genuinely its own:

- **AllGather** — `kAllGatherSdmaThresholdDefault` (32 KiB), its CTA ladder constants, `allGatherRecvSliceOffset`.
- **Broadcast** — `BcastTier` enum, ring and scatter+allgather math, its own defaults (256 KiB threshold, 2 MiB SAG, 32 MiB ring, 64 KiB/CTA ring chunk).
- **AllToAll** — new `gin_sdma_alltoall_policy.h`, extracted from `alltoall.cu`.

### Naming decision

**Keep `gin_sdma_allgather::` as-is** and have it delegate to `gin_sdma::` helpers, rather than renaming the namespace to match Broadcast. Renaming touches just-merged code for cosmetic gain and makes the refactor diff much harder to review as behaviour-preserving. Revisit only if a later collective makes the split actively confusing.

### Prior art to check first

`gin_sdma_collective_policy.h` — a single unified header — existed on `users/dondai/gin-stage2h-sdma-A1457-a2a-pull` and was split into per-collective headers on the way upstream. Confirm whether that split was a review request or just convenience **before** re-unifying, so the refactor is not silently reversing a reviewer's decision.

---

## Sequencing

The refactor must come after the in-flight stack.

1. Fork [#5](https://github.com/dlamd1dai/rocm-systems/pull/5) (Broadcast) lands — re-run the MI355 SUT gates on the rebased branch first, since the existing evidence predates the rebase.
2. [ROCm#11092](https://github.com/ROCm/rocm-systems/pull/11092) (AICOMRCCL-2217/2218) lands the conn-check restore on a properly device-linked TU.
3. Fork [#9](https://github.com/dlamd1dai/rocm-systems/pull/9) (AICOMRCCL-2220) consolidates backend env parsing in `gin_plugin_anvil_sdma.cc` and removes `gin_anvil_sdma_oss7_device.cc`. Plugin-side, so no header conflict — but it settles the convention the tests side should mirror.
4. Fork [#10](https://github.com/dlamd1dai/rocm-systems/pull/10) (AICOMRCCL-2221) — **this is where the collisions are** (see below).
5. **Refactor PR A** — introduce `gin_sdma_policy_common.h`; migrate AllGather and Broadcast onto it. Strictly behaviour-preserving.
6. **Refactor PR B** — extract `gin_sdma_alltoall_policy.h` and add its unit test. This *adds* coverage, so it is a feature change, not a pure refactor; keep it separate from A.

### Collision matrix: fork #5 vs fork #10

| File | fork #5 | fork #10 | Resolution |
|---|---|---|---|
| `projects/rccl-tests/src/CMakeLists.txt` | +11 — new rocSHMEM RDC block for `broadcast_perf` | +14 −13 — dedupes the rocSHMEM RDC link settings | Whichever lands second adopts the deduped helper |
| `projects/rccl/test/CMakeLists.txt` | +20 — `rccl-UnitTestsGinSdmaBroadcastPolicy` | +31 −3 | Textual conflict; re-apply by hand |
| `projects/rccl-tests/test/conftest.py` | untouched | +52 — shared connectivity-gate retry helper | — |
| `projects/rccl-tests/test/test_Broadcast.py` | +298 −47 | **untouched** (Broadcast was unmerged when #10 was written) | **Follow-up required:** migrate `test_Broadcast.py` onto the shared conftest helper |

---

## Verification strategy for Refactor PR A

A pure refactor must not need its tests edited. Concretely:

- [ ] `rccl-UnitTestsGinSdmaAllGatherPolicy` passes with **source unchanged**
- [ ] `rccl-UnitTestsGinSdmaBroadcastPolicy` (38 cases) passes with **source unchanged**
- [ ] `gin_sdma_policy_test` (put segmentation) passes with **source unchanged**
- [ ] `rccl-UnitTestsGinSdmaAllGatherGpu` still builds and skips cleanly without a GPU
- [ ] MI355 8-rank re-run: `alltoall_perf -D 3`, `all_gather_perf -D 3`, `broadcast_perf -D 3`, all `#wrong=0`

If a moved helper requires touching a test, treat that as the signal that the move changed behaviour and justify it explicitly in the PR.

### Host-only policy test loop (no GPU, no RCCL build)

```bash
ar rcs libgtest.a /tmp/gin_gtest_build/gtest-all.o
g++ -std=c++17 -DGIN_SDMA_HOST_ONLY \
  -I/tmp/gin_gtest_build/googletest-1.14.0/googletest/include \
  -Iprojects/rccl-tests/src \
  projects/rccl/test/gin/gin_sdma_broadcast_policy_test.cpp \
  libgtest.a /tmp/gin_gtest_build/gtest_main.o -lpthread -o bc_policy_test
./bc_policy_test
```

---

## Open decisions

1. **Threshold signature** — reconcile `pickSdmaThreshold`'s `(bool set, value)` pair against `resolveThreshold`'s sentinel form. Recommend the **sentinel form**: it composes better and AllGather already uses a sentinel for `kAllGatherCtasUnset`.
2. **Chunk-alignment argument order** — `(count, eltSize, nranks)` or `(count, nRanks, eltSize)`. Pick one and change the other call sites in the same commit.
3. **Namespace** — confirm the "keep `gin_sdma_allgather`, delegate to `gin_sdma`" recommendation above.
4. **Header location** — `gin_sdma_policy_common.h` under `projects/rccl-tests/src/` (alongside the collective headers) or under `projects/rccl/src/include/nccl_device/gin/anvil_sdma/` (alongside the backend put policy). The rccl-tests location keeps the test-harness policy separate from shipped device headers; the backend location avoids rccl-tests owning a header that `librccl` code might later want.

---

## Related

- Branch policy: [gin-stage3b-sdma-ag-branch-policy.md](./gin-stage3b-sdma-ag-branch-policy.md)
- Backport manifest: [backport-manifest.md](./backport-manifest.md)
