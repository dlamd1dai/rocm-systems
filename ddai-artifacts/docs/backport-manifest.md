# Upstream → NCCL 2.30.7 backport manifest

Target branch: `users/dondai/gin-stage3b-sdma-ag-nccl-2.30.7-wip`  
Policy: [gin-stage3b-sdma-ag-branch-policy.md](./gin-stage3b-sdma-ag-branch-policy.md)

Record selective cherry-picks from upstream `ROCm/rocm-systems` PRs. Do **not** wholesale rebase onto modern `develop`.

| Upstream PR | Branch | Upstream head | Backport status | NCCL wip commit | Notes |
|---|---|---|---|---|---|
| [#10658](https://github.com/ROCm/rocm-systems/pull/10658) | `users/dondai/gin-sdma-a2a-devtime` | `b6b7c8521b` | **Already synced — no delta** | `c1cb479e02` (audit) | A2A devtime path present via earlier merges (`5f91eb79fd`…`d97191693d`). `gin_sdma_devtime.h` identical to upstream tip; `AlltoAllDeviceTime` + CLI flags (`-B/-L/-P/-H`) match. Remaining diff vs upstream is modern-harness only (`pthread`→`std::thread`, `getDevCommRequirements(comm)` NCCL 2.29 API) — **intentionally not ported**. |
| [#10672](https://github.com/ROCm/rocm-systems/pull/10672) | `users/dondai/gin-gda-a2a-unittests` | — | Pending | — | |
| [#10675](https://github.com/ROCm/rocm-systems/pull/10675) | `users/dondai/gin-scope-fence-guard-fix` | — | Pending | — | |

## PR #10658 audit detail (2026-08-24)

**Files in upstream PR:**

- `projects/rccl-tests/src/gin_sdma_devtime.h` — byte-identical to NCCL wip tip
- `projects/rccl-tests/src/alltoall.cu` — devtime hook/CLI logic matches; diffs are NCCL 2.29 harness API only
- `projects/rccl-tests/src/common.cu` / `common.h` — devtime globals, `--device_timing`, WARN+skip, `Allreduce<double>` present on wip
- `projects/rccl-tests/src/CMakeLists.txt` — wip keeps NCCL 2.30.7 rocshmem link layout (RDC + `-rdynamic`)

**Cherry-pick dry-run:** commits `7b82305dd6`…`b6b7c8521b` conflict on wip (duplicate history). No new commit required.

**Validation:** rebuild `alltoall_perf` on MI355 (`rccl-gin-gda-sdma-713`) after manifest update.
