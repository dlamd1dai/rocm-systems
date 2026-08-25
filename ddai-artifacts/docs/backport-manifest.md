# Upstream → NCCL 2.30.7 backport manifest

Target branch: `users/dondai/gin-stage3b-sdma-ag-nccl-2.30.7-wip`
Policy: [gin-stage3b-sdma-ag-branch-policy.md](./gin-stage3b-sdma-ag-branch-policy.md)

Record selective cherry-picks from upstream `ROCm/rocm-systems` PRs. Do **not** wholesale rebase onto modern `develop`.

| Upstream PR | Branch | Upstream head | Backport status | NCCL wip commit | Notes |
|---|---|---|---|---|---|
| [#10658](https://github.com/ROCm/rocm-systems/pull/10658) | `users/dondai/gin-sdma-a2a-devtime` | `b6b7c8521b` | **Already synced — no delta** | `3993803845` | A2A devtime path present via earlier merges (`5f91eb79fd`…`d97191693d`). `gin_sdma_devtime.h` identical to upstream tip; `AlltoAllDeviceTime` + CLI flags (`-B/-L/-P/-H`) match. Remaining diff vs upstream is modern-harness only (`pthread`→`std::thread`, `getDevCommRequirements(comm)` NCCL 2.29 API) — **intentionally not ported**. |
| [#10672](https://github.com/ROCm/rocm-systems/pull/10672) | `users/dondai/gin-gda-a2a-unittests` | `11fb115c3c` | **Backported (selective)** | `162bf4379f` | Ported fixture test fixes (Flush API, `ncclGinOffsetPtr`, IPC `loadConst`), CMake GDA-under-`GIN_ANVIL_UNIT_TESTS`, CI fixtures gate. Skipped wholesale upstream `CMakeLists.txt` / `gin.sbatch` develop churn; kept NCCL 2.30.7 rccl-tests sed patches in `gin.sbatch`. |
| [#10675](https://github.com/ROCm/rocm-systems/pull/10675) | `users/dondai/gin-scope-fence-guard-fix` | `a2899ae086` | **Backported** | `8305b821c6` | Cherry-picked both commits cleanly. HIP fence guard `given>required` → `given<required` on anvil_sdma + rocshmem_gda; gdaki unchanged (libcu++ opposite ordering). Added `NCCL_GIN_THREADFENCE_SYSTEM()` seam + GDA fence-counting tests. |

## PR #10658 audit detail (2026-08-24)

**Files in upstream PR:**

- `projects/rccl-tests/src/gin_sdma_devtime.h` — byte-identical to NCCL wip tip
- `projects/rccl-tests/src/alltoall.cu` — devtime hook/CLI logic matches; diffs are NCCL 2.29 harness API only
- `projects/rccl-tests/src/common.cu` / `common.h` — devtime globals, `--device_timing`, WARN+skip, `Allreduce<double>` present on wip
- `projects/rccl-tests/src/CMakeLists.txt` — wip keeps NCCL 2.30.7 rocshmem link layout (RDC + `-rdynamic`)

**Cherry-pick dry-run:** commits `7b82305dd6`…`b6b7c8521b` conflict on wip (duplicate history). No new commit required.

**Validation:** rebuild `alltoall_perf` on MI355 (`rccl-gin-gda-sdma-713`) after manifest update.

## PR #10672 backport detail (2026-08-24)

**Upstream head:** `11fb115c3c` (6 commits)

**Ported files:**

- `projects/rccl/test/device/GinRocshmemGdaTemplate_test.cpp` — GDA template tests + Flush/signal API fixes
- `projects/rccl/test/device/GinAnvilSdmaTemplate_test.cpp` — Flush + `ncclGinOffsetPtr` fixes
- `projects/rccl/test/device/GinAnvilIpcDevice_test.cpp` — header API + `loadConst` signal checks
- `projects/rccl/test/CMakeLists.txt` — build GDA template under `(ENABLE_ROCSHMEM OR GIN_ANVIL_UNIT_TESTS)` with SDMA tests
- `projects/rccl/tools/ci/lib/gin-tests.json` — fixtures test entry
- `projects/rccl/tools/ci/lib/parse_gin_config.py` — document `fixtures` kind
- `projects/rccl/tools/ci/run-gin-ci.sh` — run fixtures without mpirun
- `projects/rccl/tools/ci/lib/gin.sbatch` — `GIN_ANVIL_UNIT_TESTS=ON` + `RCCL_FIXTURES_BIN_DIR` (NCCL 2.30.7 rccl-tests sed patches retained)

**Already present (no change):** `projects/rccl/test/device/sdma/anvil_device.hpp` (`sdma_anvil` namespace fix)

**Intentionally not ported:** upstream develop-only `CMakeLists.txt` churn (CollImplInfoTests, RMA rename, micro-test link libs, etc.)

**Validation (MI355 `smci355-ccs-aus-m03-17`, docker `rccl-gin-gda-sdma-713`, 2026-08-24):**

- Build: `cmake -DBUILD_TESTS=ON -DGIN_ANVIL_UNIT_TESTS=ON` + `rccl-UnitTestsFixtures` — **PASS**
- Run: mount backport `projects/rccl/test`, `LD_PRELOAD=ddai-rocshmem-hoststub.so` (rocSHMEM host symbols in `librccl.so`)
- **32/33 PASS:** GinRocshmemGdaTemplateTest 12/12, GinAnvilSdmaTemplateTest 12/12, GinAnvilIpcDeviceTest 8/9
- **1 FAIL (upstream test gap on gfx950):** `GinAnvilIpcDeviceTest.DetailHelpers_ChannelAndDirty` — expects `useSdmaFusedSignal()==true` on MI355 (`oss7`), but test ctx sets `signals` only (not `signal_remote_addrs`); fused stays false. Same as upstream PR tip; MI300X (gfx942) masks this. Log: `ddai-artifacts/logs/pr10672-fixtures.latest` on SUT.
- Do **not** set `HSA_OVERRIDE_GFX_VERSION=11.0.0` on MI355 (causes `HSA_STATUS_ERROR_INVALID_ISA`).

## PR #10675 backport detail (2026-08-24)

**Upstream head:** `a2899ae086` (2 commits)

**Method:** clean cherry-pick onto NCCL wip (`38f46cde40`, `a2899ae086`).

**Ported files:**

- `projects/rccl/src/include/nccl_device/gin/gin_device_common.h` — `NCCL_GIN_THREADFENCE_SYSTEM()` test seam
- `projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma.h` — flip HIP fence guard to `given < required`
- `projects/rccl/src/include/nccl_device/gin/rocshmem_gda/gin_rocshmem_gda.h` — same HIP guard fix
- `projects/rccl/src/include/nccl_device/gin/gdaki/gin_gdaki.h` — comment-only (libcu++ keeps `given > required`)
- `projects/rccl/test/device/GinRocshmemGdaTemplate_test.cpp` — fence-counting tests (G7/G7b)

**Validation:** rebuild + run `GinRocshmemGdaTemplateTest.Put_WeakerGivenScopeFencesAndPuts` and full 33-case GIN fixture filter on MI355.
