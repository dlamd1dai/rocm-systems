# GIN-SDMA A2A Docker Harness

Branch: `users/dondai/gin-sdma-a2a-mi455-wip`

Ported and reconciled from `users/dondai/gin-stage3b-sdma-ag-nccl-2.30.7-wip` for
**GIN device API AllToAll** on upstream-style RCCL (`NCCL_GIN_TYPE=6`).

## Layout

| Path | Purpose |
|------|---------|
| `ddai-artifacts/docker/Dockerfile-rccl-gin-gda-sdma` | Ubuntu 24.04 + ROCm 7.13 image with rocSHMEM SDMA + RCCL `--rocshmem-gin` + rccl-tests |
| `ddai-artifacts/scripts/docker-gin-gda-sdma-build.bash` | Build wrapper + optional GPU smoke (Test#5) |
| `ddai-artifacts/scripts/gin-sdma-a2a-test.bash` | Single-node AllToAll perf/compare harness |
| `extra-rdma-debs/` | Optional newer rdma-core/libmlx5 debs for MLX5 DMA-BUF symbols |
| `ddai-artifacts/c/ddai-rocshmem-hoststub.c` | Host stub for unit tests when rocSHMEM host lib is not linked |

## Reconciliation notes (vs stage3b branch)

| Item | stage3b (MI355 / NCCL 2.30.7) | This branch |
|------|-------------------------------|-------------|
| `NCCL_GIN_TYPE` | 5 | **6** (Anvil SDMA enum on develop) |
| Test#5 kernel | Size-hybrid `GinHybridAlltoAllKernel` + many `NCCL_GIN_ANVIL_A2A_*` env vars | **`GinAlltoAllKernel` (-D 3)** from upstream rccl-tests |
| AllGather smoke | `gin-sdma-ag-test.bash` in build script | **Removed** (A2A-only) |
| Default NP | 8 | **4** (1p4g MI455 starter) |
| Default GPU | gfx950 | **gfx1250** (override with `GPU_TARGETS=gfx950` for MI355) |
| RCCL install | `-DENABLE_ROCSHMEM_GIN=ON` cmake option | **`--rocshmem-gin`** install.sh flag |
| Image tag | `rccl-gin-gda-sdma-713` | **`rccl-gin-sdma-a2a-mi455`** |

## Quick start

From the repo root:

```bash
# MI455 (default)
bash ddai-artifacts/scripts/docker-gin-gda-sdma-build.bash

# MI355
GPU_TARGETS=gfx950 GIN_SMOKE_NP=8 bash ddai-artifacts/scripts/docker-gin-gda-sdma-build.bash

# Run A2A harness (1p4g, small + default max size)
bash ddai-artifacts/scripts/gin-sdma-a2a-test.bash 4 128M

# GIN device API only (Test#5)
RCCL_GIN_RUN_TESTS=5 bash ddai-artifacts/scripts/gin-sdma-a2a-test.bash 4 1M
```

## Test matrix (`gin-sdma-a2a-test.bash`)

| Test | Path | Description |
|------|------|-------------|
| #1 | Host `-D 0` | Ring / CE / hybrid baselines (`TEST1_MODE`) |
| #2 | GIN proxy | `NCCL_GIN_TYPE=2` |
| #4 | GIN GDA | `NCCL_GIN_TYPE=4` (auto-skipped on old bnxt fw) |
| #5 | **GIN Anvil SDMA** | `NCCL_GIN_TYPE=6`, `-D 3` GinAlltoAllKernel (default) |

Select tests with `RCCL_GIN_RUN_TESTS` (comma list), e.g. `1,5` or `5`.

### Test#5 knobs

| Variable | Default | Meaning |
|----------|---------|---------|
| `TEST5_MODE` | `d3` | `d3` = GinAlltoAllKernel, `d4` = HybridAlltoAllKernel |
| `TEST5_D3_CTA_COUNT` | `1` | `-V` for -D 3 (one CTA per signal) |
| `NCCL_GIN_ANVIL_SDMA_THRESHOLD` | `128` | Backend IPC vs SDMA threshold (bytes) |
| `NCCL_MNNVL_ENABLE` | `1` | Required for MI455 fabric/MNNVL paths |
| `TEST5_MLX5_PREFLIGHT` | `0` in smoke | Set `1` to skip Test#5 when libmlx5 lacks DMA-BUF symbols |

## Build knobs

| Variable | Default | Meaning |
|----------|---------|---------|
| `GPU_TARGETS` | `gfx1250` | Passed to rocSHMEM + RCCL + rccl-tests |
| `COLLECTIVE` | `a2a` | `a2a` = AllToAll-only device kernels; `full` = include AR/AG/BC baselines |
| `ONLY_FUNCS` | (from COLLECTIVE) | Device kernel generation filter |
| `RCCL_IMAGE_GIN_SMOKE` | `1` | Post-build Test#5 smoke on `/dev/kfd` |
| `GIN_SMOKE_NP` | `4` | Ranks for build smoke |
| `DOCKER_BUILD_NETWORK` | unset | Set to `host` if docker bridge is disabled |

## MI455 env checklist

```
NCCL_GIN_ENABLE=1
NCCL_GIN_TYPE=6
NCCL_CUMEM_ENABLE=1
NCCL_MNNVL_ENABLE=1
NCCL_DMABUF_ENABLE=1
HSA_NO_SCRATCH_RECLAIM=1
```

See also: [gin-sdma-a2a-mi455-fabric-dda-plan.md](gin-sdma-a2a-mi455-fabric-dda-plan.md)
