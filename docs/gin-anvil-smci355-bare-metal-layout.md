# GIN Anvil SDMA — bare-metal layout for `smci355-ccs-aus-m03-17`

Design reference: [`gin-anvil-sdma-backend-design.md`](gin-anvil-sdma-backend-design.md) environment **E3** (MI355X, bare metal, 8 GPUs). Unit test mapping: [`gin-anvil-sdma-unit-test-plan.md`](gin-anvil-sdma-unit-test-plan.md). Orchestration script: [`../gin-anvil-smci355-test.bash`](../gin-anvil-smci355-test.bash).

This layout installs RCCL, rocSHMEM, rccl-tests, and GTest unit binaries **on the host** under the repo tree, without Docker. Use it when you want faster iteration on unit tests, native debugging, or to bypass container MLX5/glibc bind-mount issues.

---

## Target node

| Item | Value |
|------|-------|
| Hostname | `smci355-ccs-aus-m03-17.cs-aus.dcgpu` |
| GPU | MI355X (`gfx950`) |
| Typical repo path | `~/rocm-systems` |
| ROCm | `/opt/rocm` (Conductor node default) |
| MPI | system Open MPI (`mpirun` on `PATH`) |

---

## Directory layout

All bare-metal artifacts live under **`gin-anvil-bm/`** at the repo root (created by `gin-anvil-smci355-test.bash`):

```text
~/rocm-systems/
├── gin-anvil-smci355-test.bash          # orchestrator (docker + bare-metal)
├── docker-gin-gda-sdma-*.bash           # docker harness (layout=docker)
├── docs/
│   └── gin-anvil-smci355-bare-metal-layout.md   # this file
└── gin-anvil-bm/                        # GIN_ANVIL_BM_ROOT
    ├── install/
    │   ├── rocshmem/                    # CMAKE_INSTALL_PREFIX (USE_SDMA=ON)
    │   │   ├── bin/rocshmem_info
    │   │   └── lib/librocshmem.a
    │   └── rccl/                        # RCCL install prefix
    │       ├── lib/librccl.so
    │       └── include/
    ├── build/
    │   ├── rocshmem/                    # rocSHMEM build tree (+ unit tests)
    │   ├── rccl/                        # librccl release build
    │   ├── rccl-unit/                   # BUILD_TESTS=ON (suites A–H, G)
    │   └── rccl-tests/                  # alltoall_perf
    └── logs/
        └── gin-anvil-smci355-<timestamp>.log
```

**Mapping to design doc build IDs:**

| Design ID | Bare-metal step | Artifact |
|-----------|-----------------|----------|
| **B1** | `source docker-gin-gda-sdma-preflight.bash` | tree validation |
| **B3** | `$GIN_ANVIL_BM_ROOT/install/rocshmem/bin/rocshmem_info` | SDMA enabled |
| **B5** | RCCL cmake / install | `gin_host_rocshmem_common.*` linked |
| §4.1 unit | `build/rccl-unit/test/rccl-UnitTests*` | suites A–H, G |
| §4.1 suite F | `build/rocshmem/tests/unit_tests/rocshmem_unit_tests` | factory tests |
| **C1/C2** | `build/rccl-tests/alltoall_perf` + `mpirun` | integration |

---

## Environment variables (bare-metal)

Set these before running integration tests manually, or let `gin-anvil-smci355-test.bash` export them.

### Build-time

| Variable | Default | Purpose |
|----------|---------|---------|
| `GIN_ANVIL_BM_ROOT` | `$REPO_ROOT/gin-anvil-bm` | Layout root |
| `ROCM_PATH` | `/opt/rocm` | ROCm install |
| `GIN_ANVIL_GPU_ARCH` | `gfx950` | `-DGPU_TARGETS` / `--amdgpu_targets` |
| `MPI_PREFIX` | auto-detected | Open MPI prefix; Debian layout is often `/usr/lib/x86_64-linux-gnu/openmpi` |
| `GIN_ANVIL_BUILD_SUITE_F` | `0` | **Off by default.** Set `1` to build/run rocSHMEM factory unit tests (suite F) on host |
| `GIN_ANVIL_PREFLIGHT_MPI` | `warn` (docker) / `require` (bare-metal) | `libopenmpi-dev` / `mpi.h` preflight; see `docker-gin-gda-sdma-preflight.bash` |
| `GIN_ANVIL_CLEAN` | `0` | Set `1` to wipe `gin-anvil-bm/build` before a bare-metal rebuild |

**Open MPI on host:** Preflight checks for `libopenmpi-dev` / `mpi.h`. With the default workflow (suite F off, docker integration), a missing package produces a **warning** only. Install when using bare-metal integration or opt-in suite F:

```bash
sudo apt-get install -y openmpi-bin libopenmpi-dev
```

Factory / SDMA queue coverage on MI355X is expected from **docker Test#5** (`RCCL_GIN_RUN_TESTS=5`), not host suite F.
| `ROCSHMEM_INSTALL_DIR` | `$GIN_ANVIL_BM_ROOT/install/rocshmem` | rocSHMEM prefix |
| `RCCL_INSTALL_PREFIX` | `$GIN_ANVIL_BM_ROOT/install/rccl` | RCCL prefix |

### Runtime (integration — mirrors docker harness `MPI_BASE`)

| Variable | Value |
|----------|-------|
| `LD_LIBRARY_PATH` | `$RCCL_INSTALL_PREFIX/lib:$ROCSHMEM_INSTALL_DIR/lib:$ROCM_PATH/lib` |
| `PATH` | `$ROCSHMEM_INSTALL_DIR/bin:$RCCL_INSTALL_PREFIX/bin:$ROCM_PATH/bin:$PATH` |
| `RCCL_ROCSHMEM_ENABLE` | `0` |
| `ROCSHMEM_BACKEND` | `ipc` |
| `ROCSHMEM_DISABLE_MIXED_IPC` | `1` |
| `ROCSHMEM_DEBUG_LEVEL` | `info:noversion` |
| `RCCL_ROCSHMEM_THRESHOLD` | `134217728` (128 MiB) |
| `NCCL_DEBUG` | `VERSION` (use `INFO` when debugging) |
| `NCCL_DEBUG_SUBSYS` | `INIT,NET` |
| `NCCL_CUMEM_ENABLE` | `1` |
| `RCCL_ENABLE_INTRANET` | `1` |
| `NCCL_DMABUF_ENABLE` | `1` |
| `NCCL_MSCCL_ENABLE` | `0` |
| `HSA_NO_SCRATCH_RECLAIM` | `1` |
| `NCCL_GIN_PLUGIN` | `none` |
| `HSA_FORCE_FINE_GRAIN_PCIE` | `1` (GIN tests) |

### Test#5 (Anvil SDMA)

| Variable | Value |
|----------|-------|
| `NCCL_GIN_ENABLE` | `1` |
| `NCCL_GIN_TYPE` | `5` |
| `NCCL_NET_PLUGIN` | `none` |
| `ROCSHMEM_SDMA_ENABLED` | `0` |
| `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS` | `1` (tune: 1, 2, 4, 8) |

Optional tuning passthrough: `NCCL_GIN_ANVIL_SDMA_THRESHOLD`, `NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL`.

---

## Manual build (without the orchestrator)

From `~/rocm-systems`:

```bash
export REPO_ROOT="${PWD}"
export GIN_ANVIL_BM_ROOT="${REPO_ROOT}/gin-anvil-bm"
export ROCM_PATH=/opt/rocm
export GIN_ANVIL_GPU_ARCH=gfx950
export ROCSHMEM_INSTALL_DIR="${GIN_ANVIL_BM_ROOT}/install/rocshmem"
export RCCL_INSTALL_PREFIX="${GIN_ANVIL_BM_ROOT}/install/rccl"
export MPI_PREFIX=/usr

source ./docker-gin-gda-sdma-preflight.bash

# 1) rocSHMEM (USE_SDMA=ON; BUILD_UNIT_TESTS=ON only when GIN_ANVIL_BUILD_SUITE_F=1)
mkdir -p "${GIN_ANVIL_BM_ROOT}/build/rocshmem"
cd "${GIN_ANVIL_BM_ROOT}/build/rocshmem"
"${REPO_ROOT}/projects/rocshmem/scripts/build_configs/all_backends" \
  -DUSE_SDMA=ON \
  -DGPU_TARGETS="${GIN_ANVIL_GPU_ARCH}" \
  -DCMAKE_INSTALL_PREFIX="${ROCSHMEM_INSTALL_DIR}" \
  -DMPI_ROOT="${MPI_PREFIX}" \
  -DBUILD_FUNCTIONAL_TESTS=OFF \
  -DBUILD_UNIT_TESTS=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_CTESTS=OFF \
  -DBUILD_PYTHON_TESTS=OFF

# 2) RCCL (Anvil GIN plugin)
cd "${REPO_ROOT}/projects/rccl"
sed -i \
  's/if (comm->enableRocshmem && comm->nNodes > 1 && (comm->nRanks\/comm->nNodes == 8) && comm->rocshmemThreshold <= 1048576)/if (comm->enableRocshmem)/' \
  src/rccl_wrap.cc
./install.sh \
  --amdgpu_targets="${GIN_ANVIL_GPU_ARCH}" \
  --prefix="${RCCL_INSTALL_PREFIX}" \
  --no-device-linker \
  --no_clean \
  --cmake-options \
    "-DENABLE_ROCSHMEM_GIN=ON \
     -DROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR} \
     -DROCSHMEM_SOURCE_DIR=${REPO_ROOT}/projects/rocshmem \
     -DROCSHMEM_BUILD_DIR=${GIN_ANVIL_BM_ROOT}/build/rocshmem/include/rocshmem"

# 3) rccl-tests (alltoall_perf)
mkdir -p "${GIN_ANVIL_BM_ROOT}/build/rccl-tests"
cmake -S "${REPO_ROOT}/projects/rccl-tests" -B "${GIN_ANVIL_BM_ROOT}/build/rccl-tests" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_MPI=ON \
  -DENABLE_DEVICE_API=ON \
  -DENABLE_ROCSHMEM=ON \
  -DGPU_TARGETS="${GIN_ANVIL_GPU_ARCH}" \
  -DROCSHMEM_INSTALL_DIR="${ROCSHMEM_INSTALL_DIR}" \
  -DROCSHMEM_SOURCE_DIR="${REPO_ROOT}/projects/rocshmem" \
  -DROCSHMEM_BUILD_DIR="${GIN_ANVIL_BM_ROOT}/build/rocshmem/include/rocshmem" \
  -DRCCL_SOURCE_DIR="${REPO_ROOT}/projects/rccl" \
  -DCMAKE_PREFIX_PATH="${RCCL_INSTALL_PREFIX};${MPI_PREFIX}"
cmake --build "${GIN_ANVIL_BM_ROOT}/build/rccl-tests" -j"$(nproc)"

# 4) GTest unit binaries (suites A–H, G)
cmake -S "${REPO_ROOT}/projects/rccl" -B "${GIN_ANVIL_BM_ROOT}/build/rccl-unit" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_ROCSHMEM_GIN=ON \
  -DENABLE_ROCSHMEM=ON \
  -DBUILD_TESTS=ON \
  -DGPU_TARGETS="${GIN_ANVIL_GPU_ARCH}" \
  -DROCSHMEM_INSTALL_DIR="${ROCSHMEM_INSTALL_DIR}" \
  -DROCSHMEM_SOURCE_DIR="${REPO_ROOT}/projects/rocshmem" \
  -DROCSHMEM_BUILD_DIR="${GIN_ANVIL_BM_ROOT}/build/rocshmem/include/rocshmem"
cmake --build "${GIN_ANVIL_BM_ROOT}/build/rccl-unit" \
  --target rccl-UnitTestsFixtures rccl-UnitTestsGinAnvilPlugin -j"$(nproc)"
```

---

## Manual run

### Unit tests (1 GPU)

```bash
export GIN_ANVIL_BM_ROOT=~/rocm-systems/gin-anvil-bm
export LD_LIBRARY_PATH="${GIN_ANVIL_BM_ROOT}/install/rccl/lib:${GIN_ANVIL_BM_ROOT}/install/rocshmem/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

"${GIN_ANVIL_BM_ROOT}/build/rccl-unit/test/rccl-UnitTestsFixtures" \
  --gtest_filter='GinAnvil*'
"${GIN_ANVIL_BM_ROOT}/build/rccl-unit/test/rccl-UnitTestsGinAnvilPlugin" \
  --gtest_filter='GinAnvilPluginTest.*'
"${GIN_ANVIL_BM_ROOT}/build/rocshmem/tests/unit_tests/rocshmem_unit_tests" \
  --gtest_filter='GinAnvilSdmaFactoryTest.*'
```

### Integration (8 GPUs)

```bash
GIN_ANVIL_LAYOUT=bare-metal ~/rocm-systems/gin-anvil-smci355-test.bash integration
```

Single-command C2 (Test#5 only):

```bash
mpirun -n 8 --allow-run-as-root \
  -mca pml ob1 -mca btl self,vader,tcp \
  -mca btl_vader_single_copy_mechanism none \
  -mca hwloc_base_binding_policy none \
  -x LD_LIBRARY_PATH -x PATH \
  -x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc \
  -x ROCSHMEM_DISABLE_MIXED_IPC=1 -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
  -x RCCL_ROCSHMEM_THRESHOLD=134217728 \
  -x NCCL_DEBUG=VERSION -x NCCL_DEBUG_SUBSYS=INIT,NET \
  -x NCCL_CUMEM_ENABLE=1 -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_GIN_PLUGIN=none \
  -x NCCL_NET_PLUGIN=none -x ROCSHMEM_SDMA_ENABLED=0 \
  -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=5 \
  -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 \
  "${GIN_ANVIL_BM_ROOT}/build/rccl-tests/alltoall_perf" \
  -b 128 -e 128M -f 2 -g 1 -R 2 -D 3 -A 1 -V 1
```

---

## Docker vs bare-metal

| Concern | Docker (`GIN_ANVIL_LAYOUT=docker`) | Bare metal (`GIN_ANVIL_LAYOUT=bare-metal`) |
|---------|-------------------------------------|--------------------------------------------|
| Primary use | Regression C1+C2, perf logs | Unit tests, native gdb, quick rebuild |
| Build | `docker-gin-gda-sdma-build.bash` → image `rccl-gin-gda-sdma-713` | `gin-anvil-bm/install/*` |
| MLX5 DMA-BUF | May need `extra-rdma-debs` or bind mounts | Host `libmlx5` directly |
| Unit tests | Always run on **host** (1 GPU) | Same binaries under `gin-anvil-bm/build/` |
| Integration | `./docker-gin-gda-sdma-test.bash` | `mpirun` + `alltoall_perf` from `gin-anvil-bm` |

Recommended workflow on `smci355-ccs-aus-m03-17`:

1. `./gin-anvil-smci355-test.bash all` — docker C1+C2 + host unit suites A–H, G (suite F off).
2. After code edits: `GIN_ANVIL_SKIP_DOCKER_REBUILD=1 ./gin-anvil-smci355-test.bash unit` — RCCL unit only (no `install.sh`; builds under `gin-anvil-bm/build/rccl-unit/`).
3. Optional host suite F: `GIN_ANVIL_BUILD_SUITE_F=1` after `libopenmpi-dev` install.
4. Bare-metal integration: `GIN_ANVIL_LAYOUT=bare-metal` (runs `install.sh` + `alltoall_perf`; preflight requires host MPI).

Host ROCm (e.g. 7.0.2 on smci355) may differ from the docker image (7.13). Default `all` uses docker for integration; host builds only GTest binaries.

---

*File: `docs/gin-anvil-smci355-bare-metal-layout.md`*
