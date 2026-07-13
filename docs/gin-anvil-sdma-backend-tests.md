# GIN Anvil SDMA — Docker build/test harness and `alltoall_perf` command lines

This note documents the **Docker build and test scripts** at the repository root for GIN Anvil SDMA (`NCCL_GIN_TYPE=5`): how tests are selected, what each run executes, and how to tune or debug failures. For backend design and the formal pass/fail matrix, see [`gin-anvil-sdma-backend-design.md`](gin-anvil-sdma-backend-design.md). For **GTest unit suites A–H + G** (single-GPU, no MPI), see [`gin-anvil-sdma-unit-test-plan.md`](gin-anvil-sdma-unit-test-plan.md). For **MI355 bare-metal orchestration**, see [`gin-anvil-smci355-bare-metal-layout.md`](gin-anvil-smci355-bare-metal-layout.md) and [`../gin-anvil-smci355-test.bash`](../gin-anvil-smci355-test.bash).

---

## Unit tests (GTest)

Run on **one GPU** before or alongside the Docker harness. These are **not** invoked by `docker-gin-gda-sdma-test.bash`; use `gin-anvil-smci355-test.bash unit`, `gin-anvil-ruby-test.bash unit`, or build manually.

| Suite | Class / focus | Tests | Target | Filter |
|-------|---------------|------:|--------|--------|
| A | `GinAnvilIpcTableHostTest` | 9 | `rccl-UnitTestsFixtures` | `GinAnvilIpcTableHostTest.*` |
| B–E | `GinAnvilIpcDeviceTest` | 9 | `rccl-UnitTestsFixtures` | `GinAnvilIpcDeviceTest.*` |
| H | `GinAnvilSdmaTemplateTest` | 12 | `rccl-UnitTestsFixtures` | `GinAnvilSdmaTemplateTest.*` |
| G | `GinAnvilPluginTest` | 19 | `rccl-UnitTestsGinAnvilPlugin` | `GinAnvilPluginTest.*` |
| F | `GinAnvilSdmaFactoryTest` | 12 | `rocshmem_unit_tests` | `GinAnvilSdmaFactoryTest.*` (opt-in) |

**Default:** 49 tests (30 fixtures + 19 plugin). **With suite F:** 61 tests.

```bash
# MI355 Conductor orchestrator:
./gin-anvil-smci355-test.bash unit

# Ruby MI350X orchestrator:
./gin-anvil-ruby-test.bash unit

# Manual bare-metal build (see gin-anvil-smci355-bare-metal-layout.md):
cmake -S projects/rccl -B gin-anvil-bm/build/rccl-unit \
  -DENABLE_ROCSHMEM_GIN=ON -DENABLE_ROCSHMEM=OFF -DGIN_ANVIL_UNIT_TESTS=ON \
  -DENABLE_DEVICE_LINKER=OFF -DBUILD_TESTS=ON -DGPU_TARGETS=gfx950 \
  -DROCSHMEM_INSTALL_DIR=gin-anvil-bm/install/rocshmem \
  -DROCSHMEM_SOURCE_DIR=projects/rocshmem \
  -DROCSHMEM_BUILD_DIR=gin-anvil-bm/build/rocshmem/include/rocshmem
cmake --build gin-anvil-bm/build/rccl-unit \
  --target rccl-UnitTestsFixtures rccl-UnitTestsGinAnvilPlugin
./gin-anvil-bm/build/rccl-unit/test/rccl-UnitTestsFixtures --gtest_filter='GinAnvil*'
./gin-anvil-bm/build/rccl-unit/test/rccl-UnitTestsGinAnvilPlugin --gtest_filter='GinAnvilPluginTest.*'
```

**Coverage (estimated):** ~96% weighted line / ~95% weighted branch across Anvil SDMA sources. Per-suite tables and `llvm-cov` commands: [`gin-anvil-sdma-unit-test-plan.md`](gin-anvil-sdma-unit-test-plan.md).

---

## Scripts and image

| Script | Role |
|--------|------|
| [`gin-anvil-smci355-test.bash`](../gin-anvil-smci355-test.bash) | MI355 Conductor orchestrator: preflight, build, **unit** (A–H, G, opt-in F), integration, isolation |
| [`gin-anvil-ruby-test.bash`](../gin-anvil-ruby-test.bash) | Ruby MI350X orchestrator (`sudo docker`, `gin-anvil-bm-ruby/`) |
| [`docker-gin-gda-sdma-build.bash`](../docker-gin-gda-sdma-build.bash) | Build image (`docker`, repo root) |
| [`docker-gin-gda-sdma-test.bash`](../docker-gin-gda-sdma-test.bash) | Run harness (`docker`) |
| [`docker-gin-gda-sdma-ruby-build.bash`](../docker-gin-gda-sdma-ruby-build.bash) | Build on Ruby (`sudo docker`, `--network=host`) |
| [`docker-gin-gda-sdma-ruby-test.bash`](../docker-gin-gda-sdma-ruby-test.bash) | Same harness as above (`sudo docker`) |
| [`docker-gin-gda-sdma-preflight.bash`](../docker-gin-gda-sdma-preflight.bash) | Tree validation (sourced by Ruby build; run manually before MI355 build) |

| Artifact | Value |
|----------|--------|
| Default Dockerfile | [`Dockerfile-rccl-gin-gda-sdma`](../Dockerfile-rccl-gin-gda-sdma) |
| Ruby Dockerfile | [`Dockerfile-rccl-gin-gda-sdma-ruby`](../Dockerfile-rccl-gin-gda-sdma-ruby) |
| Image tag (both) | **`rccl-gin-gda-sdma-713`** |

**Sources of truth in tree:** `projects/rccl/src/gin/`, `projects/rccl/src/include/nccl_device/net_device.h`, `projects/rccl-tests/src/alltoall.cu`.

---

## Harness overview

```text
  docker-gin-gda-sdma-test.bash [NP] [MAX_BYTES]
           │
           ├─ RCCL_GIN_RUN_TESTS filter (default 1,5)
           │
           ├─ Test#1  Host ncclAlltoAll           (-D 0)
           ├─ Test#2  GIN Ib proxy (opt-in)       (-D 3, TYPE=2)  + RDMA bind mounts
           ├─ Test#4  GIN GDA (opt-in)            (-D 3, TYPE=4)  + bnxt/fw gate
           └─ Test#5  GIN Anvil SDMA (primary)    (-D 3, TYPE=5)
```

- **`NP`** (arg 1, default **8**): `mpirun -n` rank count (one rank per GPU).
- **`MAX_BYTES`** (arg 2, default **`128M`**): passed to `alltoall_perf -e`.
- Tests run **sequentially** in one script invocation; any failing `docker run` fails the script (`set -e` on Ruby test wrapper; MI355 test relies on `docker run` exit status per block).

**Default selection:** `RCCL_GIN_RUN_TESTS=1,5` — host baseline plus Anvil SDMA only. Tests **#2** and **#4** are **not** in the default list; add them explicitly when needed.

```bash
# Default (baseline + Anvil):
./docker-gin-gda-sdma-test.bash 8 128M

# Full GIN sweep on a bnxt + MLX5-capable node:
RCCL_GIN_RUN_TESTS=1,2,4,5 ./docker-gin-gda-sdma-test.bash 8 128M

# Anvil only, force all puts through SDMA:
NCCL_GIN_ANVIL_SDMA_THRESHOLD=0 RCCL_GIN_RUN_TESTS=5 ./docker-gin-gda-sdma-test.bash 8
```

Legacy alias: **`RUN_TESTS`** is accepted if **`RCCL_GIN_RUN_TESTS`** is unset.

---

## Functional and integration tests (multi-GPU)

These tests require **8 GPUs** (default `NP=8`), **MPI**, and a built `alltoall_perf` (docker image or bare-metal `gin-anvil-bm/build/rccl-tests/`). Formal IDs and pass criteria: [design doc §6–8](gin-anvil-sdma-backend-design.md#test-plan).

| ID | Harness | What it validates | Unit-test overlap |
|----|---------|-------------------|-------------------|
| **C1** | Test#1, `-D 0` | Host `ncclAlltoAll` baseline, `#wrong==0` | — |
| **C2** | Test#5, `-D 3`, `NCCL_GIN_TYPE=5` | Full Anvil GIN AlltoAll sweep 128 B–128 MiB | Suites B–H, G (device + plugin paths) |
| **C3–C4** | Test#5, `NP=1,2,4,8` | Scale correctness | Partial (factory F11 multi-rank) |
| **D1–D3** | Test#5, default threshold | IPC small + SDMA bulk paths | C, E3, H3–H4 |
| **D4** | `THRESHOLD=0` (isolation phase) | Force all SDMA | H3, H6–H7, H11 |
| **D5** | `THRESHOLD=65536` (isolation phase) | Force all IPC | C, E3, H5 |
| **D6** | `NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL=1` | OSS7 fused copy+signal on HW | H6, `DetailHelpers` (gfx950) |
| **D7** | `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS` | Multi-queue per peer | G16, factory F7 |
| **D8** | implicit in `-D 3` kernel | Flush / quiet / dirty bitmask | H8, H12 |
| **P1–P5** | C1 vs C2 logs | Performance regression tracking | — |

**Isolation runs** (design doc D4/D5):

```bash
./gin-anvil-smci355-test.bash isolation
# Or manually:
NCCL_GIN_ANVIL_SDMA_THRESHOLD=0 RCCL_GIN_RUN_TESTS=5 ./docker-gin-gda-sdma-test.bash 8
NCCL_GIN_ANVIL_SDMA_THRESHOLD=65536 RCCL_GIN_RUN_TESTS=5 ./docker-gin-gda-sdma-test.bash 8
```

---

## GIN type vs `alltoall_perf -D`

**`NCCL_GIN_TYPE`** selects the RCCL GIN plugin. **`-D`** selects the HIP kernel in `alltoall_perf`; they are independent.

| `NCCL_GIN_TYPE` | Net device | Harness | `-D` |
|----------------:|------------|---------|------|
| 0 | (none) | Test#1 | **0** (`ncclAlltoAll`) |
| 2 | `GIN_PROXY` | Test#2 | **3** (`GinAlltoAllKernel`) |
| 4 | `GIN_ROCSHMEM_GDA` | Test#4 | **3** |
| 5 | `GIN_ANVIL_SDMA` | Test#5 | **3** |

There is **no `-D 5`**. Types **4** and **5** both use **`-D 3`**.

Other `-D` values in rccl-tests (not used by this harness): **1** / **2** NVL kernels, **4** hybrid.

---

## Build

### Preflight (`docker-gin-gda-sdma-preflight.bash`)

Run from repo root before building (Ruby build sources this automatically):

```bash
source ./docker-gin-gda-sdma-preflight.bash
```

Checks:

- Required shared GIN host sources: `gin_host_rocshmem_common.h` / `.cc`
- Removed plugin paths must **not** exist (`gin_plugin_rocshmem_api.cc`, old `rocshmem_api` headers)
- Exactly **one** `markSdmaDirty` definition in `gin_anvil_sdma.h`

**Note:** [`docker-gin-gda-sdma-build.bash`](../docker-gin-gda-sdma-build.bash) does **not** source preflight today; run it manually or use the Ruby build script.

### `docker-gin-gda-sdma-build.bash` (MI355 / default)

```bash
./docker-gin-gda-sdma-build.bash
```

Current behavior:

- **`docker build`** with **`--no-cache`** (every invocation is a full rebuild)
- Build args: `GPU_TARGETS=gfx950`, `USE_LOCAL_SRC=1`, `ROCSHMEM_USE_SDMA=1`, `RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=0` (set in script; edit script or export build-args via `docker build` manually to override)
- Creates empty **`extra-rdma-debs/`** for optional `.deb` COPY into the image
- Image tag: **`rccl-gin-gda-sdma-713`**

Override arch for MI300X: edit `TARGET_GPU_ARCH=gfx942` in the script or run `docker build` yourself with `--build-arg GPU_TARGETS=gfx942`.

### `docker-gin-gda-sdma-ruby-build.bash`

Same image tag and build args as above, plus:

- **`sudo docker`** (`DOCKER_CMD` overridable)
- **`--network="${DOCKER_BUILD_NETWORK:-host}"`** for BuildKit
- Sources **preflight** before build
- Optional log: **`BUILD_LOG`** (default `ddai-gin-rudy-build.log` — script does not tee automatically; redirect shell output if needed)

### Build requirements (image contents)

- rocSHMEM configured with **`USE_SDMA=ON`** (`ROCSHMEM_USE_SDMA=1` build-arg)
- RCCL with **`ENABLE_ROCSHMEM_GIN=ON`** and Anvil plugin
- Optional: place newer rdma-core **`.deb`** files under [`extra-rdma-debs/`](../extra-rdma-debs/) so the image exports **`mlx5dv_reg_dmabuf_mr`**
- Optional strict build: `RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=1` (fails image build if symbols missing)

Verify SDMA in image:

```bash
docker run --rm rccl-gin-gda-sdma-713 rocshmem/bin/rocshmem_info
```

---

## Test run (`docker-gin-gda-sdma-test.bash`)

Ruby: [`docker-gin-gda-sdma-ruby-test.bash`](../docker-gin-gda-sdma-ruby-test.bash) — **identical logic**; only default **`DOCKER_CMD=sudo docker`**.

### Container / MPI setup

Each test is one `docker run` + `mpirun`:

| Docker | Value |
|--------|--------|
| GPU / IPC | `--device /dev/dri`, `/dev/kfd`, `/dev/infiniband`, `--ipc host`, `--network host` |
| Limits | `--ulimit memlock=-1:-1` (disable: `DOCKER_ULIMIT_MEMLOCK=0`), `--shm-size 64G` |
| Groups | `video`, `render`; optional `rdma` (`DOCKER_RDMA_GROUP=1`) |
| uverbs | Per-host `/dev/infiniband/uverbs*`, `/dev/uverbs*` (`DOCKER_UVERBS=1`) |
| Extra | `DOCKER_EXTRA` appended to run line; `GIN_GDA_DOCKER_IT=1` adds `-it` |

Open MPI: `pml ob1`, `btl self,vader,tcp`, `btl_vader_single_copy_mechanism none`, `hwloc_base_binding_policy none`; extend with **`MPI_MCA_EXTRA`**.

Tracing: **`RCCL_GIN_ECHO=1`** (default) wraps each test in `set -x` / `set +x`.

### Shared `MPI_BASE` environment (all tests)

Passed on every `mpirun`:

| Variable | Value | Purpose |
|----------|-------|---------|
| `RCCL_ROCSHMEM_ENABLE` | `0` | Disable RCCL rocSHMEM collective path in harness |
| `ROCSHMEM_BACKEND` | `ipc` | rocSHMEM backend hint |
| `ROCSHMEM_DISABLE_MIXED_IPC` | `1` | |
| `ROCSHMEM_DEBUG_LEVEL` | `info:noversion` | |
| `RCCL_ROCSHMEM_THRESHOLD` | `134217728` (128 MiB) | |
| `NCCL_DEBUG` | `VERSION` (overridable) | Test#5 does **not** force `INFO` unless you export it |
| `NCCL_DEBUG_SUBSYS` | `INIT,NET` | |
| `NCCL_CUMEM_ENABLE` | `1` | |
| `RCCL_ENABLE_INTRANET` | `1` | |
| `NCCL_DMABUF_ENABLE` | `1` | |
| `NCCL_MSCCL_ENABLE` | `0` | |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | |

GIN tests also set **`NCCL_GIN_PLUGIN=none`** unless **`USE_EXTERNAL_PLUGIN=1`**.

### Shared `alltoall_perf` arguments

Every harness test uses:

```text
-b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -A 1 -V 1 -D <mode>
```

| Flag | Meaning |
|------|---------|
| `-b 128` | Min message 128 B |
| `-e` | Max message (`MAX_BYTES` script arg) |
| `-f 2` | Size factor 2 (powers of two) |
| `-g 1` | One GPU per rank |
| `-R 2` | Two warmup iterations |
| `-A 1` | Async error check |
| `-V 1` | **Validation on** (`#wrong` must be 0) |
| `-D` | Kernel mode (0 or 3 per test) |

---

## Per-test detail

### Test#1 — Host baseline

| | |
|--|--|
| **Selection** | In default `RCCL_GIN_RUN_TESTS=1,5` |
| **Env** | `NCCL_GIN_ENABLE=0`, `NCCL_GIN_TYPE=0`, `ROCSHMEM_SDMA_ENABLED=0` |
| **`alltoall_perf`** | **`-D 0`** |
| **Role** | CPU/host `ncclAlltoAll` reference for perf comparison (P1) |

### Test#2 — GIN Ib host proxy (optional)

| | |
|--|--|
| **Selection** | Add **`2`** to `RCCL_GIN_RUN_TESTS` (e.g. `1,2,5`) |
| **Env** | `NCCL_GIN_ENABLE=1`, `NCCL_GIN_TYPE=2`, `NCCL_NET_PLUGIN=none`, `NCCL_ENV_PLUGIN=none`, `ROCSHMEM_SDMA_ENABLED=0`, `HSA_FORCE_FINE_GRAIN_PCIE=1` |
| **`alltoall_perf`** | **`-D 3`** |
| **RDMA** | Host **bind-mounts** for verbs/mlx5 (see §Test#2 volumes) |
| **Needs** | `ncclNIbDevs > 0`, IB devices visible in container |

### Test#4 — GIN GDA (optional)

| | |
|--|--|
| **Selection** | Add **`4`** to `RCCL_GIN_RUN_TESTS`; gated by **`TEST4_MODE`** |
| **Env** | `NCCL_GIN_ENABLE=1`, `NCCL_GIN_TYPE=4`, `NCCL_NET_PLUGIN=none`, `ROCSHMEM_SDMA_ENABLED=1` |
| **`alltoall_perf`** | **`-D 3`** |
| **`TEST4_MODE=auto`** (default) | Skip if no **`bnxt_en`** NIC or firmware &lt; **`MIN_BNXT_FW_FOR_GDA`** (`233.2.104.0`) |
| **`TEST4_MODE=run`** | Force run |
| **`TEST4_MODE=skip`** | Never run |

### Test#5 — GIN Anvil SDMA (primary)

Formal matrix: **[design doc — Test plan](gin-anvil-sdma-backend-design.md#test-plan)**.

| | |
|--|--|
| **Selection** | In default `RCCL_GIN_RUN_TESTS=1,5`; gated by **`TEST5_MODE`** |
| **Env** | `NCCL_GIN_ENABLE=1`, `NCCL_GIN_TYPE=5`, `NCCL_NET_PLUGIN=none`, `ROCSHMEM_SDMA_ENABLED=0`, `HSA_FORCE_FINE_GRAIN_PCIE=1`, `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=${TEST5_NUM_CHANNELS:-1}` |
| **Optional passthrough** | `NCCL_GIN_ANVIL_SDMA_THRESHOLD` (script forwards if set) |
| **`alltoall_perf`** | **`-D 3`**, validation **`-V 1`** |
| **Image** | rocSHMEM **`USE_SDMA=ON`**; RCCL Anvil plugin present |
| **MLX5** | RCCL NET init may need **`mlx5dv_reg_dmabuf_mr`** in image (see §MLX5) |

**`TEST5_MODE=skip`** — never run Test#5.

**`TEST5_MLX5_PREFLIGHT=1`** — skip Test#5 if image `libmlx5` lacks `mlx5dv_reg_dmabuf_mr`. Default **`0`**: run Test#5 anyway (may fail at NET init if symbols missing).

---

## Environment reference

### Test selection

| Variable | Default | Effect |
|----------|---------|--------|
| **`RCCL_GIN_RUN_TESTS`** | `1,5` | Comma list: `1`, `2`, `4`, `5` |
| **`RUN_TESTS`** | — | Legacy alias for `RCCL_GIN_RUN_TESTS` |
| **`TEST4_MODE`** | `auto` | `skip` \| `run` \| `auto` |
| **`MIN_BNXT_FW_FOR_GDA`** | `233.2.104.0` | Test#4 firmware gate |
| **`TEST5_MODE`** | `run` | `skip` to skip Test#5 |
| **`TEST5_MLX5_PREFLIGHT`** | `0` | `1` = skip Test#5 when image lacks DMA-BUF symbols |
| **`TEST5_HOST_MLX5_LIB_DIR`** | unset | Bind-mount host `libmlx5*` / `libmlx5dv*` into container for Test#5 |
| **`TEST5_NUM_CHANNELS`** | `1` | Sets `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS` for Test#5 |
| **`NCCL_GIN_ANVIL_SDMA_THRESHOLD`** | unset (128 B in RCCL) | IPC vs SDMA boundary; passthrough on Test#5 |
| **`NCCL_GIN_ANVIL_SDMA_OSS7`** | unset (`1`) | `0` = legacy COPY_LINEAR + ATOMIC SDMA packets on MI355; `1` = OSS7 MI4 fused packets (`COPY_LINEAR_WAIT_SIGNAL_MI4`) when compiled for gfx950 |
| **`NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL`** | unset (`1`) | `0` = decoupled put + signal; requires `NCCL_GIN_ANVIL_SDMA_OSS7=1` for hardware fusion |
| **`RCCL_GIN_ALLTOALL_BATCHED`** | unset (`0`) | `1` = large `-D 3` / `-D 4` paths use `GinBatchedAlltoAllExchange` (legacy PR2 frame); default is clean single-lane fused frame |
| **`RCCL_GIN_ALLTOALL_HOST_TRACE`** | unset | `1` = host stderr lines for SDMA launch and batched/clean selection |

### Docker / harness

| Variable | Default | Effect |
|----------|---------|--------|
| **`DOCKER_IMAGE`** | `rccl-gin-gda-sdma-713` | Image tag |
| **`DOCKER_CMD`** | `docker` / `sudo docker` (Ruby) | Container CLI |
| **`DOCKER_UVERBS`** | `1` | `0` = skip uverbs `--device` scan |
| **`DOCKER_RDMA_GROUP`** | `1` | `0` = skip `--group-add rdma` |
| **`DOCKER_ULIMIT_MEMLOCK`** | `1` | `0` = omit memlock ulimit |
| **`DOCKER_EXTRA`** | empty | Extra `docker run` flags |
| **`GIN_GDA_DOCKER_IT`** | `0` | `1` = interactive `-it` |
| **`RCCL_GIN_ECHO`** | `1` | `0` = disable `set -x` per test |
| **`MPI_MCA_EXTRA`** | empty | Extra Open MPI MCA flags |
| **`NCCL_DEBUG`** | `VERSION` | Override for all tests |
| **`USE_EXTERNAL_PLUGIN`** | `0` | `1` = do not set `NCCL_GIN_PLUGIN=none` |

### Test#2 RDMA bind-mounts

Used when Test#2 is selected. Search paths: **`TEST2_HOST_SO_SEARCH_DIRS`** (default `/lib64`, `/usr/lib64`, `/lib/x86_64-linux-gnu`, `/usr/lib/x86_64-linux-gnu`) — internal name **`GDA_HOST_LIB_DIRS`**.

| Variable | Default | Effect |
|----------|---------|--------|
| **`TEST2_BIND_HOST_RDMA_SO`** | `1` | Mount host RDMA `.so` files |
| **`TEST2_BIND_HOST_MLX5_SO`** | `adjacent` | `0` \| `1` \| `adjacent` (mlx5 from libibverbs dir) |
| **`TEST2_BIND_HOST_IB_SYSFS`** | `1` | `/sys/class/infiniband`, `/etc/libibverbs.d` |
| **`TEST2_BIND_HOST_DEV_IFB`** | `auto` | `auto` \| `on` \| `off` for `/dev/infiniband` volume |
| **`TEST2_BIND_HOST_GNU_DIRS`** | `0` | **`1` dangerous** — whole gnu lib dirs (glibc mismatch) |
| **`TEST2_BIND_HOST_LIBDIRS`** | `/lib/x86_64-linux-gnu` … | Used when `TEST2_BIND_HOST_GNU_DIRS=1` |
| **`TEST2_BIND_HOST_RDMA_BASE`** | — | Explicit list of `.so` basenames to bind |
| **`TEST2_BIND_HOST_RDMA_EXTRA`** | — | Extra `.so` basenames |

---

## MLX5 / DMA-BUF (Test#5)

RCCL NET initialization may `dlopen` **`mlx5dv_reg_dmabuf_mr`**. Stock Ubuntu 24.04 image `libmlx5` often lacks it.

**Fix options (pick one):**

1. Bake [`extra-rdma-debs/*.deb`](../extra-rdma-debs/) into the image at build time
2. **`TEST5_HOST_MLX5_LIB_DIR=/path/to/newer/mlx5`** at test run time
3. For Test#2-style debugging: **`TEST2_BIND_HOST_MLX5_SO=adjacent`**

Do **not** bind-mount all of **`/lib/x86_64-linux-gnu`** (host glibc vs image mismatch).

---

## Performance tuning

1. **`GPU_TARGETS`** — `gfx950` (MI355X), `gfx942` (MI300X); must match node ISA
2. **`MAX_BYTES`** — `128M` default; use smaller for smoke (`4M`, `1M`)
3. **`TEST5_NUM_CHANNELS`** / **`NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS`** — 1, 2, 4, 8
4. **`NCCL_GIN_ANVIL_SDMA_THRESHOLD`** — default 128 B; `0` = all SDMA, `65536` = all IPC
5. **`ROCSHMEM_USE_SDMA=1`** at image build (required for Anvil queues)

---

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| Test#5 skipped at start | `TEST5_MLX5_PREFLIGHT=1` and image lacks symbols; set `0` or fix MLX5 |
| `ginType NONE` with `-D 3` | `NCCL_GIN_ENABLE=1`, `NCCL_GIN_TYPE=5`, `NCCL_DEBUG=INFO` NET lines |
| `dlvsym` / `mlx5dv_reg_dmabuf_mr` | `extra-rdma-debs`, `TEST5_HOST_MLX5_LIB_DIR`, or `TEST2_BIND_HOST_MLX5_SO` |
| GPU fault at 128 B (Test#5) | Signal VA not peer-mapped — design doc LSA + IPC table |
| Hang in AlltoAll | `TEST5_NUM_CHANNELS=1`; SDMA dirty / quiet |
| Wrong results small sizes | Lower threshold / isolation: `NCCL_GIN_ANVIL_SDMA_THRESHOLD=0` |
| Wrong results large sizes | IPC table; force IPC: `THRESHOLD=65536` |
| `gin_anvil_sdma_create failed` | Image built with `USE_SDMA=ON`, xGMI visible, HIP devices |
| Stale script on node | Rsync `docker-gin-gda-sdma-test.bash`; `docker run` should show `render`, uverbs |
| Preflight / build fails | `source docker-gin-gda-sdma-preflight.bash`; fix missing or duplicate sources |
| `#wrong != 0` | `-V 1` is on; check init logs and isolation runs (IPC vs SDMA) |

---

## Quick reference

| Test | `RCCL_GIN_RUN_TESTS` | `NCCL_GIN_ENABLE` | `NCCL_GIN_TYPE` | `-D` | Backend |
|------|------------------------|-------------------|-----------------|------|---------|
| #1 | default | 0 | 0 | 0 | Host `ncclAlltoAll` |
| #2 | opt-in (`2`) | 1 | 2 | 3 | GIN Ib proxy |
| #4 | opt-in (`4`) | 1 | 4 | 3 | GDA `QueuePair` |
| #5 | default | 1 | 5 | 3 | **GIN Anvil SDMA** |

```bash
# Recommended default regression:
./docker-gin-gda-sdma-test.bash 8 128M

# Log to file:
./docker-gin-gda-sdma-test.bash 8 128M 2>&1 | tee ddai-gin-perf.log
```

---

*File: `docs/gin-anvil-sdma-backend-tests.md` — harness and command-line reference for GIN Anvil SDMA. Unit tests: [`gin-anvil-sdma-unit-test-plan.md`](gin-anvil-sdma-unit-test-plan.md).*
