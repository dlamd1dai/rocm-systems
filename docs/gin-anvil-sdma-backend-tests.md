# GIN Anvil SDMA — Docker build/test harness and `alltoall_perf` command lines

This note documents the **Docker build and test scripts** for GIN Anvil SDMA (`NCCL_GIN_TYPE=6`), how each harness test maps to GIN backends and **`alltoall_perf -D`** modes, and practical tuning / troubleshooting. For backend design and the formal test matrix, see [`gin-anvil-sdma-backend-design.md`](gin-anvil-sdma-backend-design.md).

**Scripts at repo root:**

| Script | Use |
|--------|-----|
| **`docker-gin-gda-sdma-build.bash`** | Build image on MI355X / general nodes (`docker`) |
| **`docker-gin-gda-sdma-test.bash`** | Run Test#1–#5 harness (`docker`) |
| **`docker-gin-gda-sdma-ruby-build.bash`** | Build on Ruby cluster (`sudo docker`, BuildKit `--network=host`) |
| **`docker-gin-gda-sdma-ruby-test.bash`** | Same harness as above (`sudo docker`) |
| **`docker-gin-gda-sdma-preflight.bash`** | Sourced by build scripts; validates tree before `docker build` |

**Dockerfiles / image:**

| Artifact | Value |
|----------|--------|
| MI355 / default Dockerfile | **`Dockerfile-rccl-gin-gda-sdma`** |
| Ruby Dockerfile | **`Dockerfile-rccl-gin-gda-sdma-ruby`** |
| Image tag (both) | **`rccl-gin-gda-sdma-713`** |

**Sources of truth:** `projects/rccl/src/gin/`, `projects/rccl/src/include/nccl_device/net_device.h`, `projects/rccl-tests/src/alltoall.cu`.

---

## 1. GIN design overview

**GIN** is a **device-visible RMA abstraction**: device code issues **`gin.put`**, signals, and **`flush`** over a registered path selected by a **GIN plugin**.

| Knob | Meaning |
|------|--------|
| **`NCCL_GIN_ENABLE`** | Master switch for GIN plugin paths. |
| **`NCCL_GIN_TYPE`** | Selects which GIN net device / plugin RCCL wires up. |

**`NCCL_GIN_TYPE` is not the same as `alltoall_perf -D`.** `-D` picks the HIP kernel (`ncclAlltoAll` vs `GinAlltoAllKernel`, etc.). There is **no `-D 5`**; **`NCCL_GIN_TYPE=5`** (GDA) still pairs with **`-D 3`**.

---

## 2. GIN backends (`NCCL_GIN_TYPE`)

| Value | Name | Role |
|------:|------|------|
| **0** | (none) | No GIN net device. |
| **2** | **`NCCL_NET_DEVICE_GIN_PROXY`** | Host proxy GIN (Ib progress on host). Test#2. |
| **3** | **`NCCL_NET_DEVICE_GIN_GDAKI`** | GDAKI path; not in default harness. |
| *(removed)* | *was **4** = rocSHMEM API GIN* | Use **6** (Anvil SDMA) or **5** (GDA). |
| **5** | **`NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA`** | rocSHMEM GDA `QueuePair` GIN. Test#4 (bnxt gate). |
| **6** | **`NCCL_NET_DEVICE_GIN_ANVIL_SDMA`** | Direct Anvil SDMA GIN (`gin_plugin_anvil_sdma.cc`). **Test#5.** |

---

## 3. `alltoall_perf -D` (rccl-tests)

| **`-D`** | Kernel |
|---------|--------|
| **0** | Host **`ncclAlltoAll`** |
| **1** | `NvlAlltoAllKernel` |
| **2** | `NvlAlltoAllKernelOptimized` |
| **3** | **`GinAlltoAllKernel`** |
| **4** | `HybridAlltoAllKernel` |

Tests **#2, #4, #5** use **`-D 3`**. Test **#1** uses **`-D 0`**.

---

## 4. Build scripts

### `docker-gin-gda-sdma-build.bash`

**Purpose:** Build **`rccl-gin-gda-sdma-713`** from **`Dockerfile-rccl-gin-gda-sdma`**.

```bash
USE_LOCAL_SRC=1 ./docker-gin-gda-sdma-build.bash      # cached build
USE_LOCAL_SRC=1 ./docker-gin-gda-sdma-build.bash 1    # --no-cache
```

- **`DOCKER_CMD`**: `docker` (override via env).
- **`GPU_TARGETS`**: default **`gfx950`** (`TARGET_GPU_ARCH`).
- **`USE_LOCAL_SRC=1`**: build from local `projects/` tree.
- **`ROCSHMEM_USE_SDMA=1`**: required for Test#5 (`USE_SDMA=ON` in rocSHMEM).
- **`ROCSHMEM_CACHE_BUST` / `RCCL_CACHE_BUST`**: bump to invalidate Docker layers.
- **`RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS`**: optional strict MLX5 symbol check at image build (default **0**).
- Sources **`docker-gin-gda-sdma-preflight.bash`** before build.
- Log: **`ddai-gin-build.log`** (override **`BUILD_LOG`**).

### `docker-gin-gda-sdma-ruby-build.bash`

Same build args and image tag; Ruby-specific:

- **`sudo docker`** (override **`DOCKER_CMD`**).
- **`Dockerfile-rccl-gin-gda-sdma-ruby`**.
- **`--network="${DOCKER_BUILD_NETWORK:-host}"`** for BuildKit on hosts without `docker0`.
- Log: **`ddai-docker-ruby-build.log`**.

---

## 5. Test scripts

### Usage

```bash
# MI355X / default (8 GPUs, 128M max message):
./docker-gin-gda-sdma-test.bash 8 128M

# Ruby cluster:
./docker-gin-gda-sdma-ruby-test.bash 8 128M

# Test#5 only, with isolation:
NCCL_GIN_ANVIL_SDMA_THRESHOLD=0 RCCL_GIN_RUN_TESTS=5 ./docker-gin-gda-sdma-test.bash 8
```

**Arguments:** `[NP]` (default **8**), **`[MAX_BYTES]`** (default **`128M`**).

**Defaults:** **`RCCL_GIN_RUN_TESTS=1,5`** (baseline + Anvil SDMA). Override with **`RCCL_GIN_RUN_TESTS`** or legacy **`RUN_TESTS`**.

**Image:** **`DOCKER_IMAGE=rccl-gin-gda-sdma-713`** (overridable on MI355 script; Ruby defaults the same).

### Shared harness behavior

Both **`docker-gin-gda-sdma-test.bash`** and **`docker-gin-gda-sdma-ruby-test.bash`** share the same test logic; only **`DOCKER_CMD`** differs (`docker` vs `sudo docker`).

**Docker / MPI defaults:**

- **`--ulimit memlock=-1:-1`**, **`--shm-size 64G`**, **`--network host`**, **`/dev/dri`**, **`/dev/kfd`**, **`/dev/infiniband`**, **`--ipc host`**, **`--group-add video` + `render`**, **`--privileged`**.
- Per-host **`/dev/infiniband/uverbs*`** / **`/dev/uverbs*`** via **`DOCKER_UVERBS`** (default **1**); optional **`--group-add rdma`**.
- Open MPI: `pml ob1`, `btl self,vader,tcp`, **`btl_vader_single_copy_mechanism none`**, **`hwloc_base_binding_policy none`**; extend with **`MPI_MCA_EXTRA`**.
- **`NCCL_GIN_PLUGIN=none`** on GIN tests (unless **`USE_EXTERNAL_PLUGIN=1`**).

**Test selection env (both scripts):**

| Variable | Default | Effect |
|----------|---------|--------|
| **`TEST4_MODE`** | `auto` | `skip` \| `run` \| `auto` (skip GDA when no `bnxt_en` or fw too old) |
| **`MIN_BNXT_FW_FOR_GDA`** | `233.2.104.0` | Firmware gate for Test#4 `auto` |
| **`TEST5_MODE`** | `run` | Set **`skip`** to skip Test#5 |
| **`TEST5_MLX5_PREFLIGHT`** | `0` | Set **`1`** to skip Test#5 if image `libmlx5` lacks `mlx5dv_reg_dmabuf_mr` |
| **`TEST5_HOST_MLX5_LIB_DIR`** | unset | Bind-mount newer host `libmlx5*` / `libmlx5dv*` for Test#5 |
| **`TEST5_NUM_CHANNELS`** | `1` | Maps to **`NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS`** |
| **`NCCL_GIN_ANVIL_SDMA_THRESHOLD`** | unset (128 B in RCCL) | Passthrough for IPC vs SDMA isolation |

**Test#2 RDMA bind-mount env:**

| Variable | Default | Effect |
|----------|---------|--------|
| **`TEST2_HOST_SO_SEARCH_DIRS`** | `/lib64` … `/usr/lib/x86_64-linux-gnu` | Host lib search paths |
| **`TEST2_BIND_HOST_RDMA_SO`** | `1` | Per-file RDMA `.so` bind mounts |
| **`TEST2_BIND_HOST_MLX5_SO`** | `adjacent` | `0` \| `1` \| `adjacent` (mlx5 from same dir as `libibverbs.so.1`) |
| **`TEST2_BIND_HOST_IB_SYSFS`** | `1` | Bind `/sys/class/infiniband`, `/etc/libibverbs.d` |
| **`TEST2_BIND_HOST_DEV_IFB`** | `auto` | `auto` \| `on` \| `off` for `/dev/infiniband` volume |
| **`DOCKER_UVERBS`** | `1` | `0` to skip uverbs `--device` discovery |
| **`DOCKER_RDMA_GROUP`** | `1` | `0` to skip `--group-add rdma` |

---

## 6. Per-test validity

### Test#1 — Host baseline

- **Env:** `NCCL_GIN_ENABLE=0`, `NCCL_GIN_TYPE=0`, `ROCSHMEM_SDMA_ENABLED=0`.
- **`alltoall_perf`:** **`-D 0`**, **`-R 2`**.

### Test#2 — GIN Ib host proxy

- **Env:** `NCCL_GIN_ENABLE=1`, **`NCCL_GIN_TYPE=2`**, `NCCL_GIN_PLUGIN=none`, `NCCL_NET_PLUGIN=none`, `HSA_FORCE_FINE_GRAIN_PCIE=1`, `NCCL_DEBUG=INFO`.
- **`alltoall_perf`:** **`-D 3`** (`GinAlltoAllKernel`).
- Needs **`ncclNIbDevs > 0`** and verbs visible in container (see §8).

### Test#4 — GIN GDA (optional)

- **Env:** `NCCL_GIN_ENABLE=1`, **`NCCL_GIN_TYPE=5`**, `ROCSHMEM_SDMA_ENABLED=1`.
- **`alltoall_perf`:** **`-D 3`**.
- Skipped when **`TEST4_MODE=auto`** and host has no **`bnxt_en`** or firmware &lt; **`MIN_BNXT_FW_FOR_GDA`**.

### Test#5 — GIN Anvil SDMA (primary)

Formal matrix: **[`gin-anvil-sdma-backend-design.md` — Test plan](gin-anvil-sdma-backend-design.md#test-plan)**.

- **Env:** `NCCL_GIN_ENABLE=1`, **`NCCL_GIN_TYPE=6`**, `ROCSHMEM_SDMA_ENABLED=0`, **`NCCL_DEBUG=INFO`** (script default), `NCCL_GIN_PLUGIN=none`, `NCCL_NET_PLUGIN=none`, `HSA_FORCE_FINE_GRAIN_PCIE=1`.
- Optional: **`NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS`**, **`NCCL_GIN_ANVIL_SDMA_THRESHOLD`**.
- **`alltoall_perf`:** **`-D 3`**, **`-R 2`**, **`-V 1`** (validation).
- Requires rocSHMEM **`USE_SDMA=ON`** in image and RCCL Anvil plugin; **`libmlx5`** DMA-BUF symbols for RCCL NET init (see **`extra-rdma-debs/`** or **`TEST5_HOST_MLX5_LIB_DIR`**).

---

## 7. Performance tuning

1. **`GPU_TARGETS` / `TARGET_GPU_ARCH`** — must match node ISA (`gfx950` MI355X, `gfx942` MI300X).
2. **`MAX_BYTES`** script arg — smoke vs peak sweep (`128M` default).
3. **`NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS`** / **`TEST5_NUM_CHANNELS`** — channel A/B on MI-class hosts.
4. **`NCCL_GIN_ANVIL_SDMA_THRESHOLD`** — IPC vs SDMA boundary (default 128 B).
5. **`ROCSHMEM_USE_SDMA=1`** at image build for Test#5.
6. **`MPI_MCA_EXTRA`** — only if you understand Docker MPI trade-offs.

---

## 8. Troubleshooting

1. **Stale script on node** — `docker run` must show `--group-add render`, `/dev/dri`, `/dev/kfd`, uverbs devices. Rsync **`docker-gin-gda-sdma-test.bash`** or **`docker-gin-gda-sdma-ruby-test.bash`** from this tree.

2. **`ginType NONE` with `-D 3`** — set `NCCL_GIN_ENABLE=1`, correct `NCCL_GIN_TYPE`, run with **`NCCL_DEBUG=INFO`**.

3. **MLX5 / DMA-BUF (`dlvsym` … `mlx5dv_reg_dmabuf_mr`)** — use **`TEST2_BIND_HOST_MLX5_SO=adjacent`**, **`extra-rdma-debs/*.deb`** at build, or **`TEST5_HOST_MLX5_LIB_DIR`**. Do not bind-mount entire **`/lib/x86_64-linux-gnu`** (breaks glibc vs image).

4. **Test#5 skipped** — **`TEST5_MLX5_PREFLIGHT=1`** without symbols; set **`TEST5_MLX5_PREFLIGHT=0`** or fix MLX5 libs.

5. **GPU fault at 128 B on Test#5** — signal VA not peer-mapped; see design doc (LSA resource window + IPC table).

6. **Hang in AlltoAll** — try **`NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1`**; check SDMA dirty / quiet in logs.

---

## 9. Quick reference

| Test | `NCCL_GIN_ENABLE` | `NCCL_GIN_TYPE` | `-D` | Backend |
|------|-------------------|-----------------|------|---------|
| #1 | 0 | 0 | 0 | Host `ncclAlltoAll` |
| #2 | 1 | 2 | 3 | GIN Ib proxy + `GinAlltoAllKernel` |
| #4 | 1 | 5 | 3 | GDA `QueuePair` + `GinAlltoAllKernel` |
| #5 | 1 | 6 | 3 | **GIN Anvil SDMA** + `GinAlltoAllKernel` |

Default run: **`RCCL_GIN_RUN_TESTS=1,5`** via **`docker-gin-gda-sdma-test.bash`** or **`docker-gin-gda-sdma-ruby-test.bash`**.

---

*File: `docs/gin-anvil-sdma-backend-tests.md` — harness and command-line reference for GIN Anvil SDMA.*
