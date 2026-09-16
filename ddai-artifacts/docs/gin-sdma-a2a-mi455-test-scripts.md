# MI455 GIN-SDMA AllToAll test scripts

WIP branch: `users/dondai/gin-sdma-a2a-mi455-fabric-wip`.

These notes cover the **MI455** launchers under `ddai-artifacts/scripts/`.
Docker image tag: `rccl-gin-sdma-a2a-mi455`. Device-API kernel: `GinAlltoAllKernel`
(`-D 3`), `NCCL_GIN_TYPE=6`.

Default 1-node SUT: `ctheliosp-rck-g02-j07-01.rck.dcgpu` (ssh alias `mi455-sut`).
2p8g pair: `ctheliosp-rck-g02-j07-01.rck.dcgpu` + `ctheliosp-rck-g02-j07-02.rck.dcgpu` (`mi455-sut` / `mi455-sut-2`).

| Script | Mode | Shape | What it runs |
|--------|------|-------|----------------|
| `gin-sdma-a2a-mi455-baremetal-func.bash` | host `mpirun` | 1p4g | GIN A2A 128 B–4 GiB, validation on |
| `gin-sdma-a2a-mi455-baremetal-2p8g.bash` | host `mpirun` | 2p8g | Same sweep across two 4-GPU nodes |
| `gin-sdma-a2a-mi455-2p8g.bash` | wrapper | 2p8g | `exec` of `…-baremetal-2p8g.bash` |
| `gin-sdma-a2a-test.bash` | `docker run` | 1p4g | Test#1 / #2 / #4 / #5 matrix |
| `gin-sdma-a2a-mi455-run-all.bash` | docker + host gtests | 1p4g | Policy UTs + harness 1M + Test#5 4 GiB |

Copy of scripts on the SUT (optional): `~/gin-a2a-runs/`.

---

## Bare-metal 1p4g

`ddai-artifacts/scripts/gin-sdma-a2a-mi455-baremetal-func.bash`

Host OpenMPI against `/dev/kfd`. **No** `docker run` of `alltoall_perf`. Docker is
used only to stage PR binaries into `${GIN_A2A_STAGE:-$HOME/gin-a2a-baremetal-stage}`
(RCCL, `alltoall_perf`, OpenMPI, PMIx, image ROCm).

```bash
# Full functional sweep (128 B–4 GiB, -D 3)
bash ddai-artifacts/scripts/gin-sdma-a2a-mi455-baremetal-func.bash

# Short debug sweep
MAX_BYTES=256K WARMUP=0 ITERS=1 bash ddai-artifacts/scripts/gin-sdma-a2a-mi455-baremetal-func.bash

# Host-path control (not GIN kernel)
D_MODE=0 NCCL_GIN_ENABLE=0 NCCL_GIN_TYPE=0 MAX_BYTES=256K \
  bash ddai-artifacts/scripts/gin-sdma-a2a-mi455-baremetal-func.bash

# Skip ~9 GB image ROCm copy; use host /opt/rocm (ABI must match the image)
ROCM_FROM_HOST=1 bash ddai-artifacts/scripts/gin-sdma-a2a-mi455-baremetal-func.bash

# Stage only (used by 2p8g); does not launch AllToAll
GIN_A2A_STAGE_ONLY=1 bash ddai-artifacts/scripts/gin-sdma-a2a-mi455-baremetal-func.bash
```

Pass criteria: `Out of bounds values : 0 OK`, no `Memory access fault` /
`GPU HANG` / `HSA_STATUS_ERROR` / `Test failure` / `NCCL error`.

### 1p4g knobs

| Variable | Default | Meaning |
|----------|---------|---------|
| `NP` | `4` | MPI ranks (1 GPU each) |
| `HIP_VISIBLE_DEVICES` | `0,1,2,3` | GPUs |
| `MIN_BYTES` / `MAX_BYTES` | `128` / `4G` | rccl-tests `-b` / `-e` |
| `WARMUP` / `ITERS` | `1` / `2` | `-w` / `-n` |
| `D_MODE` | `3` | `-D` (`3` = GinAlltoAllKernel) |
| `DOCKER_IMAGE` | `rccl-gin-sdma-a2a-mi455` | Staging source |
| `GIN_A2A_STAGE` | `$HOME/gin-a2a-baremetal-stage` | Staged tree |
| `ALLTOALL_PERF` / `RCCL_LIBDIR` | (from stage) | Override to skip docker staging |
| `NCCL_IB_DISABLE` | `1` | Avoid broken verbs `ibv_get_device_list` SEGV |
| `GIN_A2A_STAGE_ONLY` | unset | `1` = stage and exit |

---

## Bare-metal 2p8g

`ddai-artifacts/scripts/gin-sdma-a2a-mi455-baremetal-2p8g.bash`

Two nodes, **4 ranks per node**, 8 GPUs total. Launch from **node 1**. The stage
directory must exist at the **same path** on both nodes (`rsync` unless
`SYNC_STAGE=0` or the path is NFS).

Keep `gin-sdma-a2a-mi455-baremetal-func.bash` next to this script (staging).

```bash
HOST2=ctheliosp-rck-g02-j07-02.rck.dcgpu \
  bash ddai-artifacts/scripts/gin-sdma-a2a-mi455-baremetal-2p8g.bash

HOST2=peer MAX_BYTES=64M WARMUP=0 ITERS=1 \
  bash ddai-artifacts/scripts/gin-sdma-a2a-mi455-baremetal-2p8g.bash

HOSTS=host1:4,host2:4 bash ddai-artifacts/scripts/gin-sdma-a2a-mi455-baremetal-2p8g.bash

HOSTFILE=/path/hostfile bash ddai-artifacts/scripts/gin-sdma-a2a-mi455-baremetal-2p8g.bash
```

`gin-sdma-a2a-mi455-2p8g.bash` is a compatibility wrapper that execs the
bare-metal 2p8g script.

Fabric LL arms only if all **8** GPUs are one MNNVL clique
(`clique.size == nRanks`). Otherwise RCCL falls back to gin.put/SDMA (or fails
if there is no inter-node net).

### 2p8g knobs (in addition to 1p4g)

| Variable | Default | Meaning |
|----------|---------|---------|
| `HOST2` | (required unless `HOSTS`/`HOSTFILE`) | Peer hostname |
| `HOST1` | `hostname -s` | Launch node |
| `HOSTS` | `HOST1:4,HOST2:4` | OpenMPI `-H` list |
| `HOSTFILE` | unset | OpenMPI `--hostfile` |
| `NP` / `PPN` | `8` / `4` | ranks / ranks per node |
| `SYNC_STAGE` | `1` | `rsync` stage to remote hosts |
| `NCCL_IB_DISABLE` | `1` | Same verbs workaround; set `0` if RoCE/IB is healthy |

---

## Docker harness (1p4g)

`ddai-artifacts/scripts/gin-sdma-a2a-test.bash [NP] [MAX_BYTES]`

Runs `alltoall_perf` **inside** `docker run` with `/dev/kfd`. Default NP=4.

```bash
bash ddai-artifacts/scripts/gin-sdma-a2a-test.bash 4 128M
RCCL_GIN_RUN_TESTS=5 bash ddai-artifacts/scripts/gin-sdma-a2a-test.bash 4 1M
RCCL_GIN_RUN_TESTS=1,2,4,5 bash ddai-artifacts/scripts/gin-sdma-a2a-test.bash 4 1M
```

If `NCCL_IB_DISABLE=1`, the harness withholds verbs devices even when the
provider preflight did not flag a broken `.so` (avoids `ibv_get_device_list`
SEGV on MI455).

| Test | Path |
|------|------|
| #1 | Host `-D 0` RING / CE / hybrid (`TEST1_MODE`) |
| #2 | GIN proxy `NCCL_GIN_TYPE=2` |
| #4 | GIN GDA `NCCL_GIN_TYPE=4` (may skip on old bnxt fw) |
| #5 | GIN Anvil-SDMA `NCCL_GIN_TYPE=6`, `-D 3` |

More knobs: [gin-sdma-a2a-harness.md](gin-sdma-a2a-harness.md).

---

## Run-all matrix (1p4g)

`ddai-artifacts/scripts/gin-sdma-a2a-mi455-run-all.bash`

IB withheld (`NCCL_IB_DISABLE=1`). Logs under `$HOME/gin-a2a-runs/<timestamp>/`.

```bash
bash ddai-artifacts/scripts/gin-sdma-a2a-mi455-run-all.bash
```

Steps: host-compiled policy gtests if present in the image (often SKIP), then
harness Tests 1,2,4,5 to 1 MiB, then Test#5 to 4 GiB, then MPI
`Alltoall_FabricLL_*` if `rccl-UnitTestsMPI` exists (usually SKIP on this image).

---

## Env checklist (GIN Anvil-SDMA)

```
NCCL_GIN_ENABLE=1
NCCL_GIN_TYPE=6
NCCL_CUMEM_ENABLE=1
NCCL_MNNVL_ENABLE=1
NCCL_DMABUF_ENABLE=1
HSA_NO_SCRATCH_RECLAIM=1
```

Compact GIN LL scratch uses a **threshold-sized** tail and
`ginFabricLlA2ASlotPkts` (not the 16 MiB host-DDA slot). `alltoall_perf` must
be rebuilt from that header; staging an old binary will page-fault at 128 B.
