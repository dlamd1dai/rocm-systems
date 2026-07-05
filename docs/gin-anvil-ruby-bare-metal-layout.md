# GIN Anvil SDMA — bare-metal layout for Ruby cluster (MI350X / cv350-rck-*)

Design reference: [`gin-anvil-sdma-backend-design.md`](gin-anvil-sdma-backend-design.md) environment **E1/E4** (MI350X on Ruby, `gfx950`). Unit test mapping: [`gin-anvil-sdma-unit-test-plan.md`](gin-anvil-sdma-unit-test-plan.md). Orchestration script: [`../gin-anvil-ruby-test.bash`](../gin-anvil-ruby-test.bash).

This layout mirrors [`gin-anvil-smci355-bare-metal-layout.md`](gin-anvil-smci355-bare-metal-layout.md) but uses a **separate artifact tree** so Ruby and Conductor bare-metal builds do not collide on the same checkout.

---

## Target nodes

| Item | Value |
|------|-------|
| Hostname pattern | `cv350-rck-*.rck.dcgpu` (example: `cv350-rck-g03-e09-08.rck.dcgpu`) |
| GPU | AMD Instinct **MI350X** (`gfx950` in docker/unit builds) |
| Typical repo path | `~/rocm-systems` |
| ROCm | `/opt/rocm` |
| Docker | `sudo docker` (Ruby nodes) |

---

## Directory layout

```text
~/rocm-systems/
├── gin-anvil-ruby-test.bash             # orchestrator (Ruby defaults)
├── gin-anvil-smci355-test.bash            # Conductor MI355 orchestrator
├── gin-anvil-test-common.bash           # shared implementation
├── docker-gin-gda-sdma-ruby-*.bash      # Ruby docker build/test
└── gin-anvil-bm-ruby/                   # GIN_ANVIL_BM_ROOT (Ruby)
    ├── install/
    │   ├── rocshmem/
    │   └── rccl/
    ├── build/
    │   ├── rocshmem/
    │   ├── rccl-unit/
    │   └── rccl-tests/
    └── logs/
        └── gin-anvil-ruby-<timestamp>.log
```

---

## Quick start

```bash
cd ~/rocm-systems

# Full regression (docker C1+C2 + 49 host unit tests):
./gin-anvil-ruby-test.bash all

# Unit tests only (after first build):
GIN_ANVIL_SKIP_DOCKER_REBUILD=1 ./gin-anvil-ruby-test.bash unit

# Integration only:
RCCL_GIN_RUN_TESTS=5 ./gin-anvil-ruby-test.bash integration

# Threshold isolation (D4/D5):
./gin-anvil-ruby-test.bash isolation
```

---

## Ruby vs smci355 orchestrator

| Setting | `gin-anvil-ruby-test.bash` | `gin-anvil-smci355-test.bash` |
|---------|---------------------------|-------------------------------|
| `GIN_ANVIL_BM_ROOT` | `gin-anvil-bm-ruby/` | `gin-anvil-bm/` |
| Docker CLI | `sudo docker` | `docker` |
| Build script | `docker-gin-gda-sdma-ruby-build.bash` | `docker-gin-gda-sdma-build.bash` |
| Test script | `docker-gin-gda-sdma-ruby-test.bash` | `docker-gin-gda-sdma-test.bash` |
| Default GPU arch | `gfx950` | `gfx950` |
| Host check | `cv350-rck-*` | `smci355-*` |

All CMake flags, unit test binaries, and integration env vars are identical to the smci355 bare-metal profile. See [`gin-anvil-smci355-bare-metal-layout.md`](gin-anvil-smci355-bare-metal-layout.md) for manual build commands (substitute `gin-anvil-bm-ruby` and `gin-anvil-ruby-test.bash`).

---

## Environment overrides

| Variable | Ruby default |
|----------|--------------|
| `GIN_ANVIL_BM_ROOT` | `$REPO_ROOT/gin-anvil-bm-ruby` |
| `GIN_ANVIL_GPU_ARCH` | `gfx950` |
| `DOCKER_CMD` | `sudo docker` |
| `DOCKER_IMAGE` | `rccl-gin-gda-sdma-713` |
| `GIN_ANVIL_NP` | `8` |
| `GIN_ANVIL_BUILD_SUITE_F` | `0` (opt-in factory tests) |

---

*File: `docs/gin-anvil-ruby-bare-metal-layout.md`*
