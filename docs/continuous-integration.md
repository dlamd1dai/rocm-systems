# Continuous Integration

> [!IMPORTANT]
> This document is currently in **draft** and may be subject to change.

This document is to detail the various continuous integration (CI) systems that are run on the rocm-systems super-repo.

## Table of Contents
1. [Azure Pipelines](#azure-pipelines)
    1. [Overview](#az-overview)
    2. [PR Workflow](#az-workflow)
    3. [Interpreting Results](#az-results)
    4. [Build and Test Coverage](#az-coverage)
    5. [Downstream Job Triggers](#az-downstream)
2. [Math CI](#math-ci)
    1. [Overview](#math-overview)
3. [Windows CI](#windows-ci)
    1. [Overview](#win-overview)
4. [TheRock CI](#therock-ci)
    1. [Overview](#rock-overview)
5. [RCCL GIN Anvil SDMA (local validation)](#rccl-gin-anvil-sdma-local-validation)

## Azure Pipelines

### Overview <a id="az-overview"></a>

The ROCm Azure Pipelines CI (also known as External CI) is a public-facing CI system that builds and tests against latest public source code. It encompasses almost all of the ROCm stack, typically pulling source code from the `develop` or `amd-staging` branch on a component's GitHub repository. The CI's main source is publicly available at [ROCm/ROCm/.azuredevops](https://github.com/ROCm/ROCm/tree/develop/.azuredevops).

See the [Azure super-repo dashboard](https://dev.azure.com/ROCm-CI/ROCm-CI/_build?definitionScope=%5Csuper-repo) for a full list of pipelines running in the super-repo.

For commits, the pipelines will run based on the conditions defined in the trigger files under [/.azuredevops](https://github.com/ROCm/rocm-systems/tree/develop/.azuredevops).

For PRs, the [`Dispatch Azure CI`](https://github.com/ROCm/rocm-systems/blob/develop/.github/workflows/azure-ci-dispatcher.yml) GitHub Action will be run, which will analyze a PR's contents and determine which pipelines to run. This action will report the final results of each Azure run it dispatches.

### PR Workflow <a id="az-workflow"></a>

1. PR is submitted
2. `Dispatch Azure CI` is run on the PR
    1. Analyzes the PR's contents, determines which pipelines to run
    2. Sends request(s) to Azure API to start runs
3. Azure CI builds and tests the PR against latest public source
4. `Dispatch Azure CI` waits until all runs are finished and reports their overall status

URLs for individual Azure runs can be found in the logs of the `Dispatch Azure CI` action, under the `Wait for and report Azure CI` step.

### Interpreting Results <a id="az-results"></a>

Any errors or warnings during a run will be highlighted on the run's main page on Azure, and clicking on those will bring you directly to the offending logs.

Azure runs can have the following statuses: `Success`, `Failed`, or `Warning`. This corresponds to GitHub status checks as follows:

| Azure Status | GitHub PR Status | Explanation |
|-|-|-|
| ✅ Success | ✅ Succeeded | The job was successful. |
| ⚠️ Warning | ✅ Succeeded with issues | An allowed failure occurred and the job continued on without further issue. |
| ❌ Failed | ❌ Failing | The job failed. |
| Did not run | ⬛ Neutral | The job did not run, likely due to not fulfilling the trigger requirements. |

Warnings can occur if a step fails but was marked as being allowed to fail, so a job will continue running in the event of a warning.

In particular, steps are allowed to fail if they have the property `continueOnError: true` ([reference](https://learn.microsoft.com/en-us/azure/devops/pipelines/process/tasks?view=azure-devops&tabs=yaml#task-control-options)).

### Build and Test Coverage <a id="az-coverage"></a>

Azure CI builds and tests primarily on Ubuntu 22.04 LTS and for `gfx942` and `gfx90a` architectures, and adding build support for more architectures and operating systems is in progress.

Build coverage:
| | Ubuntu 22.04 | Almalinux 8 |
|-|-|-|
| **gfx942** | ✅ Supported | ✅ Supported |
| **gfx90a** | ✅ Supported | ✅ Supported |
| **gfx1201** | 🚧 In progress | 🚧 In progress |
| **gfx1100** | 🚧 In progress | 🚧 In progress |
| **gfx1030** | 🚧 In progress | 🚧 In progress |

Test coverage:
| | Ubuntu 22.04 | Almalinux 8 |
|-|-|-|
| **gfx942** | ✅ Supported | ❌ Unsupported |
| **gfx90a** | ✅ Supported | ❌ Unsupported |
| **gfx1201** | ❌ Unsupported | ❌ Unsupported |
| **gfx1100** | ❌ Unsupported | ❌ Unsupported |
| **gfx1030** | ❌ Unsupported | ❌ Unsupported |

For testing, the majority of components use `ctest` or `gtest`. Component-specific details such as build flags and test configurations can be viewed in a component's main pipeline file in [ROCm/ROCm/.azuredevops/components](https://github.com/ROCm/ROCm/tree/develop/.azuredevops/components).

### Downstream Job Triggers <a id="az-downstream"></a>

Azure CI runs for a component will trigger runs for downstream components (provided that they are fully migrated onto the super-repo). The end goal is to catch upstream breaking changes before they are merged and to ensure the super-repo is always in a valid state.

For example: a rocPRIM PR will trigger a rocPRIM job. If successful, it will then continue to run hipCUB and rocThrust jobs.

Currently, the following downstream trigger paths are enabled:

```mermaid
graph TD;
  rocPRIM-->hipCUB;
  rocPRIM-->rocThrust;
  rocRAND-->hipRAND;
  hipBLAS-common-->hipBLASLt
```

## Math CI

### Overview <a id="math-overview"></a>

## Windows CI

### Overview <a id="win-overview"></a>

## TheRock CI

### Overview <a id="rock-overview"></a>

## RCCL GIN Anvil SDMA (local validation) <a id="rccl-gin-anvil-sdma-local-validation"></a>

The **GIN Anvil SDMA** backend (`NCCL_GIN_TYPE=5`, `NCCL_NET_DEVICE_GIN_ANVIL_SDMA`) is validated on **single-node, multi-GPU xGMI** hosts through **repo-root Docker scripts**, not through Azure Pipelines or the default TheRock RCCL workflows today. Use this harness when changing RCCL GIN plugins, rocSHMEM Anvil SDMA factory code, or related device templates.

### Documentation

| Document | Role |
|----------|------|
| [`gin-anvil-sdma-backend-design.md`](./gin-anvil-sdma-backend-design.md) | Design rationale, datapath (IPC vs SDMA), signal/IPC table model, formal test matrix |
| [`gin-anvil-sdma-backend-tests.md`](./gin-anvil-sdma-backend-tests.md) | Script reference, `NCCL_GIN_TYPE` vs `alltoall_perf -D`, per-test env |
| [`../extra-rdma-debs/README.md`](../extra-rdma-debs/README.md) | Optional newer `libmlx5` debs when the image lacks `mlx5dv_reg_dmabuf_mr` |

### Build image

From the repository root (MI355X default `gfx950`; use `GPU_TARGETS=gfx942` on MI300X):

```bash
# MI355 — always full rebuild (--no-cache); run preflight first:
source ./docker-gin-gda-sdma-preflight.bash
./docker-gin-gda-sdma-build.bash

# Ruby — preflight sourced automatically, BuildKit --network=host:
./docker-gin-gda-sdma-ruby-build.bash
```

See [`gin-anvil-sdma-backend-tests.md`](./gin-anvil-sdma-backend-tests.md) for build-args, `extra-rdma-debs/`, and image verification.

**Build requirements:**

- `ROCSHMEM_USE_SDMA=1` (default build-arg) — rocSHMEM `USE_SDMA=ON` for Anvil queues
- RCCL built with **`ENABLE_ROCSHMEM_GIN=ON`** in the image
- Image tag: **`rccl-gin-gda-sdma-713`**

Optional: `RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=1` (strict DMA-BUF symbol check at image build).

### Run tests

```bash
./docker-gin-gda-sdma-test.bash 8 128M
# Ruby: ./docker-gin-gda-sdma-ruby-test.bash 8 128M
```

Default **`RCCL_GIN_RUN_TESTS=1,5`** runs only Test#1 (host `-D 0`) and Test#5 (Anvil `NCCL_GIN_TYPE=5`, `-D 3`). Tests **#2** and **#4** are opt-in via `RCCL_GIN_RUN_TESTS=1,2,4,5`.

All harness tests use **`alltoall_perf -V 1`** (validation); `#wrong` must be **0**.

Anvil-only:

```bash
RCCL_GIN_RUN_TESTS=5 ./docker-gin-gda-sdma-test.bash 8 128M
```

### Harness test map

| Harness | `NCCL_GIN_TYPE` | `alltoall_perf -D` | Backend |
|---------|-----------------|--------------------|---------|
| Test#1 | 0 | 0 | Host `ncclAlltoAll` (baseline) |
| Test#2 | 2 | 3 | GIN Ib host proxy |
| Test#4 | 4 | 3 | GIN rocSHMEM GDA (`QueuePair`) |
| Test#5 | **5** | 3 | **GIN Anvil SDMA** (primary) |

### Preflight and common failures

- **`docker-gin-gda-sdma-preflight.bash`** — run manually before MI355 build (`source ./docker-gin-gda-sdma-preflight.bash`); Ruby build sources it automatically. See harness doc for checks.
- **Test#5 skipped** — `TEST5_MLX5_PREFLIGHT=1` and image lacks DMA-BUF symbols; default is `0` (run anyway). Fix with [`extra-rdma-debs`](../extra-rdma-debs/) or `TEST5_HOST_MLX5_LIB_DIR`.
- **GPU fault at 128 B** — signal VA not peer-mapped; verify LSA resource-window bind and IPC table (design doc **N7**).
- **Hang in AlltoAll** — try `NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1` or `TEST5_NUM_CHANNELS=1`.

### Relation to other CI

| System | GIN Anvil SDMA coverage |
|--------|-------------------------|
| Azure Pipelines | Not in default super-repo RCCL GIN matrix (see [Build and Test Coverage](#az-coverage)) |
| TheRock RCCL CI | Standard RCCL/TheRock tests; does not run this Docker GIN harness by default |
| Local Docker harness | **Authoritative** correctness/perf gate for `NCCL_GIN_TYPE=5` before merge |

For full objectives, environment matrix (E1–E4), and release checklist, see the [test plan](./gin-anvil-sdma-backend-design.md#test-plan) in the design doc.
