# ROCm Systems

Welcome to the ROCm Systems super-repo. This repository consolidates multiple ROCm systems projects into a single repository to streamline development, CI, and integration. The first set of projects focuses on requirements for building PyTorch.

# Super-repo Status and CI Health

This table provides the current status of the migration of specific ROCm systems projects as well as a pointer to their current CI health.

**Key:**
- **Completed**: Fully migrated and integrated. This super-repo should be considered the source of truth for this project. The old repo may still be used for release activities.
- **In Progress**: Ongoing migration, tests, or integration. Please refrain from submitting new pull requests on the individual repo of the project, and develop on the super-repo.
- **Pending**: Not yet started or in the early planning stages. The individual repo should be considered the source of truth for this project.

| Component              | Source of Truth | Migration Status | Component CI Status                   |
|------------------------|-----------------|------------------|---------------------------------------|
| `amdsmi`               | Public          | Completed        |                                       |
| `aqlprofile`           | Public          | Completed        | [![CodeQL](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-codeql.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-codeql.yml) <br> [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-continuous_integration.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-continuous_integration.yml) |
| `clr`                  | Public          | Completed         |                                       |
| `hip`                  | Public          | Completed         |                                       |
| `hipfile`              | Public          | Completed         |                                       |
| `hipother`             | Public          | Completed         |                                       |
| `hip-tests`            | Public          | Completed         |                                       |
| `rdc`                  | Public          | Completed        |                                       |
| `rocdbgapi`            | Public          | Completed        | None                                  |
| `rocdecode`            | Public          | Completed        | [![Media Libs CI](https://github.com/ROCm/rocm-systems/actions/workflows/media-libs-ci.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/media-libs-ci.yml) |
| `rocjpeg`              | Public          | Completed        | [![Media Libs CI](https://github.com/ROCm/rocm-systems/actions/workflows/media-libs-ci.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/media-libs-ci.yml) |
| `rocm-core`            | Public          | Completed         |                                       |
| `rocminfo`             | Public          | Completed         |                                       |
| `rocm-smi-lib`         | Public          | Completed         |                                       |
| `rocprofiler`          | Public          | Completed         |                                       |
| `rocprofiler-compute`  | Public          | Completed         | [![Formatting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-formatting.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-formatting.yml) <br> [![ rhel-8](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-rhel-8.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-rhel-8.yml) <br> [![tarball](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-tarball.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-tarball.yml) <br> [![ubuntu jammy](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-ubuntu-jammy.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-ubuntu-jammy.yml) |
| `rocprofiler-register` | Public          | Completed         | [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-register-continuous-integration.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-register-continuous-integration.yml) |
| `rocprofiler-sdk`      | Public          | Completed        | [![Code Coverage Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-code_coverage.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-code_coverage.yml) <br> [![CodeQL](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-codeql.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-codeql.yml) <br> [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-continuous_integration.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-continuous_integration.yml) <br> [![Documentation](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-docs.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-docs.yml) <br> [![Formatting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-formatting.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-formatting.yml) <br> [![Python Linting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-python.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-python.yml) <br> [![Restrictions](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-restrictions.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-restrictions.yml) <br> [![Release Compatibility](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-rocm_release_compatibility.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-rocm_release_compatibility.yml) |
| `rocprofiler-systems`  | Public          | Completed         | [![Containers](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-containers.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-containers.yml) <br> [![rocprofiler-systems GHCR Packages for CI Images](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ghcr.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ghcr.yml) <br> [![Formatting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-formatting.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-formatting.yml) <br> [![Python Linting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-python.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-python.yml) <br> [![RedHat Linux](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-redhat.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-redhat.yml) <br> [![Ubuntu Jammy](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-jammy.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-jammy.yml) <br> [![Ubuntu Noble](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-noble.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-noble.yml) |
| `rocr-debug-agent`     | Public          | Completed        | None                                  |
| `rocr-runtime`         | Public          | Completed        |                                       |
| `rocshmem`             | Public          | Completed        |                                       |
| `roctracer`            | Public          | Completed        |                                       |


## Tentative migration schedule

| Component              | Tentative Date |
|------------------------|----------------|


*Remaining schedule to be determined.

# TheRock CI Status

Note TheRock CI performs multi-component testing on top of builds leveraging [TheRock](https://github.com/ROCm/TheRock) build system.

[![The Rock CI](https://github.com/ROCm/rocm-systems/actions/workflows/therock-ci.yml/badge.svg?branch%3Adevelop+event%3Apush)](https://github.com/ROCm/rocm-systems/actions/workflows/therock-ci.yml?query=branch%3Adevelop+event%3Apush)

---

## Nomenclature

Project names have been standardized to match the casing and punctuation of released packages. This removes inconsistent camel-casing and underscores used in legacy repositories.

## Structure

The repository is organized as follows:

```
projects/
  amdsmi/
  aqlprofile/
  clr/
  hip/
  hipfile/
  hipother/
  hip-tests/
  rccl/
  rccl-tests/
  rdc/
  rocdbgapi/
  rocdecode/
  rocjpeg/
  rocm-core
  rocminfo/
  rocmsmilib/
  rocprofiler/
  rocprofiler-compute/
  rocprofiler-register/
  rocprofiler-sdk/
  rocprofiler-systems/
  rocr-debug-agent/
  rocrruntime/
  rocshmem/
  roctracer/
```

- Each folder under `projects/` corresponds to a ROCm systems project that was previously maintained in a standalone GitHub repository and released as distinct packages.
- Each folder under `shared/` contains code that existed in its own repository and is used as a dependency by multiple projects, but does not produce its own distinct packages in previous ROCm releases.

## Goals

- Enable unified build and test workflows across ROCm libraries.
- Facilitate shared tooling, CI, and contributor experience.
- Improve integration, visibility, and collaboration across ROCm library teams.

## Documentation

Super-repo guides live under [`docs/`](./docs/). Highlights for **RCCL GPU-Initiated Networking (GIN)** and the **GIN Anvil SDMA** intra-node backend:

| Document | Description |
|----------|-------------|
| [`docs/gin-anvil-sdma-backend-design.md`](./docs/gin-anvil-sdma-backend-design.md) | Design, architecture, and formal test plan (`NCCL_GIN_TYPE=5`) |
| [`docs/gin-anvil-sdma-backend-tests.md`](./docs/gin-anvil-sdma-backend-tests.md) | Docker build/test harness, `alltoall_perf -D` mapping, env tuning |
| [`docs/gin-anvil-sdma-unit-test-plan.md`](./docs/gin-anvil-sdma-unit-test-plan.md) | GTest unit suites A–H + G: test inventory, coverage, build/run |
| [`docs/gin-anvil-smci355-bare-metal-layout.md`](./docs/gin-anvil-smci355-bare-metal-layout.md) | MI355 Conductor bare-metal install layout and CMake profile |
| [`docs/gin-anvil-ruby-bare-metal-layout.md`](./docs/gin-anvil-ruby-bare-metal-layout.md) | Ruby MI350X bare-metal layout (`gin-anvil-bm-ruby/`) |
| [`docs/continuous-integration.md`](./docs/continuous-integration.md#rccl-gin-anvil-sdma-local-validation) | CI overview and local Docker validation workflow |

**Quick start (single-node MI300/MI355, 8 GPUs):**

```bash
source ./docker-gin-gda-sdma-preflight.bash   # Ruby build does this automatically
./docker-gin-gda-sdma-build.bash
./docker-gin-gda-sdma-test.bash 8 128M        # integration: Test#1 + Test#5
```

**Unit tests (1 GPU, before integration):**

```bash
./gin-anvil-smci355-test.bash unit            # MI355 Conductor (49 GTest cases)
./gin-anvil-ruby-test.bash unit                # Ruby MI350X (same suites, separate tree)
```

Default run executes Test#1 (host baseline) and Test#5 (`NCCL_GIN_ENABLE=1`, `NCCL_GIN_TYPE=5`, `alltoall_perf -D 3 -V 1`). See [`docs/gin-anvil-sdma-backend-tests.md`](./docs/gin-anvil-sdma-backend-tests.md) for full harness options.

Related: [`extra-rdma-debs/README.md`](./extra-rdma-debs/README.md) (newer `libmlx5` for RCCL NET init in the test image).

## Getting Started

To begin contributing or building, see the [CONTRIBUTING.md](./CONTRIBUTING.md) guide. It includes setup instructions, sparse-checkout configuration, development workflow, and pull request guidelines.

## License

This super-repo contains multiple subprojects, each of which retains the license under which it was originally published.

📁 Refer to the `LICENSE`, `LICENSE.md`, or `LICENSE.txt` file within each `projects/` or `shared/` directory for specific license terms.
📄 Refer to the header notice in individual files outside `projects/` or `shared/` folders for their specific license terms.

> **Note**: The root of this repository does not define a unified license across all components.

## Questions or Feedback?

- 💬 [Start a discussion](https://github.com/ROCm/rocm-systems/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-systems/issues)

We're happy to help!
