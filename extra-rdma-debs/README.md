# Optional rdma-core / provider packages

Place extra RDMA userspace packages here when the stock image lacks symbols or
NIC providers required for GIN Anvil-SDMA Test#5.

| Base OS | Package format | Typical use |
|---------|----------------|-------------|
| **CentOS Stream 9** (default image) | `*.rpm` | Broadcom `libbng_re` RPMs from MI455 SUT; optional rdma-core updates |
| Ubuntu 24.04 (`BASE_OS=ubuntu`) | `*.deb` | Newer rdma-core/libmlx5 with `mlx5dv_reg_dmabuf_mr` |

The Dockerfile copies this directory and installs any matching packages for the
selected base OS. Leave empty to skip (default).

### MI455 (Broadcom Thor Ultra, `bng_re`)

Copy from the SUT (after `dnf install rdma-core libibverbs`):

```bash
# On SUT — example RPMs to bake into the image
scp ctheliosp-1b112-a43-1.mnb.dcgpu:/usr/lib64/libbng_re*.rpm extra-rdma-debs/  # if repackaged
# Or copy the installed RPM:
rpm -qa | grep libbng_re
scp ctheliosp-...:/path/to/libbng_re-140*.rpm extra-rdma-debs/
```

CentOS Stream 9 base image already ships **rdma-core 61** with
`libmlx5.so.1.25` (MLX5_1.25 DMA-BUF symbols). You mainly need **`libbng_re`**
RPMs here for in-container `bng_re` IB provider support.

### Runtime bind-mount (alternative)

Instead of baking RPMs, bind-mount host libs on the SUT:

```bash
export TEST5_HOST_MLX5_LIB_DIR=/usr/lib64
export TEST2_BIND_HOST_RDMA_SO=1
export MPI_MCA_EXTRA="-x NCCL_IB_HCA=bng_re"
```

See `ddai-artifacts/docs/gin-sdma-a2a-harness.md`.

If Test#5 is skipped with an MLX5 preflight message:

- CentOS image: set `RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=1` at build time to verify symbols
- Ubuntu legacy: add MOFED/newer rdma-core `.deb` files here and rebuild
- Or set `TEST5_MLX5_PREFLIGHT=0`, or `TEST5_HOST_MLX5_LIB_DIR` to a host directory with suitable `libmlx5*.so*`
