# Optional rdma-core / libmlx5 `.deb` packages

Place newer `rdma-core`, `libmlx5`, or `libmlx5dv` packages here when the stock
Ubuntu 24.04 pocket lacks MLX5 DMA-BUF symbols required by GIN Anvil-SDMA
(`mlx5dv_reg_dmabuf_mr`, `mlx5dv_get_data_direct_sysfs_path`).

The Dockerfile copies this directory and installs any `*.deb` files found.
Leave the directory empty to skip (default).

If Test#5 is skipped at runtime with an MLX5 preflight message, either:

- add MOFED/newer rdma-core debs here and rebuild, or
- run with `TEST5_MLX5_PREFLIGHT=0`, or
- set `TEST5_HOST_MLX5_LIB_DIR` to a host directory with suitable `libmlx5*.so*`.
