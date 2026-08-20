# Optional rdma-core / libmlx5 `.deb` overlays

Place newer `rdma-core`, `libmlx5`, or related `.deb` files here before running
`ddai-artifacts/scripts/docker-gin-gda-sdma-build.bash`. The Dockerfile installs
any `*.deb` in this directory during the image build.

Stock Ubuntu 24.04 `libmlx5` often lacks `mlx5dv_reg_dmabuf_mr` (MLX5_1.25).
Without newer libs, Test#5 can be skipped at runtime via `TEST5_MLX5_PREFLIGHT=1`
or by bind-mounting host MLX5 libs with `TEST5_HOST_MLX5_LIB_DIR`.
