#! /usr/bin/env bash
# Runs inside rccl-ginsdma101. Overlay skip/always Anvil headers and rebuild
# alltoall_perf (GIN device templates are compiled into the test binary).
# Mount $OUT at /pr12137 with gin_anvil_sdma.{skipquiet,alwaysquiet}.h present.
set -euo pipefail
HDR=/workspace/rccl/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma.h
WHICH="${1:-both}"

rebuild_variant() {
  local name="$1" src="$2"
  echo "===== rebuild $name from $src ====="
  cp -a "$src" "$HDR"
  cd /workspace/rccl-tests
  find . -name 'alltoall*.o' -delete || true
  rm -f alltoall.o src/alltoall.o || true
  make alltoall_perf -j"$(nproc)"
  mkdir -p "/pr12137/ab/$name"
  cp -a /workspace/rccl-tests/alltoall_perf "/pr12137/ab/$name/alltoall_perf"
  if [ ! -s "/pr12137/ab/$name/librccl.so.1" ] || [ -L "/pr12137/ab/$name/librccl.so.1" ]; then
    LIB=$(find /opt/rocm/lib /workspace/rccl -name 'librccl.so.1' -type f 2>/dev/null | head -1)
    echo "copying LIB=$LIB"
    rm -f "/pr12137/ab/$name/librccl.so" "/pr12137/ab/$name/librccl.so.1" "/pr12137/ab/$name/librccl.so.1.0"
    cp -a "$LIB" "/pr12137/ab/$name/librccl.so.1"
    ln -sfn librccl.so.1 "/pr12137/ab/$name/librccl.so"
  fi
  ls -l "/pr12137/ab/$name"
}

case "$WHICH" in
  skip) rebuild_variant skip /pr12137/gin_anvil_sdma.skipquiet.h ;;
  always) rebuild_variant always /pr12137/gin_anvil_sdma.alwaysquiet.h ;;
  both)
    rebuild_variant skip /pr12137/gin_anvil_sdma.skipquiet.h
    rebuild_variant always /pr12137/gin_anvil_sdma.alwaysquiet.h
    ;;
  *) echo "usage: skip|always|both" >&2; exit 1 ;;
esac
echo REBUILD_AB_OK
