#! /usr/bin/env bash
# Host wrapper: docker-run gin-anvil-sdma-ipc-rebuild-inside.bash against rccl-ginsdma101.
# Usage: OUT=/home/dondai/pr12137 ./gin-anvil-sdma-ipc-rebuild-overlay.bash [skip|always|both]
set -euo pipefail
OUT="${OUT:-/home/dondai/pr12137}"
IMAGE="${DOCKER_IMAGE:-rccl-ginsdma101}"
WHICH="${1:-both}"
HERE=$(cd "$(dirname "$0")" && pwd)

docker run --rm --name pr12137-rebuild-overlay \
  --shm-size 16G --network host \
  --device /dev/dri --device /dev/kfd \
  --ipc host --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  -v "$OUT:/pr12137" \
  -v "$HERE/gin-anvil-sdma-ipc-rebuild-inside.bash:/pr12137/rebuild-inside.bash:ro" \
  "$IMAGE" \
  bash /pr12137/rebuild-inside.bash "$WHICH"
echo HOST_REBUILD_DONE
