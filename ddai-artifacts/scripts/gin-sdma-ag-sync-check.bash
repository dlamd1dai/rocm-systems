#!/usr/bin/env bash
# Verify GIN-SDMA AllGather manifest files match between two branches.
# Usage (from repo root):
#   ddai-artifacts/scripts/gin-sdma-ag-sync-check.bash [REF_A] [REF_B]
# Defaults: users/dondai/gin-stage3b-sdma-ag vs users/dondai/gin-stage3b-sdma-ag-nccl-2.30.7-wip

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "error: run from inside the rocm-systems git repository" >&2
  exit 1
fi
cd "${REPO_ROOT}"

REF_A="${1:-users/dondai/gin-stage3b-sdma-ag}"
REF_B="${2:-users/dondai/gin-stage3b-sdma-ag-nccl-2.30.7-wip}"
MANIFEST="${REPO_ROOT}/ddai-artifacts/scripts/gin-sdma-ag-manifest.txt"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "error: manifest not found: ${MANIFEST}" >&2
  exit 1
fi

for ref in "${REF_A}" "${REF_B}"; do
  if ! git rev-parse --verify -q "${ref}^{commit}" >/dev/null; then
    echo "error: unknown git ref: ${ref}" >&2
    exit 1
  fi
done

echo "GIN-SDMA AllGather manifest sync check"
echo "  A: ${REF_A}"
echo "  B: ${REF_B}"
echo

fail=0
while IFS= read -r line || [[ -n "${line}" ]]; do
  line="${line%%#*}"
  line="$(echo "${line}" | xargs)"
  [[ -z "${line}" ]] && continue

  if ! git cat-file -e "${REF_A}:${line}" 2>/dev/null; then
    echo "MISSING on ${REF_A}: ${line}"
    fail=1
    continue
  fi
  if ! git cat-file -e "${REF_B}:${line}" 2>/dev/null; then
    echo "MISSING on ${REF_B}: ${line}"
    fail=1
    continue
  fi

  hash_a="$(git rev-parse "${REF_A}:${line}")"
  hash_b="$(git rev-parse "${REF_B}:${line}")"
  if [[ "${hash_a}" != "${hash_b}" ]]; then
    echo "DIFF: ${line}"
    fail=1
  else
    echo "OK:   ${line}"
  fi
done < "${MANIFEST}"

# all_gather.cu cannot be in the manifest: the NCCL 2.30.7 wip branch carries its
# own version-guarded ncclTestEngine / devComm-requirements layout, so the file is
# never byte-identical. The AllGather kernel lives in that file, which made the
# design itself unverifiable here -- a per-CTA signal-slot regression that
# deadlocked every grid wider than one CTA sat on one branch and not the other
# while this check reported all-clear. Compare the kernel region on its own so
# design drift fails even when the surrounding file legitimately differs.
AG_KERNEL_FILE="projects/rccl-tests/src/all_gather.cu"
AG_KERNEL_BEGIN="__device__ void ginAllGatherBody"

extract_ag_kernel() {  # $1 = git ref
  git show "${1}:${AG_KERNEL_FILE}" 2>/dev/null \
    | awk -v begin="${AG_KERNEL_BEGIN}" '
        index($0, begin) { inside = 1 }
        inside           { print }
        inside && /^\}/  { exit }
      '
}

echo
body_a="$(extract_ag_kernel "${REF_A}")"
body_b="$(extract_ag_kernel "${REF_B}")"

if [[ -z "${body_a}" || -z "${body_b}" ]]; then
  # A rename or refactor must fail loudly rather than silently pass.
  [[ -z "${body_a}" ]] && echo "MISSING kernel region on ${REF_A}: '${AG_KERNEL_BEGIN}' in ${AG_KERNEL_FILE}"
  [[ -z "${body_b}" ]] && echo "MISSING kernel region on ${REF_B}: '${AG_KERNEL_BEGIN}' in ${AG_KERNEL_FILE}"
  fail=1
elif [[ "$(printf '%s' "${body_a}" | sha256sum)" != "$(printf '%s' "${body_b}" | sha256sum)" ]]; then
  echo "DIFF: ${AG_KERNEL_FILE} :: ginAllGatherBody (kernel design drift)"
  diff <(printf '%s\n' "${body_a}") <(printf '%s\n' "${body_b}") | sed 's/^/      /' || true
  fail=1
else
  echo "OK:   ${AG_KERNEL_FILE} :: ginAllGatherBody (kernel design)"
fi

echo
if [[ "${fail}" -ne 0 ]]; then
  echo "FAIL: manifest files differ between ${REF_A} and ${REF_B}" >&2
  exit 1
fi
echo "PASS: all manifest files match"
