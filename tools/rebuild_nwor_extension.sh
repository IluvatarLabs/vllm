#!/usr/bin/env bash
# Incrementally rebuild the CUDA extension and copy the artifact into
# vllm/_C.abi3.so so the Python package immediately sees the new kernels.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

MAX_JOBS="${MAX_JOBS:-2}"

pushd "${REPO_ROOT}" > /dev/null

if [ ! -f build/build.ninja ]; then
  cat >&2 <<'EOF'
[NWOR] build/build.ninja not found.
[NWOR] Run the full vLLM build once (e.g. python3 -m pip install -e . --no-build-isolation)
[NWOR] before using tools/rebuild_nwor_extension.sh for incremental rebuilds.
EOF
  exit 1
fi

echo "[NWOR] Building _C.abi3.so via ninja -j${MAX_JOBS}" >&2
ninja -C build _C -j"${MAX_JOBS}"

echo "[NWOR] Copying fresh extension into vllm/_C.abi3.so" >&2
cp build/_C.abi3.so vllm/_C.abi3.so

popd > /dev/null

echo "[NWOR] Done"
