#!/usr/bin/env bash
# Launch the vLLM NWOR development container with local sources mounted and
# precompiled wheels disabled so that the rebuilt CUDA extension is used.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

DOCKER_IMAGE="${DOCKER_IMAGE:-vllm-nwor:fixed}"

docker run --rm --gpus all \
  -v "${REPO_ROOT}":/workspace/vllm \
  -w /workspace/vllm \
  -e PYTHONPATH=/workspace/vllm \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib \
  -e VLLM_USE_PRECOMPILED=0 \
  --entrypoint=/bin/bash \
  "${DOCKER_IMAGE}" \
  "$@"
