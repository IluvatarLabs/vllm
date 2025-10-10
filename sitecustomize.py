# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python site customizations for local vLLM development.

This hook ensures that the CUDA extension is loaded eagerly whenever a Python
interpreter starts inside the repo (e.g. from Docker exec). Without this, the
`torch.ops._C_cache_ops` namespace would remain empty until
`vllm._custom_ops` is imported manually, which makes quick smoke checks harder.

If the extension cannot be imported (for example when running on a CPU-only
machine), we silently ignore the failure.
"""

from __future__ import annotations

import os

if not os.environ.get("VLLM_DISABLE_SITECUSTOMIZE"):
    try:  # pragma: no cover - runs during interpreter bootstrap
        # Prefer locally built extensions when running from the source tree.
        os.environ.setdefault("VLLM_USE_PRECOMPILED", "0")

        import vllm._custom_ops  # noqa: F401 - side effect: registers ops
    except Exception:
        # Import errors are expected in contexts where the extension is
        # unavailable; keep startup resilient.
        pass
