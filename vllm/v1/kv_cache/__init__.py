# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deferred KV cache staging utilities for NWOR (No-Write-On-Reject)."""

from .deferred import (  # noqa: F401
    DeferredWriteManager,
    get_global_deferred_manager,
    record_or_write_kv_cache,
    ShouldFallback,
    set_global_deferred_manager,
)

__all__ = [
    "DeferredWriteManager",
    "get_global_deferred_manager",
    "record_or_write_kv_cache",
    "ShouldFallback",
    "set_global_deferred_manager",
]
