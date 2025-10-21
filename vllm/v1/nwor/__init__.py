"""NWOR (No-Write-On-Reject) Draft Commit Implementation.

This module implements a greenfield NWOR architecture that defers KV cache writes
during speculative decoding until after acceptance is determined. The core idea is
to stage lightweight pointers during the forward pass, then commit only accepted
tokens via a single CUDA kernel per layer.

Key components:
- DraftCommitManager: Manages draft staging and commit lifecycle
- commit_draft_layer: CUDA kernel that scatters accepted tokens to KV cache
"""

from vllm.v1.nwor.draft_manager import (
    DraftCommitManager,
    get_draft_manager,
)

__all__ = [
    "DraftCommitManager",
    "get_draft_manager",
]
