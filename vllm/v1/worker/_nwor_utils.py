from __future__ import annotations

from typing import Sequence

import torch

from vllm.v1.kv_cache.interceptor import KVCacheInterceptor
from vllm.v1.kv_cache.staging import StagingBuffers


def reset_nwor_staging_buffers(
    interceptor: KVCacheInterceptor,
    kv_caches: Sequence[torch.Tensor],
    staging_capacity: int,
) -> None:
    """Rebuild NWOR staging buffers to align with the current layer layout.

    The KV cache list (`kv_caches`) may change after drafter layers are bound or
    tensor-parallel ranks initialize. This helper reallocates staging buffers
    and re-registers layer caches so they match exactly the caches owned by this
    rank. It should be called whenever `kv_caches` changes.
    """
    if interceptor is None or not interceptor.nwor_enabled:
        return

    if staging_capacity <= 0:
        interceptor.set_staging_buffers(None)
        interceptor.register_layer_caches([], [], [])
        return

    if not kv_caches:
        interceptor.set_staging_buffers(None)
        interceptor.register_layer_caches([], [], [])
        return

    # Build layer shapes and tensor lists from the actual caches for this rank.
    layer_shapes: list[tuple[int, int]] = []
    key_tensors: list[torch.Tensor] = []
    value_tensors: list[torch.Tensor] = []

    # Use the dtype/device of the first layer (all layers share layout per rank).
    first_key, _ = kv_caches[0].unbind(0)
    dtype = first_key.dtype
    device = first_key.device

    for layer_cache in kv_caches:
        layer_key, layer_value = layer_cache.unbind(0)
        num_kv_heads = int(layer_key.shape[-2])
        head_dim = int(layer_key.shape[-1])
        layer_shapes.append((num_kv_heads, head_dim))
        key_tensors.append(layer_key)
        value_tensors.append(layer_value)

    staging_buffers = StagingBuffers.allocate(
        layer_shapes=layer_shapes,
        capacity=staging_capacity,
        dtype=dtype,
        device=device,
    )

    interceptor.set_staging_buffers(staging_buffers)
    interceptor.register_layer_caches(key_tensors, value_tensors, [])
