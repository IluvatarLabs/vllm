"""Staging utilities for NWOR (No-Write-On-Reject)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

@dataclass
class LayerStagingBuffers:
    key: torch.Tensor
    value: torch.Tensor
    slots: torch.Tensor

@dataclass
class StagingBuffers:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    slots: list[torch.Tensor]
    metadata: torch.Tensor  # shape [num_layers, 2]; columns: [count, error_flag]
    capacity: int
    device: torch.device

    @classmethod
    def allocate(
        cls,
        *,
        layer_shapes: list[tuple[int, int]],
        capacity: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "StagingBuffers":
        if capacity <= 0:
            raise ValueError("Staging capacity must be > 0")
        num_layers = len(layer_shapes)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        slots: list[torch.Tensor] = []
        for (num_kv_heads, head_dim) in layer_shapes:
            shape = (capacity, num_kv_heads, head_dim)
            keys.append(torch.empty(shape, dtype=dtype, device=device))
            values.append(torch.empty(shape, dtype=dtype, device=device))
            slots.append(torch.empty(capacity, dtype=torch.int32, device=device))
        metadata = torch.zeros((num_layers, 2), dtype=torch.int32, device=device)
        return cls(keys=keys, values=values, slots=slots, metadata=metadata,
                   capacity=capacity, device=device)

    def reset(self, *, stream: Optional[torch.cuda.Stream] = None) -> None:
        if stream is None:
            self.metadata.zero_()
        else:
            with torch.cuda.stream(stream):
                self.metadata.zero_()

    def layer_buffers(self, layer_idx: int) -> LayerStagingBuffers:
        return LayerStagingBuffers(
            key=self.keys[layer_idx],
            value=self.values[layer_idx],
            slots=self.slots[layer_idx],
        )

    def free(self) -> None:
        self.keys.clear()
        self.values.clear()
        self.slots.clear()
        self.metadata = torch.tensor([], dtype=torch.int32, device=self.device)


class StagingBuffer:
    """Legacy CPU-side staging buffer used by unit/integration tests."""

    def __init__(
        self,
        n_layers: int,
        max_tokens: int,
        n_heads: int,
        head_dim: int,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if n_layers <= 0 or max_tokens <= 0:
            raise ValueError("n_layers and max_tokens must be > 0")

        self.n_layers = n_layers
        self.max_tokens = max_tokens
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype

        shape = (n_layers, max_tokens, n_heads, head_dim)
        self.k_buffer = torch.empty(shape, device=self.device, dtype=dtype)
        self.v_buffer = torch.empty(shape, device=self.device, dtype=dtype)
        self.slot_buffer = torch.empty(max_tokens, device=self.device,
                                       dtype=torch.int64)

        self.token_mask = torch.zeros(max_tokens, device=self.device,
                                      dtype=torch.bool)
        self.stage_count = 0
        self.reset()

    def reset(self) -> None:
        self.token_mask.zero_()
        self.stage_count = 0
        self.slot_buffer.zero_()

    def is_busy(self) -> bool:
        return bool(self.token_mask.any().item())

    def unique_tokens(self) -> int:
        return int(self.token_mask.count_nonzero().item())

    def stage(
        self,
        layer_idx: int,
        token_idx: int,
        k_slice: torch.Tensor,
        v_slice: torch.Tensor,
        slot_tensor: torch.Tensor,
    ) -> None:
        if not (0 <= layer_idx < self.n_layers):
            raise ValueError("Invalid layer index")
        if not (0 <= token_idx < self.max_tokens):
            raise ValueError("Invalid token index")

        k = k_slice.to(self.device, self.dtype)
        v = v_slice.to(self.device, self.dtype)
        self.k_buffer[layer_idx, token_idx].copy_(k)
        self.v_buffer[layer_idx, token_idx].copy_(v)

        if layer_idx == 0:
            slot_value = int(slot_tensor.squeeze().item())
            self.slot_buffer[token_idx] = slot_value

        self.token_mask[token_idx] = True
        self.stage_count += 1

    def commit(
        self,
        accepted_len: int,
        kv_cache_ops,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
    ) -> int:
        if accepted_len <= 0:
            return 0

        if accepted_len > self.max_tokens:
            return 0

        if not self.token_mask[:accepted_len].all():
            return 0

        slots = self.slot_buffer[:accepted_len].to(key_cache.device)

        for layer_idx in range(self.n_layers):
            k = self.k_buffer[layer_idx, :accepted_len].to(key_cache.device)
            v = self.v_buffer[layer_idx, :accepted_len].to(value_cache.device)
            kv_cache_ops.reshape_and_cache_flash(
                k,
                v,
                key_cache,
                value_cache,
                slots,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )

        return accepted_len
