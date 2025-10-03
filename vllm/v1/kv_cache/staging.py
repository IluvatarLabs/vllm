"""Device-side staging buffers for NWOR (No-Write-On-Reject)."""

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
