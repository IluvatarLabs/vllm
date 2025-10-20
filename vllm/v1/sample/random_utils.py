# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utilities for CUDA-graph-safe random number generation."""

from __future__ import annotations

import secrets
from typing import Dict

import torch

_GRAPH_GENERATORS: Dict[torch.device, torch.Generator] = {}


def _get_graph_generator(device: torch.device) -> torch.Generator:
    generator = _GRAPH_GENERATORS.get(device)
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(secrets.randbits(64))
        _GRAPH_GENERATORS[device] = generator
    return generator


def graph_uniform(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = _get_graph_generator(device)
    return torch.rand(shape, device=device, dtype=dtype, generator=generator)


def graph_exponential(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # Sample from U(0,1) and map via -log(U) to obtain Exp(1).
    uniform = graph_uniform(shape, device=device, dtype=dtype)
    eps = torch.finfo(uniform.dtype).tiny
    uniform.clamp_(min=eps)
    return uniform.neg_().log_()
