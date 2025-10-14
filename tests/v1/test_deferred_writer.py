# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.kv_cache.deferred import DeferredWriteManager
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _make_metadata(draft_token_ids: list[int], per_request: list[int]) -> SpecDecodeMetadata:
    total = len(draft_token_ids)
    cu = torch.tensor(per_request, dtype=torch.int32)
    cu = torch.cumsum(cu, dim=0)
    return SpecDecodeMetadata(
        draft_token_ids=torch.tensor(draft_token_ids, dtype=torch.int32),
        num_draft_tokens=list(per_request),
        cu_num_draft_tokens=cu,
        target_logits_indices=torch.zeros(total, dtype=torch.int32),
        bonus_logits_indices=torch.zeros(len(per_request), dtype=torch.int32),
        logits_indices=torch.zeros(total + len(per_request), dtype=torch.int32),
    )


def test_deferred_manager_commit_partial_acceptance():
    manager = DeferredWriteManager()
    assert manager.begin_window([2])

    writes: list[tuple[torch.Tensor, torch.Tensor]] = []

    def writer(key, value, key_cache, value_cache, slot_mapping, *_):
        writes.append((key.clone(), slot_mapping.clone()))

    key = torch.arange(4, dtype=torch.float32).view(2, 1, 2)
    value = torch.arange(4, dtype=torch.float32).view(2, 1, 2)
    slot_mapping = torch.tensor([3, 7], dtype=torch.int32)
    key_cache = torch.empty_like(key)
    value_cache = torch.empty_like(value)

    manager.stage_layer(
        layer_id="layer0",
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype="fp16",
        k_scale=None,
        v_scale=None,
        writer=writer,
    )

    mask = torch.tensor([True, False])
    manager.commit(mask)

    assert len(writes) == 1
    committed_key, committed_slots = writes[0]
    assert committed_key.shape[0] == 1
    assert committed_slots.tolist() == [3]
    metrics = manager.get_metrics()
    assert metrics["tokens_committed"] == 1
    assert metrics["tokens_rejected"] == 1


def test_deferred_manager_cancel_flush_writes_all():
    manager = DeferredWriteManager()
    assert manager.begin_window([1, 1])

    writes: list[tuple[str, torch.Tensor]] = []

    def writer(key, value, *_args):  # pragma: no cover - signature compatibility
        writes.append(("commit", key.clone()))

    key = torch.randn(1, 1, 2)
    value = torch.randn(1, 1, 2)
    slot_mapping = torch.tensor([5], dtype=torch.int32)
    key_cache = torch.empty_like(key)
    value_cache = torch.empty_like(value)

    manager.stage_layer(
        layer_id="layer0",
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype="fp16",
        k_scale=None,
        v_scale=None,
        writer=writer,
    )
    manager.stage_layer(
        layer_id="layer1",
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype="fp16",
        k_scale=None,
        v_scale=None,
        writer=writer,
    )

    manager.cancel_and_flush("test_cancel")
    assert len(writes) == 2
    assert all(tensor.shape[0] == 1 for _tag, tensor in writes)


def test_build_acceptance_mask_matches_expected():
    metadata = _make_metadata([10, 11, 20], [2, 1])
    sampled = torch.tensor(
        [
            [10, 99, 0],  # second token rejected
            [20, 0, 0],
        ],
        dtype=torch.int32,
    )

    runner = GPUModelRunner.__new__(GPUModelRunner)
    mask = runner._build_nwor_acceptance_mask(metadata, sampled)
    expected = torch.tensor([True, False, True], dtype=torch.bool)
    assert torch.equal(mask.cpu(), expected)


def test_nwor_disabled_env(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_NWOR", "1")

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.speculative_config = object()
    runner._deferred_write_manager = DeferredWriteManager()

    metadata = _make_metadata([1, 2], [2])
    runner._maybe_begin_nwor_window(metadata)

    assert not runner._deferred_write_manager.window_active
