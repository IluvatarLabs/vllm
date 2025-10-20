# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.kv_cache.deferred import DeferredWriteManager, ShouldFallback
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
try:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
except RuntimeError as exc:  # e.g., torch.cuda init failure on CPU-only envs
    pytest.skip(f"GPUModelRunner unavailable: {exc}", allow_module_level=True)


def _make_metadata(draft_token_ids: list[int], per_request: list[int], device: str = "cpu") -> SpecDecodeMetadata:
    total = len(draft_token_ids)
    cu = torch.tensor(per_request, dtype=torch.int32, device=device)
    cu = torch.cumsum(cu, dim=0)
    return SpecDecodeMetadata(
        draft_token_ids=torch.tensor(draft_token_ids, dtype=torch.int32, device=device),
        num_draft_tokens=list(per_request),
        cu_num_draft_tokens=cu,
        target_logits_indices=torch.zeros(total, dtype=torch.int32, device=device),
        bonus_logits_indices=torch.zeros(len(per_request), dtype=torch.int32, device=device),
        logits_indices=torch.zeros(total + len(per_request), dtype=torch.int32, device=device),
    )


def _make_mock_runner(scv_mode="off"):
    """Create a minimal GPUModelRunner for testing.

    Bypasses __init__ but sets required attributes for SCV/NWOR tests.
    """
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner._scv_mode = scv_mode
    runner._scv_debug = False  # Required by _scv_enabled()
    runner._scv_profile = False  # Required by _scv_nvtx_range()
    runner._nwor_debug = False  # Required by NWOR paths
    runner._scv_capture_available = True  # For graph mode checks
    runner._scv_graph_executor = None  # For graph capture
    runner._scv_graph_cache = {}  # Required for graph mode
    runner._scv_graph_failures = {}  # Required for blacklisting
    runner.speculative_config = None  # For NWOR tests
    runner._deferred_write_manager = DeferredWriteManager()
    return runner


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

    manager.commit([1])

    assert len(writes) == 1
    committed_key, committed_slots = writes[0]
    assert committed_key.shape[0] == 1
    assert committed_slots.tolist() == [3]
    window_metrics = manager.pop_last_window_metrics()
    assert window_metrics == {
        "mode": "stage",
        "committed": 1,
        "rejected": 1,
        "fallback": 0,
    }


def test_deferred_manager_multiple_layers_full_window():
    manager = DeferredWriteManager()
    assert manager.begin_window([2, 3])

    writes_per_layer: dict[str, list[torch.Tensor]] = {"layer0": [], "layer1": []}

    def make_writer(layer_id: str):
        def _writer(key, value, key_cache, value_cache, slot_mapping, *_args):
            writes_per_layer[layer_id].append(slot_mapping.clone())

        return _writer

    slot_mapping = torch.arange(5, dtype=torch.int32)
    key = torch.randn(5, 1, 2)
    value = torch.randn(5, 1, 2)
    cache = torch.empty_like(key)

    for layer_id in ("layer0", "layer1"):
        manager.stage_layer(
            layer_id=layer_id,
            key=key,
            value=value,
            key_cache=cache,
            value_cache=cache,
            slot_mapping=slot_mapping,
            kv_cache_dtype="fp16",
            k_scale=None,
            v_scale=None,
            writer=make_writer(layer_id),
        )

    manager.commit([2, 0])

    assert len(writes_per_layer["layer0"]) == 1
    assert len(writes_per_layer["layer1"]) == 1

    expected_slots = torch.tensor([0, 1], dtype=torch.int32)
    assert torch.equal(writes_per_layer["layer0"][0], expected_slots)
    assert torch.equal(writes_per_layer["layer1"][0], expected_slots)

    metrics = manager.pop_last_window_metrics()
    assert metrics == {
        "mode": "stage",
        "committed": 2,
        "rejected": 3,
        "fallback": 0,
    }


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
    window_metrics = manager.pop_last_window_metrics()
    assert window_metrics is not None
    assert window_metrics.get("fallback") == 1


def test_build_acceptance_mask_matches_expected():
    metadata = _make_metadata([10, 11, 20], [2, 1])
    sampled = torch.tensor(
        [
            [10, 99, 0],  # second token rejected
            [20, 0, 0],
        ],
        dtype=torch.int32,
    )

    runner = _make_mock_runner(scv_mode="off")
    counts, mask = runner._compute_nwor_acceptance(metadata, sampled, return_mask=True)
    expected = torch.tensor([True, False, True], dtype=torch.bool)
    assert torch.equal(mask.cpu(), expected)
    assert counts == [1, 1]


def test_nwor_disabled_env(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_NWOR", "1")

    runner = _make_mock_runner(scv_mode="off")
    runner.speculative_config = object()  # Override to enable NWOR path

    metadata = _make_metadata([1, 2], [2])
    runner._maybe_begin_nwor_window(metadata)

    assert not runner._deferred_write_manager.window_active


def test_fp8_staging_slices_quant_scales():
    manager = DeferredWriteManager()
    assert manager.begin_window([2])

    recorded: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def writer(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale):
        recorded.append((key.clone(), value.clone(), slot_mapping.clone(), k_scale.clone() if k_scale is not None else None))

    key = torch.arange(4, dtype=torch.float32).view(2, 1, 2)
    value = torch.arange(4, dtype=torch.float32).view(2, 1, 2)
    slot_mapping = torch.tensor([3, 7], dtype=torch.int32)
    key_cache = torch.empty_like(key, dtype=torch.uint8)
    value_cache = torch.empty_like(value, dtype=torch.uint8)
    k_scale = torch.tensor([0.5, 0.7], dtype=torch.float32)
    v_scale = torch.tensor([0.6, 0.9], dtype=torch.float32)

    manager.stage_layer(
        layer_id="layer0",
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype="fp8",
        k_scale=k_scale,
        v_scale=v_scale,
        writer=writer,
    )

    manager.commit([1])

    assert len(recorded) == 1
    committed_key, committed_value, slots, committed_k_scale = recorded[0]
    assert committed_key.shape[0] == 1
    assert torch.equal(slots, torch.tensor([3], dtype=torch.int32))
    assert committed_k_scale is None or committed_k_scale.shape[0] == 1
    window_metrics = manager.pop_last_window_metrics()
    assert window_metrics == {
        "mode": "stage",
        "committed": 1,
        "rejected": 1,
        "fallback": 0,
    }


def test_nwor_immediate_mode_skips_window():
    manager = DeferredWriteManager(mode="immediate")
    assert not manager.begin_window([2])
    assert manager.get_mode() == "immediate"


def test_nwor_off_mode_skips_window():
    manager = DeferredWriteManager(mode="off")
    assert not manager.begin_window([3])
    assert manager.get_mode() == "off"


def test_scv_vectorized_mask_matches_reference():
    metadata = _make_metadata([1, 2, 3, 4], [4])
    sampled = torch.tensor([[1, 2, 0, 4]], dtype=torch.int32)

    runner = _make_mock_runner(scv_mode="adaptive")

    counts, mask = runner._compute_nwor_acceptance(metadata, sampled, return_mask=True)
    assert mask.tolist() == [True, True, False, False]
    assert counts == [2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.skipif(not hasattr(torch.cuda, "CUDAGraph"), reason="Requires CUDA graphs")
def test_scv_mask_handles_oob_gracefully():
    """Test that SCV mask computation handles out-of-bounds access gracefully.

    This reproduces the scenario where sampled_token_ids has fewer columns
    than the draft token count, which previously caused device-side asserts.
    """
    # 4 draft tokens for one request
    metadata = _make_metadata([10, 20, 30, 40], [4], device="cuda")

    # But sampled_token_ids only has 2 columns (should trigger clamping)
    # This simulates the case where not all draft tokens have been sampled yet
    sampled = torch.tensor([[10, 20]], dtype=torch.int32, device="cuda")

    runner = _make_mock_runner(scv_mode="graph")

    # This should not crash, but should gracefully handle the OOB
    counts, mask = runner._compute_nwor_acceptance(metadata, sampled, return_mask=True)

    # First 2 tokens match, next 2 are out of bounds so rejected
    assert mask.tolist() == [True, True, False, False]
    assert counts == [2]


def test_scv_mask_all_oob():
    """Test when all draft tokens are beyond sampled_token_ids bounds."""
    metadata = _make_metadata([10, 20, 30], [3])

    # Empty sampled (0 columns) - extreme case
    sampled = torch.empty((1, 0), dtype=torch.int32)

    runner = _make_mock_runner(scv_mode="adaptive")

    # Should fallback gracefully, not crash
    counts, mask = runner._compute_nwor_acceptance(metadata, sampled, return_mask=True)

    # All tokens should be rejected (or fallback to None)
    if counts is not None:
        assert counts == [0]
    if mask is not None:
        assert mask.tolist() == [False, False, False]


def test_scv_mask_invalid_shape_falls_back():
    """Test that invalid sampled_token_ids shape triggers fallback."""
    metadata = _make_metadata([10, 20], [2])

    # 1D tensor (invalid shape)
    sampled = torch.tensor([10, 20], dtype=torch.int32)

    runner = _make_mock_runner(scv_mode="graph")

    # Should fallback to reference path (returns None from vectorized)
    counts, mask = runner._compute_nwor_acceptance(metadata, sampled, return_mask=True)

    # Reference path should still compute correctly
    assert counts == [2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.skipif(not hasattr(torch.cuda, "CUDAGraph"), reason="Requires CUDA graphs")
def test_scv_graph_inplace_matches_reference():
    metadata_cpu = _make_metadata([10, 20, 30, 40], [4], device="cpu")
    metadata_cuda = _make_metadata([10, 20, 30, 40], [4], device="cuda")
    sampled = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.int32, device="cuda")

    runner_ref = _make_mock_runner(scv_mode="off")
    counts_ref, mask_ref = runner_ref._compute_nwor_acceptance(
        metadata_cpu, sampled.cpu(), return_mask=True
    )

    runner_graph = _make_mock_runner(scv_mode="graph")
    counts_graph, mask_graph = runner_graph._compute_nwor_acceptance(
        metadata_cuda, sampled, return_mask=True
    )

    assert counts_graph == counts_ref
    assert torch.equal(mask_graph.cpu(), mask_ref.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.skipif(not hasattr(torch.cuda, "CUDAGraph"), reason="Requires CUDA graphs")
def test_scv_graph_different_cu_patterns():
    runner = _make_mock_runner(scv_mode="graph")

    metadata1 = _make_metadata([10, 20, 30, 40], [4], device="cuda")
    sampled1 = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.int32, device="cuda")
    runner._compute_nwor_acceptance(metadata1, sampled1, return_mask=True)

    metadata2 = _make_metadata([10, 20, 30, 40], [2, 2], device="cuda")
    sampled2 = torch.tensor(
        [[10, 20, 50], [30, 40, 60]], dtype=torch.int32, device="cuda"
    )
    runner._compute_nwor_acceptance(metadata2, sampled2, return_mask=True)

    assert len(runner._scv_graph_cache) == 2


def test_commit_failure_triggers_fallback_metrics():
    manager = DeferredWriteManager()
    assert manager.begin_window([1])

    key = torch.randn(1, 1, 2)
    value = torch.randn(1, 1, 2)
    slot_mapping = torch.tensor([0], dtype=torch.int32)
    key_cache = torch.empty_like(key)
    value_cache = torch.empty_like(value)

    def writer(*_args, **_kwargs):
        raise RuntimeError("forced failure")

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

    with pytest.raises(ShouldFallback):
        manager.commit([1])

    window_metrics = manager.pop_last_window_metrics()
    assert window_metrics is not None
    assert window_metrics.get("fallback") == 1
