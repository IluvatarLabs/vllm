import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch may be unavailable
    torch = None


if torch is None:

    class CommitPruneTraceTest(unittest.TestCase):

        @unittest.skip("PyTorch is required for NWOR commit prune tests")
        def test_placeholder(self):
            pass

else:

    from vllm.v1.kv_cache.nwor import NWORController, _PendingLayer

    class CacheCallRecorder:
        def __init__(self):
            self.calls = []

        def __call__(self, key, value, key_cache, value_cache, slot_mapping,
                     kv_cache_dtype, k_scale, v_scale):
            self.calls.append(int(key.shape[0]))

    @contextmanager
    def nwor_env(**env_vars):
        original = {k: os.environ.get(k) for k in env_vars}
        for key, value in env_vars.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        try:
            yield
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _make_pending(controller: NWORController,
                      num_tokens: int) -> _PendingLayer:
        controller._total_tokens = num_tokens
        controller._draft_total = num_tokens
        controller._layer_names = ["layer0"]
        controller._staging_buffers.ensure(num_layers=1,
                                           num_tokens=num_tokens,
                                           num_heads=1,
                                           head_dim=1,
                                           dtype=torch.float32,
                                           device=torch.device("cpu"))
        staging_keys = controller._staging_buffers.keys[0]
        staging_values = controller._staging_buffers.values[0]
        base = torch.arange(num_tokens, dtype=torch.float32).view(-1, 1, 1)
        staging_keys[:num_tokens] = base
        staging_values[:num_tokens] = base + 100.0
        slot_mapping = torch.arange(num_tokens, dtype=torch.long)
        return _PendingLayer(
            layer_name="layer0",
            layer_index=0,
            slot_mapping=slot_mapping,
            key_cache=torch.zeros(1, dtype=torch.float32),
            value_cache=torch.zeros(1, dtype=torch.float32),
            kv_cache_dtype="fp16",
            k_scale=None,
            v_scale=None,
            request_indices=torch.zeros(num_tokens, dtype=torch.int32),
            staged_tokens=num_tokens,
        )

    def _accepted_mask(num_tokens: int, accepted: int) -> torch.Tensor:
        mask = torch.zeros(num_tokens, dtype=torch.bool)
        mask[:accepted] = True
        return mask

    class CommitPruneTraceTest(unittest.TestCase):

        def test_commit_write_counts_drop_when_pruned(self):
            with nwor_env(VLLM_NWOR_TRACE_WRITES="1",
                          VLLM_NWOR_DEBUG_ASSERTS="1",
                          VLLM_NWOR_PRUNE_COMMIT="0"):
                controller = NWORController(enabled=True)
            controller._enable_trace = True
            controller._trace_window_begin()
            pending = _make_pending(controller, num_tokens=2)
            accepted_mask = _accepted_mask(2, accepted=1)
            recorder = CacheCallRecorder()
            with patch(
                    "vllm.v1.kv_cache.nwor.torch.ops._C_cache_ops.reshape_and_cache_flash",
                    side_effect=recorder):
                controller._commit_layer(pending, accepted_mask)
            self.assertEqual(recorder.calls, [1])
            self.assertEqual(controller._window_trace["writes_commit"], 1)

        def test_commit_write_counts_zero_with_prune(self):
            with nwor_env(VLLM_NWOR_TRACE_WRITES="1",
                          VLLM_NWOR_DEBUG_ASSERTS="1",
                          VLLM_NWOR_PRUNE_COMMIT="1"):
                controller = NWORController(enabled=True)
            controller._enable_trace = True
            controller._trace_window_begin()
            pending = _make_pending(controller, num_tokens=2)
            pending.cache_layout = None
            accepted_mask = _accepted_mask(2, accepted=1)
            recorder = CacheCallRecorder()
            with patch(
                    "vllm.v1.kv_cache.nwor.torch.ops._C_cache_ops.reshape_and_cache_flash",
                    side_effect=recorder):
                controller._commit_layer(pending, accepted_mask)
            self.assertEqual(recorder.calls, [])
            self.assertEqual(controller._window_trace["writes_commit"], 0)


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    unittest.main()
