import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch may be unavailable
    torch = None


if torch is None:

    class StagingSnapshotTest(unittest.TestCase):

        @unittest.skip("PyTorch is required for NWOR staging snapshot tests")
        def test_placeholder(self):
            pass

else:

    from vllm.v1.kv_cache.nwor import NWORController, get_cache_view_for_layer

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

    def _make_chunk(slot_offset: int, length: int) -> tuple[torch.Tensor, torch.Tensor,
                                                            torch.Tensor]:
        key = torch.arange(slot_offset,
                           slot_offset + length,
                           dtype=torch.float32).view(length, 1, 1)
        value = key + 10.0
        slot_mapping = torch.arange(slot_offset,
                                    slot_offset + length,
                                    dtype=torch.long)
        return key, value, slot_mapping

    class StagingSnapshotTest(unittest.TestCase):

        def setUp(self) -> None:  # pragma: no cover - simple fixture
            self.key_cache = torch.zeros((1, 16, 1, 1), dtype=torch.float32)
            self.value_cache = torch.zeros((1, 16, 1, 1), dtype=torch.float32)

        def test_ranges_recorded_across_chunks(self):
            with nwor_env(VLLM_NWOR_TRACE_WRITES="0",
                         VLLM_NWOR_PRUNE_COMMIT="1",
                         VLLM_NWOR_DEBUG_ASSERTS="1",
                         VLLM_NWOR_DEFER_WRITE="1"):
                controller = NWORController(enabled=True)
            controller.begin_window([4])

            chunk1 = _make_chunk(0, 2)
            chunk2 = _make_chunk(2, 2)
            req_indices = torch.tensor([0, 0], dtype=torch.int32)

            def fake_cache_op(*_args, **_kwargs):  # pragma: no cover - simple stub
                return None

            with patch(
                    "vllm.v1.kv_cache.nwor.torch.ops._C_cache_ops.reshape_and_cache_flash",
                    side_effect=fake_cache_op,
                    create=True):
                controller.record_layer(
                    layer_name="layer0",
                    key=chunk1[0],
                    value=chunk1[1],
                    slot_mapping=chunk1[2],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    kv_cache_dtype="fp16",
                    k_scale=torch.ones(2, 1, 1),
                    v_scale=torch.ones(2, 1, 1),
                    token_request_indices=req_indices,
                )

                acc = controller._current_accumulator
                self.assertIsNotNone(acc)
                self.assertEqual(acc.staging_ranges, [(0, 2)])
                overlay_key, overlay_value = get_cache_view_for_layer(
                    "layer0", self.key_cache, self.value_cache)
                # When torch is available, overlay clones should exist.
                self.assertIsNotNone(acc.overlay_key_cache)
                self.assertIsNotNone(acc.overlay_value_cache)
                self.assertTrue(torch.equal(overlay_key,
                                            acc.overlay_key_cache))
                self.assertTrue(torch.equal(overlay_value,
                                            acc.overlay_value_cache))

                controller.record_layer(
                    layer_name="layer0",
                    key=chunk2[0],
                    value=chunk2[1],
                    slot_mapping=chunk2[2],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    kv_cache_dtype="fp16",
                    k_scale=torch.ones(2, 1, 1),
                    v_scale=torch.ones(2, 1, 1),
                    token_request_indices=req_indices,
                )

            self.assertIsNone(controller._current_accumulator)
            self.assertEqual(len(controller._pending_layers), 1)
            pending = controller._pending_layers[0]
            self.assertEqual(pending.staging_ranges, [(0, 2), (2, 4)])


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    unittest.main()
