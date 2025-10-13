import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch may be unavailable
    torch = None

if torch is not None:  # pragma: no cover - depends on torch
    from vllm.v1.kv_cache.nwor import NWORController


@contextmanager
def nwor_env(**env_vars):
    original = {k: os.environ.get(k) for k in env_vars}
    try:
        for key, value in env_vars.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if torch is None:  # pragma: no cover - skip when torch missing

    class DeferWriteTest(unittest.TestCase):

        @unittest.skip("PyTorch is required for defer-write tests")
        def test_placeholder(self):
            pass

else:

    class DeferWriteTest(unittest.TestCase):

        def setUp(self) -> None:  # pragma: no cover - simple CPU fixtures
            self.key_cache = torch.zeros((1, 2, 1, 1))
            self.value_cache = torch.zeros((1, 2, 1, 1))

        def _stage_sample(self, controller: NWORController) -> None:
            key = torch.randn((2, 1, 1))
            value = torch.randn((2, 1, 1))
            slots = torch.tensor([0, 1], dtype=torch.long)
            controller.record_layer(
                layer_name="layer0",
                key=key,
                value=value,
                slot_mapping=slots,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                kv_cache_dtype="fp16",
                k_scale=torch.ones((2, 1, 1)),
                v_scale=torch.ones((2, 1, 1)),
                token_request_indices=None,
            )

        def test_defer_write_skips_staging_write(self):
            with nwor_env(VLLM_NWOR_PRUNE_COMMIT="1",
                          VLLM_NWOR_DEBUG_ASSERTS="1",
                          VLLM_NWOR_DEFER_WRITE="1"):
                controller = NWORController(enabled=True)

            controller.begin_window([2])

            with patch("vllm.v1.kv_cache.nwor.torch.ops._C_cache_ops.reshape_and_cache_flash") as mock_op:
                self._stage_sample(controller)
                self.assertEqual(mock_op.call_count, 0)
                controller.commit_window([1])
                self.assertEqual(mock_op.call_count, 1)

        def test_defer_write_fallback_flushes_once(self):
            with nwor_env(VLLM_NWOR_PRUNE_COMMIT="1",
                          VLLM_NWOR_DEBUG_ASSERTS="1",
                          VLLM_NWOR_DEFER_WRITE="1"):
                controller = NWORController(enabled=True)

            controller.begin_window([2])

            with patch("vllm.v1.kv_cache.nwor.torch.ops._C_cache_ops.reshape_and_cache_flash") as mock_op:
                self._stage_sample(controller)
                self.assertEqual(mock_op.call_count, 0)
                controller.commit_all_pending()
                self.assertEqual(mock_op.call_count, 1)


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    unittest.main()
