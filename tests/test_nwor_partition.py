import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


if torch is None:

    class TestNWORPartition(unittest.TestCase):

        @unittest.skip("PyTorch is required for NWOR partition tests")
        def test_placeholder(self):
            pass

else:

    from vllm.v1.kv_cache.nwor import NWORController

    class TestNWORPartition(unittest.TestCase):

        def setUp(self) -> None:
            self.controller = NWORController(enabled=True)

        def test_partition_basic(self) -> None:
            canonical = torch.tensor([0, 0, 1], dtype=torch.int32, device="cpu")
            chunk_requests = torch.tensor([1, 0, 0], dtype=torch.int32, device="cpu")
            stage, tail = self.controller._partition_chunk_positions(
                canonical_slice=canonical,
                chunk_requests=chunk_requests,
                chunk_len=3,
                num_requests=2,
            )
            self.assertTrue(torch.equal(stage.cpu(), torch.tensor([1, 2, 0])))
            self.assertEqual(tail.numel(), 0)

        def test_partition_with_invalid_requests(self) -> None:
            canonical = torch.tensor([0, 0], dtype=torch.int32)
            chunk_requests = torch.tensor([-1, 0, 0], dtype=torch.int32)
            stage, tail = self.controller._partition_chunk_positions(
                canonical_slice=canonical,
                chunk_requests=chunk_requests,
                chunk_len=3,
                num_requests=2,
            )
            self.assertTrue(torch.equal(stage.cpu(), torch.tensor([1, 2])))
            self.assertTrue(torch.equal(tail.cpu(), torch.tensor([0])))

        def test_partition_insufficient_tokens(self) -> None:
            canonical = torch.tensor([0, 0], dtype=torch.int32)
            chunk_requests = torch.tensor([0], dtype=torch.int32)
            result = self.controller._partition_chunk_positions(
                canonical_slice=canonical,
                chunk_requests=chunk_requests,
                chunk_len=1,
                num_requests=1,
            )
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
