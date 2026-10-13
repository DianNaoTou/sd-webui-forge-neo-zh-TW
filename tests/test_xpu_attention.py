"""Numerical regression tests for bounded XPU Math SDPA; run on CPU."""
import unittest
from unittest.mock import patch

try:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from backend.xpu_attention import math_sdpa
except ImportError:  # torch is not installed in CPU-only CI/test venvs
    torch = None


@unittest.skipIf(torch is None, "requires torch")
class XPUMathAttentionTests(unittest.TestCase):
    def compare(self, q, k, v, **kwargs):
        original = torch.nn.functional.scaled_dot_product_attention
        with sdpa_kernel(SDPBackend.MATH):
            expected = original(q, k, v, **kwargs)
        with patch("torch.nn.functional.scaled_dot_product_attention", wraps=original) as call:
            actual = math_sdpa(q, k, v, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        return call

    def inputs(self, queries=513, keys=41):
        torch.manual_seed(17)
        return (torch.randn(2, 4, queries, 8),
                torch.randn(2, 4, keys, 8), torch.randn(2, 4, keys, 8))

    def test_slices_queries_and_retains_all_keys(self):
        q, k, v = self.inputs()
        call = self.compare(q, k, v, scale=0.2)
        self.assertEqual(call.call_count, 3)
        self.assertEqual([c.args[0].shape[-2] for c in call.call_args_list], [256, 256, 1])
        self.assertTrue(all(c.args[1].shape[-2] == 41 for c in call.call_args_list))

    def test_masks(self):
        q, k, v = self.inputs()
        for mask in (torch.rand(513, 41) > 0.2, torch.randn(2, 1, 513, 41),
                     torch.randn(1, 1, 1, 41)):
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                self.compare(q, k, v, attn_mask=mask)

    def test_gqa_and_broadcast_batch(self):
        q, k, v = self.inputs()
        self.compare(q, k[:1, :2], v[:1, :2], enable_gqa=True)

    def test_short_and_causal_calls_preserved(self):
        q, k, v = self.inputs(13, 13)
        self.assertEqual(self.compare(q, k, v).call_count, 1)
        q, k, v = self.inputs(513, 513)
        self.assertEqual(self.compare(q, k, v, is_causal=True).call_count, 1)

    def test_budget_reduces_query_chunk(self):
        q = torch.randn(1, 4, 513, 2)
        k = torch.randn(1, 4, 16385, 2)
        v = torch.randn(1, 4, 16385, 2)
        call = self.compare(q, k, v)
        self.assertTrue(all(c.args[0].shape[-2] <= 127 for c in call.call_args_list))


if __name__ == "__main__":
    unittest.main()
