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


@unittest.skipIf(torch is None, "requires torch")
class XPUAttentionModeTests(unittest.TestCase):
    def test_native_delegates_once_without_math_context_or_slicing(self):
        from backend import xpu_attention as xpu
        q, k, v = (torch.randn(1, 2, 513, 8) for _ in range(3))
        mask = torch.randn(513, 513)
        original = torch.nn.functional.scaled_dot_product_attention
        expected = original(q, k, v, mask, 0.0, False, scale=0.2)
        with patch.object(xpu, 'sdpa_kernel', side_effect=AssertionError('native must not force Math')), patch.object(xpu, 'math_sdpa', side_effect=AssertionError('native must not slice')), patch('torch.nn.functional.scaled_dot_product_attention', wraps=original) as call:
            actual = xpu.xpu_sdpa(q, k, v, mask, 0.0, False, scale=0.2)
        torch.testing.assert_close(actual, expected)
        call.assert_called_once()
        self.assertIs(call.call_args.args[3],mask)
        self.assertEqual(call.call_args.kwargs,{'scale':0.2})

    def test_math_restores_original_unsliced_path(self):
        from backend import xpu_attention as xpu
        from contextlib import nullcontext
        q, k, v = (torch.randn(1, 2, 513, 8) for _ in range(3))
        original = torch.nn.functional.scaled_dot_product_attention
        with patch.object(xpu, 'sdpa_kernel', return_value=nullcontext()) as context, patch.object(xpu, 'math_sdpa', side_effect=AssertionError('math must not slice')), patch('torch.nn.functional.scaled_dot_product_attention', wraps=original) as call:
            actual = xpu.xpu_sdpa(q,k,v,mode='math')
        context.assert_called_once_with(SDPBackend.MATH)
        call.assert_called_once()
        self.assertEqual(call.call_args.args[0].shape[-2],513)
        torch.testing.assert_close(actual,original(q,k,v))

    def test_sliced_keeps_test5_implementation(self):
        from backend import xpu_attention as xpu
        q = torch.randn(1,2,513,8)
        with patch.object(xpu,'math_sdpa',return_value=q) as old:
            self.assertIs(xpu.xpu_sdpa(q,q,q,mode='sliced',enable_gqa=True),q)
        old.assert_called_once_with(q,q,q,enable_gqa=True)

    def test_native_oom_or_nonfinite_output_is_not_silently_retried(self):
        from backend import xpu_attention as xpu
        q = torch.randn(1,2,8,8)
        with patch.object(xpu,'math_sdpa') as fallback:
            for error in (torch.OutOfMemoryError('test OOM'),RuntimeError('test kernel failure')):
                with patch('torch.nn.functional.scaled_dot_product_attention',side_effect=error):
                    with self.assertRaises(type(error)) as caught:
                        xpu.xpu_sdpa(q,q,q)
                    self.assertIs(caught.exception,error)
            with patch('torch.nn.functional.scaled_dot_product_attention',return_value=torch.full_like(q,float('nan'))):
                self.assertTrue(torch.isnan(xpu.xpu_sdpa(q,q,q)).all())
            fallback.assert_not_called()

    def test_invalid_mode_is_rejected(self):
        from backend import xpu_attention as xpu
        q = torch.randn(1,2,8,8)
        with self.assertRaisesRegex(ValueError,'Unknown XPU SDPA mode'):
            xpu.xpu_sdpa(q,q,q,mode='typo')

    def test_modes_are_logged_once_per_process(self):
        from backend import xpu_attention as xpu
        q = torch.randn(1,2,8,8)
        with patch.object(xpu,'_reported_modes',set()),patch.object(xpu.logger,'info') as log:
            xpu.xpu_sdpa(q,q,q)
            xpu.xpu_sdpa(q,q,q)
            self.assertEqual(log.call_count,1)
            self.assertEqual(log.call_args.args[1],'native')

    def test_cli_default_and_explicit_choices(self):
        import subprocess,sys
        for mode in (None,'native','math','sliced'):
            command=[sys.executable,'-c','from backend.args import args; print(args.xpu_sdpa)']
            if mode:
                command += ['--xpu-sdpa',mode]
            self.assertEqual(subprocess.check_output(command,text=True).strip(),mode or 'native')
        result=subprocess.run([sys.executable,'-c','import backend.args','--xpu-sdpa','typo'],capture_output=True)
        self.assertNotEqual(result.returncode,0)

    def test_dispatch_is_xpu_only_and_preserves_masked_gqa(self):
        import types
        from test_vae_flash_regressions import load_nodes
        select = unittest.mock.Mock(return_value='xpu-output')
        functional = unittest.mock.Mock(return_value='cpu-or-cuda-output')
        repeated_k,repeated_v = object(),object()
        repeat = unittest.mock.Mock(return_value=(repeated_k,repeated_v))
        ns = dict(torch=types.SimpleNamespace(nn=types.SimpleNamespace(functional=types.SimpleNamespace(scaled_dot_product_attention=functional))),
                  backend_args=types.SimpleNamespace(xpu_sdpa='native'),repeat_kv_for_gqa=repeat)
        load_nodes('backend/operations.py',['scaled_dot_product_attention'],ns)
        q=types.SimpleNamespace(device=types.SimpleNamespace(type='xpu'),shape=(1,4,8,8))
        k,v,mask=object(),object(),object()
        with patch.dict('sys.modules',{'backend.xpu_attention':types.SimpleNamespace(xpu_sdpa=select)}):
            self.assertEqual(ns['scaled_dot_product_attention'](q,k,v,attn_mask=mask,enable_gqa=True),'xpu-output')
        select.assert_called_once_with(q,repeated_k,repeated_v,mode='native',attn_mask=mask,enable_gqa=False)
        repeat.assert_called_once_with(k,v,4,-3)
        for device in ('cpu','cuda'):
            q.device.type=device
            self.assertEqual(ns['scaled_dot_product_attention'](q,k,v),'cpu-or-cuda-output')
        self.assertEqual(functional.call_count,2)


if __name__ == "__main__":
    unittest.main()
