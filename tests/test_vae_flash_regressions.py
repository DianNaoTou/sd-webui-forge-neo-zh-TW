"""CPU numerical checks of production code without initializing GPU/model loaders.

AST extraction isolates the real functions from Forge's import-time device setup.
The Flash kernel is replaced; these tests do not validate a CUDA Flash kernel.
"""
import ast
import itertools
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def load_nodes(path, names, namespace, class_name=None, optimize=0):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    source = tree.body
    if class_name:
        source = next(n for n in source if isinstance(n, ast.ClassDef) and n.name == class_name).body
    nodes = [n for n in source if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != len(names):
        raise ValueError("missing production function")
    for node in nodes:
        node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec", optimize=optimize), namespace)
    return namespace


@unittest.skipIf(torch is None, "requires torch")
class TiledVAEChannelsTests(unittest.TestCase):
    def test_three_and_four_channel_decoders_all_tile_orientations(self):
        ns = load_nodes("backend/patcher/vae.py", ["tiled_scale", "tiled_scale_multidim"],
                        {"torch": torch, "itertools": itertools})
        original = ns["tiled_scale"]
        ns["tiled_scale"] = Mock(wraps=original)
        load_nodes("backend/patcher/vae.py", ["decode_tiled_"], ns, class_name="VAE")
        # Both a one-tile input and overlapping tiles, non-square and batched.
        for channels in (3, 4):
            for height, width in ((2, 3), (13, 17)):
                with self.subTest(channels=channels, shape=(height, width)):
                    samples = torch.arange(2 * height * width, dtype=torch.float32).reshape(2, 1, height, width)
                    def decode(a):
                        return torch.cat([a + c for c in range(channels)], dim=1).repeat_interleave(2, -2).repeat_interleave(2, -1)
                    vae = SimpleNamespace(first_stage_model=SimpleNamespace(decode=decode),
                                          vae_dtype=torch.float32, device="cpu", output_device="cpu",
                                          upscale_ratio=2, output_channels=channels,
                                          process_output=lambda x: x)
                    ns["tiled_scale"].reset_mock()
                    actual = ns["decode_tiled_"](vae, samples, tile_x=8, tile_y=8, overlap=2)
                    torch.testing.assert_close(actual, decode(samples))
                    calls = ns["tiled_scale"].call_args_list
                    self.assertEqual(len(calls), 3)
                    self.assertEqual([(c.args[2], c.args[3]) for c in calls], [(4, 16), (16, 4), (8, 8)])
                    self.assertTrue(all(c.kwargs.get("out_channels", 3) == channels for c in calls))


@unittest.skipIf(torch is None, "requires torch")
class FlashMaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        self.q = torch.randn(1, 3, 8)
        self.k = torch.randn(1, 7, 8)
        self.v = torch.randn(1, 7, 8)

    def function(self, optimize=0, fail=False):
        sdpa = Mock(wraps=torch.nn.functional.scaled_dot_product_attention)
        def fake_flash(q, k, v, **kwargs):
            if fail:
                raise RuntimeError("simulated CUDA kernel failure")
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=kwargs["dropout_p"], is_causal=kwargs["causal"]).transpose(1, 2)
        flash = Mock(side_effect=fake_flash)
        ns = load_nodes("backend/attention.py", ["attention_flash"],
                        {"torch": torch, "operations": SimpleNamespace(scaled_dot_product_attention=sdpa),
                         "flash_attn_wrapper": flash, "logger": logging.getLogger("regression")},
                        optimize=optimize)
        return ns["attention_flash"], flash, sdpa

    def expected(self, mask=None):
        q, k, v = [t.view(1, -1, 2, 4).transpose(1, 2) for t in (self.q, self.k, self.v)]
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(1, 3, 8)

    def test_explicit_masks_preserved_even_with_causal_hint_and_optimization(self):
        causal = torch.ones(3, 7, dtype=torch.bool).tril(4)
        padding = causal.clone()
        padding[:, 1] = False
        additive = torch.zeros(1, 2, 3, 7)
        additive[..., -1] = -4.0
        masks = (causal, padding, additive, padding[None], padding[None, None])
        for optimize in (0, 1):
            for mask in masks:
                with self.subTest(optimize=optimize, shape=mask.shape, dtype=mask.dtype):
                    fn, flash, sdpa = self.function(optimize)
                    actual = fn(self.q, self.k, self.v, 2, mask=mask, causal=True)
                    torch.testing.assert_close(actual, self.expected(mask))
                    flash.assert_not_called()
                    self.assertEqual(sdpa.call_count, 1)
                    self.assertFalse(sdpa.call_args.kwargs["is_causal"])
                    torch.testing.assert_close(sdpa.call_args.kwargs["attn_mask"].squeeze(), mask.squeeze())

    def test_unmasked_flash_and_kernel_failure_fallback(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                fn, flash, sdpa = self.function(fail=fail)
                torch.testing.assert_close(fn(self.q, self.k, self.v, 2), self.expected())
                self.assertEqual(flash.call_count, 1)
                self.assertEqual(sdpa.call_count, int(fail))

    def test_head_shaped_input_and_output(self):
        fn, flash, _ = self.function()
        q, k, v = [t.view(1, -1, 2, 4).transpose(1, 2) for t in (self.q, self.k, self.v)]
        mask = torch.ones(3, 7, dtype=torch.bool).tril(4)
        out = fn(q, k, v, 2, mask=mask, skip_reshape=True, skip_output_reshape=True)
        torch.testing.assert_close(out, torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask))
        flash.assert_not_called()

    def test_qwen_production_segments_and_rectangular_alignment(self):
        ns = load_nodes("backend/nn/qwen21.py", ["build_sequence"], {"torch": torch},
                        class_name="QwenImage21Transformer2DModel")
        model = SimpleNamespace(txt_in=lambda t: t, img_in=lambda t: t,
                                pe_embedder=lambda t: t.unsqueeze(-1))
        x = torch.zeros(1, 3, 2, 2)
        context = torch.zeros(1, 5, 3)
        _, _, segments = ns["build_sequence"](model, x, context, [torch.zeros(1, 3, 1, 2)], [2])
        self.assertEqual([(s, e) for s, e, _ in segments], [(0, 2), (2, 4), (4, 7), (7, 11)])
        for start, end, mask in segments:
            if mask is None:
                continue
            queries, keys = end - start, end
            bottom_right = torch.arange(keys)[None, :] <= torch.arange(queries)[:, None] + keys - queries
            self.assertTrue(torch.equal(mask, bottom_right))
            if start:
                top_left = torch.arange(keys)[None, :] <= torch.arange(queries)[:, None]
                self.assertFalse(torch.equal(mask, top_left))
                # Different outputs prove that using pre-2.1 Flash alignment is not equivalent.
                q = torch.zeros(1, 1, queries, 1)
                k = torch.zeros(1, 1, keys, 1)
                v = torch.arange(keys, dtype=torch.float32).view(1, 1, keys, 1)
                correct = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                wrong = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                self.assertFalse(torch.allclose(correct, wrong))
        # Exercise the production segment dispatcher and ensure exact masks survive.
        calls = []
        def attention(q, k, v, heads, **kwargs):
            calls.append((q.shape[1], k.shape[1], kwargs["mask"]))
            return q
        ns["attention_function"] = attention
        load_nodes("backend/nn/qwen21.py", ["block_causal_attention"], ns)
        q = torch.randn(1, 11, 1, 3)
        torch.testing.assert_close(ns["block_causal_attention"](segments)(q, q, q, 1), q.flatten(2))
        for (queries, keys, mask), (start, end, expected) in zip(calls, segments):
            self.assertEqual((queries, keys), (end - start, end))
            self.assertIs(mask, expected)


if __name__ == "__main__":
    unittest.main()
