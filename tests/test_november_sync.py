"""Exercise upstream precision and model-card edge cases without GPU startup."""
import ast
import json
from pathlib import Path
import types
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]

class NovemberSyncTests(unittest.TestCase):
    def test_attention_precision_returns_override_or_requested_dtype(self):
        tree = ast.parse((ROOT/'backend/attention.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_attn_precision')
        ns = dict(torch=types.SimpleNamespace(dtype=object),
                  memory_management=types.SimpleNamespace(force_upcast_attention_dtype=lambda: {'half':'full'}))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'attention-test', 'exec'), ns)
        self.assertEqual(ns['get_attn_precision']('requested', 'half'), 'full')
        self.assertEqual(ns['get_attn_precision']('requested', 'other'), 'requested')

    def test_checkpoint_cards_accept_empty_metadata_and_map_long_names(self):
        tree = ast.parse((ROOT/'modules/ui_settings.py').read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_change_model')
        opts = types.SimpleNamespace(sd_modules_dropdown_use_short=True)
        ns = dict(json=json, gr=types.SimpleNamespace(update=lambda **kw:kw, skip=lambda:None, Warning=Mock()),
                  shared=types.SimpleNamespace(opts=opts),
                  main_entry=types.SimpleNamespace(module_map={'vae':'folder/vae'}),
                  sd_models=types.SimpleNamespace(get_closet_checkpoint_match=lambda x:types.SimpleNamespace(name=x)))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'card-test', 'exec'), ns)
        change = ns['_change_model']
        self.assertEqual(change('sdxl',''), [{'value':'sdxl'},None])
        self.assertEqual(change('sdxl','["vae"]')[1], {'value':['vae']})
        opts.sd_modules_dropdown_use_short=False
        self.assertEqual(change('sdxl','["vae"]')[1], {'value':['folder/vae']})
        self.assertEqual(change('sdxl','{}'), [None,None])
