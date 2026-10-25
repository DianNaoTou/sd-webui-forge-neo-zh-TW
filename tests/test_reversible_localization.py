"""English-source UI and zh_Hant dictionary contracts; no GPU or weights."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import patch

import gradio as gr
from fastapi import HTTPException

from test_layerdiffuse_integration import LayerMethod, load_helper
from test_vae_flash_regressions import load_nodes

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'extensions/sd-forge-layerdiffuse/scripts/forge_layerdiffusion.py'
ZH = json.loads((ROOT/'localizations/zh_Hant.json').read_text(encoding='utf-8'))


class ReversibleLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse((ROOT/SCRIPT).read_text(encoding='utf-8'))
        self.labels = next(ast.literal_eval(n.value) for n in self.tree.body
                           if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='METHOD_LABELS')
        self.resize = next(ast.literal_eval(n.value) for n in self.tree.body
                           if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='RESIZE_LABELS')

    def test_layerdiffuse_ui_source_is_english_and_dictionary_covers_its_labels(self):
        cls=next(n for n in self.tree.body if isinstance(n,ast.ClassDef) and n.name=='LayerDiffusionForForge')
        ui=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='ui')
        strings=[n.value for n in ast.walk(ui) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        self.assertFalse([s for s in strings if re.search(r'[\u4e00-\u9fff]',s)])
        for value in [*self.labels.values(),*self.resize.values(),'LayerDiffuse','Generation Method','Weight (SDXL)',
                      'Layer Effect Stop Fraction','Image Resize Mode','Additional Foreground Prompt']:
            self.assertIn(value,ZH,value)
            self.assertRegex(ZH[value],r'[\u4e00-\u9fff]')

    def test_real_gradio_choices_preserve_english_values_and_control_order(self):
        ns=load_nodes(SCRIPT,['title','ui'],dict(gr=gr,LayerMethod=LayerMethod,
                      ResizeMode=load_helper('enums').ResizeMode,METHOD_LABELS=self.labels,RESIZE_LABELS=self.resize,
                      ForgeCanvas=lambda **kwargs:types.SimpleNamespace(background=gr.Image())),class_name='LayerDiffusionForForge')
        script=type('UIOnly',(),{'title':ns['title'],'ui':ns['ui']})()
        with gr.Blocks(analytics_enabled=False) as demo:
            controls=script.ui()
        self.assertEqual(len(controls),12)
        self.assertEqual(script.title(),'LayerDiffuse')
        self.assertEqual(controls[1].value,LayerMethod.FG_ONLY_ATTN.value)
        self.assertEqual(controls[1].choices,[(e.value,e.value) for e in LayerMethod])
        self.assertEqual(controls[7].choices,[(e.value,e.value) for e in load_helper('enums').ResizeMode])
        self.assertTrue(any(c['props'].get('label')=='Generation Method' for c in demo.config['components']))

    def test_backend_locale_switch_produces_zh_dictionary_then_empty_english(self):
        spec=importlib.util.spec_from_file_location('i18n_dictionary_test',ROOT/'modules/localization.py')
        module=importlib.util.module_from_spec(spec)
        import modules
        fake_scripts=types.SimpleNamespace(list_scripts=lambda *a:[])
        with patch.multiple(modules,create=True,scripts=fake_scripts,errors=types.SimpleNamespace(report=lambda *a,**k:None)):
            spec.loader.exec_module(module)
            module.list_localizations(str(ROOT/'localizations'))
            chinese=json.loads(module.localization_js('zh_Hant').partition(' = ')[2])
            english=json.loads(module.localization_js('None').partition(' = ')[2])
        self.assertEqual(chinese['LayerDiffuse'],'LayerDiffuse 圖層透明生成')
        self.assertEqual(english,{})


class ScriptAliasTests(unittest.TestCase):
    def function(self):
        return load_nodes('modules/api/api.py',['script_name_to_index'],{'HTTPException':HTTPException})['script_name_to_index']

    def test_canonical_and_legacy_layerdiffuse_api_names_resolve_same_script(self):
        tree=ast.parse((ROOT/SCRIPT).read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LayerDiffusionForForge')
        aliases=next(ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.Assign) and n.targets[0].id=='api_aliases')
        script=types.SimpleNamespace(title=lambda:'LayerDiffuse',api_aliases=aliases)
        for name in ('LayerDiffuse','layerdiffuse',*aliases):
            self.assertEqual(self.function()(name,[script]),0)

    def test_canonical_titles_take_priority_over_aliases_and_missing_is_422(self):
        scripts=[types.SimpleNamespace(title=lambda:'First',api_aliases=('Second',)),types.SimpleNamespace(title=lambda:'Second')]
        self.assertEqual(self.function()('Second',scripts),1)
        with self.assertRaises(HTTPException) as caught:
            self.function()('missing',scripts)
        self.assertEqual(caught.exception.status_code,422)
