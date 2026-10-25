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


class BundledPluginLocalizationTests(unittest.TestCase):
    def locale_module(self):
        spec=importlib.util.spec_from_file_location('plugin_locale_test',ROOT/'modules/localization.py')
        module=importlib.util.module_from_spec(spec)
        import modules
        opts=types.SimpleNamespace(localization='zh_Hant')
        self.opts=opts
        context=patch.multiple(modules,create=True,
                               scripts=types.SimpleNamespace(list_scripts=lambda *a:[]),
                               shared=types.SimpleNamespace(opts=opts),
                               errors=types.SimpleNamespace(report=lambda *a,**kw:None))
        context.start();self.addCleanup(context.stop)
        spec.loader.exec_module(module)
        module.list_localizations(str(ROOT/'localizations'))
        return module

    def test_callback_messages_switch_languages_without_translating_user_data(self):
        locale=self.locale_module()
        value='my-model-{0}-測試.safetensors'
        self.assertEqual(locale.translate('Model already exists: {0}',value),'模型已存在：'+value)
        self.opts.localization='None'
        self.assertEqual(locale.translate('Model already exists: {0}',value),'Model already exists: '+value)
        self.opts.localization='zh_Hant'
        self.assertEqual(locale.translate('Model already exists: {0}',value),'模型已存在：'+value)
        self.assertEqual(locale.translate('unregistered {0}',value),'unregistered '+value)

    def test_bad_translation_format_falls_back_without_breaking_callback(self):
        locale=self.locale_module()
        with patch.object(locale,'_read_translation_file',return_value={'Error: {0}':'錯誤：{missing}'}):
            self.assertEqual(locale.translate('Error: {0}','offline'),'Error: offline')

    def test_callback_format_fields_match_dictionary_templates(self):
        import string
        formatter=string.Formatter()
        for folder in ['extensions/forge-neo-image2prompt','extensions/sd-webui-civitai-downloader']:
            for path in (ROOT/folder).rglob('*.py'):
                tree=ast.parse(path.read_text(encoding='utf-8-sig'))
                for node in ast.walk(tree):
                    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='ui_text':
                        key=ast.literal_eval(node.args[0])
                        self.assertIn(key,ZH,(path,key))
                        fields=lambda s:sorted((name,spec,conv) for _,name,spec,conv in formatter.parse(s) if name is not None)
                        self.assertEqual(fields(key),fields(ZH[key]),key)

    def test_real_image2prompt_ui_has_english_source_and_legacy_inputs_still_work(self):
        locale=self.locale_module()
        source='extensions/forge-neo-image2prompt/scripts/image2prompt.py'
        tree=ast.parse((ROOT/source).read_text(encoding='utf-8'))
        constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n,ast.Assign) and n.targets[0].id in ('MODELS','STYLES','LEGACY_LABELS')}
        ns=load_nodes(source,['on_ui_tabs','generate_prompt'],dict(gr=gr,ui_text=locale.translate,
                     **constants,_load_model=lambda *a:('model','processor'),
                     _run_qwen=lambda *a:'original prompt, untouched',_free_model=lambda:None,
                     unload_clicked=lambda:None))
        with patch.dict(sys.modules,{'modules.ui':types.SimpleNamespace(switch_values_symbol='x')}):
            ui,title,_=ns['on_ui_tabs']()[0]
        self.assertEqual(title,'Image2Prompt')
        props=[x['props'] for x in ui.config['components']]
        for component in props:
            for field in ['label','info','placeholder','value']:
                text=component.get(field)
                if isinstance(text,str):
                    self.assertNotRegex(text,r'[\u4e00-\u9fff]')
        from PIL import Image
        old_model=next(k for k in constants['LEGACY_LABELS'] if k.startswith('Qwen2-VL-2B'))
        old_style='SD 提示詞／自然語言'
        image=Image.new('RGB',(1,1));self.addCleanup(image.close)
        prompt,chinese=ns['generate_prompt'](image,old_model,old_style,'',64,True,True)
        self.assertEqual(prompt,'original prompt, untouched');self.assertIn('模型已從顯存卸載',chinese)
        self.opts.localization='None'
        prompt,english=ns['generate_prompt'](image,old_model,old_style,'',64,True,True)
        self.assertEqual(prompt,'original prompt, untouched');self.assertIn('Model unloaded',english)
        self.assertNotRegex(english,r'[\u4e00-\u9fff]')
        self.assertEqual(ns['generate_prompt'](None,old_model,old_style,'',64,True,True)[1],'Please upload an image first.')

    def test_real_civitai_ui_labels_stay_english_for_reload_and_legacy_preview_is_accepted(self):
        import os
        from unittest.mock import Mock
        source='extensions/sd-webui-civitai-downloader/scripts/tab.py'
        ns=load_nodes(source,['on_ui_tabs','download_model'],dict(gr=gr,os=os,VERSION='test',
                     _shared=types.SimpleNamespace(opts=types.SimpleNamespace(civitai_enable_folder_selector=True)),
                     get_model_folders=lambda:{'LORA':'models/Lora'},
                     ui_text=lambda text,*args:text.format(*args),
                     download_civitai_model_with_progress=Mock(return_value=iter([('progress','output')])) ))
        ui,title,_=ns['on_ui_tabs']()[0]
        self.assertEqual(title,'Model Downloader')
        for component in ui.config['components']:
            for key in ['label','info','placeholder','value']:
                value=component['props'].get(key)
                if isinstance(value,str):self.assertNotRegex(value,r'[\u4e00-\u9fff]')
        result=list(ns['download_model'](('42','17'),('one','two'),'預覽圖 2',None))
        self.assertEqual(result,[('progress','output')])
        self.assertEqual(ns['download_civitai_model_with_progress'].call_args.args[3],'two')

    def test_civitai_progress_and_missing_state_use_real_gradio_updates(self):
        locale=self.locale_module()
        from unittest.mock import Mock
        source='extensions/sd-webui-civitai-downloader/scripts/tab.py'
        ns=load_nodes(source,['download_model','download_civitai_model_with_progress'],dict(
            gr=gr,ui_text=locale.translate,DOWNLOAD_CANCEL_FLAGS={},
            get_civitai_api_key=lambda:'',get_civitai_model_info=Mock(side_effect=RuntimeError('offline'))))
        missing=list(ns['download_model'](None,None,'Preview 1',None))
        self.assertEqual(missing[0][0]['__type__'],'update')
        self.assertEqual(missing[0][1]['value'],'尚未檢查模型，請先按「檢查模型」。')
        rows=list(ns['download_civitai_model_with_progress']('42'))
        self.assertEqual(rows[0][0]['value'],'準備開始…')
        self.assertEqual(rows[-1][1]['value'],'錯誤：offline')
        self.assertEqual(ns['DOWNLOAD_CANCEL_FLAGS'],{})
        self.opts.localization='None'
        rows=list(ns['download_civitai_model_with_progress']('42'))
        self.assertEqual(rows[-1][1]['value'],'Error: offline')

    def test_bundled_ui_constructors_do_not_embed_chinese_labels(self):
        paths=['extensions/forge-neo-image2prompt/scripts/image2prompt.py',
               'extensions/stable-diffusion-webui-tagger-fork/tagger/ui.py',
               'extensions-builtin/sd_forge_lora/ui_edit_user_metadata.py',
               'extensions/sd-webui-civitai-downloader/scripts/tab.py',
               'extensions/sd-webui-civitai-downloader/scripts/settings.py']
        for path in paths:
            tree=ast.parse((ROOT/path).read_text(encoding='utf-8-sig'))
            for call in ast.walk(tree):
                if not isinstance(call,ast.Call):continue
                name=ast.unparse(call.func)
                values=[k.value for k in call.keywords if k.arg in ('label','info','placeholder')]
                if name in ('gr.Markdown','gr.Button','gr.Accordion','gr.Tab'):values+=call.args[:1]
                if name.endswith('.OptionInfo'):values+=call.args[1:2]
                for value in values:
                    if isinstance(value,ast.Constant) and isinstance(value.value,str):
                        self.assertNotRegex(value.value,r'[\u4e00-\u9fff]',(path,value.lineno))
