"""Review gates against released neo 2.30-tw.1, independent of GPU imports."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'tests/fixtures/ui_2_30_tw_1.json'


def controls(source):
    protected = {'txt2img_prompt','txt2img_neg_prompt','img2img_prompt','img2img_neg_prompt',
                 'txt2img_hires_fix_row3','hr_checkpoint','hr_vae_te','hr_sampler',
                 'forge_ui_preset','setting_sd_model_checkpoint','setting_sd_modules','forge_ui_dtype',
                 'txt2img_sampling','img2img_sampling','txt2img_steps','img2img_steps',
                 'txt2img_width','txt2img_height','img2img_width','img2img_height',
                 'txt2img_generate','img2img_generate','settings_show_all_pages','settings_show_one_page'}
    result = {}
    def visit(node, parents):
        if isinstance(node, ast.Call):
            elem_node = next((k.value for k in node.keywords if k.arg=='elem_id'),None)
            elem = elem_node.value if isinstance(elem_node,ast.Constant) else None
            if isinstance(elem_node,ast.JoinedStr) and any(isinstance(v,ast.Constant) and v.value.endswith(('_prompt','_neg_prompt','_generate','_sampling','_steps')) for v in elem_node.values):
                elem = ast.unparse(elem_node)
                protected.add(elem)
            if elem in protected:
                result[elem] = {'call':ast.dump(node,include_attributes=False),'parents':parents}
        if isinstance(node,ast.With):
            context = [ast.dump(i.context_expr,include_attributes=False) for i in node.items]
            for child in node.body:
                visit(child,parents+context)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child,parents)
    visit(ast.parse(source),[])
    return result


class UIFreezeTests(unittest.TestCase):
    def test_released_control_hierarchy_labels_and_defaults(self):
        baseline = json.loads(BASELINE.read_text())
        for name, expected in baseline['controls'].items():
            self.assertTrue(expected, name)
            self.assertEqual(controls((ROOT/name).read_text()),expected,name)

    def test_released_css_and_settings_navigation(self):
        for name, digest in json.loads(BASELINE.read_text())['sha256'].items():
            # Normalize checkout line endings on Windows and Linux.
            actual = hashlib.sha256((ROOT/name).read_text().encode()).hexdigest()
            self.assertEqual(actual,digest,name+' changed: review UI Freeze before updating baseline')

    def test_ui_launch_includes_assets_without_backend_switch(self):
        tree = ast.parse((ROOT/'webui.py').read_text())
        launch = next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='launch' and any(k.arg=='allowed_paths' for k in n.keywords))
        value = next(k.value for k in launch.keywords if k.arg=='allowed_paths')
        self.assertIn('ui_asset_paths()',ast.unparse(value))
        self.assertNotIn('xpu',ast.unparse(value))
        opts = (ROOT/'modules/shared_options.py').read_text()
        self.assertIn('"localization": OptionInfo("zh_Hant"',opts)

    def test_hr_local_refresh_preserves_upstream_shared_refresh(self):
        from unittest.mock import Mock
        import types
        tree = ast.parse((ROOT/'modules/ui.py').read_text())
        refresh = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='refresh_model_and_modules')
        main = types.SimpleNamespace(refresh_models=Mock())
        update = Mock(return_value=['checkpoints','modules'])
        ns = dict(main_entry=main,_refresh_models=update)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[refresh],type_ignores=[])),'hr-refresh','exec'),ns)
        self.assertEqual(ns['refresh_model_and_modules'](),['checkpoints','modules'])
        main.refresh_models.assert_called_once_with()
        update.assert_called_once_with()
        calls = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertTrue(any(c.startswith('hr_checkpoint_refresh.click(') and 'refresh_model_and_modules' in c for c in calls))
        self.assertTrue(any(c.startswith('main_entry.on_refresh_checkpoint(') for c in calls))

    def test_wd14_cross_tab_bindings_keep_real_prompt_and_image_sources(self):
        # Execute the shipped registration block with recorder objects; no ONNX/GPU.
        tree = ast.parse((ROOT/'extensions/stable-diffusion-webui-tagger-fork/tagger/ui.py').read_text(encoding='utf-8-sig'))
        fn = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='on_ui_tabs')
        nodes = []
        for n in ast.walk(fn):
            if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Attribute) and isinstance(n.iter.func.value,ast.Dict):
                if any(isinstance(k,ast.Constant) and k.value=='txt2img' for k in n.iter.func.value.keys):
                    nodes.append(n)
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='register_paste_params_button':
                # The loop's call is already executed by its For node.
                if any(k.arg=='source_image_component' for arg in n.value.args if isinstance(arg,ast.Call) for k in arg.keywords):
                    nodes.append(n)
        import types
        bindings = []
        ns = dict(parameters_copypaste=types.SimpleNamespace(ParamBinding=lambda **k:k,register_paste_params_button=bindings.append),
                  send_to_txt2img=object(),send_to_img2img=object(),send_image_to_img2img=object(),tags=object(),image=object())
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'wd14-bindings','exec'),ns)
        self.assertEqual([b['tabname'] for b in bindings],['txt2img','img2img','img2img'])
        self.assertTrue(all(b['source_text_component'] is ns['tags'] for b in bindings))
        self.assertIs(bindings[-1]['source_image_component'],ns['image'])
        self.assertTrue(all(b['paste_field_names']==['Prompt'] for b in bindings))


if __name__ == '__main__':
    unittest.main()
