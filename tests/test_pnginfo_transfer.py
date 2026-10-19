"""Real Gradio PNG metadata transfers with Forge inference services isolated."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
try:
    import gradio as gr
    from PIL import Image, PngImagePlugin
except ImportError:
    gr = None


@unittest.skipIf(gr is None, 'requires pinned Gradio UI dependencies')
class PNGInfoTransferTests(unittest.TestCase):
    def setUp(self):
        import modules
        import modules_forge
        self.opts = types.SimpleNamespace(
            infotext_skip_pasting=[], strip_whitespaces=True, infotext_styles='Ignore',
            use_old_hires_fix_width_height=False, sd_model_checkpoint='',
            forge_additional_modules=[], send_size=True)
        shared = types.SimpleNamespace(opts=self.opts, cmd_opts=types.SimpleNamespace(
            hide_ui_dir_config=True, no_prompt_history=True))
        services = dict(errors=types.SimpleNamespace(report=lambda *a, **k: None),
                        images=types.SimpleNamespace(), processing=types.SimpleNamespace(),
                        script_callbacks=types.SimpleNamespace(infotext_pasted_callback=lambda *a: None),
                        shared=shared, ui_tempdir=types.SimpleNamespace())
        main = types.SimpleNamespace(module_list={})
        spec = importlib.util.spec_from_file_location('pnginfo_transfer_test', ROOT/'modules/infotext_utils.py')
        self.mod = importlib.util.module_from_spec(spec)
        with patch.multiple(modules, create=True, **services), patch.object(modules_forge, 'main_entry', main, create=True), patch.dict('sys.modules', {
            **{'modules.'+k: v for k, v in services.items()},
            'modules.paths': types.SimpleNamespace(data_path='/unused'),
            'backend.text_processing.emphasis': types.SimpleNamespace(uses_emphasis=lambda x: False),
            'modules_forge.main_entry': main}):
            spec.loader.exec_module(self.mod)
        # Use the production legacy-event adapter without loading GPU services.
        tree = ast.parse((ROOT/'modules/gradio_extensions.py').read_text())
        wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EventWrapper')
        namespace = {}
        exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<EventWrapper>', 'exec'), namespace)
        self.event_wrapper = namespace['EventWrapper']

    def test_png_prompts_and_parameters_reach_both_fresh_tabs(self):
        info = ('繁中測試, a cat\nNegative prompt: blurry\n'
                'Steps: 28, Sampler: Euler, CFG scale: 7, Seed: 42, Size: 832x1216')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'metadata.png'
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text('parameters', info)
            Image.new('RGB', (832, 1216), 'red').save(path, pnginfo=metadata)
            for tab in ('txt2img', 'img2img'):
                with self.subTest(tab=tab), gr.Blocks(analytics_enabled=False) as demo:
                    source = gr.Image(type='pil', image_mode='RGBA')
                    text = gr.Textbox(visible=False)
                    prompt, negative = gr.Textbox(), gr.Textbox()
                    width, height = gr.Slider(value=512), gr.Slider(value=512)
                    steps, cfg, seed = gr.Slider(value=20), gr.Slider(value=7), gr.Number(value=-1)
                    dest = gr.Image(type='pil') if tab == 'img2img' else None
                    button = gr.Button()
                    button.click = self.event_wrapper(button.click)
                    fields = [self.mod.PasteField(c, k) for c, k in [
                        (prompt, 'Prompt'), (negative, 'Negative prompt'), (width, 'Size-1'),
                        (height, 'Size-2'), (steps, 'Steps'), (cfg, 'CFG scale'), (seed, 'Seed')]]
                    self.mod.paste_fields[tab] = dict(fields=fields, init_img=dest, override_settings_component=None)
                    self.mod._connect_paste_params_buttons(self.mod.ParamBinding(button, tab, text, source))
                    image = source.preprocess(gr.FileData(path=str(path), orig_name=path.name))
                    paste = next(f.fn for f in demo.fns.values() if getattr(f.fn, '__name__', '') == 'paste_func')
                    values = paste(image.info['parameters'])
                    self.assertIsNone(prompt.value)  # reproduces the initial empty Gradio field
                    self.assertEqual([v.get('value') for v in values], [
                        '繁中測試, a cat', 'blurry', 832, 1216, 28, 7, 42])
                    self.assertTrue(any(d['js'] == f'switch_to_{tab}' for d in demo.get_config_file()['dependencies']))
                    if dest is not None:
                        transfer = next(f.fn for f in demo.fns.values() if f.outputs == [dest, width, height])
                        import modules
                        with patch.dict('sys.modules', {'modules.ui': types.SimpleNamespace(sRound=lambda x: x)}):
                            moved, w, h = transfer(image)
                        self.assertEqual((moved.size, moved.mode, w, h), ((832, 1216), 'RGB', 832, 1216))

    def test_nullable_fields_and_existing_types_remain_usable(self):
        with gr.Blocks(analytics_enabled=False):
            for field, value, expected in [
                (gr.Textbox(), 'hello', 'hello'),
                (gr.Dropdown(choices=['Euler']), 'Euler', 'Euler'),
                (gr.Checkbox(value=True), 'False', False),
                (gr.Slider(value=20), '28', 28),
                (gr.Textbox(value='existing'), 'replacement', 'replacement')]:
                self.assertEqual(self.mod._parse_info(field, 'value', {'value': value})['value'], expected)
            field = gr.Textbox()
            self.assertNotIn('value', self.mod._parse_info(field, 'missing', {}))
