"""Real Gradio HTTP tests, with only Forge GPU services replaced.

Run in a CPU environment with gradio==4.40.0, fastapi==0.127.1,
httpx==0.28.1 and huggingface-hub>=0.34,<1.0 installed.
"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
try:
    import gradio as gr
    from fastapi.testclient import TestClient
except ImportError:
    gr = None


@unittest.skipIf(gr is None, 'requires pinned Gradio/FastAPI UI dependencies')
class UIAssetServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / 'isolated data'
        self.state.mkdir()
        self.active = [types.SimpleNamespace(path=str(p)) for p in (ROOT / 'extensions').iterdir() if p.is_dir()]
        self.active += [types.SimpleNamespace(path=str(p)) for p in (ROOT / 'extensions-builtin').iterdir() if p.is_dir()]
        def scripts(folder, suffix):
            bases = [ROOT, *(Path(e.path) for e in self.active)]
            return [types.SimpleNamespace(path=str(p),filename=p.name) for b in bases for p in sorted((b / folder).glob('*' + suffix)) if p.is_file()]
        def named(name):
            return [str(b / name) for b in [ROOT, *(Path(e.path) for e in self.active)] if (b / name).is_file()]
        script_service = types.SimpleNamespace(list_scripts=scripts, list_files_with_name=named)
        # Load the actual localization merger against the shipped resources.
        spec = importlib.util.spec_from_file_location('ui_test_localization', ROOT / 'modules/localization.py')
        loc = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'modules.errors': types.SimpleNamespace(report=lambda *a, **k: None), 'modules.scripts':script_service}):
            spec.loader.exec_module(loc)
        loc.list_localizations(str(ROOT / 'localizations'))
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(localization='zh_Hant'), cmd_opts=types.SimpleNamespace(theme=None))
        spec = importlib.util.spec_from_file_location('ui_test_assets', ROOT / 'modules/ui_gradio_extensions.py')
        self.assets = importlib.util.module_from_spec(spec)
        # Set attributes on modules too: another test may already have imported it.
        import modules
        services = dict(extensions=types.SimpleNamespace(active=lambda:self.active), localization=loc,
                        scripts=script_service, shared=shared,
                        util=types.SimpleNamespace(truncate_path=lambda p:os.path.relpath(p,ROOT) if Path(p).is_relative_to(ROOT) else str(p)))
        with patch.multiple(modules, create=True, **services), patch.dict('sys.modules', {
            **{'modules.'+k:v for k,v in services.items()},
            'modules.paths':types.SimpleNamespace(data_path=str(self.state),script_path=str(ROOT))}):
            spec.loader.exec_module(self.assets)

    def client(self, paths):
        with gr.Blocks(analytics_enabled=False) as demo:
            gr.Textbox(elem_id='txt2img_prompt')
        demo.allowed_paths = paths
        return TestClient(gr.routes.App.create_app(demo))

    def test_isolated_data_dir_reproduces_403_then_assets_and_plugins_load(self):
        urls = [self.assets.webpath(ROOT/'style.css'), self.assets.webpath(ROOT/'script.js'),
                self.assets.webpath(ROOT/'javascript/localization.js')]
        for ext in ('sd-webui-prompt-all-in-one-neo','stable-diffusion-webui-tagger-fork'):
            urls += [self.assets.webpath(p) for p in (ROOT/'extensions'/ext/'javascript').glob('*.js')]
        before = self.client([str(self.state)])
        after = self.client([str(self.state), *self.assets.ui_asset_paths()])
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(before.get('/'+url).status_code,403)
                response = after.get('/'+url)
                self.assertEqual(response.status_code,200)
                self.assertTrue(response.content)
                self.assertIn('text/css' if '.css?' in url else 'javascript',response.headers['content-type'])
        for private in ('config.json','extensions/sd-webui-prompt-all-in-one-neo/storage/private.json'):
            self.assertEqual(after.get('/file='+str(ROOT/private)).status_code,403)

    def test_translation_and_all_injected_frontend_files_are_allowed(self):
        html = self.assets.javascript_html()
        self.assertLess(html.index('window.localization'),html.index('src='))
        loc = json.loads(html.split('window.localization = ',1)[1].split('</script>',1)[0])
        self.assertEqual(loc['Generate'],'生成')
        import re
        with patch.dict("sys.modules", {"modules.shared_gradio_themes":types.SimpleNamespace(resolve_var=lambda key:"white")}):
            css = self.assets.css_html()
        allowed = set(self.assets.ui_asset_paths())
        for url in re.findall(r'(?:src|href)="([^"]+)"', html + css):
            name = unquote(url.removeprefix('file=').split('?',1)[0])
            self.assertIn(str((ROOT/name).resolve()),allowed)

    def test_linked_extension_assets_use_both_path_spellings(self):
        extension = ROOT/'extensions/sd-webui-prompt-all-in-one-neo'
        alias = Path(self.temp.name)/'linked extension'
        try:
            alias.symlink_to(extension, target_is_directory=True)
        except OSError:
            self.skipTest('host cannot create test symlink')
        self.active[:] = [types.SimpleNamespace(path=str(alias))]
        paths = self.assets.ui_asset_paths()
        script = next((alias/'javascript').glob('*.js'))
        response = self.client([str(self.state), *paths]).get('/'+self.assets.webpath(script))
        self.assertEqual(response.status_code,200)
        self.assertIn(str(script.absolute()),paths)
        self.assertIn(str(script.resolve()),paths)

    def test_special_characters_in_asset_paths_and_inactive_extension(self):
        special = self.state/'漢字 # & prompt.js'
        special.write_text('window.test = true;',encoding='utf-8')
        response = self.client([str(self.state)]).get('/'+self.assets.webpath(special))
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.text,'window.test = true;')
        self.active.clear()
        self.assertFalse(any('/extensions/' in p or '/extensions-builtin/' in p for p in self.assets.ui_asset_paths()))


if __name__ == '__main__':
    unittest.main()
