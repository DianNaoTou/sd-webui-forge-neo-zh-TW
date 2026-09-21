"""CPU-only regression tests. Run: python -m unittest discover -s tests -v"""
import ast
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

from modules import dependency_utils as deps

ROOT = Path(__file__).resolve().parents[1]


class DependencyPolicyTests(unittest.TestCase):
    def test_version_bounds_pins_and_markers(self):
        with patch.object(deps.metadata, 'version', return_value='1.1.0'):
            self.assertFalse(deps.requirement_met('huggingface-hub>=0.34,<1'))
        with patch.object(deps.metadata, 'version', return_value='12.3.0'):
            self.assertFalse(deps.requirement_met('Pillow==10.4.0'))
        with patch.object(deps.metadata, 'version', return_value='6.33.6'):
            self.assertTrue(deps.requirement_met('protobuf>=6.31.1,<7'))
        with patch.object(deps.metadata, 'version', side_effect=AssertionError('must skip marker')):
            self.assertTrue(deps.requirement_met('missing; python_version<"2"'))

    def test_missing_unpinned_requirement(self):
        with patch.object(deps.metadata, 'version', side_effect=deps.metadata.PackageNotFoundError):
            self.assertFalse(deps.requirement_met('onnxruntime'))

    def test_extras_checked(self):
        def version(name):
            if name == 'onnxruntime-gpu':
                return '1.30.0'
            raise deps.metadata.PackageNotFoundError(name)
        with patch.object(deps.metadata, 'version', side_effect=version), patch.object(deps.metadata, 'requires', return_value=['nvidia-cudnn-cu12; extra == "cudnn"']):
            self.assertFalse(deps.requirement_met('onnxruntime-gpu[cudnn]>=1.21,<2'))

    def test_xpu_never_selects_cuda_when_driver_unavailable(self):
        fake = types.SimpleNamespace(version=types.SimpleNamespace(xpu='2026', cuda=None))
        with patch.dict('sys.modules', {'torch': fake}):
            self.assertEqual(deps.torch_backend(), 'xpu')
        self.assertNotIn('gpu', deps.onnx_requirement('xpu'))
        self.assertIn('[cuda,cudnn]', deps.onnx_requirement('cuda'))

    def test_onnx_fresh_and_second_launch(self):
        for backend, wanted, other in [('xpu', 'onnxruntime', 'onnxruntime-gpu'), ('cuda', 'onnxruntime-gpu', 'onnxruntime')]:
            installed = {}
            commands = []
            def version(name):
                if name in installed:
                    return installed[name]
                raise deps.metadata.PackageNotFoundError(name)
            def install(command, description):
                commands.append(command)
                if command.startswith('uninstall'):
                    self.assertIn(other, command)
                    installed.pop(other, None)
                else:
                    self.assertIn(wanted, command)
                    installed[wanted] = '1.30.0'
            runner = Mock(side_effect=install)
            with patch.object(deps, 'torch_backend', return_value=backend), patch.object(deps.metadata, 'version', side_effect=version), patch.object(deps.metadata, 'requires', return_value=[]), patch.dict('os.environ', {}, clear=True):
                deps.ensure_onnxruntime(runner)
                deps.ensure_onnxruntime(runner)
                self.assertEqual(runner.call_count, 1)
                installed[other] = '1.30.0'
                deps.ensure_onnxruntime(runner)
                self.assertEqual(runner.call_count, 3)
                self.assertTrue(commands[-2].startswith('uninstall'))
                self.assertIn('--force-reinstall', commands[-1])

    def test_constraints_quote_paths_with_spaces(self):
        with patch.object(deps, 'ROOT', Path('/project with spaces')):
            self.assertEqual(deps.pip_constraint_args(),
                             '--constraint "/project with spaces/constraints-common.txt" '
                             '--constraint "/project with spaces/requirements.txt"')

    def test_pip_options_only_apply_to_install_operations(self):
        constraints = deps.pip_constraint_args()
        install = deps.prepare_pip_command('install package', 'https://index.example/simple')
        self.assertIn(constraints, install)
        self.assertIn('--prefer-binary', install)
        self.assertIn('--index-url https://index.example/simple', install)

        upgrade = deps.prepare_pip_command('install --upgrade package', 'https://index.example/simple')
        self.assertIn(constraints, upgrade)
        self.assertIn('--prefer-binary', upgrade)

        uninstall = deps.prepare_pip_command('uninstall -y package', 'https://index.example/simple')
        self.assertEqual(uninstall, 'uninstall -y package')

    def test_explicit_install_index_is_preserved(self):
        command = 'install --pre package --index-url https://nightly.example/simple/'
        prepared = deps.prepare_pip_command(command, 'https://default.example/simple')
        self.assertIn('--index-url https://nightly.example/simple/', prepared)
        self.assertNotIn('https://default.example/simple', prepared)
        self.assertEqual(prepared.count('--prefer-binary'), 1)

    def test_existing_prefer_binary_is_not_duplicated(self):
        prepared = deps.prepare_pip_command('install --prefer-binary package')
        self.assertEqual(prepared.count('--prefer-binary'), 1)

    def test_empty_checkpoint_and_vae_defaults(self):
        tree = ast.parse((ROOT / 'extensions/ADetailer-Neo/lib_adetailer/ui.py').read_text())
        expressions = [kw.value for node in ast.walk(tree) if isinstance(node, ast.Call) for kw in node.keywords if kw.arg == 'value' and isinstance(kw.value, ast.IfExp) and 'webui_info' in ast.unparse(kw.value)]
        self.assertEqual(len(expressions), 2)
        for values in ([], ['model']):
            info = types.SimpleNamespace(checkpoints_list=values, vae_list=values)
            for expr in expressions:
                self.assertEqual(eval(compile(ast.Expression(expr), '<test>', 'eval'), {'webui_info': info}), values[0] if values else None)


# Load the real downloader while replacing only Forge/GPU-specific imports.
def load_downloader():
    fake_shared = types.ModuleType('modules.shared')
    fake_shared.cmd_opts = types.SimpleNamespace()
    fake_utils = types.ModuleType('adtest.utils')
    fake_utils.NUM = float
    fake_utils.print = Mock()
    spec = importlib.util.spec_from_file_location('adtest.detection.common', ROOT / 'extensions/ADetailer-Neo/lib_adetailer/detection/common.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'modules.shared': fake_shared, 'adtest.utils': fake_utils, spec.name: module}):
        spec.loader.exec_module(module)
    return module


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.downloader = load_downloader()

    def response(self, body=b'model', length='5'):
        response = io.BytesIO(body)
        response.headers = {'Content-Length': length}
        return response

    def test_success_cache_and_stale_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'model.pt'
            stale = Path(str(target) + '.dead.partial')
            stale.write_bytes(b'incomplete')
            with patch.object(self.downloader, 'urlopen', return_value=self.response()) as request:
                self.downloader._download_model('https://example.test/model', target)
                self.downloader._download_model('https://example.test/model', target)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(target.read_bytes(), b'model')
            self.assertFalse(stale.exists())
            self.assertFalse(list(Path(folder).glob('*.partial')))

    def test_truncation_cleanup_and_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'model.pt'
            with patch.object(self.downloader, 'urlopen', return_value=self.response(b'bad', '100')):
                with self.assertRaises(OSError):
                    self.downloader._download_model('https://example.test/model', target)
            self.assertFalse(target.exists())
            self.assertFalse(list(Path(folder).glob('*.partial')))
            with patch.object(self.downloader, 'urlopen', return_value=self.response()):
                self.downloader._download_model('https://example.test/model', target)
            self.assertEqual(target.read_bytes(), b'model')

    def test_failure_reason_and_continue(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(self.downloader, 'urlopen', side_effect=[TimeoutError('timed out'), self.response()]), patch.object(self.downloader, 'print') as log:
            self.downloader._download(folder, {'bad.pt': 'https://example.test/a', 'good.pt': 'https://example.test/b'})
            self.assertTrue((Path(folder) / 'good.pt').exists())
            self.assertTrue(any('TimeoutError: timed out' in call.args[0] for call in log.call_args_list))

    def test_interrupt_cleanup(self):
        response = self.response()
        response.read = Mock(side_effect=KeyboardInterrupt)
        with tempfile.TemporaryDirectory() as folder, patch.object(self.downloader, 'urlopen', return_value=response):
            target = Path(folder) / 'model.pt'
            with self.assertRaises(KeyboardInterrupt):
                self.downloader._download_model('https://example.test/model', target)
            self.assertFalse(target.exists())
            self.assertFalse(list(Path(folder).glob('*.partial')))


if __name__ == '__main__':
    unittest.main()
