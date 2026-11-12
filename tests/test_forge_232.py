"""2.32 startup and cross-device image embedding regressions; no GPU required."""
import ast
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class AcceleratorProbeTests(unittest.TestCase):
    def probe(self, torch, optimize=0):
        tree = ast.parse((ROOT / 'modules/launch_utils.py').read_text())
        code = next(n.value.value for n in ast.walk(tree)
                    if isinstance(n, ast.AnnAssign) and getattr(n.target, 'id', None) == 'TORCH_CHECK')
        with patch.dict(sys.modules, {'torch': torch}):
            exec(compile(code, 'accelerator-probe', 'exec', optimize=optimize), {})

    def test_cuda_xpu_mps_execute_operation_and_copy_to_cpu(self):
        for device in ('cuda', 'xpu', 'mps'):
            with self.subTest(device=device):
                torch = Mock()
                torch.accelerator.current_accelerator.return_value = device
                self.probe(torch)
                torch.accelerator.current_accelerator.assert_called_once_with(check_available=True)
                torch.ones.assert_called_once_with([1], device=device)
                torch.ones.return_value.mul_.assert_called_once_with(2.0)
                torch.ones.return_value.mul_.return_value.cpu.assert_called_once_with()

    def test_no_accelerator_fails_even_with_optimized_python(self):
        for optimize in (0, 2):
            torch = Mock()
            torch.accelerator.current_accelerator.return_value = None
            with self.subTest(optimize=optimize), self.assertRaisesRegex(RuntimeError, 'No compute device'):
                self.probe(torch, optimize)
            torch.ones.assert_not_called()

    def test_driver_or_kernel_error_is_not_silently_accepted(self):
        torch = Mock()
        torch.ones.return_value.mul_.side_effect = RuntimeError('kernel failure')
        with self.assertRaisesRegex(RuntimeError, 'kernel failure'):
            self.probe(torch)


class VisionEmbeddingTests(unittest.TestCase):
    def test_vision_runs_on_encoder_device_then_moves_all_features(self):
        tree = ast.parse((ROOT / 'backend/nn/llm/llama.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3VL')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_embed')
        image, merged, a, b = Mock(), Mock(), Mock(), Mock()
        visual = Mock(return_value=(merged, [a, b]))
        qwen = Mock()
        qwen.process_qwen2vl_images.return_value = (image, 'grid')
        ns = dict(qwen_vl=qwen, torch=types.SimpleNamespace(float32='float32'),
                  text_encoder_device=lambda: 'encoder')
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'vision-embed', 'exec'), ns)
        result, extra = ns['preprocess_embed'](types.SimpleNamespace(visual=visual),
                                              {'type': 'image', 'data': 'image-data'}, 'target')
        image.to.assert_called_once_with('encoder', dtype='float32')
        visual.assert_called_once_with(image.to.return_value, 'grid')
        for tensor in (merged, a, b):
            tensor.to.assert_called_once_with('target')
        self.assertIs(result, merged.to.return_value)
        self.assertEqual(extra, {'grid': 'grid', 'deepstack': [a.to.return_value, b.to.return_value]})
        visual.reset_mock()
        self.assertEqual(ns['preprocess_embed'](types.SimpleNamespace(visual=visual),
                                               {'type': 'text'}, 'target'), (None, None))
        visual.assert_not_called()


class PreprocessorInstallTests(unittest.TestCase):
    def test_single_shared_ort_selection_and_pinned_insightface(self):
        launch = types.ModuleType('launch')
        launch.is_installed = lambda name: name != 'insightface'
        launch.run_pip = Mock()
        deps = types.ModuleType('modules.dependency_utils')
        deps.ensure_onnxruntime = Mock()
        deps.requirement_met = lambda _: True
        errors = types.ModuleType('modules.errors')
        errors.display = Mock()
        with patch.dict(sys.modules, {'launch': launch, 'modules.errors': errors,
                                      'modules.dependency_utils': deps}):
            runpy.run_path(str(ROOT / 'extensions-builtin/forge_legacy_preprocessors/install.py'))
        deps.ensure_onnxruntime.assert_called_once_with(launch.run_pip)
        launch.run_pip.assert_called_once_with('install insightface==2.1 --no-deps',
                                              'insightface (for ControlNet Preprocessor)')
