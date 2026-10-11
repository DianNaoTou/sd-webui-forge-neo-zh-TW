"""Regression tests for Issue #4: NVIDIA users ending up with CPU-only PyTorch.

Run without torch/GPU: python -m unittest discover -s tests -v
"""
import re
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from modules import dependency_utils as deps

ROOT = Path(__file__).resolve().parents[1]
NVIDIA_BAT = ROOT / 'webui-user.bat'
ARC_BAT = ROOT / 'webui-user-intel-arc.bat'


def bat_torch_command(path):
    m = re.search(r'set "TORCH_COMMAND=(.+?)"', path.read_text(encoding='utf-8'))
    return m.group(1)


def fake_versions(mapping):
    def version(name):
        if name in mapping:
            return mapping[name]
        raise deps.metadata.PackageNotFoundError(name)
    return version


class RequirementFilesTests(unittest.TestCase):
    def test_no_torch_in_common_requirements(self):
        files = [ROOT / 'requirements.txt', ROOT / 'constraints-common.txt']
        files += sorted((ROOT / 'extensions').glob('*/requirements*.txt'))
        files += sorted((ROOT / 'extensions-builtin').glob('*/requirements*.txt'))
        for path in files:
            for line in path.read_text(encoding='utf-8').splitlines():
                name = re.split(r'[\s<>=!~;\[]', line.strip(), maxsplit=1)[0].lower()
                with self.subTest(file=path.name, line=line):
                    self.assertNotIn(name, deps.TORCH_PACKAGES,
                                     f'{path} must not install PyTorch; the launcher chooses the backend')


class BackendDetectionTests(unittest.TestCase):
    def test_cpu_build_is_mismatch_for_nvidia(self):
        expected = deps.expected_torch_backend(bat_torch_command(NVIDIA_BAT))
        self.assertEqual(expected, 'cuda')
        for version in ('2.13.0+cpu', '2.13.0'):
            with self.subTest(version=version), patch.object(deps.metadata, 'version', fake_versions({'torch': version})):
                self.assertEqual(deps.torch_backend(), 'cpu')
                self.assertTrue(deps.torch_backend_mismatch(expected, deps.torch_backend()))
        self.assertTrue(deps.torch_backend_mismatch(expected, None))  # cuda None / not installed

    def test_cuda_build_ok_for_nvidia(self):
        with patch.object(deps.metadata, 'version', fake_versions({'torch': '2.13.0+cu130'})):
            self.assertEqual(deps.torch_backend(), 'cuda')
            self.assertFalse(deps.torch_backend_mismatch('cuda', deps.torch_backend()))

    def test_intel_arc_flow_requires_xpu(self):
        expected = deps.expected_torch_backend(bat_torch_command(ARC_BAT))
        self.assertEqual(expected, 'xpu')
        for version, bad in (('2.14.0+xpu', False), ('2.14.0+cpu', True), ('2.13.0+cu130', True)):
            with self.subTest(version=version), patch.object(deps.metadata, 'version', fake_versions({'torch': version})):
                self.assertEqual(deps.torch_backend_mismatch(expected, deps.torch_backend()), bad)

    def test_no_enforcement_without_explicit_command(self):
        self.assertIsNone(deps.expected_torch_backend('pip install torch==2.13.0+cu130', explicit=False))
        self.assertIsNone(deps.expected_torch_backend('pip install torch'))
        self.assertFalse(deps.torch_backend_mismatch(None, 'cpu'))

    def test_repair_commands_target_gpu_indexes(self):
        nvidia = bat_torch_command(NVIDIA_BAT)
        self.assertIn('torch==2.13.0+cu130', nvidia)
        self.assertIn('--index-url https://download.pytorch.org/whl/cu130', nvidia)
        self.assertIn('torchvision==', nvidia)
        arc = bat_torch_command(ARC_BAT)
        self.assertIn('+xpu', arc)
        self.assertIn('--index-url https://download.pytorch.org/whl/xpu', arc)

    def test_launcher_uses_shared_guard(self):
        src = (ROOT / 'modules' / 'launch_utils.py').read_text(encoding='utf-8')
        self.assertIn('dependency_utils.expected_torch_backend(', src)
        self.assertIn('repair_torch_backend or not is_installed("torch")', src)
        self.assertIn('still {torch_backend()} after installation', src)
        default = re.search(r'TORCH_COMMAND", f"(pip install torch==[^"]+)"', src).group(1)
        self.assertIn('+cu130', default)


class TorchPinConstraintTests(unittest.TestCase):
    def test_installed_torch_is_pinned_for_later_pip_installs(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(deps, 'ROOT', Path(tmp)), \
                patch.object(deps.metadata, 'version', fake_versions({'torch': '2.13.0+cu130', 'torchvision': '0.28.0+cu130'})):
            cmd = deps.prepare_pip_command('install ultralytics')
            path = Path(tmp) / 'tmp' / 'constraints-torch.txt'
            self.assertIn(f'--constraint "{path}"', cmd)
            text = path.read_text(encoding='utf-8')
            self.assertIn('torch==2.13.0+cu130', text)
            self.assertIn('torchvision==0.28.0+cu130', text)

    def test_no_torch_constraint_before_torch_install(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(deps, 'ROOT', Path(tmp)), \
                patch.object(deps.metadata, 'version', fake_versions({})):
            self.assertNotIn('constraints-torch', deps.pip_constraint_args())

    def test_uninstall_untouched(self):
        self.assertEqual(deps.prepare_pip_command('uninstall -y onnxruntime'), 'uninstall -y onnxruntime')


if __name__ == '__main__':
    unittest.main()
