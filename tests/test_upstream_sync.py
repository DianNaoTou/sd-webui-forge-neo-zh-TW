"""CPU checks of real update paths, isolated from Forge's GPU initialization."""
import ast
import importlib.util
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from modules import dependency_utils as deps
from test_vae_flash_regressions import load_nodes, torch

ROOT = Path(__file__).resolve().parents[1]


class InstallationAndAPITests(unittest.TestCase):
    def test_nunchaku_abi_matrix(self):
        for platform in ('win_amd64', 'linux_x86_64'):
            self.assertIsNone(deps.nunchaku_wheel('2.13', 'cu13.0', 'cp310', platform))
            self.assertIsNone(deps.nunchaku_wheel('2.14', 'xpu', 'cp310', platform))
            self.assertIsNone(deps.nunchaku_wheel('2.8', 'cu13.0', 'cp310', platform))
            self.assertIsNone(deps.nunchaku_wheel('2.11', 'cu13.0', 'cp314', platform))
            self.assertIn('cu13.0torch2.11-cp310-cp310-' + platform,
                          deps.nunchaku_wheel('2.11', 'cu13.0', 'cp310', platform))
        source = (ROOT / 'modules/launch_utils.py').read_text()
        self.assertNotIn('torch==2.11.0+cu130', source)
        self.assertIn('install --no-deps {nunchaku_package}', source)

    def test_api_defaults_reuse_built_controls_without_calling_ui(self):
        ui = Mock(side_effect=AssertionError('UI must not be rebuilt'))
        script = types.SimpleNamespace(args_from=1, args_to=3, ui=ui,
                    api_info=types.SimpleNamespace(args=[types.SimpleNamespace(value=0), types.SimpleNamespace(value=False)]))
        empty = types.SimpleNamespace(args_from=3, args_to=3, ui=ui, api_info=None)
        ns = load_nodes('modules/api/api.py', ['init_default_script_args'], {}, class_name='Api')
        self.assertEqual(ns['init_default_script_args'](None, types.SimpleNamespace(scripts=[script, empty])), [0, 0, False])
        ui.assert_not_called()

    def test_refresh_callbacks_chain_once_and_reset_on_ui_reload(self):
        ns = {'Callable': object, 'gr': types.SimpleNamespace(components=types.SimpleNamespace(Component=object)),
              '_refresh_callbacks': [], '_CALLBACK_SET': False}
        load_nodes('modules_forge/main_entry.py', ['capture_callbacks', 'on_refresh_checkpoint', '_refreshed_callback'], ns)
        events = []
        class Event:
            def then(self, **kwargs):
                events.append(kwargs)
                return self
        ns['_event_refresh'] = Event()
        first, second = Mock(), Mock()
        ns['on_refresh_checkpoint'](first, outputs=['refiner'])
        ns['on_refresh_checkpoint'](second, outputs=['hires'])
        ns['_refreshed_callback']()
        self.assertEqual([e['fn'] for e in events], [first, second])
        with self.assertRaises(RuntimeError):
            ns['on_refresh_checkpoint'](Mock())
        ns['capture_callbacks']()
        self.assertEqual(ns['_refresh_callbacks'], [])
        ns['on_refresh_checkpoint'](first)


@unittest.skipIf(torch is None, 'requires torch')
class SpectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the real package and forecaster. Only device services are CPU stand-ins.
        backend = types.ModuleType('backend')
        backend.stream = types.SimpleNamespace(should_use_stream=lambda: False)
        memory = types.ModuleType('backend.memory_management')
        memory.get_torch_device = lambda: torch.device('cpu')
        memory.should_use_bf16 = lambda device: False
        memory.sync_stream = Mock()
        log = types.ModuleType('backend.logging')
        log.setup_logger = lambda logger: None
        path = ROOT / 'extensions-builtin/sd_forge_spectrum/lib_spectrum'
        spec = importlib.util.spec_from_file_location('spectrum_test_package', path / '__init__.py', submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'backend': backend, 'backend.memory_management': memory,
                                      'backend.logging': log, spec.name: module}):
            spec.loader.exec_module(module)
        cls.module = module

    def test_batched_condition_histories_forecast_and_new_pass(self):
        class Model:
            def clone(self):
                return self
            def set_model_unet_function_wrapper(self, wrapper):
                self.wrapper = wrapper
        model = self.module.SpectrumNode.patch(Model(), 20, 0.25, 1, 1e-5, 2, 0.0, 3, 0.9)
        calls = []
        def forward(x, timestep, **conditions):
            calls.append(x.shape[0])
            self.assertEqual(conditions['scalar'].ndim, 0)
            return x * 2
        # Transition between batched CFG and cond-only as negative skipping toggles.
        for sequence in ([[0, 1]] * 12, [[0, 1], [0]] * 6, [[0]] * 12):
            for step, flags in enumerate(sequence):
                n = 2 * len(flags)
                x = torch.arange(1, n + 1, dtype=torch.float32).reshape(n, 1, 1, 1)
                result = model.wrapper(forward, {'input': x, 'timestep': torch.full((n,), 20-step),
                    'c': {'transformer_options': {'cond_or_uncond': flags}, 'scalar': torch.tensor(1.)}})
                self.assertEqual(result.shape, x.shape)
                self.assertTrue(torch.isfinite(result).all())
                # Constant histories should remain close when forecasting.
                torch.testing.assert_close(result, x*2, rtol=0.02, atol=0.02)
        self.assertLess(len(calls), 36)

    def test_weight_error_classification_preserves_real_failure(self):
        safe = types.SimpleNamespace(safe_open=Mock())
        ns = load_nodes('backend/utils.py', ['load_torch_file'],
                        {'torch': torch, 'safetensors': safe, 'DISABLE_MMAP': False})
        for message in ('HeaderTooLarge', 'MetadataIncompleteBuffer'):
            safe.safe_open.side_effect = RuntimeError(message)
            with self.assertRaisesRegex(ValueError, 'corrupted or incomplete'):
                ns['load_torch_file']('model.safetensors')
        for error in (PermissionError('access denied'), RuntimeError('out of memory')):
            safe.safe_open.side_effect = error
            with self.assertRaises(type(error)) as caught:
                ns['load_torch_file']('model.safetensors')
            self.assertIs(caught.exception, error)

    def test_highres_zero_steps_inherits_first_pass(self):
        node = Mock()
        ns = load_nodes('extensions-builtin/sd_forge_spectrum/scripts/spectrum.py',
                ['process_before_every_sampling'], {'SpectrumNode': node}, class_name='SpectrumForForge')
        p = types.SimpleNamespace(txt2img_upscale=True, hr_second_pass_steps=0, steps=20,
            sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=object())), extra_generation_params={})
        ns['process_before_every_sampling'](None, p, True, *range(7))
        self.assertEqual(node.patch.call_args.args[1], 20)
        p.hr_second_pass_steps = 10
        ns['process_before_every_sampling'](None, p, True, *range(7))
        self.assertEqual(node.patch.call_args.args[1], 10)
