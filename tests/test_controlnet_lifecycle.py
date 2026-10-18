"""Exercise production ControlNet lifecycle on CPU without importing GPU/UI services."""
import ast
from collections import OrderedDict
import gc
from contextlib import nullcontext
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch
import weakref

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
CNET = 'extensions-builtin/sd_forge_controlnet/scripts/controlnet.py'


def extract(path, name, namespace, methods=None, keep_bases=False):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(x for x in tree.body if getattr(x, 'name', None) == name)
    if isinstance(node, ast.ClassDef):
        if methods is not None:
            node.body = [x for x in node.body if isinstance(x, ast.FunctionDef) and x.name in methods]
        if not keep_bases:
            node.bases = []
        for func in node.body:
            if isinstance(func, ast.FunctionDef):
                func.decorator_list = [d for d in func.decorator_list if isinstance(d, ast.Name) and d.id in ('staticmethod', 'classmethod', 'property')]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), 'exec'), namespace)
    return namespace[name]


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.abort = type('ScriptAbortError', (RuntimeError,), {})
        self.ns = {'scripts': types.SimpleNamespace(ScriptAbortError=self.abort), 'logger': Mock(),
                   'Infotext': types.SimpleNamespace(write_infotext=Mock()), 'torch': torch}
        extract(CNET, 'ControlNetCachedParameters', self.ns)
        self.cls = extract(CNET, 'ControlNetForForgeOfficial', self.ns,
            {'process', '_abort', 'cleanup_processing', 'process_before_every_sampling',
             'postprocess_batch_list', 'validate_conditioning'})
        self.script = self.cls()
        self.script.get_enabled_units = lambda args: [x for x in args if x.enabled]
        self.script.bound_check_params = Mock()
        self.script.process_unit_before_every_sampling = Mock()
        self.script.process_unit_after_every_sampling = Mock()
        self.p = types.SimpleNamespace()
        self.units = [types.SimpleNamespace(enabled=True, module='canny', model='control') for _ in range(2)]

    def test_initialization_failure_aborts_all_units_and_clears_partial_state(self):
        resources = []
        def init(p, unit, params, *args, **kw):
            params.control_cond = object()
            params.model = object()
            resources.append(params)
            if unit._idx == 1:
                raise ValueError('model missing')
        self.script.process_unit_after_click_generate = init
        with self.assertRaisesRegex(self.abort, 'model missing'):
            self.script.process(self.p, *self.units)
        self.assertEqual(self.script.current_params, {})
        self.assertTrue(all(x.model is None and x.control_cond is None for x in resources))
        self.script.process_unit_before_every_sampling.assert_not_called()

    def test_multiunit_success_highres_callbacks_and_next_request_recovery(self):
        self.script.process_unit_after_click_generate = Mock()
        self.script.process(self.p, *self.units)
        self.script.process_before_every_sampling(self.p, *self.units)
        self.p.is_hr_pass = True
        self.script.process_before_every_sampling(self.p, *self.units)
        self.script.postprocess_batch_list(self.p, object(), *self.units)
        self.assertEqual(self.script.process_unit_before_every_sampling.call_count, 4)
        self.assertEqual(self.script.process_unit_after_every_sampling.call_count, 2)
        self.script.cleanup_processing(self.p)
        self.script.cleanup_processing(self.p)  # idempotent
        self.assertEqual(self.script.current_params, {})
        self.script.process(self.p, *self.units)
        self.assertEqual(len(self.script.current_params), 2)

    def test_sampling_callback_failure_drops_state_and_can_retry(self):
        self.script.process_unit_after_click_generate = Mock()
        self.script.process(self.p, *self.units)
        self.script.process_unit_before_every_sampling.side_effect = ValueError('bad condition')
        with self.assertRaisesRegex(self.abort, 'bad condition'):
            self.script.process_before_every_sampling(self.p, *self.units)
        self.assertEqual(self.script.current_params, {})
        self.script.process_unit_before_every_sampling.side_effect = None
        self.script.process(self.p, *self.units)
        self.script.process_before_every_sampling(self.p, *self.units)

    def test_disabled_nested_adetailer_request_does_not_clear_parent(self):
        self.script.process_unit_after_click_generate = Mock()
        self.script.process(self.p, *self.units)
        original = self.script.current_params
        nested = types.SimpleNamespace(control_net_disabled=True)
        self.script.process(nested, *self.units)
        self.script.process_before_every_sampling(nested, *self.units)
        self.script.postprocess_batch_list(nested, object(), *self.units)
        self.script.cleanup_processing(nested)
        self.assertIs(self.script.current_params, original)
        self.assertEqual(len(original), 2)

    def test_missing_initialization_is_explicit_abort_not_keyerror(self):
        self.script.current_params = {}
        with self.assertRaises(self.abort):
            self.script.process_before_every_sampling(self.p, *self.units)
        self.script.postprocess_batch_list(self.p, object(), *self.units)
        self.script.process_unit_after_every_sampling.assert_not_called()

    def test_cleanup_restores_unet_even_if_sampling_cleanup_fails(self):
        self.script.process_unit_after_click_generate = Mock()
        self.script.process(self.p, *self.units)
        original = object()
        target = types.SimpleNamespace(unet=object())
        self.script._patch_target = target
        self.script._unet_before_controlnet = original
        module = types.ModuleType('backend.sampling.sampling_function')
        module.sampling_cleanup = Mock(side_effect=RuntimeError('cleanup failure'))
        with patch.dict('sys.modules', {'backend.sampling.sampling_function': module}):
            with self.assertRaisesRegex(RuntimeError, 'cleanup failure'):
                self.script.cleanup_processing(self.p)
        self.assertIs(target.unet, original)
        self.assertEqual(self.script.current_params, {})

    def test_core_runner_only_propagates_explicit_abort_and_runs_all_cleanups(self):
        errors = Mock()
        ns = {'ScriptAbortError': self.abort, 'errors': errors}
        runner_cls = extract('modules/scripts.py', 'ScriptRunner', ns,
                             {'process', 'process_before_every_sampling', 'cleanup_processing'})
        runner = runner_cls()
        script = types.SimpleNamespace(args_from=0, args_to=0, filename='test',
                    process=Mock(side_effect=self.abort('invalid control')),
                    process_before_every_sampling=Mock(side_effect=self.abort('invalid sampling')),
                    cleanup_processing=Mock(side_effect=RuntimeError('cleanup')))
        second = types.SimpleNamespace(cleanup_processing=Mock())
        runner.ordered_scripts = lambda method: [script]
        runner.alwayson_scripts = [script]
        p = types.SimpleNamespace(script_args=[])
        for method in ('process', 'process_before_every_sampling'):
            with self.assertRaises(self.abort):
                getattr(runner, method)(p)
        script.process.side_effect = ValueError('ordinary extension error')
        runner.process(p)  # historical handling unchanged
        errors.report.assert_called()
        runner.alwayson_scripts.append(second)
        runner.cleanup_processing(p)
        second.cleanup_processing.assert_called_once_with(p)

    def test_process_images_finally_cleans_up_on_success_and_sampling_error(self):
        # Execute the actual production try/finally with only its body isolated.
        tree = ast.parse((ROOT / 'modules/processing.py').read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process_images')
        block = next(n for n in func.body if isinstance(n, ast.Try))
        block.body = [ast.Expr(ast.Call(func=ast.Name('sample', ast.Load()), args=[], keywords=[]))]
        fn = ast.FunctionDef(name='run', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[block], decorator_list=[])
        cleanup = Mock()
        sample = Mock()
        ns = {'p': types.SimpleNamespace(scripts=types.SimpleNamespace(cleanup_processing=cleanup), override_settings_restore_afterwards=False),
              '_vae_override': None, 'sample': sample}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<production-finally>', 'exec'), ns)
        ns['run']()
        cleanup.assert_called_once_with(ns['p'])
        sample.side_effect = RuntimeError('CUDA error')
        with self.assertRaisesRegex(RuntimeError, 'CUDA error'):
            ns['run']()
        self.assertEqual(cleanup.call_count, 2)


@unittest.skipIf(torch is None, 'requires torch')
class CompatibilityTests(unittest.TestCase):
    def test_condition_contracts_allow_images_ipadapter_and_unknown_extensions(self):
        ns = {'torch': torch}
        cls = extract(CNET, 'ControlNetForForgeOfficial', ns, {'validate_conditioning'})
        unit = types.SimpleNamespace(module='test-preprocessor', model='test-model')
        for kind, value in [('image', torch.zeros(1,3,4,4)), ('ipadapter', {'image': object()}), (None, object())]:
            cls.validate_conditioning(types.SimpleNamespace(conditioning_kind=kind), value, unit)
        for kind, value in [('image', {'image': object()}), ('ipadapter', torch.zeros(1,3,4,4))]:
            with self.assertRaises(ValueError):
                cls.validate_conditioning(types.SimpleNamespace(conditioning_kind=kind), value, unit)

    def test_missing_and_incompatible_models_fail_before_preprocessor_download(self):
        ns = {'torch': torch, 'ControlModelPatcher': type('ControlModelPatcher', (), {}),
              'os': __import__('os'), 'StableDiffusionProcessingTxt2Img': type('Txt2Img', (), {})}
        cls = extract(CNET, 'ControlNetForForgeOfficial', ns, {'process_unit_after_click_generate', 'validate_conditioning'})
        obj = cls()
        obj.get_target_dimensions = lambda p: (512,512,512,512)
        obj.get_input_data = Mock(side_effect=AssertionError('preprocessor must not run'))
        pre = types.SimpleNamespace(do_not_need_model=False, model_conditioning_kind='ipadapter')
        ns['global_state'] = types.SimpleNamespace(get_preprocessor=lambda name: pre, get_controlnet_filename=lambda model: __file__)
        ns['try_load_supported_control_model'] = lambda filename: types.SimpleNamespace(conditioning_kind='image')
        unit = types.SimpleNamespace(use_preview_as_input=False, module='CLIP-ViT-H (IPAdapter)', model='None')
        for selected in ('None', 'generic-control-model'):
            unit.model = selected
            with self.assertRaises(ValueError):
                obj.process_unit_after_click_generate(object(), unit, types.SimpleNamespace())
        obj.get_input_data.assert_not_called()


@unittest.skipIf(torch is None, 'requires torch')
class ModelLifetimeTests(unittest.TestCase):
    def setUp(self):
        class Wrapper:
            def __init__(self, model, **kw):
                self.model = model
        self.ns = {'torch': torch, 'ModelPatcher': Wrapper,
                   'memory_management': types.SimpleNamespace(get_torch_device=lambda: torch.device('cpu'),
                         unet_offload_device=lambda: torch.device('cpu'), unload_model=Mock())}
        path = 'backend/patcher/controlnet.py'
        extract(path, 'ControlBase', self.ns,
                {'__init__', 'copy_to', 'get_models', 'cleanup'})
        self.control = extract(path, 'ControlNet', self.ns,
                {'__init__', 'copy', 'get_models', 'cleanup'}, keep_bases=True)

    def test_copies_share_managed_wrapper_but_not_conditioning_state(self):
        base = self.control(torch.nn.Linear(1,1))
        base.cond_hint = torch.ones(1)
        first, second = base.copy(), base.copy()
        self.assertIs(first.control_model_wrapped, base.control_model_wrapped)
        self.assertIs(second.get_models()[0], first.get_models()[0])
        self.assertIsNone(first.cond_hint)
        first.cond_hint = torch.ones(2)
        first.cleanup()
        self.assertIsNone(first.cond_hint)
        self.assertIsNotNone(base.cond_hint)
        self.ns['memory_management'].unload_model.assert_not_called()

    def test_control_lora_cleanup_is_safe_before_prepare_and_when_repeated(self):
        cls = extract('backend/patcher/controlnet.py', 'ControlLora', self.ns,
                      {'__init__', 'cleanup'}, keep_bases=True)
        model = cls({})
        model.cleanup()  # prepare never ran
        model.control_model = torch.nn.Linear(1,1)
        model.cleanup()
        model.cleanup()  # sampler already cleaned up, request finally runs again
        self.assertIsNone(model.control_model)

    def test_actual_loader_cache_preserves_wrapper_and_respects_dtype_key(self):
        class FakeNet(torch.nn.Module):
            created = 0
            def __init__(self, **kw):
                super().__init__()
                FakeNet.created += 1
            def load_state_dict(self, data, strict=False):
                return [], []
        mm = self.ns['memory_management']
        mm.unet_dtype = lambda: torch.float16
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(data={}),
            sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=types.SimpleNamespace(model=types.SimpleNamespace(computation_dtype=torch.float16)))))
        self.ns.update({'OrderedDict': OrderedDict, 'shared': shared, 'logger': Mock(),
            'os': __import__('os'), 'using_forge_operations': lambda **kw: nullcontext(),
            'cldm': types.SimpleNamespace(ControlNet=FakeNet),
            'model_config_from_unet': lambda *a: types.SimpleNamespace(unet_config={'out_channels':4}),
            'load_t2i_adapter': lambda data: None})
        extract('modules_forge/supported_controlnet.py', 'ModelCache', self.ns)
        self.ns['_CONTROL_MODEL_CACHE'] = self.ns['ModelCache']()
        extract('modules_forge/supported_controlnet.py', 'ControlModelPatcher', self.ns, {'__init__'})
        cls = extract('modules_forge/supported_controlnet.py', 'ControlNetPatcher', self.ns,
                      {'__init__', 'try_build_from_state_dict'}, keep_bases=True)
        def load(path):
            return cls.try_build_from_state_dict({'zero_convs.0.0.weight':torch.zeros(1),
                'input_hint_block.0.weight':torch.zeros(1,3,1,1)}, path).model_patcher
        first = load('test.safetensors')
        wrapper = weakref.ref(first.control_model_wrapped)
        raw = first.control_model
        del first
        gc.collect()
        self.assertIsNotNone(wrapper())
        second = load('test.safetensors')
        self.assertIs(second.control_model_wrapped, wrapper())
        self.assertIs(second.control_model, raw)
        self.assertEqual(FakeNet.created, 1)
        shared.sd_model.forge_objects.unet.model.computation_dtype = torch.float32
        third = load('test.safetensors')
        self.assertIsNot(third.control_model_wrapped, wrapper())
        self.assertEqual(FakeNet.created, 2)
        self.ns['_CONTROL_MODEL_CACHE'].cache.clear()
        del second, third
        gc.collect()
        self.assertIsNone(wrapper())
