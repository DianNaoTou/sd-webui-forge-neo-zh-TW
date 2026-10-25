"""CPU-only tests of production LayerDiffuse guards, patches and request lifecycle.

No Forge startup, dependency installation, network requests or model weights.
"""
import ast
from enum import Enum
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'extensions/sd-forge-layerdiffuse'
SCRIPT = EXT / 'scripts/forge_layerdiffusion.py'


def load_helper(name):
    spec = importlib.util.spec_from_file_location('layerdiffuse_test_' + name, EXT / 'lib_layerdiffusion' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = load_helper('compat')
tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
enum_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LayerMethod')
namespace = {'Enum': Enum}
exec(compile(ast.Module(body=[enum_node], type_ignores=[]), str(SCRIPT), 'exec'), namespace)
LayerMethod = namespace['LayerMethod']


class Tensor:
    """Only emulate transport operations; values use NumPy to verify broadcasting."""
    def __init__(self, array):
        self.array = np.asarray(array)
        self.shape = self.array.shape

    def repeat(self, *repetitions):
        return Tensor(np.tile(self.array, repetitions))

    def to(self, reference):
        return self


def request(sd15=False, batch=1):
    unet = types.SimpleNamespace(model_options={}, add_patches=Mock(), add_object_patch=Mock(),
                                add_conditioning_modifier=Mock(), detach=Mock())
    return types.SimpleNamespace(sd_model=types.SimpleNamespace(is_sd1=sd15, is_sdxl=not sd15,
        is_inpaint=False, forge_objects=types.SimpleNamespace(unet=unet)),
        batch_size=batch, scripts=types.SimpleNamespace(alwayson_scripts=[]), script_args=[])


class RequestTests(unittest.TestCase):
    def test_model_family_and_missing_images_are_rejected(self):
        p = request()
        self.assertEqual(compat.validate_request(p, LayerMethod.FG_ONLY_ATTN, None, None, None), 'sdxl')
        with self.assertRaisesRegex(ValueError, '模型不符'):
            compat.validate_request(p, LayerMethod.JOINT_SD15, None, None, None)
        with self.assertRaisesRegex(ValueError, '尚未提供'):
            compat.validate_request(p, LayerMethod.FG_TO_BLEND, None, None, None)
        compat.validate_request(p, LayerMethod.FG_TO_BLEND, np.zeros((8,8,4)), None, None)
        p.sd_model.is_inpaint = True
        with self.assertRaisesRegex(ValueError, 'Inpainting'):
            compat.validate_request(p, LayerMethod.FG_ONLY_ATTN, None, None, None)

    def test_sd15_frames_and_attention_conflicts(self):
        p = request(True, 2)
        with self.assertRaisesRegex(ValueError, '3'):
            compat.validate_request(p, LayerMethod.JOINT_SD15, None, None, None)
        p.batch_size = 3
        compat.validate_request(p, LayerMethod.JOINT_SD15, None, None, None)
        p.sd_model.forge_objects.unet.model_options = {'transformer_options': {'patches_replace': {'attn1': {}}}}
        with self.assertRaisesRegex(ValueError, 'Attention'):
            compat.validate_request(p, LayerMethod.JOINT_SD15, None, None, None)

    def test_adetailer_guard_uses_actual_enable_predicate(self):
        p = request()
        enabled = Mock(return_value=True)
        p.scripts.alwayson_scripts = [types.SimpleNamespace(is_ad_enabled=enabled, args_from=1, args_to=3)]
        p.script_args = ['other', True, {'ad_model': 'face'}]
        with self.assertRaisesRegex(ValueError, 'ADetailer'):
            compat.validate_request(p, LayerMethod.FG_ONLY_ATTN, None, None, None)
        enabled.assert_called_once_with(True, {'ad_model': 'face'})
        enabled.return_value = False
        compat.validate_request(p, LayerMethod.FG_ONLY_ATTN, None, None, None)

    def test_invalid_api_and_rgb_inputs_fail_before_resource_loading(self):
        p = request()
        with self.assertRaisesRegex(ValueError, 'RGBA'):
            compat.validate_request(p, LayerMethod.FG_TO_BLEND, np.zeros((8,8,3)), None, None)
        p.sd_model.forge_objects.unet.detach = None
        with self.assertRaisesRegex(ValueError, 'detach'):
            compat.validate_request(p, LayerMethod.FG_ONLY_ATTN, None, None, None)

    def test_concat_broadcast_preserves_values_and_rejects_mismatch(self):
        tensor = Tensor(np.arange(16).reshape(1,4,2,2))
        reference = Tensor(np.zeros((3,4,2,2)))
        expanded = compat.match_batch(tensor, reference)
        self.assertEqual(expanded.shape, (3,4,2,2))
        for row in expanded.array:
            np.testing.assert_array_equal(row, tensor.array[0])
        with self.assertRaises(ValueError):
            compat.match_batch(tensor, Tensor(np.zeros((3,4,4,4))))
        with self.assertRaises(ValueError):
            compat.match_batch(Tensor(np.zeros((2,4,2,2))), reference)

    def test_condition_injection_preserves_other_plugins_and_handles_cfg_one(self):
        original = [{'model_conds': {'cross_attn': object()}, 'control': object()}]
        tensor = object()
        result = compat.attach_concat(original, tensor, lambda x: x)
        self.assertIs(result[0]['model_conds']['c_concat'], tensor)
        self.assertNotIn('c_concat', original[0]['model_conds'])
        self.assertIs(result[0]['control'], original[0]['control'])
        self.assertIs(result[0]['model_conds']['cross_attn'], original[0]['model_conds']['cross_attn'])
        self.assertIsNone(compat.attach_concat(None, tensor, lambda x: x))

    def test_codecs_are_separate_by_request_family_and_role(self):
        p, other = request(), request()
        loader = Mock(side_effect=lambda: object())
        first = compat.get_transparent_codec(p, 'sd15', 'decoder', loader)
        self.assertIs(first, compat.get_transparent_codec(p, 'sd15', 'decoder', loader))
        self.assertIsNot(first, compat.get_transparent_codec(p, 'sdxl', 'decoder', loader))
        self.assertIsNot(first, compat.get_transparent_codec(p, 'sd15', 'encoder', loader))
        self.assertIsNot(first, compat.get_transparent_codec(other, 'sd15', 'decoder', loader))
        self.assertEqual(loader.call_count, 4)

    def test_cleanup_restores_multiple_hr_patches_and_is_idempotent(self):
        p = request()
        target = p.sd_model.forge_objects
        original = target.unet
        first, second = object(), object()
        compat.record_patch(p, target, 'unet', original, first)
        compat.record_patch(p, target, 'unet', first, second)
        target.unet = second
        codec = types.SimpleNamespace(model=types.SimpleNamespace(detach=Mock()))
        p._layerdiffuse_codecs = {('sdxl','decoder'): codec}
        p._layerdiffuse_error = 'sampling failed'
        mm = types.ModuleType('backend')
        mm.memory_management = types.SimpleNamespace(unload_model=Mock())
        utils = types.ModuleType('lib_layerdiffusion.utils')
        utils.unload_model_clones_compat = Mock()
        with patch.dict('sys.modules', {'backend':mm, 'lib_layerdiffusion.utils':utils}):
            compat.cleanup_request(p)
            compat.cleanup_request(p)
        self.assertIs(target.unet, original)
        codec.model.detach.assert_called_once_with(unpatch_all=True)
        self.assertFalse(hasattr(p, '_layerdiffuse_codecs'))
        self.assertFalse(hasattr(p, '_layerdiffuse_error'))

    def test_cleanup_failure_still_restores_model_and_drops_references(self):
        p = request()
        target = p.sd_model.forge_objects
        original, replacement = target.unet, object()
        compat.record_patch(p, target, 'unet', original, replacement)
        target.unet = replacement
        mm = types.ModuleType('backend')
        mm.memory_management = types.SimpleNamespace(unload_model=Mock())
        utils = types.ModuleType('lib_layerdiffusion.utils')
        utils.unload_model_clones_compat = Mock(side_effect=RuntimeError('detach failed'))
        with patch.dict('sys.modules', {'backend':mm, 'lib_layerdiffusion.utils':utils}):
            with self.assertRaisesRegex(RuntimeError, 'detach failed'):
                compat.cleanup_request(p)
        self.assertIs(target.unet, original)
        self.assertFalse(hasattr(p, '_layerdiffuse_patches'))

    def test_hook_failure_stays_failed_until_final_cleanup(self):
        abort = type('ScriptAbortError', (RuntimeError,), {})
        modules = types.ModuleType('modules')
        modules.scripts = types.SimpleNamespace(ScriptAbortError=abort)
        called = Mock(side_effect=ValueError('bad weight'))
        guarded = compat.fail_closed(called)
        p = request()
        with patch.dict('sys.modules', {'modules': modules}):
            for _ in range(2):
                with self.assertRaisesRegex(abort, 'bad weight'):
                    guarded(object(), p)
        self.assertEqual(called.call_count, 1)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        source = ast.parse((EXT / 'lib_layerdiffusion/utils.py').read_text(encoding='utf-8'))
        fn = next(n for n in source.body if getattr(n,'name',None) == 'load_frozen_patcher_compat')
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<production-adapter>', 'exec'), ns)
        self.load = ns[fn.name]
        self.adapter = type('LoRAAdapter', (), {'name':'lora', '__init__': lambda obj, **kw: obj.__dict__.update(kw)})
        self.module = types.ModuleType('modules_forge.packages.comfy')
        self.module.weight_adapter = types.SimpleNamespace(adapters=[self.adapter])
        self.unet = types.SimpleNamespace(model=types.SimpleNamespace(state_dict=lambda:{'weight':np.zeros((2,4,1,1))}),
            add_patches=Mock(side_effect=lambda **kw:list(kw['patches'])))

    def load_patch(self, weights):
        with patch.dict('sys.modules', {'modules_forge.packages.comfy':self.module}):
            self.load(self.unet, 'test.safetensors', weights, 0.8)

    def test_lora_builds_current_adapter_and_retains_strength(self):
        up, down = np.zeros((2,1)), np.zeros((1,4))
        self.load_patch({'weight::lora::0':up, 'weight::lora::1':down})
        args = self.unet.add_patches.call_args.kwargs
        self.assertIsInstance(args['patches']['weight'], self.adapter)
        self.assertIs(args['patches']['weight'].weights[0], up)
        self.assertEqual(args['strength_patch'], 0.8)

    def test_diff_expands_input_channels_and_rejects_other_shape_changes(self):
        self.load_patch({'weight::diff::0':np.zeros((2,8,1,1))})
        patch_value = self.unet.add_patches.call_args.kwargs['patches']['weight']
        self.assertEqual(patch_value[0], 'diff')
        self.assertTrue(patch_value[1][1]['pad_weight'])
        with self.assertRaisesRegex(ValueError, '形狀'):
            self.load_patch({'weight::diff::0':np.zeros((3,8,1,1))})

    def test_malformed_missing_and_partial_patches_fail_closed(self):
        for data in ({}, {'invalid':np.zeros(1)}, {'weight::lora::-1':np.zeros(1)},
                     {'missing::diff::0':np.zeros(1)}, {'weight::lora::0':np.zeros(1)},
                     {'weight::unsupported::0':np.zeros(1)}):
            with self.subTest(data=list(data)):
                with self.assertRaises(ValueError):
                    self.load_patch(data)
        self.unet.add_patches.return_value = []
        self.unet.add_patches.side_effect = None
        with self.assertRaisesRegex(ValueError, '未套用'):
            self.load_patch({'weight::diff::0':np.zeros((2,4,1,1))})


class DownloadTests(unittest.TestCase):
    def test_pinned_download_and_existing_corrupt_file_preservation(self):
        downloads = load_helper('downloads')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.safetensors'
            path.write_bytes(b'good')
            expected = hashlib.sha256(b'good').hexdigest()
            downloads.MANIFEST = {'repository':'LayerDiffusion/layerdiffusion-v1',
                'files':{'test.safetensors':{'sha256':expected,'size':4}}}
            module = types.ModuleType('modules.util')
            module.load_file_from_url = Mock(return_value=str(path))
            url = f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{downloads.REVISION}/test.safetensors'
            with patch.dict('sys.modules', {'modules.util':module}):
                self.assertEqual(downloads.load_file_from_url(url=url, model_dir=folder, file_name=path.name),str(path))
                self.assertEqual(module.load_file_from_url.call_args.kwargs['hash_prefix'], expected)
                path.write_bytes(b'bad!')
                with self.assertRaisesRegex(ValueError, '驗證失敗'):
                    downloads.load_file_from_url(url=url, model_dir=folder, file_name=path.name)
                self.assertEqual(path.read_bytes(), b'bad!')
                with self.assertRaises(ValueError):
                    downloads.load_file_from_url(url='https://example.com/evil',model_dir=folder,file_name=path.name)

    def test_manifest_covers_all_downloads_and_ui_keeps_stable_values(self):
        downloads = load_helper('downloads')
        names = {kw.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'load_file_from_url'
                 for kw in node.keywords if kw.arg == 'file_name'}
        self.assertEqual(names, set(downloads.MANIFEST['files']))
        labels = next(n.value for n in tree.body if isinstance(n,ast.Assign)
                      and getattr(n.targets[0],'id',None) == 'METHOD_LABELS')
        self.assertEqual(set(ast.literal_eval(labels)), {e.name for e in LayerMethod})
        self.assertFalse((EXT/'install.py').exists())
        self.assertFalse((EXT/'requirements.txt').exists())
        self.assertFalse((EXT/'.git').exists())
        # A global inpainting flag write would leak into the next request/HR pass.
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    self.assertFalse(isinstance(target,ast.Attribute) and target.attr == 'is_inpaint')


class SamplingModifierTests(unittest.TestCase):
    def test_actual_modifier_handles_hr_batches_and_stop_with_no_unconditional(self):
        script = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'LayerDiffusionForForge')
        method = next(n for n in script.body if getattr(n,'name',None) == 'process_before_every_sampling')
        functions = [n for n in method.body if isinstance(n,ast.FunctionDef)]
        original_model, patched_model = object(), object()
        prepare = Mock()
        ns = {'sigma_end':5, 'original_unet':types.SimpleNamespace(model=original_model),
              'unet':types.SimpleNamespace(extra_concat_condition=Tensor(np.ones((1,4,2,2)))),
              'is_model_loaded':lambda model:False, 'sampling_prepare':prepare,
              'match_batch':compat.match_batch,'attach_concat':compat.attach_concat,'Condition':lambda x:x}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<production-conditioning-modifier>','exec'),ns)
        fn = ns['conditioning_modifier']
        cond = [{'model_conds':{}}]
        x = Tensor(np.zeros((2,4,2,2)))
        active = fn(patched_model,x,np.array([10]),None,cond,1,{},123)
        self.assertIs(active[0],patched_model)
        self.assertIsNone(active[3])
        self.assertEqual(active[4][0]['model_conds']['c_concat'].shape,(2,4,2,2))
        stopped = fn(patched_model,x,np.array([1]),None,active[4],1,{},123)
        self.assertIs(stopped[0],original_model)
        self.assertNotIn('c_concat',stopped[4][0]['model_conds'])
        self.assertIn('c_concat',active[4][0]['model_conds'])
        prepare.assert_called_once()


if __name__ == '__main__':
    unittest.main()
