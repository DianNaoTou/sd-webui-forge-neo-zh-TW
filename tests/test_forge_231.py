"""2.31 CPU regressions: real production functions, no model downloads or GPU.

AST loading isolates Forge device initialization. MOSS uses real PyTorch and
Kitchen eager kernels with CPU-only weight transport; this is not GPU validation.
"""
import ast
from contextlib import nullcontext, redirect_stdout
import hashlib
import io
import json
import lzma
import math
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from PIL import Image

from test_vae_flash_regressions import load_nodes

ROOT = Path(__file__).resolve().parents[1]


class StateDictTests(unittest.TestCase):
    def test_prefix_filter_never_treats_string_as_characters(self):
        from backend.state_dict import load_state_dict
        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(2, 2)
        model.layers = torch.nn.Linear(2, 2)
        for prefixes in ('lm_head.', ['lm_head.']):
            output = io.StringIO()
            with redirect_stdout(output):
                load_state_dict(model, {}, ignore_start=prefixes)
            self.assertIn('layers.weight', output.getvalue())
            self.assertNotIn('lm_head.weight', output.getvalue())

    def test_multi_prefix_and_meta_materialization(self):
        from backend.state_dict import load_state_dict
        model = torch.nn.Linear(2, 2, device='meta')
        with redirect_stdout(io.StringIO()):
            load_state_dict(model, {'weight': torch.ones(2, 2)}, ignore_start=['bias', 'visual.'])
        self.assertFalse(model.bias.is_meta)
        torch.testing.assert_close(model.weight, torch.ones(2, 2))
        torch.testing.assert_close(model.bias, torch.zeros(2))


class AlphaTests(unittest.TestCase):
    def process(self, image, threshold):
        ns = load_nodes('modules/images.py', ['_process_alpha'],
                        {'Image': Image, 'opts': types.SimpleNamespace(alpha_threshold=threshold)})
        return ns['_process_alpha'](image)

    def test_disabled_threshold_preserves_semitransparent_pixels(self):
        image = Image.new('RGBA', (4, 1))
        image.putdata([(10, 20, 30, a) for a in (0, 1, 15, 128)])
        self.assertEqual(self.process(image, 0).tobytes(), image.tobytes())

    def test_threshold_zeros_rgb_only_below_cutoff(self):
        image = Image.new('RGBA', (4, 1))
        image.putdata([(10, 20, 30, a) for a in (0, 15, 16, 255)])
        result = self.process(image, 16)
        self.assertEqual(list(result.getdata()), [(0, 0, 0, 0), (0, 0, 0, 0), (10, 20, 30, 16), (10, 20, 30, 255)])

    def test_opaque_and_rgb_images_keep_expected_modes(self):
        self.assertEqual(self.process(Image.new('RGBA', (1, 1), (1, 2, 3, 255)), 16).mode, 'RGB')
        image = Image.new('RGB', (1, 1))
        self.assertIs(self.process(image, 16), image)


class OverrideTests(unittest.TestCase):
    def test_xyz_vae_none_removes_vaes_without_removing_text_encoders(self):
        vaes = {'old.safetensors': '/vae/old.safetensors', 'new.safetensors': '/vae/new.safetensors'}
        opts = types.SimpleNamespace(forge_additional_modules=['/te/clip.safetensors', vaes['old.safetensors']])
        ns = load_nodes('scripts/xyz_grid.py', ['find_vae', 'apply_vae'],
                        {'sd_vae': types.SimpleNamespace(vae_dict=vaes), 'opts': opts, 'StableDiffusionProcessing': object})
        for selection, expected in [('None', []), ('new.safetensors', [vaes['new.safetensors']])]:
            p = types.SimpleNamespace(override_settings={})
            ns['apply_vae'](p, selection, ['None', 'new.safetensors'])
            self.assertEqual(p.override_settings['forge_additional_modules'], ['/te/clip.safetensors', *expected])
        self.assertEqual(opts.forge_additional_modules, ['/te/clip.safetensors', vaes['old.safetensors']])

    def process_namespace(self, cleanup):
        opts = types.SimpleNamespace(data={'forge_additional_modules': ['clip', 'old']})
        opts.forge_additional_modules = opts.data['forge_additional_modules']
        set_config = Mock()
        sd_vae = types.SimpleNamespace(vae_dict={'old': 'old'}, reload_vae_weights=Mock(return_value=True), restore_vae_weights=Mock())
        ns = dict(StableDiffusionProcessing=object, Processed=object, opts=opts,
                  shared=types.SimpleNamespace(opts=opts), sd_vae=sd_vae, set_config=set_config,
                  sd_models=types.SimpleNamespace(checkpoint_aliases={}),
                  manage_model_and_prompt_cache=Mock(), sd_samplers=types.SimpleNamespace(fix_p_invalid_sampler_and_scheduler=Mock()),
                  profiling=types.SimpleNamespace(Profiler=nullcontext), process_images_inner=Mock(return_value='result'),
                  _overridden_modules=None)
        load_nodes('modules/processing.py', ['process_images'], ns)
        p = types.SimpleNamespace(scripts=types.SimpleNamespace(before_process=Mock(), cleanup_processing=cleanup),
                                  override_settings={'sd_vae':'new', 'forge_additional_modules':['temporary']},
                                  override_settings_restore_afterwards=True)
        return ns, p

    def test_cleanup_failure_still_restores_options_and_vae(self):
        ns, p = self.process_namespace(Mock(side_effect=RuntimeError('cleanup failed')))
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            ns['process_images'](p)
        self.assertEqual(ns['set_config'].call_args.args[0], {'forge_additional_modules':['clip','old']})
        ns['sd_vae'].restore_vae_weights.assert_called_once()
        self.assertIsNone(ns['_overridden_modules'])

    def test_generation_failure_still_restores_options_and_vae(self):
        ns, p = self.process_namespace(Mock())
        ns['process_images_inner'].side_effect = RuntimeError('generation failed')
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            ns['process_images'](p)
        ns['sd_vae'].restore_vae_weights.assert_called_once()
        p.scripts.cleanup_processing.assert_called_once_with(p)


class MOSSNumericsTests(unittest.TestCase):
    def test_qwen_delta_rule_matches_sequential_recurrence(self):
        ns = load_nodes('backend/nn/llm/llama.py', ['torch_chunk_gated_delta_rule'], {'torch':torch,'F':F})
        torch.manual_seed(231)
        for length in (1, 3, 9):
            q, k, v = [torch.randn(2, length, 2, 4) for _ in range(3)]
            g = -torch.rand(2, length, 2)
            beta = torch.rand(2, length, 2)
            initial = torch.randn(2, 2, 4, 4)
            actual, state = ns['torch_chunk_gated_delta_rule'](q,k,v,g,beta,chunk_size=4,initial_state=initial,output_final_state=True)
            q, k = F.normalize(q,dim=-1)/2, F.normalize(k,dim=-1)
            expected, recurrent = [], initial.clone()
            for i in range(length):
                recurrent = recurrent * g[:,i,:,None,None].exp()
                delta = (v[:,i] - torch.einsum('bhk,bhkv->bhv', k[:,i], recurrent)) * beta[:,i,:,None]
                recurrent = recurrent + k[:,i,:,:,None] * delta[:,:,None,:]
                expected.append(torch.einsum('bhk,bhkv->bhv',q[:,i],recurrent))
            torch.testing.assert_close(actual,torch.stack(expected,dim=1),atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(state,recurrent,atol=2e-6,rtol=2e-5)

    def test_moss_tiny_forward_uses_kitchen_eager_and_block_hook(self):
        import comfy_kitchen as ck
        from einops import rearrange
        ns = dict(torch=torch, nn=torch.nn, math=math, rearrange=rearrange,
                  memory_management=types.SimpleNamespace(is_device_mps=lambda d:False,is_intel_xpu=lambda:False,is_directml_enabled=lambda:False))
        flux = ast.parse((ROOT/'backend/nn/flux.py').read_text())
        nodes = [n for n in flux.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('rope','timestep_embedding','EmbedND')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'flux-cpu','exec'),ns)
        def attention(q,k,v,heads,**kwargs):
            batch, length, width = q.shape
            out = F.scaled_dot_product_attention(*[a.reshape(batch,length,heads,-1).transpose(1,2) for a in (q,k,v)])
            return out.transpose(1,2).reshape(batch,length,width)
        ns.update(ck=ck, attention_function=attention,
                  weights_manual_cast=lambda module,x:(module.weight,None,None),
                  main_stream_worker=lambda *a:nullcontext(),
                  linear_input_act=lambda layer,x,act:layer((4*torch.tanh(x.chunk(2,-1)[1]/4)*torch.sigmoid(x.chunk(2,-1)[1]))*(25*torch.tanh(x.chunk(2,-1)[0]/25))))
        tree = ast.parse((ROOT/'backend/nn/moss_image.py').read_text())
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))],type_ignores=[]),'moss-cpu','exec'),ns)
        model = ns['MOSSImage'](dim=32,ffn_dim=48,num_heads=2,num_layers=2,text_dim=24,latent_channels=8)
        options = {'patches_replace':{'dit':{('single_block',1):Mock(side_effect=lambda args, extra:args)}}}
        with torch.inference_mode():
            result = model(torch.randn(2,8,2,3),torch.ones(2),torch.randn(2,5,24),transformer_options=options)
        self.assertEqual(result.shape,(2,8,2,3))
        self.assertTrue(torch.isfinite(result).all())
        self.assertNotIn('block_index',options)
        options['patches_replace']['dit'][('single_block',1)].assert_called_once()

    def test_tokenizer_archive_matches_pinned_hash(self):
        folder=ROOT/'backend/huggingface'
        raw=lzma.decompress((folder/'moss.tokenizer.json.xz').read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),'0a04a9fd30a9fd98e9ce0328986f698c66329c3e4367bed61586f2be0a9b2bec')
        self.assertIsInstance(json.loads(raw),dict)


class CLITests(unittest.TestCase):
    def test_legacy_onnx_option_remains_parseable(self):
        import subprocess, sys
        result=subprocess.check_output([sys.executable,'-c','from backend.args import args; print(args.onnxruntime_gpu)','--onnxruntime-gpu'],text=True)
        self.assertEqual(result.strip(),'True')

    def test_cpu_never_claims_gpu_quantization_kernels(self):
        ns = load_nodes('backend/memory_management.py',
                        ['supports_fp8_compute','supports_nvfp4_compute','supports_mxfp8_compute','supports_int8_compute'],
                        {'torch':torch,'is_device_cpu':lambda d:d.type=='cpu','SUPPORT_FP8_OPS':True,
                         'is_nvidia':Mock(side_effect=AssertionError('CPU must return before GPU checks'))})
        for name in ('supports_fp8_compute','supports_nvfp4_compute','supports_mxfp8_compute','supports_int8_compute'):
            self.assertFalse(ns[name](torch.device('cpu')))

    def test_numeric_and_scheduled_lora_strengths(self):
        networks=types.SimpleNamespace(load_networks=Mock(),available_networks={})
        shared=types.SimpleNamespace(opts=types.SimpleNamespace(sd_lora='None',lora_add_hashes_to_infotext=False))
        ns=load_nodes('extensions-builtin/sd_forge_lora/extra_networks_lora.py',['activate'],
                      {'networks':networks,'shared':shared},class_name='ExtraNetworkLora')
        for value,expected in [(0.5,0.5),('0.7',0.7),('0.5@0.2',0.0)]:
            params=types.SimpleNamespace(items=['test',value],positional=['test',value],named={})
            ns['activate'](types.SimpleNamespace(errors={}),object(),[params])
            networks.load_networks.assert_called_with(['test'],[expected],[expected])


class SLGTests(unittest.TestCase):
    def test_sampling_dispatch_respects_callback_order(self):
        seen=[]
        first=types.SimpleNamespace(args_from=0,args_to=0,filename='first',process_before_every_sampling=lambda *a,**k:seen.append('first'))
        second=types.SimpleNamespace(args_from=0,args_to=0,filename='second',process_before_every_sampling=lambda *a,**k:seen.append('second'))
        runner=types.SimpleNamespace(alwayson_scripts=[first,second],ordered_scripts=lambda name:[second,first])
        ns=load_nodes('modules/scripts.py',['process_before_every_sampling'],
                      {'ScriptAbortError':RuntimeError,'errors':Mock()},class_name='ScriptRunner')
        ns['process_before_every_sampling'](runner,types.SimpleNamespace(script_args=[]))
        self.assertEqual(seen,['second','first'])

    def test_sigma_window_cfg_one_and_target_layer_patches(self):
        def replace(options,fn,kind,block,index):
            return {**options,'skipped':options.get('skipped',[])+[index]}
        forward=Mock(return_value=('cond','uncond'))
        ns=load_nodes('modules/processing_scripts/slg.py',['make_calc_cond_batch_function'],
                      {'set_model_options_patch_replace':replace,'_calc_cond_uncond_batch':forward},class_name='SkipLayerGuidance')
        callback=ns['make_calc_cond_batch_function']([], [22,24], 1.0, 0.5)
        options={'keep':True}
        self.assertEqual(callback(None,['positive'],['negative'],None,torch.tensor([0.75]),options),['cond','uncond'])
        self.assertEqual(forward.call_count,2)
        self.assertEqual(forward.call_args.args[-1]['skipped'],[22,24])
        self.assertNotIn('skipped',options)
        for sigma,uncond in [(0.1,['negative']),(0.75,None)]:
            forward.reset_mock()
            callback(None,['positive'],uncond,None,torch.tensor([sigma]),options)
            forward.assert_called_once()
            self.assertIs(forward.call_args.args[-1],options)

    def test_slg_leaves_non_moss_models_untouched(self):
        ns=load_nodes('modules/processing_scripts/slg.py',['process_before_every_sampling'],
                      {'dynamic_args':types.SimpleNamespace(moss=False),'opts':types.SimpleNamespace(slg_enable=True)},class_name='SkipLayerGuidance')
        # A model-less request must return without touching its UNet.
        self.assertIsNone(ns['process_before_every_sampling'](None,object()))
