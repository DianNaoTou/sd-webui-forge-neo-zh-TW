"""Reference retry, GGUF CPU numerics and real extension UI integration checks."""
import html
import json
import logging
import os
from pathlib import Path
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

import gradio as gr
import numpy as np
import torch
from PIL import Image

from test_vae_flash_regressions import load_nodes

ROOT = Path(__file__).resolve().parents[1]
STITCH = 'extensions-builtin/sd_forge_image_stitch/scripts/image_stitch.py'


class ReferenceTests(unittest.TestCase):
    def stitch(self, encode):
        class Processing:
            pass
        dynamic = types.SimpleNamespace(can_reference=lambda: True, wan=False, is_referencing=False)
        opts = types.SimpleNamespace(reference_max_side=1024, klein_do_reference=True,
            qwen21_do_reference=True, anima_do_reference=False, krea2_do_reference=False,
            moss_do_reference='disabled', img2img_background_color='white')
        ns = dict(torch=torch, np=np, Image=Image, StableDiffusionProcessing=Processing,
            StableDiffusionProcessingTxt2Img=Processing, dynamic_args=dynamic, opts=opts,
            sd_models=types.SimpleNamespace(model_data=types.SimpleNamespace(forge_loading_parameters={})),
            images=types.SimpleNamespace(flatten=lambda im, bg:im), device='cpu',
            images_tensor_to_samples=encode, logger=logging.getLogger(__name__))
        load_nodes(STITCH, ['process','reset_references','extract_images','hash_image'],ns,class_name='ImageStitch')
        cls=type('ImageStitch',(),{name:staticmethod(ns[name]) if name != 'process' else ns[name]
             for name in ['process','reset_references','extract_images','hash_image']})
        cls.cached_parameters=None
        ns['ImageStitch']=cls
        model=types.SimpleNamespace(clear_references=Mock())
        p=Processing(); p.sd_model=model; p.clear_prompt_cache=Mock()
        return cls(),p,dynamic

    def test_failed_second_reference_clears_partial_work_and_identical_retry_encodes(self):
        encode=Mock(side_effect=[None,RuntimeError('encoding failed'),None,None])
        stitch,p,dynamic=self.stitch(encode)
        refs=[(Image.new('RGB',(8,8),'red'),None),(Image.new('RGB',(8,8),'blue'),None)]
        with self.assertRaisesRegex(RuntimeError,'encoding failed'):
            stitch.process(p,True,refs)
        self.assertFalse(dynamic.is_referencing)
        self.assertIsNone(type(stitch).cached_parameters)
        self.assertEqual(p.sd_model.clear_references.call_count,2)
        stitch.process(p,True,refs)
        self.assertEqual(encode.call_count,4)
        stitch.process(p,True,refs)
        self.assertEqual(encode.call_count,4)  # successful identical work is cached
        stitch.process(p,False,refs)
        self.assertEqual(p.sd_model.clear_references.call_count,4)

    def test_color_and_alpha_changes_do_not_share_grayscale_cache(self):
        stitch,_,_=self.stitch(Mock())
        a,b=Image.new('RGB',(2,2),(255,0,0)),Image.new('RGB',(2,2),(0,130,0))
        self.assertEqual(a.convert('L').tobytes(),b.convert('L').tobytes())
        self.assertNotEqual(stitch.hash_image(a),stitch.hash_image(b))
        a=Image.new('RGBA',(2,2),(255,0,0,0));b=Image.new('RGBA',(2,2),(255,0,0,255))
        self.assertNotEqual(stitch.hash_image(a),stitch.hash_image(b))

    def test_reference_resize_rounds_dimensions_and_preserves_unchanged_tensor(self):
        from backend.misc.image_resize import resize_reference
        shared=types.ModuleType('modules.shared');shared.opts=types.SimpleNamespace(reference_max_side=0)
        with patch.dict('sys.modules',{'modules.shared':shared}):
            im=torch.rand(1,4,32,48)
            self.assertIs(resize_reference(im),im)
            self.assertEqual(resize_reference(torch.rand(1,3,1,2),32).shape,(1,3,32,32))
            self.assertEqual(resize_reference(torch.rand(1,3,128,128),16,64).shape,(1,3,64,64))


class GGUFTests(unittest.TestCase):
    def test_quantized_weights_match_independent_gguf_decoder_on_cpu(self):
        import gguf
        from backend.operations_gguf import ParameterGGUF,QUANTS_MAPPING,dequantize_tensor
        from gguf.constants import GGML_QUANT_SIZES
        rng=np.random.default_rng(529)
        for qtype in QUANTS_MAPPING:
            with self.subTest(qtype=qtype.name):
                block_size,type_size=GGML_QUANT_SIZES[qtype]
                raw=rng.integers(0,256,size=(2,type_size),dtype=np.uint8)
                # Keep random scale encodings finite, including IQ1_M's distributed scale.
                raw &= 0x7b
                half=np.array([0.03125,0.015625],dtype=np.float16).view(np.uint8)
                if qtype.name=='BF16':
                    raw=torch.randn(2,block_size).to(torch.bfloat16).view(torch.uint8).numpy()
                elif qtype.name in ('Q2_K',):raw[:,-4:]=half
                elif qtype.name in ('Q3_K','Q6_K'):raw[:,-2:]=half[:2]
                elif qtype.name in ('Q4_1','Q5_1','Q4_K','Q5_K'):raw[:,:4]=half
                elif qtype.name=='IQ1_M':
                    # d is assembled from the upper nibbles of four trailing words.
                    words=raw[:,-8:].copy().view(np.uint16)
                    d=int(np.array([0.03125],dtype=np.float16).view(np.uint16)[0])
                    for i in range(4):words[:,i]=(words[:,i]&0x0fff)|(((d>>(4*i))&15)<<12)
                    raw[:,-8:]=words.view(np.uint8)
                else:raw[:,:2]=half[:2]
                expected=gguf.quants.dequantize(raw,qtype).reshape(2,block_size)
                parameter=ParameterGGUF(torch.from_numpy(raw.copy()),tensor_type=qtype,tensor_shape=torch.Size((2,block_size)))
                parameter.computation_dtype=torch.float16
                parameter.gguf_cls.bake(parameter)
                actual=dequantize_tensor(parameter).float().numpy()
                np.testing.assert_allclose(actual,expected,rtol=0.025,atol=0.04,equal_nan=True)
                copied=parameter.to('cpu')
                self.assertEqual(copied.shape,parameter.shape)
                torch.testing.assert_close(dequantize_tensor(copied),dequantize_tensor(parameter))

    def test_non_cuda_inputs_do_not_load_triton(self):
        from backend.operations_gguf import dequantize_tensor
        tensor=torch.randn(2,2)
        with patch.dict('sys.modules',{'modules_forge.packages.gguf.kquant_triton':None}):
            self.assertIs(dequantize_tensor(tensor,input=tensor),tensor)


class ExtensionUITests(unittest.TestCase):
    def namespace(self):
        import modules
        ui=types.ModuleType('modules.ui')
        ui.wrap_gradio_call=lambda fn,**kw:fn
        ui.wrap_gradio_call_no_job=lambda fn,**kw:fn
        ui.create_refresh_button=Mock()
        shared=types.SimpleNamespace(cmd_opts=types.SimpleNamespace(disable_extension_access=False,
            disable_all_extensions=False,disable_extra_extensions=False),
            opts=types.SimpleNamespace(disable_all_extensions='none'),
            state=types.SimpleNamespace(nextjob=Mock()))
        ns=dict(translate=lambda text, **kwargs:text.format(**kwargs),gr=gr,shared=shared,os=os,html=html,json=json,time=time,threading=threading,
            datetime=__import__('datetime').datetime,timezone=__import__('datetime').timezone,
            extensions=types.SimpleNamespace(extensions=[]),
            config_states=types.SimpleNamespace(list_config_states=Mock(),all_config_states={}),
            restart=types.SimpleNamespace(is_restartable=lambda:True),
            wrap_gradio_gpu_call=lambda fn,**kw:fn,
            available_extensions={'extensions':[]},sort_ordering=[(False,lambda x:x['name'])],
            normalize_git_url=lambda url:url.removesuffix('.git'),
            get_extension_dirname_from_url=lambda url:url.rsplit('/',1)[-1],
            ThreadPoolExecutor=__import__('concurrent.futures',fromlist=['ThreadPoolExecutor']).ThreadPoolExecutor,
            errors=types.SimpleNamespace(report=Mock()))
        functions=['create_ui','check_access','check_updates','refresh_available_extensions',
            'refresh_available_extensions_from_data','refresh_available_extensions_for_tags','get_date',
            'install_extension_from_index']
        load_nodes('modules/ui_extensions.py',functions,ns)
        for name in ['apply_and_restart','extension_table','preload_extensions_git_metadata',
                     'install_extension_from_url','save_config_state','restore_config_state','update_config_states_table']:
            ns[name]=Mock()
        self.addCleanup(patch.stopall)
        patch.dict('sys.modules',{'modules.ui':ui}).start()
        patch.object(modules,'ui',ui,create=True).start()
        return ns

    def test_real_gradio_ui_installs_index_with_matching_output_count(self):
        ns=self.namespace()
        ui=ns['create_ui']()
        dep=next(x for x in ui.config['dependencies'] if x.get('api_name')=='install_extension_from_index')
        self.assertEqual(len(dep['outputs']),4)
        ns['install_extension_from_url'].return_value=('installed','ok')
        returned=ns['install_extension_from_index']('https://example.test/a',[],'hide','any',0,'',False)
        self.assertEqual(len(returned),len(dep['outputs']))
        apply=next(x for x in ui.config['dependencies'] if x.get('js')=='extensions_apply')
        self.assertNotEqual(apply['inputs'][0],apply['inputs'][1])

    def test_unknown_or_one_cpu_can_check_extension_updates(self):
        ns=self.namespace()
        for count in (None,1,2,128):
            with self.subTest(count=count),patch.object(os,'cpu_count',return_value=count):
                ns['check_updates']('task','[]')
        ns['shared'].cmd_opts.disable_extension_access=True
        with self.assertRaises(PermissionError):ns['check_updates']('task','[]')

    def test_new_year_boundary_and_default_hidden_tags_match_render(self):
        ns=self.namespace()
        ns['available_extensions']={'extensions':[
            {'name':'boundary','url':'https://example.test/boundary','commit_time':'2025-01-01T01:00:00Z','tags':[]},
            {'name':'old','url':'https://example.test/old','commit_time':'2024-12-31T01:00:00Z','tags':[]},
            {'name':'advert','url':'https://example.test/advert','commit_time':'2026-01-01T01:00:00Z','tags':['ads']}], 'tags':{}}
        code,_=ns['refresh_available_extensions_from_data']([], 'hide','any',0,None,True)
        self.assertIn('>boundary<',code);self.assertNotIn('>old<',code)
        import io,urllib.request
        data=json.dumps(ns['available_extensions']).encode()
        with patch.object(urllib.request,'urlopen',return_value=io.BytesIO(data)):
            result=ns['refresh_available_extensions']('https://example.test/index',[],'hide','any',0,'',True)
        self.assertNotIn('>advert<',result[1]);self.assertIn('ads',result[2]['value'])


if __name__=='__main__':unittest.main()
