"""Exercise production adapters and legacy orchestration without WebUI/GPU init."""
import ast
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import typing
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from huggingface_hub.errors import LocalEntryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
TAGGER = ROOT / 'extensions/stable-diffusion-webui-tagger-fork/tagger'


def load_adapters():
    shared = types.SimpleNamespace(opts=types.SimpleNamespace(tagger_verbose=False))
    ns = dict(vars(typing), Image=Image, Path=Path, shared=shared, tqdm=lambda x, **kw: x)
    tree = ast.parse((TAGGER/'interrogator.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Interrogator']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TAGGER/'interrogator.py'), 'exec'), ns)
    legacy = types.ModuleType('tagger.interrogator')
    legacy.Interrogator = ns['Interrogator']
    legacy.Its = types.SimpleNamespace(hf_cache='existing-cache')
    legacy.onnxrt_providers = ['CPUExecutionProvider']
    legacy.get_onnxrt = Mock()
    package = types.ModuleType('tagger')
    package.interrogator = legacy
    spec = importlib.util.spec_from_file_location('tagger.new_onnx', TAGGER/'new_onnx.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'tagger':package, 'tagger.interrogator':legacy,
                                 'modules':types.SimpleNamespace(shared=shared)}):
        spec.loader.exec_module(module)
    return module, legacy, ns


class WD14ModelTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.legacy, self.ns = load_adapters()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def session(self, adapter, scores=None):
        if scores is None:
            scores = np.zeros((1, adapter.count), dtype=np.float32)
        return Mock(get_inputs=Mock(return_value=[types.SimpleNamespace(name='input', type='tensor(float)', shape=['batch_size',3,adapter.size,adapter.size])]),
                    get_outputs=Mock(return_value=[types.SimpleNamespace(name=adapter.output_name, shape=['batch_size',adapter.count])]),
                    get_providers=Mock(return_value=['CPUExecutionProvider']),
                    run=Mock(return_value=[scores]))

    def ready(self, adapter):
        adapter.tags = [('tag'+str(i), 'general') for i in range(adapter.count)]
        adapter.model = self.session(adapter)
        return adapter

    def test_construction_and_registration_never_download_or_create_sessions(self):
        with patch.object(self.mod, 'hf_hub_download', side_effect=AssertionError('network')):
            for adapter in [self.mod.PixAIInterrogator(), self.mod.CamieInterrogator()]:
                self.assertIsNone(adapter.model)
                self.assertEqual(adapter.run_mode, 2)
                self.assertTrue(adapter.require_submit)
        self.legacy.get_onnxrt.assert_not_called()
        source = (TAGGER/'utils.py').read_text()
        self.assertIn("'pixai-tagger-v0.9': PixAIInterrogator()", source)
        self.assertIn("'camie-tagger-v2': CamieInterrogator()", source)

    def test_download_local_first_pinned_and_cache_hit_never_uses_network(self):
        adapter = self.mod.PixAIInterrogator()
        with patch.object(self.mod, 'hf_hub_download', return_value='cached') as download:
            self.assertEqual(adapter._download('model.onnx'), 'cached')
        self.assertEqual(download.call_count, 1)
        self.assertEqual(download.call_args.kwargs['cache_dir'], 'existing-cache')
        self.assertEqual(download.call_args.kwargs['revision'], adapter.revision)
        self.assertTrue(download.call_args.kwargs['local_files_only'])

    def test_cache_miss_downloads_once_and_load_reuses_session(self):
        adapter = self.mod.PixAIInterrogator()
        with patch.object(self.mod, 'hf_hub_download', side_effect=[LocalEntryNotFoundError('miss'),'downloaded']) as download:
            self.assertEqual(adapter._download('model.onnx'), 'downloaded')
            self.assertNotIn('local_files_only',download.call_args.kwargs)
        with patch.object(adapter, '_download', return_value='file') as download, patch.object(adapter, '_metadata', return_value=[('tag','general')]*adapter.count), patch.object(adapter, '_session', return_value=self.session(adapter)):
            adapter.load(); adapter.load()
            self.assertEqual(download.call_count,3)
            adapter.unload()
            self.assertIsNone(adapter.model)
            self.assertIsNone(adapter.tags)

    def test_failed_download_retries_and_other_model_remains_usable(self):
        adapter = self.mod.CamieInterrogator()
        other = self.ready(self.mod.PixAIInterrogator())
        with patch.object(adapter, '_download', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(self.mod.TaggerModelError,'下載或載入失敗'):
                adapter.load()
        self.assertIsNone(adapter.model)
        self.assertIsNone(adapter.tags)
        self.assertEqual(len(other.interrogate(Image.new('RGB',(3,2)))[1]), other.count)
        with patch.object(adapter, '_download', return_value='file'), patch.object(adapter, '_metadata', return_value=[]), patch.object(adapter, '_session', return_value=self.session(adapter)):
            adapter.load()
            self.assertIsNotNone(adapter.model)

    def test_invalid_session_never_leaves_half_loaded_model(self):
        adapter = self.mod.CamieInterrogator()
        session = self.session(adapter)
        session.get_outputs.return_value[0].shape = [1,256]
        with patch.object(adapter, '_download', return_value='file'), patch.object(adapter, '_metadata', return_value=[]), patch.object(adapter, '_session', return_value=session):
            with self.assertRaises(self.mod.TaggerModelError): adapter.load()
        self.assertIsNone(adapter.model)
        self.assertIsNone(adapter.tags)

    def test_provider_selection_keeps_xpu_on_cpu_and_cuda_failure_falls_back(self):
        adapter = self.mod.CamieInterrogator()
        ort = Mock(get_available_providers=Mock(return_value=['CUDAExecutionProvider','CPUExecutionProvider']))
        self.legacy.get_onnxrt.return_value = ort
        adapter._session('model')
        self.assertEqual(ort.InferenceSession.call_args.kwargs['providers'], ['CPUExecutionProvider'])
        self.legacy.onnxrt_providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        ort.InferenceSession.side_effect = [RuntimeError('CUDA missing'), 'cpu session']
        self.assertEqual(adapter._session('model'), 'cpu session')
        self.assertEqual(ort.InferenceSession.call_args.kwargs['providers'], ['CPUExecutionProvider'])

    def test_pixai_rgb_direct_resize_normalization_and_alpha_white(self):
        adapter = self.mod.PixAIInterrogator()
        tensor = adapter.preprocess(Image.new('RGB',(1024,100),(255,0,128)))
        self.assertEqual(tensor.shape,(1,3,448,448))
        self.assertEqual(tensor.dtype,np.float32)
        np.testing.assert_allclose(tensor[0,:,0,0],[1,-1,128/255*2-1],atol=1e-6)
        np.testing.assert_allclose(adapter.preprocess(Image.new('RGBA',(2,3),(255,0,0,0))),1)

    def test_camie_rgb_letterbox_imagenet_no_bgr_or_wd14_scaling(self):
        tensor = self.mod.CamieInterrogator().preprocess(Image.new('RGB',(1024,512),(255,0,0)))
        self.assertEqual(tensor.shape,(1,3,512,512))
        mean,std=np.array([.485,.456,.406]),np.array([.229,.224,.225])
        np.testing.assert_allclose(tensor[0,:,0,0],(np.array([124,116,104])/255-mean)/std,atol=1e-6)
        np.testing.assert_allclose(tensor[0,:,256,256],(np.array([1,0,0])-mean)/std,atol=1e-6)
        self.assertTrue(np.isfinite(self.mod.CamieInterrogator().preprocess(Image.new('L',(1,100000)))).all())

    def test_pixai_first_four_tags_are_not_fabricated_ratings(self):
        adapter = self.ready(self.mod.PixAIInterrogator())
        adapter.model.run.return_value[0][0,:4] = [.9,.8,.7,.6]
        adapter.tags[0]=('1girl','general')
        adapter.tags[1]=('character_name','character')
        ratings,tags=adapter.interrogate(Image.new('RGB',(4,4)))
        self.assertEqual(ratings,{})
        self.assertAlmostEqual(tags['1girl'],.9,places=5)
        self.assertIn('character_name',tags)

    def test_camie_refined_full_vector_sigmoid_and_all_categories(self):
        adapter=self.ready(self.mod.CamieInterrogator())
        categories=['rating','general','character','copyright','artist','meta','year']
        for i,category in enumerate(categories): adapter.tags[i]=(category+'_name',category)
        adapter.model.run.return_value[0][0,:3]=[1000,0,-1000]
        ratings,tags=adapter.interrogate(Image.new('RGB',(4,4)))
        self.assertEqual(ratings,{'rating_name':1.0})
        self.assertEqual(tags['general_name'],.5)
        self.assertEqual(tags['character_name'],0)
        self.assertIn('artist_name',tags)
        self.assertEqual(adapter.model.run.call_args.args[0],['refined_predictions'])

    def test_nonfinite_wrong_shape_or_invalid_probability_unloads(self):
        for values in [np.array([[np.nan]]),np.zeros((1,2)),np.full((1,13461),1.1)]:
            adapter=self.ready(self.mod.PixAIInterrogator())
            adapter.model.run.return_value=[values]
            with self.assertRaises(self.mod.TaggerModelError): adapter.interrogate(Image.new('RGB',(2,2)))
            self.assertIsNone(adapter.model)

    def test_metadata_schema_and_category_mapping_reject_bad_indices(self):
        adapter=self.mod.CamieInterrogator()
        # A reduced fixture exercises the exact parser with every category.
        adapter.count=7
        categories=['rating','general','character','copyright','artist','meta','year']
        mapping={str(i):'tag'+str(i) for i in range(7)}
        data={'model_info':{'img_size':512},'dataset_info':{'total_tags':7,'tag_mapping':{'idx_to_tag':mapping,'tag_to_category':dict(zip(mapping.values(),categories))}}}
        path=self.path/'metadata.json';path.write_text(json.dumps(data))
        self.assertEqual([c for _,c in adapter._metadata({adapter.files[1]:path})],categories)
        del mapping['0'];path.write_text(json.dumps(data))
        with self.assertRaises(ValueError): adapter._metadata({adapter.files[1]:path})
        pix=self.mod.PixAIInterrogator();pix.count=2
        preprocess=self.path/'preprocess.json'
        preprocess.write_text(json.dumps({'stages':[{'type':'resize','size':[448,448],'interpolation':'bilinear'},{'type':'to_tensor'},{'type':'normalize','mean':[.5]*3,'std':[.5]*3}]}))
        csv=self.path/'tags.csv';csv.write_text('id,name,category\n0,1girl,0\n1,character,4\n')
        self.assertEqual(pix._metadata({'preprocess.json':preprocess,'selected_tags.csv':csv}),[('1girl','general'),('character','character')])
        csv.write_text('id,name,category\n1,1girl,0\n0,character,4\n')
        with self.assertRaises(ValueError): pix._metadata({'preprocess.json':preprocess,'selected_tags.csv':csv})

    def test_legacy_single_and_batch_paths_use_new_adapter_and_unload(self):
        adapter=self.ready(self.mod.PixAIInterrogator())
        qdata=types.SimpleNamespace(clear=Mock(),query={},in_db={},single_data=Mock(),image_dups=defaultdict(set),apply_filters=Mock(),finalize=Mock(return_value='single'),finalize_batch=Mock(return_value='batch'))
        iodata=types.SimpleNamespace(get_bytes_hash=lambda b:hashlib.sha256(b).hexdigest(),paths=[])
        self.ns.update(QData=qdata,IOData=iodata)
        adapter.interrogate_image(Image.new('RGB',(4,4)))
        self.assertEqual(qdata.apply_filters.call_count,1)
        qdata.apply_filters.reset_mock()
        for i,color in enumerate(['red','blue']):
            path=self.path/f'{i}.png';Image.new('RGB',(4,4),color).save(path)
            iodata.paths.append([path,'',''])
        self.legacy.Interrogator.input.update(large_query=True,unload_after=True)
        adapter.batch_interrogate()
        self.assertEqual(qdata.apply_filters.call_count,2)
        self.assertIsNone(adapter.model)
        self.assertEqual(self.legacy.Interrogator.output,'batch')

    def test_new_model_image_change_does_not_download_and_ui_escapes_failure(self):
        tree=ast.parse((TAGGER/'ui.py').read_text(encoding='utf-8-sig'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('on_interrogate_image','on_interrogate_image_submit','on_interrogate','search_filter')]
        adapter=self.mod.PixAIInterrogator()
        ns=dict(vars(typing),Image=Image,COMMON_OUTPUT=tuple,utils=types.SimpleNamespace(interrogators={'pix':adapter}),It=self.legacy.Interrogator,TAG_INPUTS=[],IOData=types.SimpleNamespace(update_input_glob=Mock(),err=[],update_output_dir=Mock()),QData=Mock(),TaggerModelError=self.mod.TaggerModelError,search_filter=Mock())
        from html import escape
        ns['html_esc']=escape
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(TAGGER/'ui.py'),'exec'),ns)
        with patch.object(adapter,'interrogate_image',side_effect=self.mod.TaggerModelError('<failed>')) as run:
            self.assertIn('首次',ns['on_interrogate_image'](Image.new('RGB',(1,1)),adapter.name,'')[6]);run.assert_not_called()
            self.assertEqual(ns['on_interrogate_image_submit'](Image.new('RGB',(1,1)),adapter.name,'')[6],'&lt;failed&gt;')
        with patch.object(adapter,'batch_interrogate',side_effect=self.mod.TaggerModelError('批次失敗')):
            self.assertEqual(ns['on_interrogate']('','',adapter.name,'')[6],'批次失敗')
        self.assertEqual(self.legacy.Interrogator.output,(None,None,None,'批次失敗'))
        self.assertEqual(ns['search_filter']('')[6],'批次失敗')

    def test_api_keeps_rating_separate_thresholds_and_reports_model_failure(self):
        tree=ast.parse((TAGGER/'api.py').read_text())
        api=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Api')
        node=next(n for n in api.body if isinstance(n,ast.FunctionDef) and n.name=='endpoint_interrogate')
        adapter=Mock(interrogate=Mock(return_value=({'rating_general':.8},{'keep':.9,'drop':.1})))
        class HTTPError(Exception):
            def __init__(self,status_code,detail): self.status_code,self.detail=status_code,detail
        ns=dict(vars(typing),models=types.SimpleNamespace(TaggerInterrogateRequest=object,TaggerInterrogateResponse=lambda **kw:kw),utils=types.SimpleNamespace(interrogators={'camie':adapter}),decode_base64_to_image=lambda x:x,HTTPException=HTTPError,TaggerModelError=self.mod.TaggerModelError)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(TAGGER/'api.py'),'exec'),ns)
        from contextlib import nullcontext
        owner=types.SimpleNamespace(queue_lock=nullcontext())
        request=types.SimpleNamespace(image='image',model='camie',queue='',name_in_queue='',threshold=.5)
        result=ns['endpoint_interrogate'](owner,request)
        self.assertEqual(result,{'caption':{'rating':{'rating_general':.8},'tag':{'keep':.9}}})
        adapter.interrogate.side_effect=self.mod.TaggerModelError('下載失敗')
        with self.assertRaises(HTTPError) as error: ns['endpoint_interrogate'](owner,request)
        self.assertEqual((error.exception.status_code,error.exception.detail),(503,'下載失敗'))

    def test_existing_wd14_adapter_and_prompt_send_paths_stay_intact(self):
        # Guard the existing BGR/0..255 adapter independently of new RGB loaders.
        source=(TAGGER/'interrogator.py').read_text()
        self.assertIn('image = image[:, :, ::-1]',source)
        self.assertIn('ratings = dict(tags[:4].values)',source)
        ui=(TAGGER/'ui.py').read_text(encoding='utf-8-sig')
        for target in ['txt2img','img2img']:
            self.assertIn("'"+target+"'",ui)
        self.assertIn('source_image_component=image',ui)
        self.assertIn('source_text_component=tags',ui)


if __name__ == '__main__': unittest.main()
