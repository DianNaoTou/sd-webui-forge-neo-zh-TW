"""Optional Intel GPU path: isolation/fallback contracts plus real CPU numerics.

GPU availability/performance is deliberately not claimed by CPU CI.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backend = load_file('wd14_gpu_test_backend', 'extensions/stable-diffusion-webui-tagger-fork/tagger/openvino_backend.py')
worker = load_file('wd14_gpu_test_worker', 'tools/wd14_openvino_worker.py')
launcher = load_file('wd14_gpu_test_launcher', 'tools/launch_wd14_gpu.py')
HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ('onnx', 'openvino', 'onnxruntime'))


class DeviceSelectionTests(unittest.TestCase):
    def test_gpu_only_and_discrete_selection(self):
        core = Mock(available_devices=['CPU', 'GPU.0', 'GPU.1'])
        core.get_property.side_effect = lambda device, key: 'DISCRETE' if device == 'GPU.1' else 'INTEGRATED'
        self.assertEqual(worker.select_device(core, 'GPU'), 'GPU.1')
        self.assertEqual(worker.select_device(core, 'GPU.0'), 'GPU.0')
        for device in ('CPU', 'AUTO', 'HETERO:GPU,CPU', 'GPU;CPU'):
            with self.assertRaises(ValueError):
                worker.select_device(core, device)
        with self.assertRaises(RuntimeError):
            worker.select_device(Mock(available_devices=['CPU']), 'GPU')
        with self.assertRaises(RuntimeError):
            worker.select_device(core, 'GPU.7')
        core.get_property.return_value = 'DISCRETE'; core.get_property.side_effect = None
        with self.assertRaises(RuntimeError):
            worker.select_device(core, 'GPU')

    def test_worker_environment_does_not_inherit_main_python_or_pip_configuration(self):
        base = {'PYTHONPATH': 'other-venv', 'VIRTUAL_ENV': 'main', 'PIP_TARGET': 'main',
                'UV_PROJECT_ENVIRONMENT': 'main', 'CONDA_PREFIX': 'main', 'PATH': 'drivers'}
        for make in (backend.worker_environment, launcher.environment):
            env = make(base)
            for key in base.keys() - {'PATH'}:
                self.assertNotIn(key, env)
            self.assertEqual(env['PYTHONNOUSERSITE'], '1')
            self.assertEqual(env['PATH'], 'drivers')
        self.assertEqual(base['PIP_TARGET'], 'main')

    def test_single_intel_launcher_enables_worker_and_retires_duplicate_entry(self):
        source = (ROOT / 'webui-user-intel-arc.bat').read_text(encoding='utf-8')
        self.assertIn('tools\\launch_wd14_gpu.py %*', source)
        self.assertIn('call bootstrap-python.bat', source)
        self.assertIn('setlocal', source)
        self.assertNotIn('FORGE_WD14_OPENVINO=', source)
        self.assertFalse((ROOT / 'webui-user-intel-arc-wd14-gpu.bat').exists())

    def test_launcher_delegates_to_official_profile_with_gpu_or_setup_fallback(self):
        for setup_error in (None, RuntimeError('offline')):
            with self.subTest(setup_error=setup_error), tempfile.TemporaryDirectory() as tmp:
                state = Path(tmp)
                fake_os = types.SimpleNamespace(name='nt', environ={'KEEP': 'value'})
                lock = types.SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=Mock())
                with patch.object(launcher, 'os', fake_os), \
                     patch.object(launcher, 'worker_state', return_value=state), \
                     patch.object(launcher, 'prepare', side_effect=setup_error) as prepare, \
                     patch.object(launcher.subprocess, 'call', return_value=7) as call, \
                     patch.dict(sys.modules, {'msvcrt': lock}), \
                     patch.object(sys, 'argv', ['launcher', '--wd14-device', 'GPU.1', '--model-ref', 'my-models']):
                    self.assertEqual(launcher.main(), 7)
                prepare.assert_called_once_with(ROOT)
                args = call.call_args.args[0]
                self.assertEqual(args, [sys.executable, str(ROOT / 'tools/launch_xpu_profile.py'),
                                        'official', '--model-ref', 'my-models'])
                env = call.call_args.kwargs['env']
                self.assertEqual(env['KEEP'], 'value')
                self.assertEqual(env['FORGE_WD14_OPENVINO'], '0' if setup_error else '1')
                if not setup_error:
                    self.assertEqual(env['FORGE_WD14_OV_DEVICE'], 'GPU.1')
                self.assertEqual(call.call_args.kwargs['cwd'], ROOT)

    def test_only_opt_in_tagger_runtime_uses_worker_and_cpu_device_override_wins(self):
        tree = ast.parse((ROOT / 'extensions/stable-diffusion-webui-tagger-fork/tagger/interrogator.py').read_text(encoding='utf-8'))
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_onnxrt')
        ort = types.ModuleType('onnxruntime')
        fake = types.ModuleType('tagger.openvino_backend'); fake.OpenVinoRuntime = lambda module: ('gpu', module)
        ns = dict(os=os, torch=types.SimpleNamespace(version=types.SimpleNamespace(cuda=None)),
                  shared=types.SimpleNamespace(cmd_opts=types.SimpleNamespace(additional_device_ids=None)), use_cpu=False)
        exec(compile(ast.Module(body=[func], type_ignores=[]), '<tagger runtime>', 'exec'), ns)
        with patch.dict(sys.modules, {'onnxruntime': ort, 'tagger.openvino_backend': fake}):
            with patch.dict(os.environ, {'FORGE_WD14_OPENVINO': '0'}):
                self.assertIs(ns['get_onnxrt'](), ort)
            with patch.dict(os.environ, {'FORGE_WD14_OPENVINO': '1'}):
                self.assertEqual(ns['get_onnxrt'](), ('gpu', ort))
                ns['shared'].cmd_opts.additional_device_ids = 'cpu:0'
                self.assertIs(ns['get_onnxrt'](), ort)

    def test_load_failure_creates_cpu_session_with_no_gpu_provider_options(self):
        ort = Mock(); cpu = ort.InferenceSession.return_value
        with patch.object(backend, 'GpuSession', side_effect=RuntimeError('no Intel GPU')):
            session = backend.FallbackSession(ort, 'cached-model.onnx', providers=['CUDAExecutionProvider'], provider_options=[{}])
        self.assertEqual(session.get_providers(), cpu.get_providers.return_value)
        ort.InferenceSession.assert_called_once_with('cached-model.onnx', providers=['CPUExecutionProvider'])
        session.run(['tags'], {'input': 'unchanged'})
        cpu.run.assert_called_once_with(['tags'], {'input': 'unchanged'})

    def test_runtime_failure_retries_current_input_on_cpu_once_then_stays_cpu(self):
        class Broken:
            def run(self, *args, **kwargs):
                raise RuntimeError('GPU device lost')
            def close(self):
                self.closed = True
        ort = Mock(); ort.InferenceSession.return_value.run.return_value = ['cpu-result']
        with patch.object(backend, 'GpuSession', Broken):
            with patch.object(Broken, '__init__', return_value=None):
                session = backend.FallbackSession(ort, 'model.onnx')
            active = session.model
            self.assertEqual(session.run(['tags'], {'input': 'same-input'}), ['cpu-result'])
            self.assertTrue(active.closed)
            self.assertEqual(session.run(['tags'], {'input': 'next-input'}), ['cpu-result'])
        ort.InferenceSession.assert_called_once()
        self.assertEqual(ort.InferenceSession.return_value.run.call_count, 2)

    def test_installer_rejects_redirected_environment_before_installing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); (root/'runtime').mkdir(); (root/'elsewhere').mkdir()
            try:
                (root/'runtime/wd14-openvino').symlink_to(root/'elsewhere', target_is_directory=True)
            except OSError:
                self.skipTest('Symlink privilege unavailable')
            with patch.object(launcher.subprocess, 'run') as run:
                with self.assertRaises(RuntimeError):
                    launcher.prepare(root)
                run.assert_not_called()

    def test_installer_checks_owned_prefix_and_leaves_main_venv_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); venv = root/'runtime/wd14-openvino/venv'
            executable = venv/'Scripts/python.exe'
            executable.parent.mkdir(parents=True); executable.touch()
            probe = dict(prefix=str(venv), base_prefix='bootstrap',
                         packages={'openvino':launcher.OPENVINO, 'numpy':launcher.NUMPY})
            report = dict(openvino=launcher.OPENVINO, devices={'GPU.0':'Intel test device'})
            replies = [types.SimpleNamespace(returncode=0,stdout=json.dumps(probe)),
                       types.SimpleNamespace(returncode=0,stdout=json.dumps(report))]
            with patch.object(launcher.subprocess, 'run', side_effect=replies) as run:
                self.assertEqual(launcher.prepare(root), executable)
            self.assertTrue(all(call.args[0][0] == str(executable) for call in run.call_args_list))
            self.assertFalse(any('pip' in call.args[0] for call in run.call_args_list))
            probe['prefix'] = 'main-venv'
            with patch.object(launcher.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0,stdout=json.dumps(probe))) as run:
                with self.assertRaises(RuntimeError):launcher.prepare(root)
                self.assertEqual(run.call_count, 1)

    def test_engine_refuses_silent_cpu_execution(self):
        core = Mock(available_devices=['GPU.0'])
        core.read_model.return_value.inputs = ['input']
        core.read_model.return_value.outputs = ['output']
        core.compile_model.return_value.get_property.return_value = ['CPU']
        fake = types.SimpleNamespace(Core=lambda:core)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {'openvino':fake}), patch.object(worker, 'port_info',return_value=dict(name='pixels',shape=[1,3,2,2],type='tensor(float)')):
                with self.assertRaisesRegex(RuntimeError, 'Expected GPU execution'):
                    worker.Engine('model.onnx','GPU',Path(directory))
        self.assertEqual(core.compile_model.call_args.args[1],'GPU.0')
        self.assertEqual(core.compile_model.call_args.args[2]['INFERENCE_PRECISION_HINT'],'f32')


@unittest.skipUnless(HAS_RUNTIME, 'Real worker runtime is required in the dedicated CI job')
class RealOpenVinoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve()
        import onnx
        from onnx import helper, TensorProto
        inputs = helper.make_tensor_value_info('pixels', TensorProto.FLOAT, [1, 3, 2, 2])
        outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 3]) for name in ('scores', 'logits')]
        nodes = [helper.make_node('ReduceMean', ['pixels'], ['logits'], axes=[2, 3], keepdims=0),
                 helper.make_node('Sigmoid', ['logits'], ['scores'])]
        model = helper.make_model(helper.make_graph(nodes, 'tiny-tagger', [inputs], outputs),
                                  opset_imports=[helper.make_opsetid('', 17)])
        model.ir_version = 9
        self.model_path = self.path/'tiny.onnx'; onnx.save(model, self.model_path)

    def test_real_engine_array_path_preserves_names_output_order_and_matches_ort_cpu(self):
        import openvino as ov
        import onnxruntime as ort
        # Exercise production Engine.run using the real CPU plugin. GPU selection
        # and the GPU-only guard are checked separately; this is not a GPU test.
        engine = worker.Engine.__new__(worker.Engine)
        core = ov.Core(); model = core.read_model(str(self.model_path))
        engine.inputs = [worker.port_info(p) for p in model.inputs]
        engine.outputs = [worker.port_info(p) for p in model.outputs]
        engine.compiled = core.compile_model(model, 'CPU', {'INFERENCE_PRECISION_HINT': 'f32'})
        engine.request = engine.compiled.create_infer_request()
        values = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2) / 12
        np.save(self.path/'input.npy', values, allow_pickle=False)
        names = ['logits', 'scores']
        engine.run(self.path, names)
        expected = ort.InferenceSession(str(self.model_path), providers=['CPUExecutionProvider']).run(names, {'pixels': values})
        with np.load(self.path/'output.npz', allow_pickle=False) as actual:
            for i, result in enumerate(expected):
                np.testing.assert_allclose(actual[f'output_{i}'], result, rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError):
            engine.run(self.path, ['not-an-output'])
        np.save(self.path/'input.npy', np.zeros((1, 3, 3, 3), np.float32))
        with self.assertRaises(ValueError):
            engine.run(self.path, names)

    def test_real_no_gpu_worker_exits_cleans_up_and_falls_back_to_cpu(self):
        import openvino as ov
        import onnxruntime as ort
        if any(d.startswith('GPU') for d in ov.Core().available_devices):
            self.skipTest('This regression specifically exercises a machine without Intel GPU')
        root = self.path/'repo'; (root/'tools').mkdir(parents=True)
        shutil.copyfile(ROOT/'tools/wd14_openvino_worker.py', root/'tools/wd14_openvino_worker.py')
        original = backend.GpuSession
        with patch.object(backend, 'GpuSession', side_effect=lambda path: original(path, root=root, python=sys.executable, timeout=60)):
            session = backend.FallbackSession(ort, str(self.model_path))
        self.assertEqual(session.get_providers(), ['CPUExecutionProvider'])
        values = np.zeros((1, 3, 2, 2), np.float32)
        np.testing.assert_allclose(session.run(['scores'], {'pixels': values})[0], 0.5)
        self.assertFalse(list((root/'runtime/wd14-openvino/io').iterdir()))
        self.assertFalse(backend.ACTIVE)


class WorkerTransportTests(unittest.TestCase):
    def test_successful_worker_roundtrip_uses_current_arrays_and_unload_cleans_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); (root/'tools').mkdir()
            # Controlled subprocess stands in for GPU hardware, exercising the
            # actual parent IPC, current-input handling and lifecycle on both OSes.
            (root/'tools/wd14_openvino_worker.py').write_text('''
import argparse, json, sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--directory');p.add_argument('--device');p.add_argument('--cache');a=p.parse_args()
d=Path(a.directory)
for line in sys.stdin:
    request=json.loads(line)
    if request['op']=='load':
        response=dict(ok=True,inputs=[dict(name='pixels',shape=[1,3,2,2],type='tensor(float)')],outputs=[dict(name='scores',shape=[1,3,2,2],type='tensor(float)')],device='GPU.0',device_name='test fixture',execution_devices=['GPU.0'])
    else:
        values=np.load(d/'input.npy',allow_pickle=False)
        np.savez(d/'output.npz',output_0=values*2)
        response=dict(ok=True)
    print(json.dumps(response),flush=True)
''', encoding='utf-8')
            session = backend.GpuSession('fixture.onnx', root=root, python=sys.executable, timeout=30)
            self.addCleanup(session.close)
            process = session._process; data_directory = session.directory
            self.assertEqual(session.get_inputs()[0].name, 'pixels')
            for value in (1, 7):
                data = np.full((1, 3, 2, 2), value, np.float32)
                np.testing.assert_array_equal(session.run(['scores'], {'pixels': data})[0], data*2)
                self.assertFalse((data_directory/'input.npy').exists())
                self.assertFalse((data_directory/'output.npz').exists())
            session.close(); session.close()
            self.assertIsNotNone(process.poll())
            self.assertFalse(data_directory.exists())
            self.assertFalse(backend.ACTIVE)

    def test_unresponsive_worker_is_terminated_instead_of_leaking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); (root/'tools').mkdir()
            (root/'tools/wd14_openvino_worker.py').write_text('import time; time.sleep(60)\n', encoding='utf-8')
            with self.assertRaises(TimeoutError):
                backend.GpuSession('fixture.onnx', root=root, python=sys.executable, timeout=0.05)
            self.assertFalse(list((root/'runtime/wd14-openvino/io').iterdir()))
            self.assertFalse(backend.ACTIVE)


if __name__ == '__main__':
    unittest.main()
