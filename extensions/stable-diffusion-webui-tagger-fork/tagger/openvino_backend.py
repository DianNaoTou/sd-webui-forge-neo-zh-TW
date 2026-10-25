"""Optional ONNX-session facade backed by an isolated OpenVINO GPU process.

Ordinary launchers never import OpenVINO or start this worker. Model-specific
preprocessing, tag metadata, thresholds and output selection stay in the tagger.
"""
import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import weakref

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = weakref.WeakSet()


def worker_environment(base=None):
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key.startswith(('PIP_', 'UV_')) or key in (
            'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'PYTHONUSERBASE',
        ):
            env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONIOENCODING='utf-8', PIP_CONFIG_FILE=os.devnull)
    return env


def read_replies(stream, replies):
    try:
        for line in stream:
            replies.put(line)
    finally:
        replies.put(None)


class GpuSession:
    def __init__(self, model_path, *, root=ROOT, device=None, python=None, timeout=600):
        self._lock = threading.RLock()
        self._process = None
        self._temporary = None
        self._timeout = timeout
        state = Path(root) / 'runtime' / 'wd14-openvino'
        python = python or state / 'venv' / 'Scripts' / 'python.exe'
        if not Path(python).is_file():
            raise RuntimeError('WD14 OpenVINO environment is missing; use the GPU launcher')
        device = device or os.environ.get('FORGE_WD14_OV_DEVICE', 'GPU')
        io_dir = state / 'io'
        io_dir.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix='session-', dir=io_dir)
        self.directory = Path(self._temporary.name)
        ACTIVE.add(self)
        try:
            print('[WD14 GPU] Loading cached ONNX model in the isolated worker; first GPU compilation may take several minutes.', flush=True)
            self._process = subprocess.Popen([
                str(python), '-u', str(Path(root) / 'tools' / 'wd14_openvino_worker.py'),
                '--directory', str(self.directory), '--device', device,
                '--cache', str(state / 'cache'),
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8',
                env=worker_environment(), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self._replies = queue.Queue()
            threading.Thread(target=read_replies, args=(self._process.stdout, self._replies), daemon=True).start()
            response = self._request(dict(op='load', model=str(Path(model_path).resolve())))
            self._inputs = [SimpleNamespace(**p) for p in response['inputs']]
            self._outputs = [SimpleNamespace(**p) for p in response['outputs']]
            self._provider = f"OpenVINO GPU worker: {response['device']} ({response['device_name']})"
            print(f'[WD14 GPU] {self._provider}; execution={response["execution_devices"]}', flush=True)
        except Exception:
            self.close()
            raise

    def _request(self, message, timeout=None):
        if self._process is None:
            raise RuntimeError('WD14 GPU worker is closed')
        self._process.stdin.write(json.dumps(message) + '\n')
        self._process.stdin.flush()
        try:
            line = self._replies.get(timeout=self._timeout if timeout is None else timeout)
        except queue.Empty as exc:
            raise TimeoutError('WD14 OpenVINO worker timed out') from exc
        if line is None:
            raise RuntimeError('WD14 OpenVINO worker exited unexpectedly')
        response = json.loads(line)
        if not response.get('ok'):
            raise RuntimeError(response.get('error', 'OpenVINO inference failed'))
        return response

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return [self._provider]

    def run(self, output_names, input_feed, *args, **kwargs):
        with self._lock:
            if set(input_feed) != {self._inputs[0].name}:
                raise ValueError('WD14 GPU input name mismatch')
            names = output_names or [p.name for p in self._outputs]
            try:
                np.save(self.directory / 'input.npy', input_feed[self._inputs[0].name], allow_pickle=False)
                self._request(dict(op='run', outputs=names), timeout=min(self._timeout, 120))
                with np.load(self.directory / 'output.npz', allow_pickle=False) as outputs:
                    return [outputs[f'output_{i}'].copy() for i in range(len(names))]
            finally:
                for name in ('input.npy', 'output.npz'):
                    (self.directory / name).unlink(missing_ok=True)

    def close(self):
        with self._lock:
            process, self._process = self._process, None
            if process is not None:
                if process.stdin:
                    try:
                        process.stdin.close()  # EOF also handles a closed WebUI.
                    except OSError:
                        pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                if process.stdout:
                    process.stdout.close()
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
            ACTIVE.discard(self)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FallbackSession:
    def __init__(self, ort, model_path, *args, **kwargs):
        self.ort, self.model_path = ort, model_path
        self.args, self.kwargs = args, kwargs
        self._lock = threading.RLock()
        self.model = None
        try:
            self.model = GpuSession(model_path)
        except Exception as exc:
            self._cpu(exc)

    def _cpu(self, reason):
        print(f'[WD14 GPU] OpenVINO failed; using CPU ONNX Runtime: {reason}', flush=True)
        if self.model is not None:
            self.model.close()
            self.model = None
        kwargs = dict(self.kwargs, providers=['CPUExecutionProvider'])
        kwargs.pop('provider_options', None)
        # Existing tagger call sites pass providers as a keyword, not positionally.
        self.model = self.ort.InferenceSession(self.model_path, *self.args, **kwargs)

    def get_inputs(self):
        return self.model.get_inputs()

    def get_outputs(self):
        return self.model.get_outputs()

    def get_providers(self):
        return self.model.get_providers()

    def run(self, output_names, input_feed, *args, **kwargs):
        with self._lock:
            try:
                return self.model.run(output_names, input_feed, *args, **kwargs)
            except Exception as exc:
                if not isinstance(self.model, GpuSession):
                    raise
                self._cpu(exc)
                return self.model.run(output_names, input_feed, *args, **kwargs)

    def close(self):
        with self._lock:
            if isinstance(self.model, GpuSession):
                self.model.close()
            self.model = None


class OpenVinoRuntime:
    def __init__(self, ort):
        self.ort = ort

    def __getattr__(self, name):
        return getattr(self.ort, name)

    def InferenceSession(self, model_path, *args, **kwargs):
        return FallbackSession(self.ort, model_path, *args, **kwargs)


def close_workers():
    for session in list(ACTIVE):
        session.close()


atexit.register(close_workers)
