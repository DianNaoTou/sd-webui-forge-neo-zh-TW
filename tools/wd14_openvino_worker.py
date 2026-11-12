"""Private WD14 OpenVINO worker. No WebUI/Torch imports, network or listener.

Only control messages cross stdin/stdout. Arrays use a private session directory,
with pickle disabled. EOF closes the model when the parent exits.
"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import sys


def select_device(core, requested):
    if not re.fullmatch(r'GPU(?:\.\d+)?', requested):
        raise ValueError('WD14 device must be GPU or GPU.<index>')
    devices = [d for d in core.available_devices if d.startswith('GPU')]
    if not devices:
        raise RuntimeError('OpenVINO found no Intel GPU; check the Intel graphics driver')
    if requested != 'GPU':
        if requested not in devices:
            raise RuntimeError(f'{requested} is unavailable; found {devices}')
        return requested
    if len(devices) == 1:
        return devices[0]
    discrete = [d for d in devices if 'DISCRETE' in str(core.get_property(d, 'DEVICE_TYPE')).upper()]
    if len(discrete) == 1:
        return discrete[0]
    raise RuntimeError(f'Multiple GPUs found: {devices}; use --wd14-device GPU.<index>')


def port_info(port):
    types = {'f32': 'tensor(float)', 'f16': 'tensor(float16)', 'i64': 'tensor(int64)'}
    shape = [int(d.get_length()) if d.is_static else None for d in port.partial_shape]
    return dict(name=port.get_any_name(), shape=shape,
                type=types.get(port.element_type.get_type_name(), str(port.element_type)))


class Engine:
    def __init__(self, model_path, device, cache):
        import openvino as ov
        self.core = ov.Core()
        self.device = select_device(self.core, device)
        model = self.core.read_model(str(model_path))
        self.inputs = [port_info(p) for p in model.inputs]
        self.outputs = [port_info(p) for p in model.outputs]
        if len(self.inputs) != 1 or self.inputs[0]['type'] != 'tensor(float)':
            raise ValueError('WD14 OpenVINO requires one float32 image input')
        cache.mkdir(parents=True, exist_ok=True)
        self.compiled = self.core.compile_model(model, self.device, {
            'CACHE_DIR': str(cache), 'INFERENCE_PRECISION_HINT': 'f32',
            'PERFORMANCE_HINT': 'LATENCY',
        })
        self.execution_devices = list(self.compiled.get_property('EXECUTION_DEVICES'))
        if not self.execution_devices or any(not d.startswith('GPU') for d in self.execution_devices):
            raise RuntimeError(f'Expected GPU execution, got {self.execution_devices}')
        self.device_name = self.core.get_property(self.device, 'FULL_DEVICE_NAME')
        self.request = self.compiled.create_infer_request()

    def run(self, directory, output_names):
        import numpy as np
        known = [p['name'] for p in self.outputs]
        if not isinstance(output_names, list) or not output_names or any(n not in known for n in output_names):
            raise ValueError('Unknown output names')
        values = np.load(directory / 'input.npy', allow_pickle=False)
        if values.dtype != np.float32 or values.ndim != 4 or not np.isfinite(values).all():
            raise ValueError('Expected a finite float32 image batch')
        expected = self.inputs[0]['shape']
        if len(expected) != 4 or any(d is not None and d != size for d, size in zip(expected, values.shape)):
            raise ValueError(f'Input shape mismatch: {values.shape} vs {expected}')
        result = self.request.infer({self.compiled.input(0): values})
        arrays = [np.array(result[self.compiled.output(known.index(n))], copy=True) for n in output_names]
        if any(not np.isfinite(a).all() for a in arrays):
            raise ValueError('Non-finite OpenVINO output')
        np.savez(directory / 'output.npz', **{f'output_{i}': a for i, a in enumerate(arrays)})


def serve(directory, device, cache):
    protocol = sys.stdout
    engine = None

    def reply(value):
        protocol.write(json.dumps(value, ensure_ascii=True) + '\n')
        protocol.flush()

    # Third-party import/compile output must never enter the control protocol.
    with redirect_stdout(sys.stderr):
        for line in sys.stdin:
            try:
                message = json.loads(line)
                if message.get('op') == 'load' and engine is None:
                    engine = Engine(Path(message['model']), device, cache)
                    reply(dict(ok=True, inputs=engine.inputs, outputs=engine.outputs,
                               device=engine.device, device_name=engine.device_name,
                               execution_devices=engine.execution_devices))
                elif message.get('op') == 'run' and engine is not None:
                    engine.run(directory, message['outputs'])
                    reply(dict(ok=True))
                elif message.get('op') == 'close':
                    reply(dict(ok=True))
                    return
                else:
                    raise ValueError('Invalid worker operation')
            except Exception as exc:
                reply(dict(ok=False, error=f'{type(exc).__name__}: {exc}'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--device', default='GPU')
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    serve(args.directory.resolve(), args.device, args.cache.resolve())


if __name__ == '__main__':
    main()
