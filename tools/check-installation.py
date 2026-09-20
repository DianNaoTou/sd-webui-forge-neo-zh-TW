"""Read-only installation audit; run with the hardware-specific venv's Python."""
import argparse
import importlib.metadata as metadata
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.dependency_utils import requirements_met, torch_backend, onnx_requirement, requirement_met


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True, choices=('xpu', 'cuda'))
    args = parser.parse_args()
    print(f'Python: {sys.version}\nInterpreter: {sys.executable}', flush=True)
    failed = subprocess.run([sys.executable, '-m', 'pip', 'check']).returncode != 0
    for package in ('torch', 'torchvision', 'comfy-kitchen', 'gradio', 'Pillow',
                    'pillow-heif', 'protobuf', 'transformers', 'huggingface-hub',
                    'onnx', 'onnxruntime', 'onnxruntime-gpu', 'tensorflow',
                    'ultralytics', 'mediapipe'):
        try:
            print(f'{package}: {metadata.version(package)}')
        except metadata.PackageNotFoundError:
            print(f'{package}: not installed')
    for relative in ('requirements.txt', 'extensions/ADetailer-Neo/requirements.txt',
                     'extensions/stable-diffusion-webui-tagger-fork/requirements.txt',
                     'extensions-builtin/forge_legacy_preprocessors/requirements.txt'):
        ok = requirements_met(ROOT / relative)
        print(f'{relative}: {"OK" if ok else "MISMATCH"}')
        failed |= not ok
    actual = torch_backend()
    print(f'Expected backend: {args.backend}; installed backend: {actual}')
    failed |= actual != args.backend
    import torch
    device = torch.xpu if args.backend == 'xpu' else torch.cuda
    available = device.is_available()
    print(f'GPU available: {available}')
    if available:
        print(f'GPU: {device.get_device_name(0)}')
    failed |= not available
    nvidia = sorted(dist.metadata['Name'] for dist in metadata.distributions()
                    if dist.metadata.get('Name', '').lower().startswith('nvidia-'))
    print(f'NVIDIA distributions: {nvidia}')
    if args.backend == 'xpu':
        failed |= bool(nvidia)
    wrong_ort = 'onnxruntime-gpu' if args.backend == 'xpu' else 'onnxruntime'
    failed |= requirement_met(wrong_ort)
    failed |= not requirement_met(onnx_requirement(args.backend))
    import onnxruntime as ort
    print(f'ONNX providers: {ort.get_available_providers()}')
    # Registration is not proof that a model session can load CUDA DLLs.
    print('Provider listing is not an ONNX inference test.')
    print('RESULT: ' + ('FAIL' if failed else 'PASS'))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
