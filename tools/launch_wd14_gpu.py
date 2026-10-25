"""Opt-in Intel WD14 GPU launcher; installs only the private OpenVINO venv."""
import argparse
from importlib import util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OPENVINO = '2026.3.1'
NUMPY = '2.3.5'


def environment(base=None):
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key.startswith(('PIP_', 'UV_')) or key in (
            'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'PYTHONUSERBASE',
        ):
            env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONIOENCODING='utf-8', PIP_CONFIG_FILE=os.devnull)
    return env


def worker_state(root):
    root = Path(root).resolve()
    state = root / 'runtime' / 'wd14-openvino'
    venv = state / 'venv'
    for path in (root / 'runtime', state, venv):
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
            raise RuntimeError('WD14 worker environment must not redirect to another directory')
    return state


def prepare(root, python=sys.executable):
    state = worker_state(root)
    venv = state / 'venv'
    executable = venv / 'Scripts' / 'python.exe'
    env = environment()
    state.mkdir(parents=True, exist_ok=True)
    if not executable.is_file():
        subprocess.run([str(python), '-m', 'venv', str(venv)], env=env, check=True)
    check = subprocess.run([str(executable), '-c', '''
from importlib.metadata import version, PackageNotFoundError
import json, sys
packages = {}
for name in ('openvino', 'numpy'):
    try:
        packages[name] = version(name)
    except PackageNotFoundError:
        packages[name] = None
print(json.dumps(dict(prefix=sys.prefix, base_prefix=sys.base_prefix, packages=packages)))
'''],
        env=env, text=True, encoding='utf-8', capture_output=True)
    expected = {'openvino': OPENVINO, 'numpy': NUMPY}
    if check.returncode != 0:
        raise RuntimeError('Cannot inspect the private WD14 Python environment')
    installed = json.loads(check.stdout)
    if Path(installed['prefix']).resolve() != venv.resolve() or installed['prefix'] == installed['base_prefix']:
        raise RuntimeError('Refusing to install outside the private WD14 venv')
    if installed['packages'] != expected:
        subprocess.run([str(executable), '-m', 'pip', 'install', '--only-binary=:all:',
                        '--index-url', 'https://pypi.org/simple',
                        'openvino==' + OPENVINO, 'numpy==' + NUMPY], env=env, check=True)
    # Import/device discovery verifies the runtime, but model downloads remain lazy.
    report = subprocess.run([str(executable), '-c',
        'import openvino as ov, json; c=ov.Core(); '
        'print(json.dumps({"openvino":ov.__version__,"devices":'
        '{d:c.get_property(d,"FULL_DEVICE_NAME") for d in c.available_devices}}))'],
        env=env, text=True, encoding='utf-8', capture_output=True, check=True)
    (state / 'runtime-report.json').write_text(report.stdout, encoding='utf-8')
    print('[WD14 GPU] Isolated OpenVINO runtime: ' + report.stdout.strip(), flush=True)
    return executable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wd14-device', default='GPU', help='GPU auto-selects the sole discrete GPU; or specify GPU.0 / GPU.1')
    args, extra = parser.parse_known_args()
    if not re.fullmatch(r'GPU(?:\.\d+)?', args.wd14_device):
        raise ValueError('Use --wd14-device GPU or GPU.<index>')
    if os.name != 'nt':
        raise RuntimeError('This launcher targets Windows Intel Arc; no changes were made')
    # Validate existing profile options before creating either environment.
    spec = util.spec_from_file_location('wd14_launch_profiles', ROOT / 'modules' / 'xpu_profiles.py')
    profiles = util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(profiles)
    _, state = profiles.paths(ROOT, 'official')
    profiles.command_args(ROOT, state, 'official', extra)
    try:
        # The project bootstrap interpreter does not import the WebUI or Torch.
        # Serialize installations across multiple GPU-launcher invocations.
        lock = worker_state(ROOT) / 'setup.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        # Filelock is not required in the bootstrap interpreter: Windows byte lock.
        import msvcrt
        with lock.open('a+b') as stream:
            if stream.seek(0, 2) == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                prepare(ROOT)
            finally:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    except Exception as exc:
        print(f'[WD14 GPU] Setup unavailable; WD14 will use CPU: {exc}', flush=True)
        env = dict(os.environ, FORGE_WD14_OPENVINO='0')
    else:
        env = dict(os.environ, FORGE_WD14_OPENVINO='1', FORGE_WD14_OV_DEVICE=args.wd14_device)
    return subprocess.call([sys.executable, str(ROOT / 'tools' / 'launch_xpu_profile.py'),
                            'official', *extra], env=env, cwd=ROOT)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'[WD14 GPU] Launcher stopped: {exc}', file=sys.stderr)
        raise SystemExit(1)
