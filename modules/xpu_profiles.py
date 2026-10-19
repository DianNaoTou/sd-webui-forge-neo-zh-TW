"""Isolated, opt-in Intel XPU comparison profiles (no GPU imports at module load)."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TORCH = '2.14.1+xpu'
VISION = '0.29.1+xpu'
KITCHEN_REV = 'c2b77e7229bcb1917a5bf3145dc45cbe8fc389a9'
KERNEL_REV = '49d2f8d074397c91f138f933e7d7d6ff033f1581'
PROFILES = {'official': ('venv-xpu-official', 7862), 'kitchen': ('venv-xpu-kitchen', 7863)}


def paths(root, profile):
    if profile not in PROFILES:
        raise ValueError('Unknown XPU profile')
    root = Path(root).resolve()
    return root / PROFILES[profile][0], root / 'runtime' / 'xpu-ab' / profile


def initialize_settings(root, state):
    state.mkdir(parents=True, exist_ok=True)
    config = state / 'config.json'
    if not config.exists():
        source = Path(root) / 'config.json'
        data = json.loads(source.read_text(encoding='utf-8')) if source.exists() else {'localization': 'zh_Hant'}
        for name in ('outdir_samples', 'outdir_txt2img_samples', 'outdir_img2img_samples',
                     'outdir_extras_samples', 'outdir_grids', 'outdir_txt2img_grids',
                     'outdir_img2img_grids', 'outdir_save', 'outdir_init_images'):
            data[name] = str(state / 'outputs' / name.removeprefix('outdir_'))
        config.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    target = state / 'ui-config.json'
    if not target.exists():
        original = Path(root) / 'ui-config.json'
        target.write_bytes(original.read_bytes() if original.exists() else b'{}')
    styles = state / 'styles.csv'
    if not styles.exists():
        original = Path(root) / 'styles.csv'
        styles.write_bytes(original.read_bytes() if original.exists() else b'name,prompt,negative_prompt\n')


def initialize_extension_link(root, state):
    """Keep shipped extension source available under an isolated data directory."""
    source = (Path(root) / 'extensions').resolve()
    link = state / 'extensions'
    if not source.is_dir():
        raise RuntimeError('Bundled extensions directory is missing')
    if link.exists() or link.is_symlink():
        if link.resolve() != source or not link.is_junction():
            raise RuntimeError('Profile extensions path is not the owned source junction; leaving it untouched.')
        return
    import _winapi
    _winapi.CreateJunction(str(source), str(link))


def requirements_for(root, state, profile):
    state.mkdir(parents=True, exist_ok=True)
    original = Path(root) / 'requirements.txt'
    if profile == 'official':
        return original
    source = original.read_text(encoding='utf-8')
    if source.count('comfy-kitchen==0.2.38') != 1:
        raise RuntimeError('Kitchen baseline changed; review XPU fork compatibility before rebuilding this profile.')
    target = state / 'requirements.txt'
    target.write_text(source.replace('comfy-kitchen==0.2.38', 'comfy-kitchen==0.2.37'), encoding='utf-8')
    return target


def environment(root, profile, base=None):
    env = dict(os.environ if base is None else base)
    venv, state = paths(root, profile)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'COMMANDLINE_ARGS', 'REQS_FILE',
                'FORGE_ORT_CUDA13', 'SAGE_PACKAGE', 'FLASH_PACKAGE', 'NUNCHAKU_PACKAGE',
                'PIP_CONSTRAINT', 'PIP_REQUIREMENT', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER',
                'PYTHONUSERBASE', 'PIP_CONFIG_FILE', 'CONDA_PREFIX', 'SKIP_VENV'):
        env.pop(key, None)
    env['FORGE_XPU_AB_PROFILE'] = profile
    env['TORCH_INDEX_URL'] = 'https://download.pytorch.org/whl/xpu'
    env['TORCH_COMMAND'] = f'pip install torch=={TORCH} torchvision=={VISION} --index-url {env["TORCH_INDEX_URL"]}'
    env['VENV_DIR'] = str(venv)
    env['PIP_CONFIG_FILE'] = os.devnull
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    for key, name in {'HF_HOME':'huggingface', 'TORCH_HOME':'torch', 'PIP_CACHE_DIR':'pip',
                      'TRITON_CACHE_DIR':'triton', 'UV_CACHE_DIR':'uv', 'GRADIO_TEMP_DIR':'gradio',
                      'TEMP':'temp', 'TMP':'temp', 'PYTHONPYCACHEPREFIX':'pycache'}.items():
        directory = state / 'cache' / name
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    return env


def command_args(root, state, profile, extra):
    protected = ('--data-dir', '--ui-settings-file', '--ui-config-file', '--port',
                 '--update-all-extensions', '--skip-prepare-environment', '--reinstall-torch',
                 '--uv', '--uv-symlink', '--uv-local-cache', '--enable-triton-backend',
                 '--sage', '--flash', '--nunchaku', '--xformers', '--onnxruntime-gpu',
                 '--use-ck-attention', '--force-xformers-vae', '--styles-file')
    for arg in extra:
        if any(arg == name or arg.startswith(name + '=') for name in protected):
            raise ValueError(f'{arg} would invalidate the isolated XPU comparison profile')
    args = ['--disable-sage', '--disable-flash', '--bf16-unet', '--fp32-vae', '--autolaunch',
            '--port', str(PROFILES[profile][1]), '--data-dir', str(state), '--ui-settings-file', str(state / 'config.json'),
            '--ui-config-file', str(state / 'ui-config.json')]
    # Share model weights (including the user's existing models junction), not settings/packages.
    if not any(x == '--model-ref' or x.startswith('--model-ref=') for x in extra):
        args += ['--model-ref', str(Path(root) / 'models')]
    styles = state / 'styles.csv'
    return args + ['--styles-file', str(styles)] + list(extra)


def run(args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)


def checkout(root, name, url, revision):
    destination = root / 'sources' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    existed = destination.exists()
    if not existed:
        run(['git', 'clone', '--no-checkout', '--filter=blob:none', url, destination])
    if existed and subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain']).strip():
        raise RuntimeError(f'Source checkout has local changes; leaving it untouched: {destination}')
    run(['git', '-C', destination, 'fetch', '--depth', '1', 'origin', revision])
    run(['git', '-C', destination, 'checkout', '--detach', revision])
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise RuntimeError('Pinned source revision mismatch')
    return destination


def pip(*args):
    run([sys.executable, '-m', 'pip', *args])


def kernel_preflight(wheel=None, build=False):
    from importlib import metadata
    if wheel:
        return
    if build:
        if not (shutil.which('icx') and shutil.which('cl')):
            raise RuntimeError('Initialize VS2022 C++/Windows SDK and oneAPI 2026.1 + oneDNN developer environment before building. No packages were changed.')
        return
    try:
        metadata.version('omni-xpu-kernel')
    except metadata.PackageNotFoundError as error:
        raise RuntimeError('Kitchen native kernel is not installed. Use build-intel-xpu-kitchen.bat from an initialized developer shell, or pass --kernel-wheel PATH --kernel-sha256 HASH. No eager substitute will be launched.') from error


def install_kernel(state, wheel=None, expected_hash=None, build=False):
    """No unverified download URL, automatic Torch downgrade or cross-platform wheel."""
    if wheel:
        wheel = Path(wheel).resolve(strict=True)
        if not expected_hash or hashlib.sha256(wheel.read_bytes()).hexdigest() != expected_hash.lower():
            raise RuntimeError('Supply --kernel-sha256 matching the local Windows kernel wheel.')
        if not wheel.name.endswith('win_amd64.whl') or 'torch214.bmg' not in wheel.name:
            raise RuntimeError('Kernel wheel must target Windows x64, Torch 2.14 and BMG.')
        pip('install', '--no-deps', '--force-reinstall', wheel)
        (state / 'kernel-source.json').write_text(json.dumps({'source':'user-supplied Windows wheel',
            'filename':wheel.name, 'sha256':expected_hash.lower()}), encoding='utf-8')
        return
    from importlib import metadata
    try:
        metadata.version('omni-xpu-kernel')
        if not build:
            return  # Loaded ABI, native target and operators are checked below.
    except metadata.PackageNotFoundError:
        pass
    if not build:
        raise RuntimeError('No matching Windows native kernel is published. Run build-intel-xpu-kitchen.bat from an MSVC/oneAPI 2026.1 developer shell, or supply --kernel-wheel PATH --kernel-sha256 HASH. No eager substitution was made.')
    if not (shutil.which('icx') and shutil.which('cl')):
        raise RuntimeError('Native build needs VS2022 C++/Windows SDK and oneAPI 2026.1 compiler + matching oneDNN development files. Initialize that developer shell first; nothing will be downgraded.')
    source = checkout(state, 'llm-scaler', 'https://github.com/intel/llm-scaler.git', KERNEL_REV)
    native = source / 'omni' / 'omni_xpu_kernel'
    wheelhouse = state / 'native-wheels'
    wheelhouse.mkdir(exist_ok=True)
    # A separate build is opt-in; no system compiler/SDK installation or DLL overwriting.
    pip('install', 'setuptools>=77', 'wheel')
    env = dict(os.environ, OMNI_XPU_DEVICE='bmg', OMNI_XPU_REQUIRE_CUTE='0', OMNI_XPU_ONEAPI_VERSION='2026.1')
    run([sys.executable, '-m', 'pip', 'wheel', native, '--no-build-isolation', '--no-deps', '--wheel-dir', wheelhouse], env=env)
    candidates = list(wheelhouse.glob('omni_xpu_kernel-*torch214.bmg*win_amd64.whl'))
    if len(candidates) != 1:
        raise RuntimeError('Native build must produce exactly one matching Windows BMG wheel; review native-wheels.')
    pip('install', '--no-deps', '--force-reinstall', candidates[0])
    (state / 'kernel-source.json').write_text(json.dumps({'revision':KERNEL_REV,
        'torch':TORCH, 'target':'bmg', 'cute':False,
        'sha256':hashlib.sha256(candidates[0].read_bytes()).hexdigest()}), encoding='utf-8')


def install_kitchen(state):
    source = checkout(state, 'kitchen', 'https://github.com/xiangyuT/comfy-kitchen-xpu.git', KITCHEN_REV)
    pip('install', 'setuptools>=77', 'wheel')
    wheels = state / 'kitchen-wheels'
    wheels.mkdir(exist_ok=True)
    run([sys.executable, '-m', 'pip', 'wheel', source, '--no-build-isolation', '--no-deps', '--wheel-dir', wheels])
    wheel = wheels / 'comfy_kitchen-0.2.37-py3-none-any.whl'
    if not wheel.is_file():
        raise RuntimeError('Expected pinned pure-Python Kitchen XPU wheel was not built')
    pip('install', '--no-deps', '--force-reinstall', wheel)
    (state / 'kitchen-source.json').write_text(json.dumps({'revision':KITCHEN_REV,
        'sha256':hashlib.sha256(wheel.read_bytes()).hexdigest()}), encoding='utf-8')


def check_native_identity(torch_version, package_torch, package_target, core_target, available, capabilities):
    if package_torch != torch_version or package_target != 'bmg' or core_target != 'bmg':
        raise RuntimeError('Native kernel Torch ABI/package/AOT target mismatch; refusing this comparison.')
    if not available or 'quantize_int8_rowwise' not in capabilities:
        raise RuntimeError('Native XPU INT8 capability is missing. Eager fallback is not an accepted Kitchen test.')


def verify_runtime(profile):
    import torch
    import comfy_kitchen as ck
    if torch.__version__ != TORCH or not torch.xpu.is_available():
        raise RuntimeError(f'Profile requires {TORCH} and a working Intel XPU')
    report = {'profile':profile, 'torch':torch.__version__, 'device':torch.xpu.get_device_name(0),
              'kitchen':getattr(ck, '__version__', 'unknown'), 'backends':ck.list_backends()}
    if profile == 'kitchen':
        if 'B580' not in report['device']:
            raise RuntimeError("This experimental BMG profile is admitted for the B580 only; review other devices separately.")
        import omni_xpu_kernel as omni
        back = report['backends'].get('xpu', {})
        check_native_identity(torch.__version__, omni.__torch_version__, omni.__xpu_target__,
                              omni.core_aot_target(), back.get('available', False), back.get('capabilities', ()))
        # Force a native route: a successful eager operation is insufficient evidence.
        x = torch.arange(256, device='xpu', dtype=torch.float32).reshape(4, 64) / 256
        with ck.use_backend('xpu'):
            q, scale = ck.quantize_int8_rowwise(x)
        torch.xpu.synchronize()
        if q.dtype != torch.int8 or not torch.isfinite(scale).all() or q.shape != x.shape:
            raise RuntimeError('Native XPU smoke test returned an invalid result')
        restored = q.float() * scale.reshape(-1, 1)
        if not torch.allclose(restored, x, atol=0.01, rtol=0.01):
            raise RuntimeError('Native INT8 smoke test failed numerical validation')
        report.update(kernel=omni.__version__, package_target=omni.__xpu_target__, native_smoke='passed')
    elif report['backends'].get('xpu', {}).get('available'):
        raise RuntimeError('Official profile unexpectedly contains a custom XPU Kitchen backend; recreate only that profile venv.')
    return report
