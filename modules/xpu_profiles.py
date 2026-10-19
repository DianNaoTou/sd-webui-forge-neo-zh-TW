"""Isolated official Intel XPU profile (no GPU imports at module load)."""
import json
import os
from pathlib import Path
import subprocess
import sys

TORCH = '2.14.1+xpu'
VISION = '0.29.1+xpu'
PROFILES = {'official': ('venv-xpu-official', 7862)}


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
    paths(root, profile)  # Reject retired profiles before writing any files.
    return Path(root) / 'requirements.txt'


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
    paths(root, profile)
    protected = ('--data-dir', '--ui-settings-file', '--ui-config-file', '--port',
                 '--update-all-extensions', '--skip-prepare-environment', '--reinstall-torch',
                 '--uv', '--uv-symlink', '--uv-local-cache', '--enable-triton-backend',
                 '--sage', '--flash', '--nunchaku', '--xformers', '--onnxruntime-gpu',
                 '--use-ck-attention', '--force-xformers-vae', '--styles-file',
                 '--build-native', '--kernel-wheel', '--kernel-sha256')
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


def pip(*args):
    run([sys.executable, '-m', 'pip', *args])


def verify_runtime(profile):
    if profile not in PROFILES:
        raise ValueError("Unknown XPU profile")
    import torch
    import comfy_kitchen as ck
    if torch.__version__ != TORCH or not torch.xpu.is_available():
        raise RuntimeError(f'Profile requires {TORCH} and a working Intel XPU')
    report = {'profile':profile, 'torch':torch.__version__, 'device':torch.xpu.get_device_name(0),
              'kitchen':getattr(ck, '__version__', 'unknown'), 'backends':ck.list_backends()}
    if report['backends'].get('xpu', {}).get('available'):
        raise RuntimeError('Official profile unexpectedly contains a custom XPU Kitchen backend; recreate only that profile venv.')
    return report
