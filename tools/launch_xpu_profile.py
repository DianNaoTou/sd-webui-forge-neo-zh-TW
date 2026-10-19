"""Windows A/B entry point. All dependency writes target the selected owned venv."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules import xpu_profiles as profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('profile', choices=profiles.PROFILES)
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--setup-only', action='store_true')
    parser.add_argument('--build-native', action='store_true')
    parser.add_argument('--kernel-wheel')
    parser.add_argument('--kernel-sha256')
    args, extra = parser.parse_known_args()
    venv, state = profiles.paths(ROOT, args.profile)
    # Validate comparison flags before creating or installing anything.
    forge_args = profiles.command_args(ROOT, state, args.profile, extra)
    if os.name != 'nt':
        raise RuntimeError('These launchers target Windows x64; no changes were made.')
    if venv.is_symlink() or venv.is_junction() or state.is_symlink() or state.is_junction():
        raise RuntimeError('Profile venv/state cannot redirect to another environment.')
    env = profiles.environment(ROOT, args.profile)
    if not args.inside:
        python = venv / 'Scripts' / 'python.exe'
        if not python.exists():
            profiles.run([sys.executable, '-m', 'venv', venv], env=env)
        return subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:], '--inside'], env=env, cwd=ROOT)
    if Path(sys.prefix).resolve() != venv.resolve():
        raise RuntimeError('Wrong interpreter: refusing to install into another environment.')
    os.environ.clear()
    os.environ.update(env)
    os.chdir(ROOT)
    profiles.initialize_settings(ROOT, state)
    profiles.initialize_extension_link(ROOT, state)
    requirement = profiles.requirements_for(ROOT, state, args.profile)
    os.environ['REQS_FILE'] = str(requirement)
    if args.profile == 'kitchen':
        profiles.kernel_preflight(args.kernel_wheel, args.build_native)
    expected = {'torch':profiles.TORCH, 'torchvision':profiles.VISION}
    installed = {}
    for name in expected:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
    if installed != expected:
        profiles.pip('install', 'torch==' + profiles.TORCH, 'torchvision==' + profiles.VISION,
                     '--index-url', 'https://download.pytorch.org/whl/xpu')
    if args.profile == 'kitchen':
        profiles.install_kernel(state, args.kernel_wheel, args.kernel_sha256, args.build_native)
        marker = state / 'kitchen-source.json'
        try:
            kitchen_version = metadata.version('comfy-kitchen')
        except metadata.PackageNotFoundError:
            kitchen_version = None
        if not marker.exists() or json.loads(marker.read_text())['revision'] != profiles.KITCHEN_REV or kitchen_version != '0.2.37':
            profiles.install_kitchen(state)
    sys.argv = ['launch.py', *forge_args, '--skip-prepare-environment']
    import launch
    # Only setup is serialized; the two UIs have separate settings/ports/outputs.
    from filelock import FileLock
    with FileLock(str(ROOT / 'runtime' / 'xpu-ab' / 'setup.lock'), timeout=600):
        launch.prepare_environment()
    report = profiles.verify_runtime(args.profile)
    from backend.args import args as backend_args
    report["xpu_sdpa"] = backend_args.xpu_sdpa
    report.update(python=sys.version, interpreter=sys.executable, port=profiles.PROFILES[args.profile][1],
                  requirements=str(requirement), kitchen_revision=profiles.KITCHEN_REV if args.profile == 'kitchen' else None)
    (state / 'runtime-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('\n[XPU A/B verified]\n' + json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    if not args.setup_only:
        launch.main()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, ImportError, subprocess.CalledProcessError) as error:
        print(f'\nXPU profile stopped: {error}\nOther profiles were not modified.', file=sys.stderr)
        raise SystemExit(1)
