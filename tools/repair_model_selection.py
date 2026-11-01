"""Narrow, reversible code overlay. No git checkout/reset/clean or model writes.

Run through repair-model-selection.ps1 on Windows. Unit tests use fake projects.
The optional replacement venv and first-launch settings live in the transaction,
so neither the old venv nor the original configuration needs restoration.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

BASE = 'bd3604b5a3e900436c7f03b9e8bb0d66913dee17'
FILES = ('modules_forge/main_entry.py', 'modules/sd_models.py',
         'modules/sysinfo.py', 'modules_forge/patch_basic.py')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(data):
    return data.replace(b'\r\n', b'\n')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def redirected(path):
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())


def inside(path, parent):
    return path == parent or parent in path.parents


def safe_target(root, name, models):
    # Windows TEMP can use an 8.3 alias (RUNNER~1). Compare canonical paths so
    # safe files are not mistaken for paths outside their own project.
    root = root.resolve()
    models = models.resolve()
    if name not in FILES:
        raise RuntimeError('Refusing a path outside the code allowlist: ' + name)
    target = root / name
    current = target
    while current != root:
        if redirected(current):
            raise RuntimeError('Code path redirects outside its expected location: ' + str(current))
        current = current.parent
    resolved = target.resolve()
    if not inside(resolved, root) or inside(resolved, models) or inside(models, resolved):
        raise RuntimeError('Code/model paths overlap; refusing to write: ' + str(target))
    return target


def model_inventory(models):
    """Directory/link metadata only. Never open, hash, move or repair a weight."""
    result = [{'path': str(models), 'resolved': str(models.resolve()), 'redirected': redirected(models)}]
    for directory, dirs, _ in os.walk(models, followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            if redirected(path):
                result.append({'path': str(path), 'resolved': str(path.resolve()), 'redirected': True})
                dirs.remove(name)
    return sorted(result, key=lambda item: item['path'])


def write_atomic(path, data):
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.forge-repair-', delete=False) as stream:
            staged = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def save_manifest(transaction, manifest):
    write_atomic(transaction / 'manifest.json', json.dumps(manifest, indent=2).encode('utf-8'))


def rollback(root, transaction):
    root = root.resolve()
    manifest = json.loads((transaction / 'manifest.json').read_text(encoding='utf-8'))
    if root.resolve() != Path(manifest['root']).resolve():
        raise RuntimeError('Backup belongs to a different project.')
    models = Path(manifest['models_resolved'])
    current_inventory = model_inventory(Path(manifest['models_path']))
    if current_inventory != manifest['model_inventory']:
        raise RuntimeError('Model/junction layout changed after repair; refusing to restore into an unverified layout.')
    # Preflight every file before restoring any. Preserve subsequent user edits.
    entries = manifest['files']
    for name, entry in entries.items():
        target = safe_target(root, name, models)
        for item in current_inventory:
            safe_target(root, name, Path(item['resolved']))
        original = (transaction / 'original' / name).read_bytes()
        if digest(original) != entry['before']:
            raise RuntimeError('Backup checksum mismatch: ' + name)
        if digest(target.read_bytes()) not in (entry['before'], entry['after']):
            raise RuntimeError('File changed after repair; refusing to overwrite your edits: ' + name)
    manifest['status'] = 'rolling-back'
    save_manifest(transaction, manifest)
    for name, entry in entries.items():
        target = safe_target(root, name, models)
        write_atomic(target, (transaction / 'original' / name).read_bytes())
        if digest(target.read_bytes()) != entry['before']:
            raise RuntimeError('Restore checksum mismatch: ' + name)
    manifest['status'] = 'rolled-back'
    save_manifest(transaction, manifest)
    print('Restored original code bytes. Original settings, branch, commit, venv and models were never replaced.')


def apply(root, revision, models, backup_parent):
    root = root.resolve()
    if git(root, 'branch', '--show-current').decode().strip() != 'neo-test':
        raise RuntimeError('Local branch must be neo-test. No automatic branch switch is performed.')
    if git(root, 'rev-parse', 'HEAD').decode().strip() != BASE:
        raise RuntimeError('This repair is validated against ' + BASE + '; a different local commit requires review, not a forced reset.')
    git(root, 'merge-base', '--is-ancestor', BASE, revision)
    if not models.is_dir():
        raise RuntimeError('Model directory was not found; refusing to guess or create one: ' + str(models))
    resolved_models = models.resolve()
    if inside(backup_parent.resolve(), root) or inside(backup_parent.resolve(), resolved_models):
        raise RuntimeError('Backups must be outside the project and model directories.')
    inventory = model_inventory(models)
    protected = [Path(item['resolved']) for item in inventory]
    for model_path in protected:
        if inside(backup_parent.resolve(), model_path):
            raise RuntimeError('Backup directory resolves inside a model junction target.')
        for name in FILES:
            safe_target(root, name, model_path)
    payload = {}
    originals = {}
    for name in FILES:
        target = safe_target(root, name, resolved_models)
        old = target.read_bytes()
        if normalized(old) != normalized(git(root, 'show', BASE + ':' + name)):
            raise RuntimeError('Local edits detected in repair target; leaving them untouched: ' + name)
        payload[name] = git(root, 'show', revision + ':' + name)
        compile(payload[name], name, 'exec')
        originals[name] = old
    transaction = backup_parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8])
    transaction.mkdir(parents=True, exist_ok=False)
    print('BACKUP / ROLLBACK DIRECTORY: ' + str(transaction), flush=True)
    manifest = dict(root=str(root), base=BASE, revision=revision,
                    models_path=str(models.absolute()), models_resolved=str(resolved_models), model_inventory=inventory,
                    branch='neo-test', status='prepared',
                    git_status=git(root, 'status', '--porcelain=v1').decode('utf-8', errors='replace'),
                    files={name: dict(before=digest(originals[name]), after=digest(payload[name])) for name in FILES})
    for name in FILES:
        backup = transaction / 'original' / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(originals[name])
        if backup.read_bytes() != originals[name]:
            raise RuntimeError('Backup verification failed before applying any patch.')
    shutil.copyfile(__file__, transaction / 'repair_model_selection.py')
    (transaction / 'repair-model-selection.ps1').write_bytes(git(root, 'show', revision + ':tools/repair-model-selection.ps1'))
    quoted_root = str(root).replace("'", "''")
    (transaction / 'rollback.ps1').write_text(
        "& (Join-Path $PSScriptRoot 'repair-model-selection.ps1') -Root '" + quoted_root + "' -Rollback $PSScriptRoot\n",
        encoding='utf-8-sig')
    print('ROLLBACK: powershell -NoProfile -ExecutionPolicy Bypass -File "' + str(transaction / 'rollback.ps1') + '"', flush=True)
    save_manifest(transaction, manifest)
    try:
        for name in FILES:
            target = safe_target(root, name, resolved_models)
            write_atomic(target, payload[name])
            if digest(target.read_bytes()) != manifest['files'][name]['after']:
                raise RuntimeError('Patched file checksum mismatch: ' + name)
        manifest['status'] = 'applied'
        save_manifest(transaction, manifest)
    except BaseException:
        rollback(root, transaction)
        raise
    return transaction


def run_logged(command, transaction, **kwargs):
    with (transaction / 'launch.log').open('a', encoding='utf-8') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', **kwargs) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            if code:
                raise subprocess.CalledProcessError(code, command)


def launch(root, transaction, models, rebuild):
    """Protected first launch: old environment/configuration are not installed into."""
    state = transaction / 'state'
    state.mkdir()
    original_state = root / 'runtime' / 'xpu-ab' / 'official'
    for name, default in [('config.json', b'{}'), ('ui-config.json', b'{}'), ('styles.csv', b'name,prompt,negative_prompt\n')]:
        source = original_state / name
        if not source.is_file():
            source = root / name
        (state / name).write_bytes(source.read_bytes() if source.is_file() else default)
    # Inventory is already recorded; this launch does not sanitize model paths.
    python = root / 'venv-xpu-official' / 'Scripts' / 'python.exe'
    check = ('import torch, gradio, fastapi, diffusers, transformers, comfy_kitchen, rich; '
             'assert torch.__version__ == "2.14.1+xpu"; assert torch.xpu.is_available(); '
             'print(torch.xpu.get_device_name(0))')
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('PIP_', 'UV_')):
            env.pop(key)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'COMMANDLINE_ARGS', 'REQS_FILE', 'PYTHONUSERBASE', 'CONDA_PREFIX'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PIP_CONFIG_FILE=os.devnull, PIP_NO_INPUT='1')
    healthy = python.is_file() and subprocess.run([str(python), '-c', check], env=env).returncode == 0
    if rebuild or not healthy:
        print('Creating a separate replacement environment; the original venv will be kept.', flush=True)
        venv = transaction / 'venv-xpu'
        run_logged([sys.executable, '-m', 'venv', str(venv)], transaction, env=env)
        python = venv / 'Scripts' / 'python.exe'
        run_logged([str(python), '-m', 'pip', 'install', 'torch==2.14.1+xpu', 'torchvision==0.29.1+xpu',
                    '--index-url', 'https://download.pytorch.org/whl/xpu'], transaction, env=env)
        # Keep Torch on its reviewed backend even when resolving other packages.
        constraints = transaction / 'constraints.txt'
        constraints.write_text('torch==2.14.1+xpu\ntorchvision==0.29.1+xpu\n', encoding='utf-8')
        run_logged([str(python), '-m', 'pip', 'install', '-c', str(constraints),
                    '-r', str(root / 'requirements.txt'), 'gradio==4.40.0', 'gradio_rangeslider==0.0.8'], transaction, env=env)
        subprocess.run([str(python), '-c', check], check=True, env=env)
    # Avoid startup dependency installers and third-party extension side effects
    # during the first verification run. Offline model access never redownloads
    # missing Hugging Face files. Extensions are unchanged on disk.
    args = [str(python), str(root / 'launch.py'), '--skip-prepare-environment', '--skip-install',
            '--disable-all-extensions', '--disable-sage', '--disable-flash', '--bf16-unet', '--fp32-vae',
            '--port', '7862', '--autolaunch', '--data-dir', str(state), '--model-ref', str(models),
            '--ui-settings-file', str(state / 'config.json'), '--ui-config-file', str(state / 'ui-config.json'),
            '--styles-file', str(state / 'styles.csv')]
    (transaction / 'launch.json').write_text(json.dumps(args, indent=2), encoding='utf-8')
    print('Protected launch: extensions disabled for this run; original configuration and environment retained.', flush=True)
    run_logged(args, transaction, cwd=root, env=env)


def validate_code(root, transaction, revision):
    tests = transaction / 'validation-tests'
    tests.mkdir()
    (tests / 'test_model_selection_sync.py').write_bytes(git(root, 'show', revision + ':tests/test_model_selection_sync.py'))
    env = dict(os.environ, FORGE_REPAIR_TEST_ROOT=str(root))
    # Require a nonempty suite; a broken download must not count as a pass.
    runner = ('import sys, unittest; suite=unittest.defaultTestLoader.discover(sys.argv[1]); '
              'assert suite.countTestCases() >= 10, "Repair regression suite is incomplete"; '
              'result=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not result.wasSuccessful())')
    run_logged([sys.executable, '-B', '-c', runner, str(tests)], transaction, cwd=root, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--revision')
    parser.add_argument('--models', type=Path)
    parser.add_argument('--rollback', type=Path)
    parser.add_argument('--no-launch', action='store_true')
    parser.add_argument('--rebuild-environment', action='store_true')
    args = parser.parse_args()
    if os.name != 'nt':
        raise RuntimeError('This entry point targets Windows. No changes made.')
    root = args.root.resolve()
    if args.rollback:
        rollback(root, args.rollback.resolve())
        return
    if not args.revision or len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        raise ValueError('A full reviewed commit SHA is required.')
    models = args.models or root / 'models'
    parent = Path(os.environ['LOCALAPPDATA']) / 'ForgeNeoRepair'
    transaction = apply(root, args.revision, models, parent)
    try:
        validate_code(root, transaction, args.revision)
        if not args.no_launch:
            launch(root, transaction, models, args.rebuild_environment)
    except BaseException:
        print('Launch/setup failed; restoring the original code. New environment/state and logs are retained in ' + str(transaction), flush=True)
        rollback(root, transaction)
        raise
    print('Repair retained. Roll back using -Rollback "' + str(transaction) + '" with the PowerShell entry point.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print('Repair stopped: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
