"""Build a self-contained, shallow Git ZIP at an explicitly validated commit.

Release automation only: users extract the ZIP and run ordinary git pull.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


def git_env():
    env = os.environ.copy()
    for name in list(env):
        if name.startswith('GIT_'):
            env.pop(name)
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT='0')
    return env


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *map(str, args)],
                                   env=git_env(), stderr=subprocess.PIPE).decode().strip()


def assert_data_policy(root):
    """Reject tracked user data, with only existing model placeholders allowed."""
    placeholders = {
        'models/ESRGAN/Put Upscaler here.txt', 'models/Lora/Put LoRA here.txt',
        'models/Stable-diffusion/Put Checkpoint here.txt', 'models/VAE/Put VAE here.txt',
        'models/embeddings/Put Textual Inversion here.txt',
        'models/text_encoder/Put Text Encoders here.txt',
    }
    forbidden_roots = {'runtime', 'outputs', 'output', 'cache', 'tmp', '.uv-cache',
                       'repositories', 'embeddings', 'config_states'}
    forbidden_files = {'config.json', 'ui-config.json', 'styles.csv', 'user.css',
                       'webui.settings.bat', 'webui-user.local.bat', 'params.txt', '.env'}
    weights = {'.bin', '.ckpt', '.gguf', '.onnx', '.pth', '.pt', '.safetensors', '.sft'}
    for name in git(root, 'ls-files', '-z').split('\0'):
        if not name:
            continue
        path = Path(name)
        first = path.parts[0]
        if (first in forbidden_roots or first == 'venv' or first.startswith('venv-')
                or name in forbidden_files or path.suffix.lower() in weights
                or (first == 'models' and name not in placeholders)):
            raise ValueError(f'Refusing to package tracked user data: {name}')


def build(source, branch, commit, output):
    if branch not in ('neo', 'neo-test'):
        raise ValueError('Only neo and neo-test are supported')
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('An exact 40-character validated commit is required')
    output = Path(output).resolve()
    if output.exists() or output.with_suffix('.zip.sha256').exists():
        raise FileExistsError('Refusing to overwrite an existing package/checksum')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='forge-git-package-') as temporary:
        root = Path(temporary) / ('sd-webui-forge-neo-zh-TW-' + branch)
        root.mkdir()
        git(root, '-c', 'init.templateDir=', 'init', '--initial-branch', branch)
        git(root, 'remote', 'add', 'origin', source)
        git(root, 'remote', 'set-branches', 'origin', branch)
        # Fetch the validated SHA, never a branch tip that can race the CI run.
        git(root, 'fetch', '--depth=1', '--no-tags', 'origin', commit)
        git(root, 'checkout', '-B', branch, commit)
        git(root, 'update-ref', 'refs/remotes/origin/' + branch, commit)
        for key, value in {
            'branch.' + branch + '.remote': 'origin',
            'branch.' + branch + '.merge': 'refs/heads/' + branch,
            'branch.' + branch + '.mergeOptions': '--no-overwrite-ignore',
            'remote.origin.tagOpt': '--no-tags', 'pull.ff': 'only',
            'pull.rebase': 'false', 'merge.autoStash': 'false',
            'rebase.autoStash': 'false', 'core.filemode': 'false',
            'core.autocrlf': 'false',
        }.items():
            git(root, 'config', '--local', key, value)
        assert_data_policy(root)
        if git(root, 'rev-parse', 'HEAD') != commit or git(root, 'status', '--porcelain'):
            raise ValueError('Package commit/worktree mismatch')
        git(root, 'fsck', '--full')
        if (root / '.git/objects/info/alternates').exists():
            raise ValueError('Package depends on external Git objects')
        config = (root / '.git/config').read_text()
        if re.search(r'extraheader|credential|worktree\s*=|hooksPath', config, re.I):
            raise ValueError('Non-portable or credential configuration')
        # Reflogs/FETCH_HEAD contain build history and are unnecessary for pull.
        shutil.rmtree(root / '.git/logs', ignore_errors=True)
        (root / '.git/FETCH_HEAD').unlink(missing_ok=True)
        files = sorted(p for p in root.rglob('*') if p.is_file())
        if any(p.is_symlink() for p in root.rglob('*')):
            raise ValueError('Symlinks require an explicit portable packaging policy')
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                archive.write(path, Path(root.name) / path.relative_to(root))
        with zipfile.ZipFile(output) as archive:
            source_bytes = sum(x.compress_size for x in archive.infolist()
                               if '/.git/' not in x.filename)
            git_bytes = sum(x.compress_size for x in archive.infolist()
                            if '/.git/' in x.filename)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n', encoding='ascii')
        return dict(commit=commit, branch=branch, zip_bytes=output.stat().st_size,
                    source_compressed_bytes=source_bytes, git_compressed_bytes=git_bytes,
                    sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--branch', choices=('neo', 'neo-test'), required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.branch, args.commit, args.output), indent=2))


if __name__ == '__main__':
    main()
