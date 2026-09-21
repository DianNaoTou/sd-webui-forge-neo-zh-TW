"""Shared dependency policy for the launcher and bundled extensions."""
import importlib.metadata as metadata
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pip_constraint_args():
    # Explicit quoted arguments also support Windows paths containing spaces.
    # Bundled extension installers all call the launcher's run_pip wrapper.
    return ' '.join(f'--constraint "{ROOT / name}"' for name in
                    ('constraints-common.txt', 'requirements.txt'))


def prepare_pip_command(command, index_url=''):
    """Add resolver options only to pip install operations.

    ``--prefer-binary``, constraints, and package indexes are install options;
    passing them to commands such as ``uninstall`` makes pip reject the entire
    operation.  Upgrade operations use ``pip install --upgrade`` and therefore
    intentionally follow the install path.
    """
    stripped = command.lstrip()
    operation = stripped.split(maxsplit=1)[0].lower() if stripped else ''
    if operation != 'install':
        return command

    prepared = f'{command} {pip_constraint_args()}'
    if not re.search(r'(^|\s)--prefer-binary(?=\s|$)', command, re.IGNORECASE):
        prepared += ' --prefer-binary'
    has_index = re.search(r'(^|\s)(--index-url(?:=|\s)|-i(?:=|\s))', command, re.IGNORECASE)
    if index_url and not has_index:
        prepared += f' --index-url {index_url}'
    return prepared


def requirement_met(spec):
    from packaging.requirements import Requirement
    req = Requirement(spec)
    if req.marker and not req.marker.evaluate():
        return True
    try:
        installed = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        return False
    if not req.specifier.contains(installed, prereleases=True):
        return False
    # Extras have no distribution of their own; check their active dependencies.
    if req.extras:
        for dep in metadata.requires(req.name) or []:
            child = Requirement(dep)
            if child.marker is None or any(child.marker.evaluate({'extra': extra}) for extra in req.extras):
                child.marker = None
                if not requirement_met(str(child)):
                    return False
    return True


def requirements_met(path):
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw.split(' #', 1)[0].strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('-r '):
            if not requirements_met(Path(path).parent / line[3:].strip()):
                return False
        elif line.startswith('-c '):
            continue  # Constraints restrict resolution; they do not install packages.
        elif not requirement_met(line):
            return False
    return True


def torch_backend():
    import torch
    # Inspect the build, not GPU availability (a driver error must not select CUDA).
    if getattr(torch.version, 'xpu', None):
        return 'xpu'
    if getattr(torch.version, 'cuda', None):
        return 'cuda'
    return 'cpu'


def onnx_requirement(backend):
    # Let the selected ORT release declare its own CUDA/cuDNN dependencies.
    # XPU uses the CPU ONNX provider; do not hard-code CUDA 11/12 packages.
    return 'onnxruntime-gpu[cuda,cudnn]>=1.30,<1.31' if backend == 'cuda' else 'onnxruntime>=1.30,<1.31'


def ensure_onnxruntime(run_pip):
    backend = torch_backend()
    wanted = 'onnxruntime-gpu' if backend == 'cuda' else 'onnxruntime'
    other = 'onnxruntime' if backend == 'cuda' else 'onnxruntime-gpu'
    wanted_was_installed = requirement_met(wanted)
    conflict = requirement_met(other)

    if conflict:
        # CPU and GPU ONNX Runtime distributions install the same Python package
        # namespace. A bundled extension (notably insightface or one of its
        # dependencies) can pull the CPU distribution into an otherwise clean
        # CUDA venv. Remove only the incompatible distribution and repair the
        # selected provider if both distributions had overlapped.
        print(f'Removing incompatible {other} from {backend} environment...')
        run_pip(f'uninstall -y "{other}"', f'remove incompatible {other}')

    spec = onnx_requirement(backend)

    # Respect the launcher's explicit CUDA 13 nightly provider. If a conflicting
    # CPU runtime overlapped an already-installed GPU runtime, reinstall the GPU
    # package once so files removed by pip are restored.
    if backend == 'cuda' and os.environ.get('FORGE_ORT_CUDA13') == '1':
        if conflict and wanted_was_installed:
            package = os.environ.get(
                'ONNX_PACKAGE',
                'onnxruntime-gpu --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/',
            )
            run_pip(f'install --force-reinstall {package}', 'repair ONNX Runtime (cuda)')
        elif requirement_met(wanted):
            return
        return

    if conflict and wanted_was_installed:
        run_pip(f'install --force-reinstall "{spec}"', f'repair ONNX Runtime ({backend})')
    elif not requirement_met(spec):
        run_pip(f'install "{spec}"', f'ONNX Runtime ({backend})')
