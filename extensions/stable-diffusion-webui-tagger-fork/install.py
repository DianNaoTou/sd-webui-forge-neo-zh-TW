"""Install WD14 dependencies once, using Forge's interpreter and policy."""
from pathlib import Path
from launch import run_pip
from modules.dependency_utils import ensure_onnxruntime, requirements_met

req_file = Path(__file__).with_name("requirements.txt")
if not requirements_met(req_file):
    run_pip(f'install -r "{req_file}"', "WD14-tagger requirements")
ensure_onnxruntime(run_pip)
