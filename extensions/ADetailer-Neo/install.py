from pathlib import Path
import launch
from modules.dependency_utils import requirements_met

req_file = Path(__file__).with_name('requirements.txt')
if not requirements_met(req_file):
    launch.run_pip(f'install -r "{req_file}"', 'ADetailer-Neo requirements')
