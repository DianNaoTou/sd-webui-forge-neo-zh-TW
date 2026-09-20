@echo off
cd /d "%~dp0"

call bootstrap-python.bat
if errorlevel 1 (
    echo.
    echo Python setup failed. Forge Neo was not started.
    pause
    exit /b 1
)

:: set GIT=
set "VENV_DIR=%~dp0venv-nvidia"

set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130"
set "TORCH_COMMAND=pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130"
set "FORGE_ORT_CUDA13="

set COMMANDLINE_ARGS=

:: --xformers --sage --uv
:: --pin-shared-memory --cuda-malloc --cuda-stream
:: --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

call webui.bat
