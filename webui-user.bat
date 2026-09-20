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

set COMMANDLINE_ARGS=

:: --xformers --sage --uv
:: --pin-shared-memory --cuda-malloc --cuda-stream
:: --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

call webui.bat
