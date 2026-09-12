@echo off
cd /d "%~dp0"

set "PYTHON=%~dp0venv313\Scripts\python.exe"
set "VENV_DIR=%~dp0venv313"
set "COMMANDLINE_ARGS=--skip-install --disable-sage --disable-flash --bf16-unet --fp32-vae --autolaunch"

if not exist "%PYTHON%" (
    echo venv313 was not found.
    echo Run this command first:
    echo powershell -NoProfile -ExecutionPolicy Bypass -File .\install-b580.ps1
    pause
    exit /b 1
)

call webui.bat
