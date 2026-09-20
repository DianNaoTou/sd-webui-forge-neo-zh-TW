@echo off
cd /d "%~dp0"

call bootstrap-python.bat
if errorlevel 1 (
    echo.
    echo Python setup failed. Forge Neo was not started.
    pause
    exit /b 1
)

rem Keep Intel Arc packages isolated from the default NVIDIA environment.
set "VENV_DIR=%~dp0venv-intel-arc"
set "TORCH_INDEX_URL=https://download.pytorch.org/whl/xpu"
set "TORCH_COMMAND=pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu"
set "COMMANDLINE_ARGS=--disable-sage --disable-flash --bf16-unet --fp32-vae --autolaunch"

call webui.bat
