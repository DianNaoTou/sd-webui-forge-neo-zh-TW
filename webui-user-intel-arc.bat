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
set "TORCH_COMMAND=pip install torch==2.14.0+xpu torchvision==0.29.0+xpu --index-url https://download.pytorch.org/whl/xpu"
set "FORGE_ORT_CUDA13="
set "COMMANDLINE_ARGS=--disable-sage --disable-flash --bf16-unet --fp32-vae --autolaunch"

call webui.bat
