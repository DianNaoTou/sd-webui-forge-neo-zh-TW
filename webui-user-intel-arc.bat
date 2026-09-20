@echo off
cd /d "%~dp0"

rem Keep Intel Arc packages isolated from the default NVIDIA environment.
set "VENV_DIR=%~dp0venv-intel-arc"
set "TORCH_INDEX_URL=https://download.pytorch.org/whl/xpu"
set "TORCH_COMMAND=pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu"
set "COMMANDLINE_ARGS=--disable-sage --disable-flash --bf16-unet --fp32-vae --autolaunch"

call webui.bat
