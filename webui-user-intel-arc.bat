@echo off
setlocal
set "FORGE_WD14_OPENVINO="
set "FORGE_WD14_OV_DEVICE="
cd /d "%~dp0"
call bootstrap-python.bat
if errorlevel 1 goto :failed
"%PYTHON%" tools\launch_xpu_profile.py official  %*
if errorlevel 1 goto :failed
endlocal
exit /b 0
:failed
echo.
echo XPU profile stopped. See the reason above.
pause
endlocal
exit /b 1
