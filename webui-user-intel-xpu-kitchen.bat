@echo off
setlocal
cd /d "%~dp0"
call bootstrap-python.bat
if errorlevel 1 goto :failed
"%PYTHON%" tools\launch_xpu_profile.py kitchen  %*
if errorlevel 1 goto :failed
endlocal
exit /b 0
:failed
echo.
echo XPU profile stopped. See the reason above.
pause
endlocal
exit /b 1
