@echo off
setlocal
cd /d "%~dp0"
call bootstrap-python.bat
if errorlevel 1 goto :failed
"%PYTHON%" tools\launch_wd14_gpu.py %*
if errorlevel 1 goto :failed
endlocal
exit /b 0
:failed
echo.
echo Intel WD14 GPU test launcher stopped. See the reason above.
pause
endlocal
exit /b 1
