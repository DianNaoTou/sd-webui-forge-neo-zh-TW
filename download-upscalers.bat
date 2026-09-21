@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Forge Neo - Upscaler Downloader

echo ================================================
echo       Forge Neo - Upscaler Downloader
echo ================================================
echo.

call bootstrap-python.bat
if errorlevel 1 (
    echo.
    echo [ERROR] Python setup failed. Upscaler models were not downloaded.
    pause
    exit /b 1
)

"%PYTHON%" -m modules.upscaler_models
set "RESULT=%ERRORLEVEL%"

echo.
if %RESULT% NEQ 0 (
    echo [ERROR] One or more downloads failed. Run this file again to retry.
) else (
    echo [DONE] All upscaler models are ready.
    echo Restart Forge Neo, then select them from the upscaler list.
)

pause
exit /b %RESULT%
