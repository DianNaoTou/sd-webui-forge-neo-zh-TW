@echo off
setlocal EnableExtensions
title Forge Neo - Upscaler Downloader

set "DIR=%~dp0models\ESRGAN"
set "FAILED=0"

echo ================================================
echo       Forge Neo - Upscaler Downloader
echo ================================================
echo.
echo Install directory:
echo %DIR%
echo.

where curl.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] curl.exe was not found.
    echo Windows 10 or later normally includes curl.exe.
    pause
    exit /b 1
)

if not exist "%DIR%" mkdir "%DIR%"
if not exist "%DIR%" (
    echo [ERROR] Could not create the install directory.
    pause
    exit /b 1
)

call :download "RealESRGAN_x4plus.pth" "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" || set /a FAILED+=1
call :download "RealESRGAN_x4plus_anime_6B.pth" "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" || set /a FAILED+=1
call :download "4x-AnimeSharp.pth" "https://huggingface.co/sudolink/upscale_models/resolve/main/4x-AnimeSharp.pth" || set /a FAILED+=1
call :download "4x-UltraSharp.pth" "https://huggingface.co/sudolink/upscale_models/resolve/main/4x-UltraSharp.pth" || set /a FAILED+=1
call :download "4x_foolhardy_Remacri.pth" "https://huggingface.co/sudolink/upscale_models/resolve/main/4x_foolhardy_Remacri.pth" || set /a FAILED+=1

echo.
if %FAILED% GTR 0 (
    echo [ERROR] %FAILED% download(s) failed. Run this file again to retry.
    echo Incomplete files keep the .download extension and are not loaded by Forge Neo.
    pause
    exit /b 1
)

echo [DONE] All upscaler models are ready.
echo Restart Forge Neo, then select them from the upscaler list.
pause
exit /b 0

:download
set "FILE=%~1"
set "URL=%~2"

echo ------------------------------------------------
echo Checking: %FILE%
echo ------------------------------------------------

if exist "%DIR%\%FILE%" (
    echo [SKIP] Already exists.
    echo.
    exit /b 0
)

echo [DOWNLOAD] Starting...
curl.exe --location --fail --retry 3 --retry-delay 3 --progress-bar --output "%DIR%\%FILE%.download" "%URL%"
if errorlevel 1 (
    echo [ERROR] Download failed: %FILE%
    echo.
    exit /b 1
)

move /Y "%DIR%\%FILE%.download" "%DIR%\%FILE%" >nul
if errorlevel 1 (
    echo [ERROR] Could not finalize: %FILE%
    echo.
    exit /b 1
)

echo [OK] %FILE%
echo.
exit /b 0
