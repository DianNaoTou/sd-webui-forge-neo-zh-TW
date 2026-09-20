@echo off
setlocal EnableExtensions

rem Project-local Python bootstrap for Forge Neo on 64-bit Windows.
rem It does not change PATH, the Windows Python launcher, or the registry.
set "FORGE_PYTHON_VERSION=3.13.12"
set "FORGE_UV_VERSION=0.12.15"
set "FORGE_UV_SHA256=477bd99a84e34891f2bd4c9152ddeb74e971accccbc59c0f0301f11f08a32d46"
set "FORGE_RUNTIME_ROOT=%~dp0runtime"
set "FORGE_UV_ROOT=%FORGE_RUNTIME_ROOT%\uv"
set "FORGE_UV_EXE=%FORGE_UV_ROOT%\uv.exe"
set "FORGE_UV_ZIP=%FORGE_RUNTIME_ROOT%\uv-windows.zip"
set "FORGE_PYTHON_PATH_FILE=%FORGE_RUNTIME_ROOT%\python-path.txt"
set "UV_PYTHON_INSTALL_DIR=%FORGE_RUNTIME_ROOT%\python"
set "UV_CACHE_DIR=%FORGE_RUNTIME_ROOT%\uv-cache"
set "UV_MANAGED_PYTHON=1"

if not exist "%FORGE_RUNTIME_ROOT%" mkdir "%FORGE_RUNTIME_ROOT%"
if errorlevel 1 goto :runtime_error

if not exist "%FORGE_UV_EXE%" call :install_uv
if errorlevel 1 goto :uv_error

echo Checking project Python %FORGE_PYTHON_VERSION%...
"%FORGE_UV_EXE%" python install "%FORGE_PYTHON_VERSION%" --install-dir "%UV_PYTHON_INSTALL_DIR%" --no-bin --no-registry
if errorlevel 1 goto :python_install_error

"%FORGE_UV_EXE%" python find "%FORGE_PYTHON_VERSION%" --managed-python --no-project > "%FORGE_PYTHON_PATH_FILE%"
if errorlevel 1 goto :python_find_error

set /p "FORGE_PYTHON_EXE="<"%FORGE_PYTHON_PATH_FILE%"
del /q "%FORGE_PYTHON_PATH_FILE%" >nul 2>nul

if not defined FORGE_PYTHON_EXE goto :python_find_error
if not exist "%FORGE_PYTHON_EXE%" goto :python_find_error

"%FORGE_PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info[:3] == (3, 13, 12) else 1)"
if errorlevel 1 goto :python_version_error

echo Using project Python: %FORGE_PYTHON_EXE%
endlocal & set "PYTHON=%FORGE_PYTHON_EXE%"
exit /b 0

:install_uv
echo Downloading project-local uv %FORGE_UV_VERSION%...
if not exist "%FORGE_UV_ROOT%" mkdir "%FORGE_UV_ROOT%"
if errorlevel 1 exit /b 1

set "FORGE_UV_PRIMARY_URL=https://releases.astral.sh/github/uv/releases/download/%FORGE_UV_VERSION%/uv-x86_64-pc-windows-msvc.zip"
set "FORGE_UV_FALLBACK_URL=https://github.com/astral-sh/uv/releases/download/%FORGE_UV_VERSION%/uv-x86_64-pc-windows-msvc.zip"

call :download_uv_curl "%FORGE_UV_PRIMARY_URL%"
if errorlevel 1 (
    echo Primary download source failed. Trying GitHub Releases...
    call :download_uv_curl "%FORGE_UV_FALLBACK_URL%"
)
if errorlevel 1 (
    echo curl download failed. Trying the PowerShell fallback...
    call :download_uv_powershell "%FORGE_UV_FALLBACK_URL%"
)
if errorlevel 1 exit /b 1

call :verify_uv_zip
if errorlevel 1 (
    echo Download checksum mismatch. Retrying once with a clean file...
    del /q "%FORGE_UV_ZIP%" >nul 2>nul
    call :download_uv_curl "%FORGE_UV_FALLBACK_URL%"
    if errorlevel 1 call :download_uv_powershell "%FORGE_UV_FALLBACK_URL%"
    if errorlevel 1 exit /b 1
    call :verify_uv_zip
    if errorlevel 1 (
        del /q "%FORGE_UV_ZIP%" >nul 2>nul
        exit /b 1
    )
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; Expand-Archive -LiteralPath $env:FORGE_UV_ZIP -DestinationPath $env:FORGE_UV_ROOT -Force"
if errorlevel 1 exit /b 1

del /q "%FORGE_UV_ZIP%" >nul 2>nul
if not exist "%FORGE_UV_EXE%" exit /b 1
exit /b 0

:download_uv_curl
where curl.exe >nul 2>nul
if errorlevel 1 exit /b 1

set "FORGE_CURL_RETRY_OPTION=--retry-all-errors"
curl.exe --help all 2>nul | findstr /C:"--retry-all-errors" >nul
if errorlevel 1 set "FORGE_CURL_RETRY_OPTION=--retry-connrefused"

echo Download source: %~1
curl.exe --location --fail --retry 10 --retry-delay 3 %FORGE_CURL_RETRY_OPTION% --retry-max-time 900 --connect-timeout 30 --speed-limit 1024 --speed-time 60 --continue-at - --output "%FORGE_UV_ZIP%" "%~1"
if errorlevel 1 exit /b 1
exit /b 0

:download_uv_powershell
set "FORGE_UV_URL=%~1"
for /L %%R in (1,1,3) do (
    echo PowerShell download attempt %%R of 3...
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri $env:FORGE_UV_URL -OutFile $env:FORGE_UV_ZIP"
    if not errorlevel 1 exit /b 0
)
exit /b 1

:verify_uv_zip
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $actual=(Get-FileHash -Algorithm SHA256 -LiteralPath $env:FORGE_UV_ZIP).Hash.ToLowerInvariant(); if ($actual -ne $env:FORGE_UV_SHA256) { throw 'uv download checksum mismatch' }"
if errorlevel 1 exit /b 1
exit /b 0

:runtime_error
echo [ERROR] Could not create the project runtime directory:
echo         %FORGE_RUNTIME_ROOT%
goto :failed

:uv_error
echo [ERROR] Could not download or extract the project-local uv runtime.
echo         Check the network connection and try again.
goto :failed

:python_install_error
echo [ERROR] Could not install project-local Python %FORGE_PYTHON_VERSION%.
echo         Check the network connection and free disk space, then try again.
goto :failed

:python_find_error
echo [ERROR] Could not locate project-local Python %FORGE_PYTHON_VERSION%.
goto :failed

:python_version_error
echo [ERROR] The project Python version is not %FORGE_PYTHON_VERSION%.
goto :failed

:failed
del /q "%FORGE_PYTHON_PATH_FILE%" >nul 2>nul
endlocal
exit /b 1
