@echo off
rem Intel Arc uses the maintained official XPU environment.
call "%~dp0webui-user-intel-xpu-official.bat" %*
exit /b %errorlevel%
