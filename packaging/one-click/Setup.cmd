@echo off
setlocal DisableDelayedExpansion
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-cli.ps1" %*
set "result=%errorlevel%"
if not "%result%"=="0" echo Installation failed. Read the error above; no automatic retry.
if not defined PIPER_SETUP_NO_PAUSE pause
exit /b %result%
