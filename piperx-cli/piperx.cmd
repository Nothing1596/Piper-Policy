@echo off
setlocal
set "PYTHONPATH=%~dp0src"
set "PYTHONUTF8=1"
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -m piperx_middleware.cli %*
) else (
  python -m piperx_middleware.cli %*
)
exit /b %errorlevel%
