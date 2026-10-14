@echo off
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Run install.ps1 first. Python 3.12 x64 is required.
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0piper.py" %*
exit /b %errorlevel%
