@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py
) else if exist "..\..\.venv\Scripts\python.exe" (
  "..\..\.venv\Scripts\python.exe" app.py
) else (
  echo Python environment not found. See the setup instructions in this folder.
  pause
  exit /b 1
)
if errorlevel 1 pause
