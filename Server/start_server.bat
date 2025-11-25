@echo off
rem Start BugClassifier server (Windows batch launcher)
rem Usage: double-click or run from cmd.exe / powershell

setlocal enabledelayedexpansion

:: Ensure Python is available
python --version >nul 2>&1
if errorlevel 1 (
  echo Python not found on PATH. Please install Python 3.9+ and retry.
  pause
  exit /b 1
)

:: Create venv if missing
if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv .venv
)

:: Upgrade pip and install requirements
echo Installing/updating dependencies (this may take a moment)...
.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install Python dependencies. Please check output above.
  pause
  exit /b 1
)

:: Run uvicorn via venv python
echo Starting BugClassifier API on http://localhost:8000
:: Run uvicorn referencing the local module when cwd is the Server folder
.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000

endlocal
pause
