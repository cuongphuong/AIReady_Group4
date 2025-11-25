<#
PowerShell launcher for BugClassifier API
Run from PowerShell:
  .\run_server.ps1

The script will:
- Ensure a .venv exists (create it if missing)
- Install/upgrade dependencies from requirements.txt
- Run the server with uvicorn
#>

$ErrorActionPreference = 'Stop'
$venvPath = Join-Path $PSScriptRoot '.venv'
$pythonExe = Join-Path $venvPath 'Scripts\python.exe'

function Ensure-Venv {
    if (-not (Test-Path $pythonExe)) {
        Write-Host "Creating virtual environment..."
        python -m venv $venvPath
    }
}

function Install-Requirements {
    Write-Host "Installing dependencies (may take a while)..."
    & $pythonExe -m pip install --upgrade pip
    & $pythonExe -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
}

function Start-Server {
    Write-Host "Starting BugClassifier API at http://localhost:8000"
    # Run uvicorn using the local module name (when running from Server/)
    & $pythonExe -m uvicorn api:app --reload --port 8000
}

try {
    Ensure-Venv
    Install-Requirements
    Start-Server
} catch {
    Write-Error "Failed to start server: $_"
    exit 1
}
