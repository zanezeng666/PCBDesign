# Restart battery_designer service
# Usage: .\scripts\restart_service.ps1

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Stopping existing Python processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.MainWindowTitle -eq "" -or $_.CommandLine -match "battery_designer"
} | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

Write-Host "[2/3] Clearing cache..." -ForegroundColor Yellow
Remove-Item -Recurse -Force "$PSScriptRoot\..\battery_designer\__pycache__" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$PSScriptRoot\..\engine\__pycache__" -ErrorAction SilentlyContinue

Write-Host "[3/3] Starting service..." -ForegroundColor Yellow
Set-Location "$PSScriptRoot\.."
Start-Process -NoNewWindow -FilePath python -ArgumentList "-m battery_designer.app"

Write-Host "Service running at http://127.0.0.1:8000" -ForegroundColor Green
