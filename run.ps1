$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing .venv. Create it and install requirements.txt first."
}
& $Python -m battery_designer.app
