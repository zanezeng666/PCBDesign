$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing .venv. Create it and install requirements.txt first."
}
# 本地 KiCad 安装路径（kicad-cli + 自带 python/pcbnew），供生成制造文件使用
if (-not $env:KICAD_BIN) { $env:KICAD_BIN = "E:\KiCad\bin" }
& $Python -m battery_designer.app
