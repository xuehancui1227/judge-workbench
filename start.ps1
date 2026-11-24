$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python (Join-Path $PSScriptRoot 'launcher.py') @args
if ($LASTEXITCODE -ne 0) { throw 'Launcher failed. See the output above.' }
