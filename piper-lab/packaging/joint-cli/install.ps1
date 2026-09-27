param([string]$PythonExe = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (Test-Path -LiteralPath '.venv') { throw 'Existing .venv found; use a fresh extraction directory.' }
$manifest = Get-Content -LiteralPath 'SHA256.json' -Raw | ConvertFrom-Json
foreach ($entry in $manifest) {
    $fullPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $entry.path))
    $prefix = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') + '\'
    if (-not $fullPath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid manifest path' }
    if ((Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash -ne $entry.sha256) { throw "Hash mismatch: $($entry.path)" }
}
if ($PythonExe) { $pythonCommand = $PythonExe; $pythonArgs = @() }
else { $pythonCommand = 'py'; $pythonArgs = @('-3.12') }
& $pythonCommand @pythonArgs check_python.py
if ($LASTEXITCODE -ne 0) { throw 'Python prerequisite failed' }
& $pythonCommand @pythonArgs -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
& $python -m pip install --no-index --find-links wheels --require-hashes -r requirements-win-py312.lock
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; keep logs and use a fresh extraction to retry' }
& $python -m pip install --no-index --no-deps wheels/piper_lab-0.2.1-py3-none-any.whl wheels/piperx_middleware-0.5.0-py3-none-any.whl
if ($LASTEXITCODE -ne 0) { throw 'Project wheel installation failed' }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& $python piper.py doctor
if ($LASTEXITCODE -ne 0) { throw 'CLI doctor failed' }
Write-Output 'Installed. Run .\piper.cmd --help. No hardware or model was started.'
