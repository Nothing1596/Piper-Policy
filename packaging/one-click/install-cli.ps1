param(
    [string]$PackageRoot = $PSScriptRoot,
    [string]$PythonExe = '',
    [string]$BinDir = (Join-Path $env:USERPROFILE '.piper\bin'),
    [switch]$NoPythonInstall,
    [switch]$NoRegister
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
# A launcher inherited from PowerShell 7 may omit Windows PowerShell modules.
$env:PSModulePath = (Join-Path $PSHOME 'Modules') + ';' + $env:PSModulePath

function Check-Python([string]$Exe) {
    if (-not $Exe -or -not (Test-Path -LiteralPath $Exe -PathType Leaf)) { return $false }
    try {
        & $Exe -c 'import sys,struct;sys.exit(0 if sys.version_info[:2]==(3,12) and struct.calcsize(chr(80))==8 else 1)' 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Find-Python {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $found = & $launcher.Source -3.12 -c 'import sys;print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and (Check-Python "$found")) { return "$found" }
        } catch { }
    }
    $locations = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:USERPROFILE 'AppData\Local\Programs\Python\Python312\python.exe')
    )
    foreach ($item in $locations) { if (Check-Python $item) { return $item } }
    return $null
}

$PackageRoot = (Resolve-Path -LiteralPath $PackageRoot).Path
if ($PackageRoot -match '[%"\r\n]') { throw 'Package path contains characters unsupported by CMD launchers.' }
$manifestFile = Join-Path $PackageRoot 'SHA256.json'
$lockFile = Join-Path $PackageRoot 'requirements-win-py312.lock'
if (-not (Test-Path -LiteralPath $manifestFile) -or -not (Test-Path -LiteralPath $lockFile)) {
    throw 'Extract the standalone offline kit first. Put Setup.cmd and install-cli.ps1 next to SHA256.json.'
}
$videoWheels = @(Get-ChildItem -LiteralPath (Join-Path $PackageRoot 'wheels') -Filter 'piper_lab-*.whl')
$robotWheels = @(Get-ChildItem -LiteralPath (Join-Path $PackageRoot 'wheels') -Filter 'piperx_middleware-*.whl')
if ($videoWheels.Count -eq 1 -and $robotWheels.Count -eq 0) { $component='video'; $module='piperlab.video_entry' }
elseif ($robotWheels.Count -eq 1 -and $videoWheels.Count -eq 0) { $component='robot'; $module='piperx_middleware.standalone_cli' }
else { throw 'Expected exactly one standalone project wheel. Combined kits are not supported by this installer.' }
$command = "piper-$component"
Write-Host "Checking $command package hashes..."
$prefix = $PackageRoot.TrimEnd('\') + '\'
foreach ($entry in (Get-Content -LiteralPath $manifestFile -Raw | ConvertFrom-Json)) {
    $file = [IO.Path]::GetFullPath((Join-Path $PackageRoot $entry.path))
    if (-not $file.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Manifest path escapes package.' }
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $entry.sha256) { throw "Hash mismatch: $($entry.path)" }
}

if ($PythonExe) {
    if (-not (Check-Python $PythonExe)) { throw '-PythonExe must be Python 3.12 x64.' }
} else { $PythonExe = Find-Python }
if (-not $PythonExe) {
    if ($NoPythonInstall) { throw 'Python 3.12 x64 not found. Install it or pass -PythonExe.' }
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) { throw 'Python and winget are missing. Install Python 3.12 x64 from python.org, then rerun Setup.cmd.' }
    Write-Host 'Installing Python 3.12 x64 for this user through winget (internet required)...'
    & $winget.Source install --id Python.Python.3.12 --exact --source winget --scope user --architecture x64 --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
    if ($LASTEXITCODE -ne 0) { throw "Python installation failed (winget exit $LASTEXITCODE)." }
    $PythonExe = Find-Python
    if (-not $PythonExe) { throw 'Python installer finished but Python 3.12 x64 could not be verified. Reopen Setup or specify -PythonExe.' }
}

$venv = Join-Path $PackageRoot '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
if (Test-Path -LiteralPath $venv) {
    if (-not (Check-Python $python)) { throw 'Existing .venv is not usable Python 3.12 x64. Preserve it and extract a fresh kit.' }
} else {
    & $PythonExe -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $python -m pip install --no-index --find-links (Join-Path $PackageRoot 'wheels') --require-hashes -r $lockFile
if ($LASTEXITCODE -ne 0) { throw 'Offline dependency installation failed. CLI was not registered.' }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
& $python -m $module --help
if ($LASTEXITCODE -ne 0) { throw 'CLI startup check failed.' }
if ($NoRegister) { Write-Host 'Installed and checked. PATH registration skipped.'; exit 0 }

$BinDir = [IO.Path]::GetFullPath($BinDir)
if ($BinDir -match '[;%"\r\n]') { throw 'Unsupported registration directory.' }
[IO.Directory]::CreateDirectory($BinDir) | Out-Null
$wrapper = Join-Path $BinDir "$command.cmd"
$marker = 'rem Managed by Piper one-click installer v1'
if ((Test-Path -LiteralPath $wrapper) -and -not (Get-Content -LiteralPath $wrapper -Raw).Contains($marker)) {
    throw "An unrelated command already exists at $wrapper; it was not overwritten."
}
$receiptFile = Join-Path $BinDir "$command.install.json"
$key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Environment')
try {
    $oldPath = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    $parts = @($oldPath -split ';' | Where-Object { $_ })
    $present = @($parts | Where-Object { [Environment]::ExpandEnvironmentVariables($_).TrimEnd('\') -ieq $BinDir.TrimEnd('\') }).Count -gt 0
    if (-not (Test-Path -LiteralPath $receiptFile)) {
        @{ user_path_before=$oldPath; bin=$BinDir; command=$command } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $BinDir "$command.path-backup.json") -Encoding UTF8
    }
    $body = "@echo off`r`n$marker`r`nsetlocal DisableDelayedExpansion`r`n`"$python`" -m $module %*`r`nexit /b %errorlevel%`r`n"
    [IO.File]::WriteAllText($wrapper, $body, [Text.Encoding]::Default)
    if (-not $present) {
        $newPath = if ($oldPath) { $oldPath.TrimEnd(';') + ';' + $BinDir } else { $BinDir }
        $key.SetValue('Path', $newPath, [Microsoft.Win32.RegistryValueKind]::ExpandString)
    }
} finally { $key.Close() }
$env:Path = $BinDir + ';' + $env:Path
& $wrapper --help
if ($LASTEXITCODE -ne 0) { throw 'Registered launcher check failed.' }
@{ component=$component; package_root=$PackageRoot; python=$python; module=$module; wrapper=$wrapper;
   wrapper_sha256=(Get-FileHash -LiteralPath $wrapper -Algorithm SHA256).Hash; installed_at=(Get-Date -Format o) } |
    ConvertTo-Json | Set-Content -LiteralPath $receiptFile -Encoding UTF8
# Notify Explorer so newly launched terminals inherit the updated user PATH.
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class PiperPathNotify { [DllImport("user32.dll", CharSet=CharSet.Auto, SetLastError=true)] public static extern IntPtr SendMessageTimeout(IntPtr h, uint m, UIntPtr w, string l, uint f, uint t, out UIntPtr r); }'
$notifyResult = [UIntPtr]::Zero
[PiperPathNotify]::SendMessageTimeout([IntPtr]0xffff, 0x1a, [UIntPtr]::Zero, 'Environment', 2, 2000, [ref]$notifyResult) | Out-Null
Write-Host "Installed and registered: $command"
Write-Host "Open a NEW terminal and run: $command --help"
Write-Host "Keep this package directory in place: $PackageRoot"
