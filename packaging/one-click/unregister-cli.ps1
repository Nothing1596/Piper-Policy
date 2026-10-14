param(
    [Parameter(Mandatory=$true)][ValidateSet('video','robot')][string]$Component,
    [string]$BinDir = (Join-Path $env:USERPROFILE '.piper\bin')
)
$ErrorActionPreference='Stop'
$env:PSModulePath = (Join-Path $PSHOME 'Modules') + ';' + $env:PSModulePath
$BinDir=[IO.Path]::GetFullPath($BinDir)
$receiptFile=Join-Path $BinDir "piper-$Component.install.json"
if (-not (Test-Path -LiteralPath $receiptFile)) { throw 'No registration receipt found.' }
$receipt=Get-Content -LiteralPath $receiptFile -Raw | ConvertFrom-Json
$wrapper=Join-Path $BinDir "piper-$Component.cmd"
if ([IO.Path]::GetFullPath($receipt.wrapper) -ne $wrapper) { throw 'Receipt path mismatch.' }
if (Test-Path -LiteralPath $wrapper) {
    if ((Get-FileHash -LiteralPath $wrapper -Algorithm SHA256).Hash -ne $receipt.wrapper_sha256) { throw 'Launcher changed; preserve it and inspect manually.' }
    Remove-Item -LiteralPath $wrapper
}
Remove-Item -LiteralPath $receiptFile
if (@(Get-ChildItem -LiteralPath $BinDir -Filter '*.cmd').Count -eq 0) {
    $key=[Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Environment')
    try {
        $value=[string]$key.GetValue('Path','',[Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        $parts=@($value -split ';' | Where-Object { [Environment]::ExpandEnvironmentVariables($_).TrimEnd('\') -ine $BinDir.TrimEnd('\') })
        $key.SetValue('Path',($parts -join ';'),[Microsoft.Win32.RegistryValueKind]::ExpandString)
    } finally { $key.Close() }
}
Write-Host "Unregistered piper-$Component. Package, Python, data and other PATH entries were preserved. Reopen your terminal."
