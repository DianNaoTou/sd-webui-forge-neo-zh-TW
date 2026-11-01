param(
    [string]$Root = 'D:\sd-webui-forge-neo-zh-TW-test',
    [string]$Revision,
    [string]$Rollback,
    [switch]$NoLaunch,
    [switch]$RebuildEnvironment,
    [string]$Python,
    [string]$ModelRoot
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path -LiteralPath $Root).Path
# Do not replace code while a project interpreter is running. This check runs
# before starting our own Python helper, and never terminates another process.
$running = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and (
        ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($Root + '\', [StringComparison]::OrdinalIgnoreCase)) -or
        ($_.CommandLine -and $_.CommandLine.IndexOf($Root, [StringComparison]::OrdinalIgnoreCase) -ge 0)
    )
})
if ($running.Count) { throw 'Close Forge/project Python processes before updating or rolling back. Nothing was changed.' }
$candidates = @()
if ($Python) { $candidates += $Python }
else {
    $managed = @(Get-ChildItem -Path (Join-Path $Root 'runtime\python\*\python.exe') -File -ErrorAction SilentlyContinue)
    $candidates += @($managed | ForEach-Object { $_.FullName })
    $candidates += @((Join-Path $Root 'venv-xpu-official\Scripts\python.exe'), (Join-Path $Root 'venv\Scripts\python.exe'))
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
}
$Python = $null
foreach ($candidate in $candidates) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
    try {
        & $candidate -I -c 'import sys, venv; sys.exit(0 if sys.version_info[:2] == (3, 13) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { $Python = $candidate; break }
    } catch { continue }
}
if (-not $Python) { throw 'A working Python 3.13 was not found. Pass -Python with your existing base interpreter. No environment was removed.' }
if ($Rollback) {
    $helper = Join-Path $Rollback 'repair_model_selection.py'
    if (-not (Test-Path -LiteralPath $helper -PathType Leaf)) { throw 'Rollback helper is missing from the backup directory.' }
    & $Python -B $helper --root $Root --rollback $Rollback
    if ($LASTEXITCODE -ne 0) { throw 'Rollback stopped; read the error above. Backups were kept.' }
    return
}
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Pass the full reviewed repair commit SHA using -Revision. Branch names are deliberately not accepted.' }
& git -C $Root fetch --no-tags https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW.git $Revision
if ($LASTEXITCODE -ne 0) { throw 'Could not fetch the pinned repair commit. No program files were changed.' }
$source = & git -C $Root show "${Revision}:tools/repair_model_selection.py"
if ($LASTEXITCODE -ne 0) { throw 'Pinned repair helper was not found.' }
$helper = Join-Path ([IO.Path]::GetTempPath()) ('forge-repair-' + [guid]::NewGuid().ToString('N') + '.py')
[IO.File]::WriteAllText($helper, ($source -join "`n"), [Text.UTF8Encoding]::new($false))
$arguments = @('-B', $helper, '--root', $Root, '--revision', $Revision)
if ($NoLaunch) { $arguments += '--no-launch' }
if ($RebuildEnvironment) { $arguments += '--rebuild-environment' }
if ($ModelRoot) { $arguments += @('--models', $ModelRoot) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Repair stopped; see the backup path and error above. Do not delete the backup.' }
