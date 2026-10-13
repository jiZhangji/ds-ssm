param(
    [string]$DataSourceHost = "gpu3090-1",
    [string]$DataSourceRoot = "/mnt/disk1/zhangji/ig_dsssm_cross_task/data/cross_task_prepared/detection",
    [string]$RemoteRoot = "/home/powerleader/ig-ds-ssm",
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$stage = "C:\Users\13207\AppData\Local\Temp\ig-ds-ssm-tar-stage"
$log = if ($LogPath) { $LogPath } else { Join-Path $stage "sync.log" }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

function Log($message) {
    $line = "[{0}] {1}" -f (Get-Date -Format o), $message
    Add-Content -LiteralPath $log -Value $line
    Write-Host $line
}

function RemoteSize($path) {
    $value = ssh -o BatchMode=yes -o ConnectTimeout=15 $DataSourceHost "stat -c '%s' '$path'"
    if ($LASTEXITCODE -ne 0) { throw "Cannot stat remote archive: $path" }
    return [int64]($value.Trim())
}

function StageArchive($dataset) {
    $remote = "/tmp/$dataset.tar"
    $local = Join-Path $stage "$dataset.tar"
    $expected = RemoteSize $remote
    if ((Test-Path -LiteralPath $local) -and (Get-Item -LiteralPath $local).Length -eq $expected) {
        Log "STAGE_OK $dataset $expected bytes"
        return $local
    }
    Remove-Item -LiteralPath $local -Force -ErrorAction SilentlyContinue
    $part = "$local.part"
    Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
    Log "STAGE_START $dataset $expected bytes"
    scp -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "${DataSourceHost}:$remote" $part 2>&1 | Add-Content -LiteralPath $log
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $part)) { throw "Stage failed: $dataset" }
    if ((Get-Item -LiteralPath $part).Length -ne $expected) { throw "Stage size mismatch: $dataset" }
    Move-Item -LiteralPath $part -Destination $local -Force
    Log "STAGE_OK $dataset $expected bytes"
    return $local
}

function InstallArchive($archive, $dataset, $target) {
    $localSize = (Get-Item -LiteralPath $archive).Length
    $remoteArchive = "/tmp/$dataset.tar"
    $remoteData = "$RemoteRoot/data/cross_task_prepared/detection/$dataset"
    Log "UPLOAD_START $dataset -> $target"
    scp -o ServerAliveInterval=30 -o ServerAliveCountMax=10 $archive "${target}:$remoteArchive" 2>&1 | Add-Content -LiteralPath $log
    if ($LASTEXITCODE -ne 0) { throw "Upload failed: $dataset -> $target" }
    $remoteSize = ssh -o BatchMode=yes $target "stat -c '%s' '$remoteArchive'"
    if ($remoteSize.Trim() -ne "$localSize") { throw "Remote size mismatch: $dataset -> $target" }
    $parent = Split-Path -Path $remoteData -Parent
    ssh -o BatchMode=yes $target "rm -rf '$remoteData' && mkdir -p '$parent' && tar -C '$parent' -xf '$remoteArchive' && test -f '$remoteData/$dataset.yaml' && rm -f '$remoteArchive'"
    if ($LASTEXITCODE -ne 0) { throw "Extract/validation failed: $dataset -> $target" }
    Log "REMOTE_DATA_OK $dataset -> $target"
}

$air = StageArchive "AIR_SARShip_2"
$srsdd = StageArchive "SRSDD_v1"
InstallArchive $air "AIR_SARShip_2" "powerleader-a6000-b"
InstallArchive $srsdd "SRSDD_v1" "a6000-d"
Log "SYNC_COMPLETE"
