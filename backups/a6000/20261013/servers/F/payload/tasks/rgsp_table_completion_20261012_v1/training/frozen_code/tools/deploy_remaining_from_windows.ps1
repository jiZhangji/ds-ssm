param(
    [string]$AirTarget = "powerleader-ts",
    [string]$SrsddTarget = "a6000-d",
    [string]$RemoteRoot = "/home/powerleader/ig-ds-ssm"
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$stage = Join-Path $env:TEMP "ig-ds-ssm-deploy-code-v2"
$srsddTar = Join-Path $env:TEMP "ig-ds-ssm-tar-stage\SRSDD_v1.tar"
$airDir = Join-Path $env:TEMP "ig-ds-ssm-chunks\AIR_SARShip_2"
$codeFiles = @(
    "source_code\work_train_single\mbyolo_train_sarair.py",
    "source_code\work_train_single\tools\run_sar4_full_suite.py",
    "source_code\work_train_single\tools\run_sarmix_proxy.py",
    "source_code\work_train_single\tools\evaluate_sarmix.py",
    "source_code\work_train_single\tools\queue_a6000_detection.sh"
)

function Check([int]$code, [string]$message) {
    if ($code -ne 0) { throw $message }
}

if (-not (Test-Path -LiteralPath $srsddTar)) { throw "Missing SRSDD archive: $srsddTar" }
$chunks = @(Get-ChildItem -LiteralPath $airDir -File | Sort-Object Name)
if ($chunks.Count -ne 9) { throw "Expected 9 AIR-SARShip chunks, found $($chunks.Count)" }

Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null
foreach ($relative in $codeFiles) {
    Copy-Item -LiteralPath (Join-Path $workspace $relative) -Destination (Join-Path $stage (Split-Path -Leaf $relative)) -Force
}

function Upload-Code([string]$target) {
    ssh -o BatchMode=yes -o ConnectTimeout=15 $target "mkdir -p '$RemoteRoot/work_train_single/tools' '$RemoteRoot/logs' '$RemoteRoot/data/cross_task_prepared/detection'"
    Check $LASTEXITCODE "Cannot prepare $target"
    scp -p (Join-Path $stage "run_sar4_full_suite.py"), (Join-Path $stage "run_sarmix_proxy.py"), (Join-Path $stage "evaluate_sarmix.py"), (Join-Path $stage "queue_a6000_detection.sh") "${target}:$RemoteRoot/work_train_single/tools/"
    Check $LASTEXITCODE "Tool upload failed on $target"
    scp -p (Join-Path $stage "mbyolo_train_sarair.py") "${target}:$RemoteRoot/work_train_single/"
    Check $LASTEXITCODE "Training entry upload failed on $target"
    ssh -o BatchMode=yes $target "chmod +x '$RemoteRoot/work_train_single/tools/queue_a6000_detection.sh'"
    Check $LASTEXITCODE "Cannot chmod queue on $target"
}

Write-Host "Preparing and uploading code to $SrsddTarget"
Upload-Code $SrsddTarget
Write-Host "Uploading SRSDD archive to $SrsddTarget"
ssh -o BatchMode=yes $SrsddTarget "rm -rf '$RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1'; mkdir -p '$RemoteRoot/data/cross_task_prepared/detection'"
scp -p $srsddTar "${SrsddTarget}:/tmp/SRSDD_v1.tar"
Check $LASTEXITCODE "SRSDD archive upload failed"
ssh -o BatchMode=yes $SrsddTarget "tar -C '$RemoteRoot/data/cross_task_prepared/detection' -xf /tmp/SRSDD_v1.tar; sed -i 's#^path: .*#path: $RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1#' '$RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1/SRSDD_v1.yaml'; test -s '$RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1/SRSDD_v1.yaml'; test \$(find '$RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1/images' -type f | wc -l) -gt 0; rm -f /tmp/SRSDD_v1.tar"
Check $LASTEXITCODE "SRSDD remote validation failed"
$srsddId = "ig_dsssm_p3_srsdd_sd16_b16_seed0_a6000d_v2"
$srsddLaunch = "nohup '$RemoteRoot/work_train_single/tools/queue_a6000_detection.sh' --experiment-id '$srsddId' --datasets SRSDD-v1.0 --state-dim 16 --batch 16 --eval-batch 16 --epochs 150 --patience 30 --workers 8 > '$RemoteRoot/logs/$srsddId.launcher.log' 2>&1 < /dev/null & echo `$!"
$srsddPid = ssh -o BatchMode=yes $SrsddTarget $srsddLaunch
Check $LASTEXITCODE "SRSDD queue launch failed"
Write-Host "STARTED SRSDD on $SrsddTarget PID=$srsddPid"

Write-Host "Preparing and uploading code to $AirTarget"
Upload-Code $AirTarget
Write-Host "Uploading AIR-SARShip chunks to $AirTarget"
ssh -o BatchMode=yes $AirTarget "rm -rf '$RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2'; rm -f /tmp/AIR_SARShip_2.part.* /tmp/AIR_SARShip_2.tar; mkdir -p '$RemoteRoot/data/cross_task_prepared/detection'"
foreach ($chunk in $chunks) {
    Write-Host "Uploading $($chunk.Name)"
    scp -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -p $chunk.FullName "${AirTarget}:/tmp/$($chunk.Name)"
    Check $LASTEXITCODE "AIR-SARShip chunk upload failed: $($chunk.Name)"
}
ssh -o BatchMode=yes $AirTarget "cat /tmp/AIR_SARShip_2.part.* > /tmp/AIR_SARShip_2.tar; tar -C '$RemoteRoot/data/cross_task_prepared/detection' -xf /tmp/AIR_SARShip_2.tar; sed -i 's#^path: .*#path: $RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2#' '$RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2/AIR_SARShip_2.yaml'; test -s '$RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2/AIR_SARShip_2.yaml'; test \$(find '$RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2/images' -type f | wc -l) -gt 0; rm -f /tmp/AIR_SARShip_2.part.* /tmp/AIR_SARShip_2.tar"
Check $LASTEXITCODE "AIR-SARShip remote validation failed"
$airId = "ig_dsssm_p3_airship_sd16_b16_seed0_a6000a_v2"
$airLaunch = "nohup '$RemoteRoot/work_train_single/tools/queue_a6000_detection.sh' --experiment-id '$airId' --datasets AIR-SARShip-2.0 --state-dim 16 --batch 16 --eval-batch 16 --epochs 150 --patience 30 --workers 8 > '$RemoteRoot/logs/$airId.launcher.log' 2>&1 < /dev/null & echo `$!"
$airPid = ssh -o BatchMode=yes $AirTarget $airLaunch
Check $LASTEXITCODE "AIR-SARShip queue launch failed"
Write-Host "STARTED AIR-SARShip on $AirTarget PID=$airPid"
