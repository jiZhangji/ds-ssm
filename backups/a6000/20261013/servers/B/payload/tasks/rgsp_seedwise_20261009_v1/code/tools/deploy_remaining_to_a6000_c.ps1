param(
    [string]$Target = "a6000-c-lan",
    [string]$RemoteRoot = "/home/powerleader/ig-ds-ssm",
    [string]$Stage = "C:\Users\13207\AppData\Local\Temp\ig-ds-ssm-c-deploy"
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$dataStage = "C:\Users\13207\AppData\Local\Temp\ig-ds-ssm-chunks"
$airDir = Join-Path $dataStage "AIR_SARShip_2"
$srsddTar = Join-Path $env:TEMP "ig-ds-ssm-tar-stage\SRSDD_v1.tar"

function Check($code, $message) {
    if ($code -ne 0) { throw $message }
}

New-Item -ItemType Directory -Force -Path $Stage | Out-Null
$tools = @(
    "source_code\work_train_single\mbyolo_train_sarair.py",
    "source_code\work_train_single\tools\run_sar4_full_suite.py",
    "source_code\work_train_single\tools\run_sarmix_proxy.py",
    "source_code\work_train_single\tools\evaluate_sarmix.py",
    "source_code\work_train_single\tools\queue_a6000_detection.sh"
)
$stagedTools = @()
foreach ($relative in $tools) {
    $name = Split-Path -Leaf $relative
    $destination = Join-Path $Stage $name
    Copy-Item -LiteralPath (Join-Path $workspace $relative) -Destination $destination -Force
    $stagedTools += $destination
}

ssh -o BatchMode=yes -o ConnectTimeout=15 $Target "mkdir -p '$RemoteRoot/work_train_single/tools' '$RemoteRoot/logs' '$RemoteRoot/data/cross_task_prepared/detection'"
Check $LASTEXITCODE "Cannot prepare A6000-C"
scp -p @($stagedTools[1..4]) "${Target}:$RemoteRoot/work_train_single/tools/"
Check $LASTEXITCODE "Code upload failed"
scp -p $stagedTools[0] "${Target}:$RemoteRoot/work_train_single/"
Check $LASTEXITCODE "Training entry upload failed"
ssh -o BatchMode=yes $Target "chmod +x '$RemoteRoot/work_train_single/tools/queue_a6000_detection.sh'"

Write-Host "Uploading SRSDD to A6000-C"
ssh -o BatchMode=yes $Target "rm -rf '$RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1'"
scp -p $srsddTar "${Target}:/tmp/SRSDD_v1.tar"
Check $LASTEXITCODE "SRSDD upload failed"
ssh -o BatchMode=yes $Target "tar -C '$RemoteRoot/data/cross_task_prepared/detection' -xf /tmp/SRSDD_v1.tar; sed -i 's#^path: .*#path: $RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1#' '$RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1/SRSDD_v1.yaml'; test -d '$RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1/images/train'; test -d '$RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1/images/val'; test -f '$RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1/SRSDD_v1.yaml'; rm -f /tmp/SRSDD_v1.tar"
Check $LASTEXITCODE "SRSDD validation failed"

Write-Host "Uploading AIR-SARShip chunks to A6000-C"
ssh -o BatchMode=yes $Target "rm -rf '$RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2'; rm -f /tmp/AIR_SARShip_2.part.* /tmp/AIR_SARShip_2.tar"
$chunks = Get-ChildItem -LiteralPath $airDir -File | Sort-Object Name
if ($chunks.Count -ne 9) { throw "Expected 9 AIR-SARShip chunks, found $($chunks.Count)" }
foreach ($chunk in $chunks) {
    Write-Host "Uploading $($chunk.Name)"
    scp -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -p $chunk.FullName "${Target}:/tmp/$($chunk.Name)"
    Check $LASTEXITCODE "AIR-SARShip chunk upload failed: $($chunk.Name)"
}
ssh -o BatchMode=yes $Target "cat /tmp/AIR_SARShip_2.part.* > /tmp/AIR_SARShip_2.tar; tar -C '$RemoteRoot/data/cross_task_prepared/detection' -xf /tmp/AIR_SARShip_2.tar; sed -i 's#^path: .*#path: $RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2#' '$RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2/AIR_SARShip_2.yaml'; test -d '$RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2/images/train'; test -d '$RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2/images/val'; test -f '$RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2/AIR_SARShip_2.yaml'; rm -f /tmp/AIR_SARShip_2.part.* /tmp/AIR_SARShip_2.tar"
Check $LASTEXITCODE "AIR-SARShip validation failed"

$experiment = "ig_dsssm_p3_remaining_c_v1"
$command = "nohup '$RemoteRoot/work_train_single/tools/queue_a6000_detection.sh' --experiment-id '$experiment' --datasets SRSDD-v1.0 AIR-SARShip-2.0 --state-dim 16 --batch 16 --eval-batch 16 --epochs 150 --patience 30 --workers 8 > '$RemoteRoot/logs/$experiment.launcher.log' 2>&1 < /dev/null & echo `$!"
$pid = ssh -o BatchMode=yes $Target $command
Check $LASTEXITCODE "Detection queue launch failed"
Write-Host "STARTED SERIAL SRSDD THEN AIR-SARShip ON A6000-C PID=$pid"
