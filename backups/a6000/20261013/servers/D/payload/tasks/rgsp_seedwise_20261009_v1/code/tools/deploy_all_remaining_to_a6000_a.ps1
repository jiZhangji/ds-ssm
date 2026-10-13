param(
    [string]$Target = "powerleader-ts",
    [string]$RemoteRoot = "/home/powerleader/ig-ds-ssm"
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$stage = Join-Path $env:TEMP "ig-ds-ssm-deploy-code-v3"
$srsddTar = Join-Path $env:TEMP "ig-ds-ssm-tar-stage\SRSDD_v1.tar"
$airDir = Join-Path $env:TEMP "ig-ds-ssm-chunks\AIR_SARShip_2"
$relativeFiles = @(
    "source_code\work_train_single\mbyolo_train_sarair.py",
    "source_code\work_train_single\tools\run_sar4_full_suite.py",
    "source_code\work_train_single\tools\run_sarmix_proxy.py",
    "source_code\work_train_single\tools\evaluate_sarmix.py",
    "source_code\work_train_single\tools\queue_a6000_detection.sh"
)

function Check([int]$code, [string]$message) { if ($code -ne 0) { throw $message } }
function Send-File([string]$target, [string]$localPath, [string]$remotePath) {
    # Windows OpenSSH SFTP is unreliable on these hosts; stream bytes via ssh stdin.
    $command = "ssh -o BatchMode=yes -o ConnectTimeout=20 $target `"cat > '$remotePath'`" < `"$localPath`""
    cmd.exe /d /c $command
    Check $LASTEXITCODE "Binary upload failed: $localPath -> $target`:$remotePath"
}
if (-not (Test-Path -LiteralPath $srsddTar)) { throw "Missing SRSDD archive" }
$chunks = @(Get-ChildItem -LiteralPath $airDir -File | Sort-Object Name)
if ($chunks.Count -ne 9) { throw "Expected 9 AIR-SARShip chunks, found $($chunks.Count)" }

Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null
foreach ($relative in $relativeFiles) {
    Copy-Item -LiteralPath (Join-Path $workspace $relative) -Destination (Join-Path $stage (Split-Path -Leaf $relative)) -Force
}

ssh -o BatchMode=yes -o ConnectTimeout=15 $Target "mkdir -p '$RemoteRoot/work_train_single/tools' '$RemoteRoot/logs' '$RemoteRoot/data/cross_task_prepared/detection'"
Check $LASTEXITCODE "Cannot prepare A6000-A"
Send-File $Target (Join-Path $stage "run_sar4_full_suite.py") "$RemoteRoot/work_train_single/tools/run_sar4_full_suite.py"
Send-File $Target (Join-Path $stage "run_sarmix_proxy.py") "$RemoteRoot/work_train_single/tools/run_sarmix_proxy.py"
Send-File $Target (Join-Path $stage "evaluate_sarmix.py") "$RemoteRoot/work_train_single/tools/evaluate_sarmix.py"
Send-File $Target (Join-Path $stage "queue_a6000_detection.sh") "$RemoteRoot/work_train_single/tools/queue_a6000_detection.sh"
Send-File $Target (Join-Path $stage "mbyolo_train_sarair.py") "$RemoteRoot/work_train_single/mbyolo_train_sarair.py"
ssh -o BatchMode=yes $Target "chmod +x '$RemoteRoot/work_train_single/tools/queue_a6000_detection.sh'"
Check $LASTEXITCODE "Cannot chmod queue"

Write-Host "Uploading SRSDD"
ssh -o BatchMode=yes $Target "rm -rf '$RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1'; rm -f /tmp/SRSDD_v1.tar"
Send-File $Target $srsddTar "/tmp/SRSDD_v1.tar"
ssh -o BatchMode=yes $Target "tar -C '$RemoteRoot/data/cross_task_prepared/detection' -xf /tmp/SRSDD_v1.tar; sed -i 's#^path: .*#path: $RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1#' '$RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1/SRSDD_v1.yaml'; test \$(find '$RemoteRoot/data/cross_task_prepared/detection/SRSDD_v1/images' -type f | wc -l) -gt 0; rm -f /tmp/SRSDD_v1.tar"
Check $LASTEXITCODE "SRSDD validation failed"

Write-Host "Uploading AIR-SARShip chunks"
ssh -o BatchMode=yes $Target "rm -rf '$RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2'; rm -f /tmp/AIR_SARShip_2.part.* /tmp/AIR_SARShip_2.tar"
foreach ($chunk in $chunks) {
    Write-Host $chunk.Name
    Send-File $Target $chunk.FullName "/tmp/$($chunk.Name)"
}
ssh -o BatchMode=yes $Target "cat /tmp/AIR_SARShip_2.part.* > /tmp/AIR_SARShip_2.tar; tar -C '$RemoteRoot/data/cross_task_prepared/detection' -xf /tmp/AIR_SARShip_2.tar; sed -i 's#^path: .*#path: $RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2#' '$RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2/AIR_SARShip_2.yaml'; test \$(find '$RemoteRoot/data/cross_task_prepared/detection/AIR_SARShip_2/images' -type f | wc -l) -gt 0; rm -f /tmp/AIR_SARShip_2.part.* /tmp/AIR_SARShip_2.tar"
Check $LASTEXITCODE "AIR-SARShip validation failed"

$id = "ig_dsssm_p3_remaining_a6000a_v1"
$launch = "nohup '$RemoteRoot/work_train_single/tools/queue_a6000_detection.sh' --experiment-id '$id' --datasets SRSDD-v1.0 AIR-SARShip-2.0 --state-dim 16 --batch 16 --eval-batch 16 --epochs 150 --patience 30 --workers 8 > '$RemoteRoot/logs/$id.launcher.log' 2>&1 < /dev/null & echo `$!"
$pid = ssh -o BatchMode=yes $Target $launch
Check $LASTEXITCODE "Queue launch failed"
Write-Host "STARTED SERIAL QUEUE ON A6000-A PID=$pid"
