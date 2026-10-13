param(
    [string]$Target = "gpu3090-1",
    [string]$RemoteRoot = "/mnt/disk1/zhangji/ds_ssm_sarmix",
    [string]$DataRoot = "/mnt/disk1/zhangji/ig_dsssm_cross_task/data/cross_task_prepared/detection"
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$stage = Join-Path $env:TEMP "ig-ds-ssm-deploy-3090-1"
$files = @(
    "source_code\work_train_single\mbyolo_train_sarair.py",
    "source_code\work_train_single\tools\run_sar4_full_suite.py",
    "source_code\work_train_single\tools\run_sarmix_proxy.py",
    "source_code\work_train_single\tools\evaluate_sarmix.py",
    "source_code\work_train_single\tools\queue_a6000_detection.sh"
)
function Check([int]$code, [string]$message) { if ($code -ne 0) { throw $message } }
function Send-File([string]$local, [string]$remote) {
    $cmd = "ssh -o BatchMode=yes -o ConnectTimeout=20 $Target `"cat > '$remote'`" < `"$local`""
    cmd.exe /d /c $cmd
    Check $LASTEXITCODE "Upload failed: $local"
}

Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null
ssh -o BatchMode=yes -o ConnectTimeout=15 $Target "mkdir -p '$RemoteRoot/work_train_single/tools' '$RemoteRoot/logs' '$RemoteRoot/experiments/cross_task_detection'"
Check $LASTEXITCODE "Cannot prepare $Target"
foreach ($relative in $files) {
    $local = Join-Path $workspace $relative
    $name = Split-Path -Leaf $relative
    Copy-Item -LiteralPath $local -Destination (Join-Path $stage $name) -Force
    $remote = if ($name -eq "mbyolo_train_sarair.py") { "$RemoteRoot/work_train_single/$name" } else { "$RemoteRoot/work_train_single/tools/$name" }
    Send-File (Join-Path $stage $name) $remote
}
ssh -o BatchMode=yes $Target "chmod +x '$RemoteRoot/work_train_single/tools/queue_a6000_detection.sh'"
Check $LASTEXITCODE "Cannot chmod queue"

$id = "ig_dsssm_p3_remaining_3090_1_v1"
$log = "$RemoteRoot/logs/$id.launcher.log"
$cmd = "ROOT='$RemoteRoot' CODE='$RemoteRoot/work_train_single' PYTHON_BIN='/mnt/disk1/zhangji/ig_dsssm_cross_task/envs/ig-ds-ssm-env/bin/python' DATA_ROOT='$DataRoot' PROJECT='$RemoteRoot/experiments/cross_task_detection' nohup '$RemoteRoot/work_train_single/tools/queue_a6000_detection.sh' --experiment-id '$id' --datasets SRSDD-v1.0 AIR-SARShip-2.0 --state-dim 16 --batch 16 --eval-batch 16 --epochs 150 --patience 30 --workers 8 > '$log' 2>&1 < /dev/null & echo `$!"
$remotePid = ssh -o BatchMode=yes $Target $cmd
Check $LASTEXITCODE "Queue launch failed"
Write-Host "STARTED SERIAL QUEUE ON 3090-1 PID=$remotePid LOG=$log"
