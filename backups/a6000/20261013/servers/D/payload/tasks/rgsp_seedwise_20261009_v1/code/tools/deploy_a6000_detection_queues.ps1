param(
    [string]$Workspace = "",
    [string]$DataSourceHost = "gpu3090-1",
    [string]$DataSourceRoot = "/mnt/disk1/zhangji/ig_dsssm_cross_task/data/cross_task_prepared/detection"
)

$ErrorActionPreference = "Stop"
$Workspace = if ($Workspace) { (Resolve-Path -LiteralPath $Workspace).Path } else { (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path }
$remoteRoot = "/home/powerleader/ig-ds-ssm"
$queue = Join-Path $Workspace "source_code\work_train_single\tools\queue_a6000_detection.sh"
$runner = Join-Path $Workspace "source_code\work_train_single\tools\run_sar4_full_suite.py"
$proxy = Join-Path $Workspace "source_code\work_train_single\tools\run_sarmix_proxy.py"
$evaluate = Join-Path $Workspace "source_code\work_train_single\tools\evaluate_sarmix.py"
$train = Join-Path $Workspace "source_code\work_train_single\mbyolo_train_sarair.py"
$stage = Join-Path $env:TEMP "ig-ds-ssm-deploy-code"
$dataStage = Join-Path $env:TEMP "ig-ds-ssm-deploy-data"

# B and D are currently reachable. A keeps its existing SADD queue; C is
# omitted until its SSH route is reliable, so deployment never blocks on it.
$jobs = @(
    @{
        Host = "powerleader-a6000-b"
        Id = "ig_dsssm_p3_airship_sd16_b16_seed0_a6000b_v1"
        Dataset = "AIR-SARShip-2.0"
        DataDir = "AIR_SARShip_2"
    },
    @{
        Host = "a6000-d"
        Id = "ig_dsssm_p3_srsdd_sd16_b16_seed0_a6000d_v1"
        Dataset = "SRSDD-v1.0"
        DataDir = "SRSDD_v1"
    }
)

foreach ($job in $jobs) {
    $hostName = $job.Host
    $dataDir = $job.DataDir
    $remoteDataRoot = "$remoteRoot/data/cross_task_prepared/detection"
    $remoteData = "$remoteDataRoot/$dataDir"
    $localData = Join-Path $dataStage $dataDir

    Write-Host "===== DEPLOY $($job.Dataset) -> $hostName ====="
    ssh -o BatchMode=yes $hostName "mkdir -p '$remoteRoot/work_train_single/tools' '$remoteRoot/logs' '$remoteDataRoot'"

    $ready = ssh -o BatchMode=yes $hostName "test -f '$remoteData/$dataDir.yaml' && echo READY || true"
    if ($ready -notmatch "READY") {
        # Stage remote data locally first. In this environment direct remote-
        # to-remote scp can return success while leaving only a partial tree.
        Remove-Item -LiteralPath $localData -Recurse -Force -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force -Path $dataStage | Out-Null
        scp -r "${DataSourceHost}:$DataSourceRoot/$dataDir" "$dataStage\"
        if (-not (Test-Path -LiteralPath (Join-Path $localData "$dataDir.yaml"))) {
            throw "Local staging failed: $localData"
        }
        ssh -o BatchMode=yes $hostName "rm -rf '$remoteData' && mkdir -p '$remoteDataRoot'"
        scp -r "$localData\" "${hostName}:$remoteDataRoot/"
    }

    $ready = ssh -o BatchMode=yes $hostName "test -f '$remoteData/$dataDir.yaml' && echo READY || true"
    if ($ready -notmatch "READY") {
        throw "Dataset synchronization failed: $hostName $remoteData"
    }

    # OpenSSH on Windows can misencode this workspace's Chinese path. Stage
    # files under an ASCII temp path before copying them to Linux.
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Copy-Item -LiteralPath $queue -Destination (Join-Path $stage "queue_a6000_detection.sh") -Force
    Copy-Item -LiteralPath $runner -Destination (Join-Path $stage "run_sar4_full_suite.py") -Force
    Copy-Item -LiteralPath $proxy -Destination (Join-Path $stage "run_sarmix_proxy.py") -Force
    Copy-Item -LiteralPath $evaluate -Destination (Join-Path $stage "evaluate_sarmix.py") -Force
    Copy-Item -LiteralPath $train -Destination (Join-Path $stage "mbyolo_train_sarair.py") -Force
    $toolFiles = @(
        (Join-Path $stage "queue_a6000_detection.sh"),
        (Join-Path $stage "run_sar4_full_suite.py"),
        (Join-Path $stage "run_sarmix_proxy.py"),
        (Join-Path $stage "evaluate_sarmix.py")
    )
    scp -p @toolFiles "${hostName}:$remoteRoot/work_train_single/tools/"
    scp -p (Join-Path $stage "mbyolo_train_sarair.py") "${hostName}:$remoteRoot/work_train_single/"
    ssh -o BatchMode=yes $hostName "chmod +x '$remoteRoot/work_train_single/tools/queue_a6000_detection.sh'"

    $launcher = "$remoteRoot/logs/$($job.Id).launcher.log"
    $command = "nohup '$remoteRoot/work_train_single/tools/queue_a6000_detection.sh' --experiment-id '$($job.Id)' --datasets '$($job.Dataset)' --state-dim 16 --batch 16 --eval-batch 16 --epochs 150 --patience 30 --workers 8 > '$launcher' 2>&1 < /dev/null & echo `$!"
    $remotePid = ssh -o BatchMode=yes $hostName $command
    Write-Host "STARTED $hostName PID=$remotePid LOG=$launcher"
}
