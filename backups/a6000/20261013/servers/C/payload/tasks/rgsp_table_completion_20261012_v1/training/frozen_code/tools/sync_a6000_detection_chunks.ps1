param(
    [string]$SourceHost = "gpu3090-1",
    [string]$SourceRoot = "/mnt/disk1/zhangji/ig_dsssm_cross_task/data/cross_task_prepared/detection",
    [string]$RemoteRoot = "/home/powerleader/ig-ds-ssm",
    [string]$ChunkBytes = "64m"
)

$ErrorActionPreference = "Stop"
$stage = "C:\Users\13207\AppData\Local\Temp\ig-ds-ssm-chunks"
New-Item -ItemType Directory -Force -Path $stage | Out-Null

function RunRemote([string]$remoteHost, [string]$command) {
    $out = ssh -o BatchMode=yes -o ConnectTimeout=20 $remoteHost $command
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed on ${remoteHost}: $command" }
    return [string]::Join("`n", @($out))
}

function SyncDataset([string]$dataset, [string]$target) {
    $prefix = "/tmp/$dataset.part."
    $archive = "/tmp/$dataset.tar"
    $localDir = Join-Path $stage $dataset
    New-Item -ItemType Directory -Force -Path $localDir | Out-Null
    Write-Host "===== ${dataset}: split on $SourceHost ====="
    RunRemote $SourceHost "test -f '$archive' || tar -C '$SourceRoot' -cf '$archive' '$dataset'"
    RunRemote $SourceHost "rm -f '${prefix}*' && split -b $ChunkBytes -d -a 3 '$archive' '$prefix'"
    $chunks = @((RunRemote $SourceHost "ls -1 '${prefix}'* | sort").Trim().Split([Environment]::NewLine) | Where-Object { $_ })
    if ($chunks.Count -eq 0) { throw "No chunks generated for $dataset" }
    $expectedFiles = [int](RunRemote $SourceHost "tar -tf '$archive' | wc -l").Trim()
    Write-Host "chunks=$($chunks.Count) expected_files=$expectedFiles"

    foreach ($remoteChunk in $chunks) {
        $name = Split-Path -Leaf $remoteChunk.Trim()
        $local = Join-Path $localDir $name
        $remoteSize = [int64](RunRemote $SourceHost "stat -c '%s' '$remoteChunk'").Trim()
        if ((Test-Path -LiteralPath $local) -and (Get-Item -LiteralPath $local).Length -eq $remoteSize) {
            Write-Host "LOCAL_OK $name"
        } else {
            $part = "$local.part"
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
            scp -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "${SourceHost}:$remoteChunk" $part
            if ($LASTEXITCODE -ne 0 -or (Get-Item -LiteralPath $part).Length -ne $remoteSize) { throw "Download mismatch: $name" }
            Move-Item -LiteralPath $part -Destination $local -Force
            Write-Host "LOCAL_OK $name"
        }
    }

    $remoteData = "$RemoteRoot/data/cross_task_prepared/detection/$dataset"
    RunRemote $target "mkdir -p '$RemoteRoot/data/cross_task_prepared/detection' '$RemoteRoot/work_train_single/tools' && rm -rf '$remoteData'"
    foreach ($remoteChunk in $chunks) {
        $name = Split-Path -Leaf $remoteChunk.Trim()
        $local = Join-Path $localDir $name
        scp -o ServerAliveInterval=30 -o ServerAliveCountMax=10 $local "${target}:/tmp/$name"
        if ($LASTEXITCODE -ne 0) { throw "Upload failed: $name -> $target" }
    }

    $remotePattern = "/tmp/$dataset.part.*"
    $parent = Split-Path -Path $remoteData -Parent
    $localArchiveSize = [int64]((Get-ChildItem -LiteralPath $localDir -File | Measure-Object Length -Sum).Sum)
    $extract = "cat $remotePattern > '$archive' && test \`$(stat -c '%s' '$archive') -eq $localArchiveSize && tar -C '$parent' -xf '$archive' && test -f '$remoteData/$dataset.yaml' && sed -i 's#^path: .*#path: $remoteData#' '$remoteData/$dataset.yaml' && test -d '$remoteData/images/train' && test -d '$remoteData/images/val' && test -d '$remoteData/images/test' && find '$remoteData' -type f | wc -l; rm -f $remotePattern '$archive'"
    $actualFiles = [int](RunRemote $target $extract).Trim()
    if ($actualFiles -lt $expectedFiles) { throw "Target file count too small: $dataset $actualFiles < $expectedFiles" }
    Write-Host "READY $dataset -> $target files=$actualFiles"
}

SyncDataset "AIR_SARShip_2" "powerleader-a6000-b"
SyncDataset "SRSDD_v1" "a6000-d"
Write-Host "SYNC_COMPLETE"

# Start only after both datasets pass the extraction checks. New IDs avoid
# inheriting the earlier failed startup records.
ssh -o BatchMode=yes powerleader-a6000-b "nohup /home/powerleader/ig-ds-ssm/work_train_single/tools/queue_a6000_detection.sh --experiment-id ig_dsssm_p3_airship_sd16_b16_seed0_a6000b_v2 --datasets AIR-SARShip-2.0 --state-dim 16 --batch 16 --eval-batch 16 --epochs 150 --patience 30 --workers 8 > /home/powerleader/ig-ds-ssm/logs/ig_dsssm_p3_airship_sd16_b16_seed0_a6000b_v2.launcher.log 2>&1 < /dev/null &"
ssh -o BatchMode=yes a6000-d "nohup /home/powerleader/ig-ds-ssm/work_train_single/tools/queue_a6000_detection.sh --experiment-id ig_dsssm_p3_srsdd_sd16_b16_seed0_a6000d_v2 --datasets SRSDD-v1.0 --state-dim 16 --batch 16 --eval-batch 16 --epochs 150 --patience 30 --workers 8 > /home/powerleader/ig-ds-ssm/logs/ig_dsssm_p3_srsdd_sd16_b16_seed0_a6000d_v2.launcher.log 2>&1 < /dev/null &"
Write-Host "QUEUES_STARTED"
