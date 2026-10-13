param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot '../../../..'),
    [string]$SsddArchive = 'D:\BaiduNetdiskDownload\SAR-Data\SARATR-X\detection\dataset\SSDD.zip'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Add-Type -AssemblyName System.IO.Compression.FileSystem
$projectPath = (Resolve-Path -LiteralPath $ProjectRoot).Path
$galleryPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$inputPath = Join-Path $galleryPath 'inputs'
New-Item -ItemType Directory -Path $inputPath -Force | Out-Null
$entries = [System.Collections.Generic.List[object]]::new()

function Copy-Source([string]$RelativeSource, [string]$DestName) {
    $source = Join-Path $projectPath $RelativeSource
    $target = Join-Path $inputPath $DestName
    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $target
    $sha = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    $copySha = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sha -ne $copySha) { throw "Copy hash mismatch: $DestName" }
    $entries.Add([pscustomobject]@{kind='verbatim_copy';source=$RelativeSource.Replace('\','/');staged=('inputs/'+$DestName.Replace('\','/'));sha256=$sha})
}

Copy-Source 'experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/detection/results_seed0.json' 'latest_seed0.json'
Copy-Source 'dsh_experiment_results/results_individual.csv' 'ssdd_legacy_individual.csv'
Copy-Source 'dsh_experiment_results/results.csv' 'ssdd_legacy_summary.csv'
Copy-Source 'dsh_experiment_results/BASELINE_PROTOCOL.md' 'ssdd_legacy_protocol.md'
Copy-Source 'paper_draft/rgsp_ieee2col_revision_20261009/supplementary/sadd_results_individual.csv' 'sadd_3seed.csv'
Copy-Source 'reports/RGSP_SEED0_RESULTS_REVIEW_20261010.md' 'seed0_review.md'
Copy-Source 'docs/project-memory/PROJECT_STATE.md' 'project_state_snapshot.md'
Copy-Source 'artifacts/dsssm_motivation_ex27_exact_v2_20260823/motivation_summary.csv' 'historical_maps_summary.csv'
Copy-Source 'artifacts/dsssm_motivation_ex27_exact_v2_20260823/manifest.json' 'historical_maps_manifest.json'
Copy-Source 'source_code/work_train_single/tools/visualize_dsssm_motivation.py' 'historical_capture_source.py'
Copy-Source 'experiments/rgsp_seedwise_20261009_v1/transfer/final_patch_v1/code/ultralytics/nn/modules/mamba_yolo.py' 'frozen_mamba_yolo.py'
Copy-Source 'experiments/rgsp_seedwise_20261009_v1/manifest.py' 'frozen_campaign_manifest.py'
Copy-Source 'paper_draft/rgsp_cn_full_experiments_20261009/main.tex' 'method_manuscript_snapshot.tex'
Copy-Source 'paper_draft/rgsp_cn_full_experiments_20261009/sections/ablation.tex' 'ablation_manuscript_snapshot.tex'
Copy-Source 'experiments/server_current_3090x2__ssdd__ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3/train/results.csv' 'historical_training.csv'
Copy-Source 'experiments/server_current_3090x2__ssdd__ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3/train/args.yaml' 'historical_training_args.yaml'

$sampleIds = @('000099','000531','000929','001081','001131','001111')
foreach ($sampleId in $sampleIds) {
    $sourceRel = "artifacts/dsssm_motivation_ex27_exact_v2_20260823/${sampleId}_dsssm_maps.npz"
    $source = Join-Path $projectPath $sourceRel
    $sourceSha = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    $zip = [System.IO.Compression.ZipFile]::OpenRead($source)
    try {
        foreach ($mapName in @('importance','input_energy','corrected_energy','residual','correction','drive_gate','delta_bias')) {
            $memberName = "raw_${mapName}.npy"
            $member = $zip.GetEntry($memberName)
            if ($null -eq $member) { throw "Missing NPZ member $sampleId / $memberName" }
            $destRel = "arrays/$sampleId/$memberName"
            $target = Join-Path $inputPath $destRel
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($member, $target, $true)
            $entries.Add([pscustomobject]@{kind='npz_member_extraction';source=$sourceRel;source_sha256=$sourceSha;member=$memberName;staged=('inputs/'+$destRel);sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()})
        }
    } finally { $zip.Dispose() }
}

$captureManifest = Get-Content -LiteralPath (Join-Path $inputPath 'historical_maps_manifest.json') -Raw | ConvertFrom-Json
$zip = [System.IO.Compression.ZipFile]::OpenRead($SsddArchive)
try {
    foreach ($sampleId in $sampleIds) {
        $memberName = "SSDD/images/test/$sampleId.jpg"
        $member = $zip.GetEntry($memberName)
        if ($null -eq $member) { throw "Missing official test image $memberName" }
        $destRel = "images/$sampleId.jpg"
        $target = Join-Path $inputPath $destRel
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($member, $target, $true)
        $sha = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        $expected = @($captureManifest.images | Where-Object { $_.path.EndsWith("/$sampleId.jpg") })
        if ($expected.Count -ne 1 -or $sha -ne $expected[0].sha256) { throw "Historical capture image mismatch $sampleId" }
        $entries.Add([pscustomobject]@{kind='zip_member_extraction';source=$SsddArchive;member=$memberName;staged=('inputs/'+$destRel);sha256=$sha;matches_historical_capture=$true})
    }
    $memberName = 'SSDD/annotations/test.json'
    $member = $zip.GetEntry($memberName)
    if ($null -eq $member) { throw 'Official COCO test annotation file missing' }
    $target = Join-Path $inputPath 'ssdd_test_coco.json'
    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($member, $target, $true)
    $entries.Add([pscustomobject]@{kind='zip_member_extraction';source=$SsddArchive;member=$memberName;staged='inputs/ssdd_test_coco.json';sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant();limitation='Official COCO boxes used for the new redraw; historical capture did not hash labels.'})
} finally { $zip.Dispose() }

$manifest = [ordered]@{
    schema='rgsp-visualization-input-manifest-v1'
    source_snapshot='2026-10-10 10:07:54-10:07:57 Asia/Shanghai'
    staging_kind='verbatim copies and ZIP member extraction only; no model execution'
    project_root=$projectPath
    entries=$entries
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $inputPath 'source_manifest.json') -Encoding utf8
[pscustomobject]@{files=$entries.Count;input_bytes=(Get-ChildItem -LiteralPath $inputPath -Recurse -File | Measure-Object Length -Sum).Sum;source_snapshot=$manifest.source_snapshot} | ConvertTo-Json
