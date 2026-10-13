# SSDD Historical Recheck and FUSAR-Map Protocol Audit

Date: 2026-08-28

## SSDD: fixed official-test recheck

The recovered historical checkpoints were evaluated in FP32 on the same 232-image SSDD official test split, with `imgsz=640`, `batch=32`, `conf=0.001`, and `iou=0.6`. No training was performed in this diagnostic.

| Evidence | Method | Train images | Epochs | Evaluation split | P | R | AP50 | AP75 | AP50:95 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| Current matched scratch | Mamba-YOLO-T baseline | 742 | 150 | official test | 0.89576 | 0.75543 | 0.83439 | 0.19393 | 0.35239 |
| Current matched scratch | P3 / IG-DS-SSM d16 | 742 | 150 | official test | 0.88231 | 0.83516 | 0.88137 | 0.33756 | 0.42991 |
| Historical checkpoint recheck | Mamba-YOLO-T baseline | 928 historical train | 300 | official test | 0.96738 | 0.96154 | 0.98605 | 0.83613 | 0.70348 |
| Historical checkpoint recheck | PFIM ex27 | 928 historical train | 300 | official test | 0.97759 | 0.95857 | 0.98688 | 0.84005 | 0.71445 |
| Historical archive | PFIM ex27 best row | 928 historical train | 300 | historical val | 0.97378 | 0.95219 | 0.98704 | not archived | 0.74022 |

Fixed-test historical gain over its matched baseline:

- AP50: `+0.00083` (`+0.083 pp`)
- AP75: `+0.00393` (`+0.393 pp`)
- AP50:95: `+0.01098` (`+1.098 pp`)

Current scratch P3 gain over its matched baseline:

- AP50: `+0.04698` (`+4.698 pp`)
- AP75: `+0.14363` (`+14.363 pp`)
- AP50:95: `+0.07753` (`+7.753 pp`)

## Why the current SSDD absolute score is lower

The main difference is the training protocol, not a disappearance of the P3 effect.

1. The current run uses 742 training images after reserving 186 images for development validation. The historical run trained with all 928 official training images.
2. The current run uses 150 epochs; the historical run uses 300 epochs.
3. The historical ex27 recipe uses AdamW, `lr0=1e-3`, five optimizer warmup epochs, batch 32, mosaic 1.0, scale 0.5, and the older ex7/PFIM topology. The current controlled run uses AdamW, `lr0=2e-4`, three optimizer warmup epochs, batch 16, mosaic 0.3, scale 0.3, and the current P3 topology.
4. The historical `0.74022` value is a best validation row. The frozen official-test recheck is `0.71445`; these values must not be treated as the same split.
5. Historical code called `model.load("yolov8n.pt")`, but the recovered compatibility audit matched only about 6 of roughly 519 tensors. It was nominal partial transfer, not meaningful Mamba-YOLO pretraining, and cannot explain the large absolute gap by itself.

The strongest evidence is the historical baseline: it reaches `0.70348` AP50:95 under the fixed test evaluator, while the current scratch baseline reaches `0.35239`. Most of the absolute gap therefore exists before adding IG-DS-SSM/PFIM.

Historical SSDD checkpoints are target-dataset-trained checkpoints. Reusing them for SSDD fine-tuning is useful only as a diagnostic and must not be described as external pretraining in the paper.

## FUSAR-Map audit

### Current local protocol

- Split: city-disjoint, 408 train / 125 val / 77 test.
- Classes: five, including a separate black `background/unknown` class.
- Input: one random 512 crop from each 1024 image per training epoch.
- Initialization: scratch.
- Latest observed progress during the audit: baseline epoch 8 and P3-d16 epoch 5 of 160. These are interim values, not final results.

Full mask audit over all 610 masks:

| Local class | RGB | Pixel fraction |
|---|---|---:|
| Black background/unknown | 0,0,0 | 46.394% |
| Building | 255,0,0 | 11.959% |
| Vegetation | 0,255,0 | 17.919% |
| Water | 0,0,255 | 18.065% |
| Road | 255,255,0 | 5.664% |

Only `0.00746%` of pixels use noncanonical colors, so corrupt masks are not the main problem.

### Published FUSAR-Map protocol differences

The original dataset paper defines four terrain classes: building, vegetation, water, and road. Black pixels are undetermined/unknown rather than a fifth reported terrain class. The paper uses 530 training images and 80 test images, seven fixed rotations plus random flips, and an optical-image-pretrained encoder for its transfer-learning model.

The released reference code also uses four output channels and 256x256 prepared patches. Its U-Net path merges black pixels into vegetation; one DeepLab path instead merges black pixels with road. That inconsistency must be resolved explicitly before reproducing author-reported metrics, but both paths differ from the current five-class evaluator.

Therefore, the current 13-15% interim mIoU cannot be directly compared with published 43-52% mIoU values. The largest causes are:

1. Five-class versus four-class evaluation, affecting 46.394% of all pixels.
2. Harder city-disjoint split versus the published 530/80 protocol.
3. Only 408 random crops per epoch versus a pre-generated 256-patch training set with extensive rotation augmentation.
4. Scratch training versus optical-pretrained encoders in published transfer-learning baselines.
5. Current runs are only at epochs 5-8 of 160.

## Required corrected FUSAR experiments

Keep the running city-disjoint experiments as a strict cross-city generalization track. Add a separate publication-aligned track rather than overwriting it:

1. Use four reported classes and freeze one explicit black-pixel policy for every method.
2. Reproduce a deterministic 530/80 image split and pre-generate 256x256 tiles.
3. Apply the same fixed augmentation set to baseline and P3.
4. Give baseline and P3 the same encoder initialization. Report scratch and pretrained tracks separately.
5. Report per-class IoU for building, vegetation, water, and road, followed by four-class mIoU, FWIoU, OA, and mPA.
6. Never compare the five-class city-disjoint result directly against the four-class author-reported table.

### Implemented count-aligned track

A separate corrected track is implemented without overwriting the running city-disjoint experiments:

- Protocol label: `deterministic-count-aligned-530-train-80-test-four-class-ignore-unknown`.
- The split uses the publication's image counts but is deterministic random seed 0; it is not claimed to be the unavailable official membership list.
- Image split: 530 train / 80 test, with all seven cities represented in both partitions.
- Tiling: non-overlapping 256x256 crops, yielding 8,480 train and 1,280 test tiles.
- Labels: black unknown pixels map to ignore index `-1`; building, vegetation, water and road map to classes 0-3.
- Model selection: fixed last epoch, because the 530/80 protocol has no validation partition. The test set is never used for checkpoint selection.
- Metrics: four-class mIoU, FWIoU, OA, mPA, Dice and per-class IoU/PA/Dice.
- Baseline and P3-d16 use the same scratch initialization seed, optimizer, input size, batch and epoch count.

Implementation files:

- `source_code/work_train_single/tools/prepare_cross_task_v2.py`
- `source_code/work_train_single/tools/run_shared_sar_tasks_v2.py`
- `source_code/work_train_single/tools/run_fusar_count_aligned.sh`
- `source_code/work_train_single/tools/queue_fusar_count_aligned_after_pid.sh`

Verification completed before deployment: 7 focused unit tests, CPU baseline train/resume/test smoke, and A6000 P3-d16 CUDA train/resume/test smoke. The generated server manifest contains the expected 530/80 images and 8,480/1,280 tiles. The formal P3 then baseline pair is queued on A6000-D after the active SSDD warm-start pair finishes.

## Evidence files

- `artifacts/ssdd_historical_checkpoint_diagnostic_20260828/mambayolo_t_baseline_official_test.json`
- `artifacts/ssdd_historical_checkpoint_diagnostic_20260828/pfim_ex27_official_test.json`
- `artifacts/ssdd_historical_checkpoint_diagnostic_20260828/fusar_map_manifest_audit_20260828.json`
- `source_code/work_train_single/tools/run_ssdd_historical_checkpoint_diagnostic.sh`
- `source_code/work_train_single/tools/audit_fusar_map_manifest.py`

Primary public references:

- FUSAR-Map official repository: https://github.com/fudanxu/FUSAR-Map
- FUSAR-Map dataset page: https://shixianzheng.github.io/FUSAR-Map/
- MASA-SegNet comparison table: https://www.mdpi.com/2072-4292/15/14/3662
