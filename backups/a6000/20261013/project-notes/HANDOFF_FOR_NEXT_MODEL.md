# Mamba-YOLO / IG-DS-SSM Research Handoff

This document is a compact handoff for continuing the AAAI-oriented SAR ship detection work in another model, another chat, or another coding session.

Start from `README_CURRENT.md` and `PROJECT_NAVIGATION.md` when entering the workspace cold. They define the active files, archive directories, and default search policy.

Additional migrated context from the former main experiment thread is preserved in `reports/thread_019f2406_integrated_context.md`. Read it before changing SSDD/H100 sweep plans, initialization protocols, or final comparison-method choices.

## 1. Current Research Position

The current paper direction is **Importance-Guided De-Speckle State Space Modeling for lightweight SAR target detection**.

The recommended mainline is:

**IG-DS-SSM / PFIM**, implemented as a SAR-aware state-space module inside a lightweight Mamba-YOLO detector.

The core idea:

- **DS-SSM** attenuates local token residuals before selective scan.
- **PFIM** estimates token-level importance from local trend and two-dimensional variation.
- In the executed CUDA path, `gamma` and `write_gate` are independently parameterized factors on the same scan drive; `delta` is the separate transition-control location. Do not describe these as three independent recurrent update sites.

Current primary controlled OGSOD-1.0 TEST result:

| Model | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|
| Mamba-YOLO-T | 0.771733 | 0.402109 | 0.423018 |
| IG-DS-SSM/PFIM-T | 0.772630 | 0.409711 | 0.436021 |

This is a matched 300-epoch, 1024-input, global-BS64, two-H200 comparison, but still only one completed run per model. The paper's primary observation is `+0.760` AP75 and `+1.300` AP50:95 percentage points.

Current best local SSDD result:

| Method | Dataset | P | R | AP50 | AP50:95 | Params | GFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|
| IG-DS-SSM / PFIM ex27 | SSDD | 97.378 | 95.219 | 98.704 | 74.022 | 5.9648M | 13.2079 |

Current SAR_AIRcraft patch-protocol result:

| Method | Split | P | R | AP50 | AP50:95 | Params | GFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|
| Mamba-YOLO-T + DS-SSM + PFIM | val | 0.909612 | 0.796257 | 0.888252 | 0.705129 | 5.9646M | 13.2 |
| Mamba-YOLO-T + DS-SSM + PFIM | test | 0.792764 | 0.637326 | 0.708220 | 0.555516 | 5.9646M | 13.2 |

The S-scale aircraft capacity experiment is complete. Its final FP32 validation P/R/AP50/AP50:95 is `0.910108/0.830495/0.893042/0.720249`. Earlier in-training validation rows were corrupted by a forced-FP16 validator path even though training itself remained healthy in FP32. Do not use the historical `0.1202/0.0842` rows.

Read `reports/sar_aircraft_experiment_20260721.md` for the full configuration, per-class metrics, protocol warning, and baseline caveat. The current SAR_AIRcraft data are 512x512 overlapping patches, not the original-image protocol described by MaDiNet. The available Mamba-YOLO-T reference also used a different optimizer, batch size, and training duration, so it is not yet a publication-grade controlled ablation.

Current SAR-ShipDataset results are recorded in `reports/sar_shipdataset_current_results.md` and `database/sar_shipdataset_current_results.csv`. The lightweight T-scale IG-DS-SSM/PFIM result is P/R/AP50/AP50:95 = `0.907854/0.889908/0.948440/0.578182`; the S-scale result is `0.922367/0.915870/0.956093/0.531767`. Baseline comparisons are archive-level evidence because the T baseline changed batch size mid-run and the B reference was not trained for 300 epochs.

Important interpretation:

- The paper should **not** claim to fully surpass MaDiNet in all metrics.
- The paper should emphasize: **near-SOTA SSDD accuracy with a much lighter detector and an interpretable SAR-aware SSM update mechanism**.
- DG-DS-SSM should be treated as an extension / ablation / future work, not the main model.

## 2. Main Competitors and SOTA Context

SSDD has no single official leaderboard. Different papers use different splits, metric conventions, and confidence thresholds. Compare carefully.

Key methods:

| Method | Main Relevance | Reported Result |
|---|---|---|
| MaDiNet | Strong SAR diffusion + Mamba SOTA reference | SSDD AP50 99.0; SAR-ShipDataset AP50 97.5; SAR-AIRcraft1.0 AP50 90.5 |
| AMANet | Important AP50:95 reference on SSDD | AP50:95 about 74.20; AP50 about 98.50 |
| CPN-YOLO | 2026 denoising-enhanced YOLO | SSDD AP50 about 98.9; verify AP50:95 before formal citation |
| SURGE | 2026 lightweight SAR ship detection via contrastive distillation | Relevant to lightweight/detection-efficiency narrative |
| LH-YOLO / YOLO-Lite / SHIP-YOLO | Lightweight YOLO-style SAR ship detectors | Useful external lightweight references, often on SAR-ShipDataset/HRSID |

Main comparison stance:

- **Against MaDiNet**: MaDiNet AP50 is higher, but our detector is much lighter and directly injects SAR priors into SSM dynamics.
- **Against AMANet**: our AP50:95 on SSDD is very close, while using a compact Mamba-YOLO-style detector.
- **Against YOLO denoising methods**: our method is not just external denoising or attention stacking; it modifies the state-space update itself.
- **Against distillation methods**: our lightweight property comes from architecture design, not only teacher-student compression.

## 3. Important Files to Read First

When continuing in another model/chat, ask the model to read these files first, in this order.

### A. High-level status and paper direction

1. `README_CURRENT.md`
2. `PROJECT_NAVIGATION.md`
3. `reports/thread_019f2406_integrated_context.md`
4. `HANDOFF_FOR_NEXT_MODEL.md`
5. `reports/current_research_mainline.md`
6. `reports/paper_innovation_and_performance_overview.md`
7. `reports/design_evidence_matrix.md`

### B. Experiment results and ranking

7. `reports/experiment_ranking_recomputed.md`
8. `database/experiment_ranking_recomputed.csv`
9. `reports/current_codex3_experiment_report.md`
10. `database/SSDD_experiment_results_summary_recomputed.csv`

### C. Complexity and lightweight comparison

11. `reports/madinet_ds_ssm_complexity_report.md`
12. `database/complexity_comparison_madinet_ds_ssm.csv`
13. `reports/lightweight_detection_comparison_madinet_datasets.md`
14. `database/lightweight_detection_comparison_madinet_datasets.csv`

### D. Current paper draft

15. `paper_draft/aaai_ds_ssm_paper_en.tex`
16. `paper_draft/aaai_ds_ssm_paper_zh.tex`
17. `paper_draft/aaai_ds_ssm_paper_en.pdf`
18. `paper_draft/aaai_ds_ssm_paper_zh.pdf`
19. `paper_draft/aaai_ds_ssm_refs.bib`

### E. MaDiNet source evidence

20. `downloads/madinet_paper/source/main_MaDiNet.tex`
21. `downloads/madinet_paper/madinet_arxiv_2411.07500.pdf`

Note: `downloads/` is ignored by default search. Use `rg -u` or search that path directly when MaDiNet source evidence is needed.

## 4. Current Paper Artifacts

Current English LaTeX draft:

`paper_draft/aaai_ds_ssm_paper_en.tex`

Current Chinese translation:

`paper_draft/aaai_ds_ssm_paper_zh.tex`

Current PDF previews:

- `paper_draft/aaai_ds_ssm_paper_en.pdf`
- `paper_draft/aaai_ds_ssm_paper_zh.pdf`

Important note:

`paper_draft/aaai26.sty` is only a **temporary local compilation shim**. Replace it with the official AAAI author-kit style before submission.

## 5. Lightweight Comparison Table

The current lightweight comparison table is saved at:

- `reports/lightweight_detection_comparison_madinet_datasets.md`
- `database/lightweight_detection_comparison_madinet_datasets.csv`

It covers:

- SSDD
- SAR-ShipDataset
- SAR-AIRcraft1.0
- HRSID as an external auxiliary reference column

Recommended usage:

- Use the **MaDiNet original table subset** for strict same-paper comparison.
- Use external methods only as literature context unless their split and metric protocol are confirmed.

## 6. Missing Experiments / Next Steps

Highest priority experiments:

1. **Complete the primary OGSOD matrix**
   - Seeds 0/1/2 for baseline, no-PFIM, global-importance PFIM, and full PFIM.
   - Report raw runs, mean/std, and seed-paired AP50/AP75/AP50:95 deltas.

2. **Complete seed-0 factor diagnostics**
   - Gamma only, write only, delta only, and gamma+write.
   - Interpret gamma/write as factors on one scan drive, not independent state locations.

3. **Analyze the dense-tank boundary**
   - Evaluate fixed checkpoints by density, median box area, and nearest-neighbor spacing.
   - Add importance maps and representative bridge/harbor successes plus tank failures.

4. **Integrate only completed controlled results**
   - Do not substitute SSDD, SAR-Ship, or aircraft archive deltas for the OGSOD causal matrix.
   - Use `reports/ogsod_ablation_matrix_plan.md` as the execution contract.

## 7. Things Not to Claim Yet

Do not claim:

- Full MaDiNet detector complexity has been measured.
- DG-DS-SSM is the best mainline.
- The method fully surpasses MaDiNet on AP50.
- PFIM branch contributions are fully isolated unless ablation results are added.
- Gamma, write, and delta are three independent recurrent operator locations.
- Local-scale prior / GDN-style gain control is part of the verified mainline.

Safe claims:

- The current strongest local SSDD result is PFIM ex27.
- The method is lightweight compared with MaDiNet's measured official MambaSAR core module.
- PFIM ex27 reaches SSDD AP50 98.704 and AP50:95 74.022 in recovered local results.
- The method injects SAR-aware de-speckling and importance guidance into SSM dynamics.
- The matched OGSOD single run improves AP75 by 0.760 pp and AP50:95 by 1.300 pp, pending multi-seed confirmation.

## 8. Suggested Prompt for Another Model

Use this prompt when starting a new model/chat:

```text
We are continuing an AAAI-oriented SAR target detection paper based on Mamba-YOLO and IG-DS-SSM/PFIM. First read README_CURRENT.md and PROJECT_NAVIGATION.md, then read HANDOFF_FOR_NEXT_MODEL.md, reports/aaai_paper_current_status.md, reports/top_conference_review_round3_20260807.md, reports/ogsod_ablation_matrix_plan.md, reports/madinet_ds_ssm_complexity_report.md, and paper_draft/aaai_ds_ssm_paper_en.tex.

The mainline is IG-DS-SSM/PFIM, not DG-DS-SSM. The cleanest evidence is one matched OGSOD run: baseline AP50/AP75/AP50:95=0.771733/0.402109/0.423018 and PFIM-T=0.772630/0.409711/0.436021. This is not yet multi-seed evidence. In the CUDA path, gamma and write multiply the same scan drive, while delta is the separate transition location. Do not reintroduce a three-independent-path claim.

Continue from the current files without deleting or overwriting experiment evidence. If making tables, separate same-protocol MaDiNet table values from external literature values. The pending OGSOD queue is now protocol-hardened as `ogsod_t_primary_3seed_v2`; do not reuse a prior matrix ID after code or data changes.
```

## 9. One-Paragraph Summary

This project has recovered and organized a multi-server Mamba-YOLO / DS-SSM / PFIM experimental archive for SAR target detection. The current AAAI-oriented mainline is IG-DS-SSM/PFIM: a lightweight module that attenuates local residuals, estimates token importance, factorizes the CUDA scan drive through gamma and write, and separately adapts delta. The cleanest result is a matched single-run OGSOD AP75/AP50:95 gain of 0.760/1.300 percentage points at about 5.96M parameters and 13.21 GFLOPs; SSDD, SAR-Ship, and aircraft results remain protocol-limited archive evidence. Strict review remains 6/10 until the prepared three-seed, no-PFIM, parameter-matched global-control, and factor-diagnostic experiments are completed.
