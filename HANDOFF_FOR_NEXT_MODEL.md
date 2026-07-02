# Mamba-YOLO / IG-DS-SSM Research Handoff

This document is a compact handoff for continuing the AAAI-oriented SAR ship detection work in another model, another chat, or another coding session.

Start from `README_CURRENT.md` and `PROJECT_NAVIGATION.md` when entering the workspace cold. They define the active files, archive directories, and default search policy.

## 1. Current Research Position

The current paper direction is **Importance-Guided De-Speckle State Space Modeling for lightweight SAR ship detection**.

The recommended mainline is:

**IG-DS-SSM / PFIM**, implemented as a SAR-aware state-space module inside a lightweight Mamba-YOLO detector.

The core idea:

- **DS-SSM** constructs de-speckled token inputs and suppresses speckle/noise propagation inside the state-space update.
- **PFIM** estimates token-level perceptual importance.
- PFIM modulates three state update paths:
  - memory retention / `gamma`
  - state writing / `write_gate`
  - transition step / `delta`

Current best local SSDD result:

| Method | Dataset | P | R | AP50 | AP50:95 | Params | GFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|
| IG-DS-SSM / PFIM ex27 | SSDD | 97.378 | 95.219 | 98.704 | 74.022 | 5.9648M | 13.2079 |

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
3. `HANDOFF_FOR_NEXT_MODEL.md`
4. `reports/current_research_mainline.md`
5. `reports/paper_innovation_and_performance_overview.md`
6. `reports/design_evidence_matrix.md`

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
16. `paper_draft/aaai_ds_ssm_paper_zh.md`
17. `paper_draft/aaai_ds_ssm_paper_en.pdf`
18. `paper_draft/aaai_ds_ssm_refs.bib`

### E. MaDiNet source evidence

19. `downloads/madinet_paper/source/main_MaDiNet.tex`
20. `downloads/madinet_paper/madinet_arxiv_2411.07500.pdf`

Note: `downloads/` is ignored by default search. Use `rg -u` or search that path directly when MaDiNet source evidence is needed.

## 4. Current Paper Artifacts

Current English LaTeX draft:

`paper_draft/aaai_ds_ssm_paper_en.tex`

Current Chinese translation:

`paper_draft/aaai_ds_ssm_paper_zh.md`

Current PDF preview:

`paper_draft/aaai_ds_ssm_paper_en.pdf`

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

1. **Verify PFIM ex27 under a clean, fixed test command**
   - Save exact command, config, weights, and output log.
   - Confirm AP50 and AP50:95.

2. **Run branch ablations if GPU time allows**
   - DS-SSM only
   - PFIM on gamma only
   - PFIM on write_gate only
   - PFIM on delta only
   - full PFIM

3. **Cross-dataset evaluation**
   - Run IG-DS-SSM/PFIM on SAR-ShipDataset.
   - Consider SAR-AIRcraft1.0 only if the paper is broadened from ship detection to SAR target detection.
   - Optionally evaluate HRSID to compare with recent lightweight ship detectors.

4. **Finish figure assets**
   - Overall framework figure.
   - DS-SSM/PFIM block figure.
   - State update modulation diagram.
   - Qualitative SSDD detection examples.

5. **Clean references**
   - Replace placeholder BibTeX with accurate entries.
   - Cite MaDiNet, AMANet, CPN-YOLO, SURGE, Mamba, Mamba-YOLO, SSDD, SAR-ShipDataset, SAR-AIRcraft1.0.

## 7. Things Not to Claim Yet

Do not claim:

- Full MaDiNet detector complexity has been measured.
- DG-DS-SSM is the best mainline.
- The method fully surpasses MaDiNet on AP50.
- PFIM branch contributions are fully isolated unless ablation results are added.
- Local-scale prior / GDN-style gain control is part of the verified mainline.

Safe claims:

- The current strongest local SSDD result is PFIM ex27.
- The method is lightweight compared with MaDiNet's measured official MambaSAR core module.
- PFIM ex27 reaches SSDD AP50 98.704 and AP50:95 74.022 in recovered local results.
- The method injects SAR-aware de-speckling and importance guidance into SSM dynamics.

## 8. Suggested Prompt for Another Model

Use this prompt when starting a new model/chat:

```text
We are continuing a SAR ship detection paper based on Mamba-YOLO and IG-DS-SSM/PFIM. First read README_CURRENT.md and PROJECT_NAVIGATION.md, then read HANDOFF_FOR_NEXT_MODEL.md, reports/current_research_mainline.md, reports/design_evidence_matrix.md, reports/experiment_ranking_recomputed.md, reports/madinet_ds_ssm_complexity_report.md, reports/lightweight_detection_comparison_madinet_datasets.md, and paper_draft/aaai_ds_ssm_paper_en.tex.

The mainline is IG-DS-SSM/PFIM, not DG-DS-SSM. The best current SSDD result is PFIM ex27: P=97.378, R=95.219, AP50=98.704, AP50:95=74.022, Params=5.9648M, GFLOPs=13.2079. The paper should emphasize lightweight SAR-aware state-space detection, not full AP50 superiority over MaDiNet.

Continue from the current files without deleting or overwriting experiment evidence. If making tables, separate same-protocol MaDiNet table values from external literature values.
```

## 9. One-Paragraph Summary

This project has recovered and organized a multi-server Mamba-YOLO / DS-SSM / PFIM experimental archive for SAR ship detection. The current AAAI-oriented mainline is IG-DS-SSM/PFIM: a lightweight SAR-aware state-space detector that builds de-speckled inputs, damps unreliable state propagation, and uses PFIM importance to modulate memory retention, state writing, and transition steps. The best current SSDD result is PFIM ex27 with 98.704 AP50 and 74.022 AP50:95 using about 5.96M parameters and 13.21 GFLOPs. MaDiNet remains a key AP50 SOTA reference, while AMANet is a key AP50:95 reference. The paper should position the method as a lightweight, interpretable SAR-aware SSM detector with near-SOTA accuracy rather than claiming universal SOTA superiority.
