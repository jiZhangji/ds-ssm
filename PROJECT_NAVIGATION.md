# Project Navigation

This file keeps the large workspace manageable without moving recovered evidence.

## Why This Exists

The repository contains active research files plus large recovered archives:

- multiple source snapshots from remote servers;
- recovered experiment folders;
- many model weights;
- downloaded third-party paper/source artifacts;
- paper drafts and generated PDFs.

Default whole-repository searches can be slow and noisy. The project now uses `.rgignore` to focus routine searches on active files.

## Active Working Area

Use these paths for daily work:

| Purpose | Path |
|---|---|
| Main handoff | `HANDOFF_FOR_NEXT_MODEL.md` |
| Current entry | `README_CURRENT.md` |
| Current research judgment | `reports/current_research_mainline.md` |
| Experiment ranking | `database/experiment_ranking_recomputed.csv` |
| Paper draft | `paper_draft/aaai_ds_ssm_paper_en.tex` |
| Single-process code copy | `source_code/work_train_single/` |
| DDP code copy | `source_code/work_train_ddp/` |
| Complexity scripts | `complexity_benchmark/` |

## Archive Area

These paths preserve evidence. Search them only when needed:

| Purpose | Path |
|---|---|
| Historical source snapshot | `source_code/server_1080ti/` |
| Historical source snapshot | `source_code/server_3090_a/` |
| Historical source snapshot | `source_code/server_3090_b/` |
| Recovered experiment artifacts | `experiments/` |
| Recovered weights | `weights/` |
| Downloaded third-party/source artifacts | `downloads/` |
| Older bundles | `archives/` |
| Logs | `logs/` |

Do not edit historical source snapshots unless the task is explicitly an audit. Do new code work in `source_code/work_train_single/` or `source_code/work_train_ddp/`.

## Recommended Searches

Fast default search:

```powershell
rg "DS_VSSBlock|DeSpeckleSSM|PFIM"
```

Search active code only:

```powershell
rg "write_gate|gamma|delta" source_code/work_train_ddp source_code/work_train_single
```

Search reports and databases:

```powershell
rg "0.74022|ex27|codex3" reports database
```

Search ignored archive paths:

```powershell
rg -u "PFIM" source_code/server_3090_b
rg --files -u weights experiments downloads
```

## Current Best Result

Use the recomputed ranking as the primary metric source:

- file: `database/experiment_ranking_recomputed.csv`
- report: `reports/experiment_ranking_recomputed.md`
- current best: `server_3090_b / ssdd/PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex27-gemini-20260603:17:45`
- best AP50:95: `0.74022`

Older reports may show old rankings based on earlier score extraction. Prefer files with `recomputed` in the name unless a task explicitly asks for historical ranking provenance.
