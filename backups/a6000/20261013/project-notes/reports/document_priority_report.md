# Document Priority Report

## Scope Status

Requested documents under `docs/00_*`, `docs/01_*`, and `docs/02_*` are not present in the current workspace. I did not read desktop paths outside `D:\A-主机文件\Codex\Mamba-YOLO-ds-ssm`, following `CODEX_RULES.md`.

Therefore this report classifies the requested documents by intended role from filename and requested path only. Content-level claims are marked `UNKNOWN` until the files are copied into the workspace `docs/` tree.

## Priority Table

| File | Type | Main content | Outdated | Implemented in code | Experiment support | Conflict with current code | Enter current mainline | Confidence |
|---|---|---|---|---|---|---|---|---|
| `docs/00_archive_original_ideas/Despeckled_Mamba.pdf` | `early_design` | Early despeckling/Mamba idea source. Content not available in workspace. | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | No automatic adoption | Low |
| `docs/00_literature/END-TO-END OPTIMIZED IMAGE COMPRESSION.pdf` | `literature_reference` | Compression literature reference. Content not available in workspace. | No, as literature | No direct implementation requirement | Not experiment evidence for this project | UNKNOWN | Theory citation only | Low |
| `docs/00_literature/VARIATIONAL IMAGE COMPRESSION WITH A SCALE HYPERPRIOR.pdf` | `literature_reference` | Scale-hyperprior literature reference. Content not available in workspace. | No, as literature | No direct implementation requirement | Not experiment evidence for this project | UNKNOWN | Theory citation only | Low |
| `docs/01_candidate_designs/基于局部尺度先验与增益控制的 DS-SSM 改造方案.docx` | `candidate_extension` | Candidate local scale prior / gain control extension. Content not available in workspace. | UNKNOWN | Not confirmed as current code | UNKNOWN | Possible if it changes PFIM/DS-SSM behavior | Only after user confirmation | Medium from filename/path |
| `docs/02_chat_summaries/核心创新：IG_DS_SSM_PFIM_方案总结.md` | `current_design_summary` | Current IG-DS-SSM / PFIM design summary. Content not available in workspace. | UNKNOWN | Partially supported by code names and modules | Partially supported by ex24/ex27 results | UNKNOWN | Likely yes, pending document import | Medium |
| `docs/02_chat_summaries/SSDD_IG_DSSSM_实验核心逻辑代码结论总结.md` | `experiment_evidence` | SSDD IG-DS-SSM experiment/code conclusion summary. Content not available in workspace. | UNKNOWN | Partially supported by recovered code and result tables | Yes, via recovered `results.csv` and recomputed ranking | UNKNOWN | Likely high-priority evidence, pending import | Medium |

## Evidence Actually Read

- Code: `source_code/work_train_single`, `source_code/work_train_ddp`.
- Experiment tables: `database/experiment_index.csv`, `database/experiment_ranking_recomputed.csv`, `database/SSDD_experiment_results_summary_recomputed.csv`.
- Reports: existing recovery, ranking, current codex3, future experiment, and training repository reports.
- Git metadata: `.git/HEAD` and `.git/logs/HEAD`; Git CLI is unavailable in this shell.

## Immediate Document Action

Copy the six requested documents into the exact workspace paths under `docs/`. After that, rerun this audit for content-level classification.
