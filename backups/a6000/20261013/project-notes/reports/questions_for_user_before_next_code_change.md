# Questions For User Before Next Code Change

1. Do you confirm `IG-DS-SSM / PFIM` as the current paper mainline?

2. Should `local scale prior / GDN-style gain control` remain a candidate extension for now, not merged into the mainline?

3. Should we continue C-new and ex29 ablations, or first stabilize/reproduce ex27?

4. Should `work_train_single` and `work_train_ddp` use the exact same model module code, with differences limited to trainer/launcher behavior?

5. Which experiment should be treated as the current best reference: recomputed ex27 mAP50-95 `0.74022`, DDP ex23, or latest codex3 DDP run?

6. Which code snapshot should be the next experiment start point: `server_3090_b` PFIM ex27 lineage, `work_train_single`, or `work_train_ddp` after model alignment?

7. Which early PDF ideas are now deprecated and should be archive-only?

8. Which candidate designs should explicitly stay out of the current paper mainline until later?

9. Should the 20-epoch warmup be considered a training technique only, or part of the claimed method?

10. Should PFIM component ablations be required before writing the method section: gamma only, write only, delta only, and combinations?

11. Should future reports cite literature PDFs only as theory motivation unless code and experiment evidence exists?

12. Please copy the six requested documents into the workspace `docs/` paths before the next document-content audit.
