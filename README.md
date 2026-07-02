# Mamba-YOLO DS-SSM Workspace

Start here:

- `README_CURRENT.md` for the current mainline, best result, and safe claims.
- `DATASETS.md` for required SSDD / SAR-AIRcraft local dataset preparation.
- `PROJECT_NAVIGATION.md` for directory zones and search policy.
- `HANDOFF_FOR_NEXT_MODEL.md` for a compact research handoff.

The active research mainline is **IG-DS-SSM / PFIM** for lightweight SAR ship detection. The current best recovered SSDD result is `server_3090_b / PFIM...ex27`, with AP50:95 `0.74022`.

Large evidence archives are kept in place locally but excluded from normal Git upload through `.gitignore` and ignored by default `rg` searches through `.rgignore`. Use `rg -u` when you need to search archived server snapshots, downloads, weights, or recovered experiments.
