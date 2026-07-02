# Design Evidence Matrix

Status labels used: `implemented`, `experiment_supported`, `candidate_only`, `deprecated`, `conflict`, `unknown`.

| Design point | Source document | Code implementation | Experiment validation | Current status | Recommendation |
|---|---|---|---|---|---|
| input de-biasing | Missing docs; code evidence | `x_tilde = x - lambda_s * residual` in `mamba_yolo.py` | Supported indirectly by DS-SSM/PFIM top runs | `implemented`, `experiment_supported` | Keep in mainline |
| energy-aware damping | Missing docs; code evidence | `gamma_logits` from energy of `x_bar` | Supported indirectly by DS-SSM/PFIM runs | `implemented`, `experiment_supported` | Keep in mainline |
| soft gamma | Missing docs; code evidence | `min_retain + (1-min_retain)*sigmoid(...)` | Supported indirectly | `implemented`, `experiment_supported` | Keep |
| lambda_init=0.1 | Missing docs; code evidence | `DeSpeckleSSM(... lambda_init=0.1 ...)`; block default may differ | UNKNOWN as exact ablation | `implemented`, `unknown` | Confirm config-level values per experiment |
| PFIM importance map | Missing current summary; code evidence | `class PFIM`, returns sigmoid importance | ex24/ex27 naming and ranking support | `implemented`, `experiment_supported` | Mainline |
| PFIM on gamma | Missing docs; code evidence | `pfim_on_gamma`, `beta_i * imp_center` | Supported by PFIM top runs as combined design | `implemented`, `experiment_supported` | Mainline, still needs ablation if publishing |
| PFIM on write_gate | Missing docs; code evidence | `write_gate = 1 + tanh(rho_i) * imp_center` | Supported as combined design | `implemented`, `experiment_supported` | Mainline, ablation desirable |
| PFIM on delta | Missing docs; code evidence | `delta_logits += eta_i * importance` | Supported as combined design | `implemented`, `experiment_supported` | Mainline, ablation desirable |
| 2D local std | Missing docs; code evidence | PFIM `use_2d_std` branch with `avg_pool2d` | Supported only as part of current PFIM | `implemented`, `experiment_supported` | Keep, but distinguish from local-scale-prior proposal |
| gamma uses importance | Missing docs; code evidence | `beta_i * imp_center` in gamma logits | Supported as combined design | `implemented`, `experiment_supported` | Keep |
| gamma uses importance - 0.5 | Missing docs; code evidence | `imp_center = importance - 0.5` | Supported as combined design | `implemented`, `experiment_supported` | Keep |
| local scale prior | Candidate doc unavailable | No confirmed distinct implementation beyond PFIM local std | No dedicated experiment found | `candidate_only`, `unknown` | Do not enter mainline yet |
| GDN-style gain normalization | Candidate doc unavailable | No confirmed active implementation found | No dedicated experiment found | `candidate_only`, `unknown` | Do not enter mainline yet |
| CUDA input damping approximation | Code evidence | CUDA path uses `u_damped = u_write * gamma` before scan | Supported by successful training, but exact equivalence not proven | `implemented`, `unknown` | Verify Python/CUDA equivalence before publication |
| Python / CUDA path consistency | Code evidence | Python path gates hidden state; CUDA path damps input before scan | Potential semantic difference | `conflict`, `unknown` | Needs unit test or analytical note |
| D * x_tilde residual outside kernel | Code evidence | `selective_scan_fn(... D=None)` then `y_ssm + D * x_tilde` | Supported by top runs | `implemented`, `experiment_supported` | Keep; document explicitly |
| 20 epoch warmup | Recent Codex change / trainer code | `_dsssm_freeze_warmup(...20)` in `work_train_ddp` | Latest codex3 run completed but lower than ex27 | `implemented`, `unknown` | Treat as DDP training technique, not model contribution yet |
| SAR-specific loss / augmentation | Requested design point | Not confirmed in current audit | UNKNOWN | `unknown` | Do not claim until code/docs verified |
