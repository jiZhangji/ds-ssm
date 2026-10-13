# Recomputed Experiment Ranking

Ranking source: recovered local `results.csv` files. Score is max `metrics/mAP50-95(B)` across epochs.

| Rank | Server | Experiment | Best epoch | P | R | mAP50 | mAP50-95 | Old score |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `server_3090_b` | `ssdd/PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex27-gemini-20260603:17:45` | `294.0` | `0.97378` | `0.95219` | `0.98704` | `0.74022` | `0.70964` |
| 2 | `server_3090_b` | `ssdd/PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex24-gemini-20260603:11:15` | `296.0` | `0.95701` | `0.95238` | `0.98287` | `0.73571` | `0.70904` |
| 3 | `server_1080ti` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex7-gemini-20260521:00:07` | `296.0` | `0.96099` | `0.95421` | `0.98226` | `0.7352` | `0.7068` |
| 4 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex7-gemini-20260521:00:07` | `296.0` | `0.96099` | `0.95421` | `0.98226` | `0.7352` | `0.7068` |
| 5 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex7-gemini-20260521:00:07` | `296.0` | `0.96099` | `0.95421` | `0.98226` | `0.7352` | `0.7068` |
| 6 | `server_1080ti` | `ssdd/ddp_mambayolo_s_ssdd_backbone_ds-ssm_300e_16bsx2-ex11-ex23-gemini-20260602:15:40` | `297.0` | `0.97115` | `0.95238` | `0.98376` | `0.73321` | `0.71534` |
| 7 | `server_1080ti` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex4-gemini-20260520:14:42` | `296.0` | `0.96138` | `0.96154` | `0.98451` | `0.73032` | `0.70556` |
| 8 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex4-gemini-20260520:14:42` | `296.0` | `0.96138` | `0.96154` | `0.98451` | `0.73032` | `0.70556` |
| 9 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex4-gemini-20260520:14:42` | `296.0` | `0.96138` | `0.96154` | `0.98451` | `0.73032` | `0.70556` |
| 10 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex16-gemini-20260528:16:14` | `295.0` | `0.97351` | `0.94215` | `0.98386` | `0.72964` | `0.69803` |
| 11 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex16-gemini-20260528:16:14` | `295.0` | `0.97351` | `0.94215` | `0.98386` | `0.72964` | `0.69803` |
| 12 | `server_1080ti` | `ssdd/mambayolo_t_ssdd_300e_16bs` | `295.0` | `0.96285` | `0.96154` | `0.98485` | `0.72852` | `0.71082` |
| 13 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_300e_16bs` | `295.0` | `0.96285` | `0.96154` | `0.98485` | `0.72852` | `0.71082` |
| 14 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_300e_16bs` | `295.0` | `0.96285` | `0.96154` | `0.98485` | `0.72852` | `0.71082` |
| 15 | `server_1080ti` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex8-gemini-20260521:10:122` | `293.0` | `0.97387` | `0.95564` | `0.9862` | `0.72836` | `0.7097` |
| 16 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex8-gemini-20260521:10:122` | `293.0` | `0.97387` | `0.95564` | `0.9862` | `0.72836` | `0.7097` |
| 17 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex8-gemini-20260521:10:122` | `293.0` | `0.97387` | `0.95564` | `0.9862` | `0.72836` | `0.7097` |
| 18 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex22-gemini-20260529:15:05` | `293.0` | `0.98104` | `0.93956` | `0.98259` | `0.7273` | `0.69944` |
| 19 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex22-gemini-20260529:15:05` | `293.0` | `0.98104` | `0.93956` | `0.98259` | `0.7273` | `0.69944` |
| 20 | `server_3090_a` | `sar_aircraft/mambayolo_t_sar_aircraft_backbone_ds-ssm_300e_16bs_1024_ex11ssdd_ex2_202606012307` | `300.0` | `0.87799` | `0.84149` | `0.89066` | `0.72591` | `0.70824` |
| 21 | `server_1080ti` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex3-gemini-20260520:11:12` | `291.0` | `0.96125` | `0.95419` | `0.98426` | `0.72425` | `0.70229` |
| 22 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex3-gemini-20260520:11:12` | `291.0` | `0.96125` | `0.95419` | `0.98426` | `0.72425` | `0.70229` |
| 23 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex3-gemini-20260520:11:12` | `291.0` | `0.96125` | `0.95419` | `0.98426` | `0.72425` | `0.70229` |
| 24 | `server_3090_b` | `ssdd/PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_36bs-ex29-gemini-20260604:14:55` | `294.0` | `0.95559` | `0.95238` | `0.97882` | `0.72415` | `0.70893` |
| 25 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex11-gemini-20260526:14:13` | `293.0` | `0.97235` | `0.96599` | `0.98623` | `0.72348` | `0.70509` |
| 26 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex11-gemini-20260526:14:13` | `293.0` | `0.97235` | `0.96599` | `0.98623` | `0.72348` | `0.70509` |
| 27 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex21-gemini-20260529:12:30` | `293.0` | `0.95865` | `0.93423` | `0.98171` | `0.72297` | `0.70213` |
| 28 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex21-gemini-20260529:12:30` | `293.0` | `0.95865` | `0.93423` | `0.98171` | `0.72297` | `0.70213` |
| 29 | `server_3090_a` | `sar_aircraft/mambayolo_t_sar_aircraft_backbone_ds-ssm_300e_8bs_1024_ex7ssdd_ex1_202605231649` | `300.0` | `0.88285` | `0.80158` | `0.88629` | `0.72243` | `0.69926` |
| 30 | `server_3090_b` | `sar_aircraft/mambayolo_t_sar_aircraft_backbone_ds-ssm_300e_8bs_1024_ex7ssdd_ex1_202605231649` | `300.0` | `0.88285` | `0.80158` | `0.88629` | `0.72243` | `0.69926` |
| 31 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex23-gemini-20260529:18:13` | `296.0` | `0.95852` | `0.94689` | `0.98691` | `0.72217` | `0.70109` |
| 32 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex23-gemini-20260529:18:13` | `296.0` | `0.95852` | `0.94689` | `0.98691` | `0.72217` | `0.70109` |
| 33 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex19-gemini-20260529:10:052` | `296.0` | `0.95236` | `0.96337` | `0.98661` | `0.72068` | `0.70083` |
| 34 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex19-gemini-20260529:10:052` | `296.0` | `0.95236` | `0.96337` | `0.98661` | `0.72068` | `0.70083` |
| 35 | `server_1080ti` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex2-gemini-20260520:0:18` | `293.0` | `0.95756` | `0.95048` | `0.98491` | `0.71976` | `0.71214` |
| 36 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex2-gemini-20260520:0:18` | `293.0` | `0.95756` | `0.95048` | `0.98491` | `0.71976` | `0.71214` |
| 37 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex2-gemini-20260520:0:18` | `293.0` | `0.95756` | `0.95048` | `0.98491` | `0.71976` | `0.71214` |
| 38 | `server_1080ti` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex6-gemini-20260520:21:42` | `293.0` | `0.95124` | `0.96481` | `0.98397` | `0.71914` | `0.69858` |
| 39 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex6-gemini-20260520:21:42` | `293.0` | `0.95124` | `0.96481` | `0.98397` | `0.71914` | `0.69858` |
| 40 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex6-gemini-20260520:21:42` | `293.0` | `0.95124` | `0.96481` | `0.98397` | `0.71914` | `0.69858` |
