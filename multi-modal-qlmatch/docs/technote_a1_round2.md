# Tech note: A1 data/sim robustness, round 2 (hit width) and light check round 1

Date: 2026-11-12. Inference only, no retraining.

Plans:
- `docs/a1_check_round_2_plan.md` (width)
- `docs/a1_light_check_round_1_plan.md` (light)

Raw outputs:
- `fm4npp-lartpc-work/a1r2/full/` (width)
- `fm4npp-lartpc-work/a1l1/full/` (light)

Each has a `summary.md` and per-run `metrics.json`.

## Summary

**Hit width (simulated-ν hits made data-like).**
- **qlfm-M CL is strongly width-sensitive.**
  - The new 3D width factors cost **0.288 in T1** (charge-weighted F1, 0.919 → 0.631) and **0.328 in the paper's per-event median T1**. That is a third more than the 2D factors of round 1 (−0.212). T2 drops 0.055.
  - Collection (Y only) accounts for −0.168 and induction (U+V only) for −0.080. These add up to less than the all-plane drop (−0.248 vs −0.288), so the planes interact.
  - Per plane, U loses most: −0.33 charge-weighted, −0.47 per-event median.
- **qlfm-M C + XF is robust.**
  - The 3D factors cost 0.038 in T1 (0.046 per-event median) and 0.004 in T2. The plane contributions are additive.
- **Diffusion bracket:** giving the data term its own drift velocity (0.1076 cm/µs) makes the `w_diff` bracket slightly milder (CL −0.435 → −0.407). Both are still harsher than the measured 3D factors.

**Light (beam-flash light made data-like).**
- **Input corrections hardly matter.**
  - **CL:** the combined correction costs ≤ 0.002 in T1. T2 is unchanged.
  - **C + XF:** the combined correction costs 0.010 in T1 (0.016 per-event median). T2 is unchanged. That is about a quarter of its width effect.
  - The 0.80 and 0.82 scales differ by about 0.001.
  - The late-tail correction alone has no effect.
- **Prompt-only light** (all light at dt ≥ 0.5 µs removed) separates the two models.
  - **CL** barely notices: T1 −0.006, and bright-subset T2 actually +0.004. CL does not use the light's time profile.
  - **C + XF** loses 0.077 in T1 (−0.111 per-event median) and 0.006–0.009 in T2. It relies on the late light.
- **T5 target test:** applying the corrections to the T5 target of simulated-light events shrinks the T5 error gap between ν events and no-ν events.
  - **Size of the gap closed:** 10–12% for the 0.80 scale alone and 18% (CL) / 22% (C + XF) for the combined correction.
  - **Direction of the head's error:** for C + XF the 0.80 scale brings the ν total-light bias from −0.11 to −0.01. For both models the combined correction overshoots, to +0.18–0.26.
  - **Reading:** roughly 10–20% of the ν / no-ν T5 gap is attributable to light mismodelling, as a lower bound. Most of the gap reflects ν events being brighter and harder to predict.

**Bottom line:** the sim/data hit-width mismatch is the dominant robustness risk, and it is serious for CL. Light mismodelling has a small effect on T1/T2 for both models. It explains a modest part of the T5 ν / no-ν difference.

## 1. Setup

- **Models:** the existing multi-task runs (yuhw, `s3/mt`), fine-tuned at 1% labels, seeds 0, 1 and 2:
  - **qlfm-M CL:** light-in-encoder pretraining; light is early-fused.
  - **qlfm-M C + XF:** charge-only pretraining, with light added late by cross-attention.
- **Evaluation:** `qlfm/eval_perturb.py`. Each variant perturbs the inputs at inference time; the trained models are unchanged. Width and light are tested separately, each against its own nominal.
- **Metrics:**
  - **T1:** ν/cosmic hit F1. Two definitions:
    - charge-weighted global F1, the in-run score;
    - per-event median F1, the paper definition.
  - **T2:** ν-event AUC, all events and the paper's bright subset (native 3–6 µs PE ≥ 200).
  - **T5:** light-head MSE.
- **Uncertainties:**
  - Values are means ± seed SD over 3 seeds.
  - Each seed also gets a paired event-bootstrap 95% CI with 1000 resamples. The bootstrap CIs are typically ±0.002–0.006, smaller than the seed spread. Per-seed brackets are tabulated in Appendix A.
- **Validation:** each job started with a smoke run and acceptance checks. All checks passed:
  - nominal reproduces the stored and round 1 scores within 1e-3;
  - round 1 deltas reproduce;
  - each perturbation acts exactly as defined.

## 2. Hit-width test (round 2)

### 2.1 Variants

All variants act on the hits of simulated neutrinos only. `hit_rms` is multiplied by f(plane, drift).

| Variant | Definition |
|---|---|
| `w_meas` | 2D measured data/sim width ratio (round 1 table; cathode Y 0.919, V 0.955, U 0.895) |
| `w_meas3d` | 3D measured ratio, all planes (cathode Y 0.903, V 0.941, U 0.880) |
| `w_meas3d_Y` | 3D ratio on Y only (f = 1 on U, V) |
| `w_meas3d_UV` | 3D ratio on U and V only (f = 1 on Y) |
| `w_diff` | Diffusion-only bracket: both terms use v = 0.1098 cm/µs (0.8745 at full drift) |
| `w_diff_vd` | Same, but the data term uses v_data = 0.1076 cm/µs (cm² → µs² conversion only); drift time t and the clamp stay on the sim timescale (0.8823 at full drift) |

The 3D per-bin factors (Sunny, p.13), on drift-bin centres 388 … 4271 ticks from the anode:

| Plane | Factors |
|---|---|
| Y | 1.000, 0.967, 0.941, 0.926, 0.915, 0.903 |
| V | 1.000, 0.980, 0.954, 0.938, 0.942, 0.941 |
| U | 1.000, 0.960, 0.922, 0.907, 0.892, 0.880 |

### 2.2 Main results (Δ vs nominal; nominal = absolute)

Mean ± seed SD (ddof = 1) over 3 seeds. Per-seed bootstrap 95% CIs of the deltas are in A.1.

**qlfm-M CL**

| Variant | T1 charge-wtd | T1 evt-median | T2 |
|---|---|---|---|
| nominal | 0.9185 ± 0.0002 | 0.9217 ± 0.0006 | 0.9932 ± 0.0028 |
| `w_meas` (2D) | −0.2120 ± 0.0113 | n/a | −0.0337 ± 0.0046 |
| `w_meas3d` | **−0.2877 ± 0.0164** | **−0.3278 ± 0.0201** | **−0.0554 ± 0.0081** |
| `w_meas3d_Y` | −0.1680 ± 0.0158 | n/a | −0.0229 ± 0.0070 |
| `w_meas3d_UV` | −0.0799 ± 0.0035 | n/a | −0.0020 ± 0.0009 |
| `w_diff` | −0.4350 ± 0.0191 | n/a | −0.1062 ± 0.0135 |
| `w_diff_vd` | −0.4066 ± 0.0188 | n/a | −0.0954 ± 0.0125 |

**qlfm-M C + XF**

| Variant | T1 charge-wtd | T1 evt-median | T2 |
|---|---|---|---|
| nominal | 0.8593 ± 0.0017 | 0.8622 ± 0.0015 | 0.9899 ± 0.0005 |
| `w_meas` (2D) | −0.0289 ± 0.0022 | n/a | −0.0027 ± 0.0002 |
| `w_meas3d` | **−0.0375 ± 0.0030** | **−0.0458 ± 0.0040** | **−0.0035 ± 0.0003** |
| `w_meas3d_Y` | −0.0234 ± 0.0017 | n/a | −0.0022 ± 0.0003 |
| `w_meas3d_UV` | −0.0142 ± 0.0008 | n/a | −0.0008 ± 0.0002 |
| `w_diff` | −0.0579 ± 0.0039 | n/a | −0.0058 ± 0.0004 |
| `w_diff_vd` | −0.0517 ± 0.0034 | n/a | −0.0052 ± 0.0003 |

The per-event median is computed from saved predictions, which exist only for nominal and `w_meas3d`.


![Hit-width variants: T1 (charge-weighted F1), absolute (§2.2). Dashed lines: nominal per model.](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1r2_width_T1_qwtd_abs.pdf)

![Hit-width variants: Δ T1 (charge-weighted) vs nominal (§2.2). Mean over 3 seeds; error bars = seed SD.](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1r2_width_T1_qwtd.pdf)


![Hit-width variants: T2 (AUC), absolute (§2.2).](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1r2_width_T2_abs.pdf)

![Hit-width variants: Δ T2 vs nominal (§2.2).](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1r2_width_T2.pdf)

**Additivity (T1 charge-weighted):**

| Model | ΔY + ΔUV | Δall | Per-seed ΔY + ΔUV − Δall (s0, s1, s2) |
|---|---|---|---|
| CL | −0.2479 ± 0.0129 | −0.2877 ± 0.0164 | +0.0323, +0.0467, +0.0404 (all planes together cost more than the sum) |
| C + XF | −0.0376 ± 0.0024 | −0.0375 ± 0.0030 | +0.0006, −0.0007, −0.0000 (additive) |

### 2.3 Per-plane T1 (seed-averaged)

Δ = change vs nominal on the same plane. Mean ± seed SD.

**qlfm-M CL**, charge-weighted:

| Plane | Nominal | `w_meas3d` | Δ | Y-only Δ | UV-only Δ |
|---|---|---|---|---|---|
| U | 0.9013 ± 0.0002 | 0.5727 ± 0.0139 | −0.3286 ± 0.0137 | −0.1043 ± 0.0149 | −0.2040 ± 0.0077 |
| V | 0.9016 ± 0.0009 | 0.6451 ± 0.0116 | −0.2566 ± 0.0119 | −0.1234 ± 0.0129 | −0.0616 ± 0.0066 |
| Y | 0.9486 ± 0.0001 | 0.6659 ± 0.0224 | −0.2827 ± 0.0223 | −0.2711 ± 0.0204 | −0.0081 ± 0.0005 |

**qlfm-M CL**, per-event median (paper definition):

| Plane | Nominal | `w_meas3d` | Δ |
|---|---|---|---|
| U | 0.8929 ± 0.0006 | 0.4230 ± 0.0147 | **−0.4699 ± 0.0153** |
| V | 0.8964 ± 0.0009 | 0.5990 ± 0.0138 | −0.2974 ± 0.0138 |
| Y | 0.9712 ± 0.0004 | 0.6831 ± 0.0335 | −0.2881 ± 0.0336 |

**qlfm-M C + XF**, charge-weighted:

| Plane | Nominal | `w_meas3d` | Δ | Y-only Δ | UV-only Δ |
|---|---|---|---|---|---|
| U | 0.8391 ± 0.0024 | 0.7920 ± 0.0049 | −0.0471 ± 0.0056 | −0.0150 ± 0.0014 | −0.0313 ± 0.0029 |
| V | 0.8435 ± 0.0032 | 0.8161 ± 0.0024 | −0.0274 ± 0.0013 | −0.0110 ± 0.0007 | −0.0154 ± 0.0003 |
| Y | 0.8907 ± 0.0004 | 0.8522 ± 0.0022 | −0.0385 ± 0.0025 | −0.0413 ± 0.0034 | +0.0006 ± 0.0002 |

**qlfm-M C + XF**, per-event median (paper definition):

| Plane | Nominal | `w_meas3d` | Δ |
|---|---|---|---|
| U | 0.8191 ± 0.0021 | 0.7455 ± 0.0078 | −0.0737 ± 0.0094 |
| V | 0.8305 ± 0.0049 | 0.7912 ± 0.0060 | −0.0393 ± 0.0029 |
| Y | 0.9334 ± 0.0012 | 0.8977 ± 0.0023 | −0.0357 ± 0.0013 |


![Per-plane T1 (charge-weighted), absolute, for nominal, `w_meas3d`, Y-only and UV-only (§2.3). Dashed segments: same-plane nominal.](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1r2_perplane_T1_qwtd_abs.pdf)

![Per-plane Δ T1 (charge-weighted) for `w_meas3d`, Y-only and UV-only (§2.3).](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1r2_perplane_T1_qwtd.pdf)


![Per-plane T1 (per-event median), absolute, for nominal and `w_meas3d` (§2.3).](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1r2_perplane_T1_evtmed_abs.pdf)

![Per-plane Δ T1 (per-event median) for `w_meas3d` (§2.3); Y-only/UV-only per-event medians were not saved.](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1r2_perplane_T1_evtmed.pdf)

- **Cross-plane effect in CL:** perturbing Y alone also costs U and V 0.10–0.12, through the model's cross-plane context.
- **Within-plane effect in C + XF:** each plane responds mainly to its own perturbation.
- **U is hit hardest in both models.** Its width factor is the smallest (0.880 at the cathode).

### 2.4 Width findings

1. The 3D measurement moves the data-like point further from simulation. CL's T1 loss grows from −0.21 (2D) to −0.29 (3D).
2. CL's sensitivity is not confined to the plane being perturbed. Y-only and U+V-only perturbations interact, and a Y-only change propagates to U and V.
3. C + XF's loss is small and additive by plane.
4. The diffusion bracket is harsher than the measured 3D factors for both models. The data drift velocity changes it only modestly: CL −0.435 → −0.407, C + XF −0.058 → −0.052.

## 3. Light check (round 1)

### 3.1 Definitions and variants

- **Per event:**
  - **t_b** is the lower edge of the 0.1 µs bin with the most ophit PE in 3.4–5.6 µs.
  - **P_b** is the summed PE in [t_b − 0.25, t_b + 0.5) µs.
  - **dt** = t − t_b.
- **Scope:** variants act on all ophits of events with a flash, because the open files do not mark simulated ophits. Hit width and charge stay native.

| Variant | Definition |
|---|---|
| `l_scale80` / `l_scale82` | PE × 0.80 / 0.82 in dt ∈ [−0.25, 3) µs (from MicroBooNE sim/data PE/cm 1.25 / 1.22) |
| `l_slow` | dt [0.5,1) / [1,1.5) / [1.5,2) / [2,3) µs × 0.83/0.65/0.61/0.64 if P_b < 600, × 0.82/0.63/0.48/0.45 if P_b ≥ 600 |
| `l_tail` | dt ≥ 8 µs × 0.37 if P_b ≥ 600 |
| `l_comb80` / `l_comb82` | scale + slow + tail combined |
| `l_prompt` | all ophits at dt ≥ 0.5 µs removed |

**Diagnostics:**
- **Flash finding:** 23 261 test events; a flash was found in 99.97% (3.4–5.6 µs). The t_b quantiles are 3.6 / 4.0 / 4.4 / 4.8 / 5.1 µs (5, 25, 50, 75, 95%).
- **P_b:** quantiles 10.2 / 98.3 / 573.6 / 1903.9 / 5818.3 PE (5, 25, 50, 75, 95%). P_b ≥ 600 in 49.13% of events (49.14% of flash events).
- **Bright subset** (native 3–6 µs PE ≥ 200, used for T2 bright): 0.7263 of events.
- **t_b histogram** (22 bins of 0.1 µs from 3.4 µs): 80, 70, 1071, 1504, 1352, 1475, 1325, 1374, 1571, 1360, 1573, 1323, 1284, 1547, 1315, 1598, 1327, 1361, 437, 96, 93, 118. The distribution is flat from 3.6 to 5.1 µs, with small tails at both ends.
- **Measured PE ratios:** summed PE after/before in each dt window (flash events, seed-0 run of the first model) matches the definitions exactly. Dim = P_b < 600, bright = P_b ≥ 600.

| Variant | Class | [−0.25, 0.5) | [0.5, 3) | [3, 8) | ≥ 8 |
|---|---|---|---|---|---|
| `l_scale80` | dim | 0.8000 | 0.8000 | 1.0000 | 1.0000 |
| `l_scale80` | bright | 0.8000 | 0.8000 | 1.0000 | 1.0000 |
| `l_scale82` | dim | 0.8200 | 0.8200 | 1.0000 | 1.0000 |
| `l_scale82` | bright | 0.8200 | 0.8200 | 1.0000 | 1.0000 |
| `l_slow` | dim | 1.0000 | 0.6792 | 1.0000 | 1.0000 |
| `l_slow` | bright | 1.0000 | 0.5326 | 1.0000 | 1.0000 |
| `l_tail` | dim | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `l_tail` | bright | 1.0000 | 1.0000 | 1.0000 | 0.3700 |
| `l_comb80` | dim | 0.8000 | 0.5434 | 1.0000 | 1.0000 |
| `l_comb80` | bright | 0.8000 | 0.4260 | 1.0000 | 0.3700 |
| `l_comb82` | dim | 0.8200 | 0.5570 | 1.0000 | 1.0000 |
| `l_comb82` | bright | 0.8200 | 0.4367 | 1.0000 | 0.3700 |
| `l_prompt` | dim | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `l_prompt` | bright | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

### 3.2 Input-light results (Δ vs nominal)

**qlfm-M CL** (nominal: T1 0.9185 ± 0.0002 / 0.9217 ± 0.0006 evt-median, T2 0.9932 ± 0.0029 all, 0.9881 ± 0.0035 bright). Per-seed CIs in A.2.

| Variant | T1 charge-wtd | T1 evt-median | T2 all | T2 bright |
|---|---|---|---|---|
| `l_scale80` | −0.0004 ± 0.0001 | −0.0009 ± 0.0002 | +0.0000 ± 0.0003 | +0.0002 ± 0.0006 |
| `l_scale82` | −0.0004 ± 0.0001 | −0.0007 ± 0.0002 | +0.0000 ± 0.0002 | +0.0002 ± 0.0006 |
| `l_slow` | −0.0003 ± 0.0000 | −0.0005 ± 0.0003 | +0.0002 ± 0.0002 | +0.0006 ± 0.0006 |
| `l_tail` | −0.0000 ± 0.0000 | −0.0000 ± 0.0001 | +0.0000 ± 0.0000 | −0.0002 ± 0.0005 |
| `l_comb80` | −0.0010 ± 0.0001 | −0.0019 ± 0.0003 | +0.0001 ± 0.0003 | +0.0005 ± 0.0008 |
| `l_comb82` | −0.0009 ± 0.0001 | −0.0017 ± 0.0004 | +0.0002 ± 0.0003 | +0.0005 ± 0.0008 |
| `l_prompt` | −0.0059 ± 0.0007 | −0.0058 ± 0.0008 | +0.0012 ± 0.0009 | **+0.0041 ± 0.0011** |

**qlfm-M C + XF** (nominal: T1 0.8593 ± 0.0017 / 0.8623 ± 0.0016 evt-median, T2 0.9899 ± 0.0005 all, 0.9689 ± 0.0004 bright). Per-seed CIs in A.2.

| Variant | T1 charge-wtd | T1 evt-median | T2 all | T2 bright |
|---|---|---|---|---|
| `l_scale80` | −0.0039 ± 0.0004 | −0.0067 ± 0.0007 | +0.0001 ± 0.0001 | +0.0001 ± 0.0003 |
| `l_scale82` | −0.0032 ± 0.0004 | −0.0057 ± 0.0006 | +0.0001 ± 0.0001 | +0.0002 ± 0.0003 |
| `l_slow` | −0.0035 ± 0.0003 | −0.0060 ± 0.0003 | +0.0003 ± 0.0000 | +0.0007 ± 0.0002 |
| `l_tail` | +0.0001 ± 0.0001 | +0.0001 ± 0.0002 | −0.0002 ± 0.0001 | −0.0004 ± 0.0002 |
| `l_comb80` | **−0.0099 ± 0.0002** | **−0.0156 ± 0.0010** | −0.0001 ± 0.0001 | −0.0000 ± 0.0005 |
| `l_comb82` | −0.0088 ± 0.0002 | −0.0143 ± 0.0012 | −0.0000 ± 0.0001 | +0.0000 ± 0.0004 |
| `l_prompt` | **−0.0771 ± 0.0135** | **−0.1106 ± 0.0222** | −0.0057 ± 0.0008 | −0.0085 ± 0.0032 |


![Input-light variants: T1 (charge-weighted F1), absolute (§3.2).](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1l1_light_T1_qwtd_abs.pdf)

![Input-light variants: Δ T1 (charge-weighted) vs nominal (§3.2). Right panel: `l_prompt` on its own scale.](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1l1_light_T1_qwtd.pdf)


![Input-light variants: T1 (per-event median), absolute (§3.2).](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1l1_light_T1_evtmed_abs.pdf)

![Input-light variants: Δ T1 (per-event median) vs nominal (§3.2).](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1l1_light_T1_evtmed.pdf)


![Input-light variants: T2 (all events), absolute (§3.2).](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1l1_light_T2_all_abs.pdf)

![Input-light variants: Δ T2 (all events) vs nominal (§3.2).](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1l1_light_T2_all.pdf)


![Input-light variants: T2 (bright subset), absolute (§3.2).](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1l1_light_T2_bright_abs.pdf)

![Input-light variants: Δ T2 (bright subset) vs nominal (§3.2).](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1l1_light_T2_bright.pdf)

**T2 by native P_b class** (nominal AUC, then Δ; mean ± seed SD; per-seed CIs in A.3):

| Model | Variant | P_b < 600 | P_b >= 600 |
|---|---|---|---|
| CL | nominal | 0.9874 ± 0.0055 | 0.9914 ± 0.0009 |
| CL | `l_slow` | −0.0001 ± 0.0004 | +0.0012 ± 0.0013 |
| CL | `l_tail` | +0.0000 ± 0.0000 | −0.0006 ± 0.0012 |
| CL | `l_comb80` | −0.0004 ± 0.0009 | +0.0013 ± 0.0014 |
| CL | `l_comb82` | −0.0003 ± 0.0008 | +0.0013 ± 0.0014 |
| CL | `l_prompt` | +0.0005 ± 0.0022 | +0.0040 ± 0.0010 |
| C + XF | nominal | 0.9869 ± 0.0010 | 0.9688 ± 0.0011 |
| C + XF | `l_slow` | +0.0000 ± 0.0001 | +0.0008 ± 0.0004 |
| C + XF | `l_tail` | −0.0000 ± 0.0000 | −0.0006 ± 0.0004 |
| C + XF | `l_comb80` | −0.0006 ± 0.0002 | −0.0004 ± 0.0012 |
| C + XF | `l_comb82` | −0.0005 ± 0.0002 | −0.0003 ± 0.0012 |
| C + XF | `l_prompt` | **−0.0129 ± 0.0015** | **−0.0076 ± 0.0026** |

Only `l_prompt` on C + XF is clearly beyond noise (the CL `l_prompt` P_b >= 600 value is positive in all seeds).

### 3.3 Input-light findings

1. **Total scale (0.80 or 0.82):** negligible for CL and small for C + XF (T1 −0.004 / −0.007). The two values differ by about 0.001, so the uncertainty in the published factor does not matter here.
2. **Slow light vs tail:** for C + XF the slow-light correction matters as much as the scale (−0.006 evt-median). The late-tail correction alone does nothing for either model. The combined effect is about the sum of its parts: −0.0156 for C + XF, against −0.0127 summed.
3. **Time profile (`l_prompt`):**
   - CL barely uses light beyond the prompt window. Removing it costs 0.006 in T1 and slightly improves bright-subset T2.
   - C + XF depends on the late light for T1 (−0.08 / −0.11), so its time-profile mismodelling is the relevant light risk. The measured corrections cost only 0.010–0.016.
4. **CL vs C + XF:** early fusion (CL) is nearly insensitive to light changes. Late fusion (C + XF) is sensitive but stays well below its width sensitivity: −0.016 vs −0.046 evt-median T1.

## 4. T5 target test

### 4.1 Method

- **Predictions:** the light head (charge → 32 PMTs × 17 log1p light features) runs once on unperturbed input over its 10 000 test events. CL keeps its trained light-off pass.
- **Targets:** each correction is applied to the target of events with a beam flash, and the MSE is recomputed. The head is unchanged.
- **Groups** (`evt_nu` class):

  | Group | Definition | n |
  |---|---|---|
  | `nu` | visible in-AV ν | 4389 |
  | `out_av` | ν outside the AV that still leaves charge | 3251 |
  | `nonu_flash` | no-ν with P_b ≥ 20 PE | 992 |
  | `nonu_dark` | no-ν with P_b < 20 PE: the real-light reference, never corrected | 948 |
  | `other` | low-visibility in-AV ν; excluded | 420 |

- **Gap** = MSE(`nu`) − MSE(`nonu_dark`).
- **Fraction closed** = (gap_nominal − gap) / gap_nominal.
- **Bias** = mean (pred − target) of the per-PMT log1p total.
- **Reference definition:** the planned reference was no-ν events with no flash at all. Only 1 of the 1 940 class-0 test events had none, so every run used the P_b < 20 PE fallback. The P_b quantiles for class 0 are 5.7 / 9.7 / 21 / 69 / 1115 PE (5, 25, 50, 75, 95%).
  Fraction of class-0 events with P_b below a threshold (seed 0; flash found in 0.999, P_b range 0 to 6597 PE):

  | P_b below (PE) | 1 | 5 | 20 | 50 | 100 | 200 |
  |---|---|---|---|---|---|---|
  | fraction | 0.001 | 0.032 | 0.489 | 0.688 | 0.806 | 0.871 |
- **Input-light check:** the input-light variants leave T5 unchanged. |ΔMSE| ≤ 1.3e-5, which is bf16 rounding.

### 4.2 Reference MSE (same under every target variant; checked identical in every seed)

| Model | MSE(`nonu_dark`) | per-seed (CIs in A.5) | per-PMT-total MSE | total bias |
|---|---|---|---|---|
| CL | **0.2158 ± 0.0011** | 0.2146, 0.2160, 0.2169 (95% CI ±0.010) | 0.5710 ± 0.0059 | +0.2683 ± 0.0034 |
| C + XF | **0.2593 ± 0.0004** | 0.2588, 0.2595, 0.2596 (95% CI ±0.012) | 0.8265 ± 0.0111 | +0.4360 ± 0.0063 |

### 4.3 Results

Mean ± seed SD. Group sizes as in 4.1 (nu 4389, out_av 3251, nonu_flash 992). The Δgap columns for `out_av` and `nonu_flash` are the change in MSE(group) − MSE(`nonu_dark`). Per-seed CIs of Δgap are in A.4.

**qlfm-M CL**

| Target variant | MSE nu | MSE out_av | MSE nonu_flash | gap | Δgap | closed | Δgap out_av | Δgap nonu_flash | bias nu |
|---|---|---|---|---|---|---|---|---|---|
| nominal | 0.6717 ± 0.0007 | 0.4765 ± 0.0016 | 0.4557 ± 0.0024 | 0.4559 ± 0.0015 | 0 | 0 | 0 | 0 | −0.0354 ± 0.0016 |
| `t_scale80` | 0.6249 ± 0.0006 | 0.4430 ± 0.0016 | 0.4202 ± 0.0024 | 0.4090 ± 0.0014 | −0.0468 ± 0.0001 | 10.3% | −0.0335 ± 0.0001 | −0.0355 ± 0.0001 | +0.0689 ± 0.0016 |
| `t_scale82` | 0.6296 ± 0.0006 | 0.4463 ± 0.0016 | 0.4238 ± 0.0024 | 0.4137 ± 0.0014 | −0.0422 ± 0.0001 | 9.2% | −0.0302 ± 0.0001 | −0.0319 ± 0.0001 | +0.0577 ± 0.0016 |
| `t_slow` | 0.6482 ± 0.0006 | 0.4544 ± 0.0016 | 0.4250 ± 0.0023 | 0.4323 ± 0.0014 | −0.0235 ± 0.0001 | 5.2% | −0.0221 ± 0.0001 | −0.0307 ± 0.0002 | +0.0404 ± 0.0016 |
| `t_tail` | 0.6453 ± 0.0008 | 0.4647 ± 0.0014 | 0.4488 ± 0.0023 | 0.4294 ± 0.0015 | −0.0264 ± 0.0001 | 5.8% | −0.0118 ± 0.0004 | −0.0069 ± 0.0002 | +0.0698 ± 0.0016 |
| `t_comb80` | 0.5883 ± 0.0006 | 0.4176 ± 0.0014 | 0.3894 ± 0.0022 | 0.3724 ± 0.0013 | **−0.0835 ± 0.0002** | **18.3%** | −0.0588 ± 0.0003 | −0.0663 ± 0.0003 | +0.2563 ± 0.0016 |
| `t_comb82` | 0.5916 ± 0.0006 | 0.4201 ± 0.0014 | 0.3923 ± 0.0022 | 0.3758 ± 0.0013 | −0.0801 ± 0.0002 | 17.6% | −0.0564 ± 0.0003 | −0.0634 ± 0.0003 | +0.2451 ± 0.0016 |

**qlfm-M C + XF**

| Target variant | MSE nu | MSE out_av | MSE nonu_flash | gap | Δgap | closed | Δgap out_av | Δgap nonu_flash | bias nu |
|---|---|---|---|---|---|---|---|---|---|
| nominal | 0.7283 ± 0.0033 | 0.5451 ± 0.0023 | 0.4760 ± 0.0020 | 0.4690 ± 0.0032 | 0 | 0 | 0 | 0 | −0.1118 ± 0.0053 |
| `t_scale80` | 0.6746 ± 0.0029 | 0.5058 ± 0.0023 | 0.4431 ± 0.0020 | 0.4153 ± 0.0028 | −0.0537 ± 0.0004 | 11.5% | −0.0393 ± 0.0002 | −0.0328 ± 0.0001 | **−0.0076 ± 0.0053** |
| `t_scale82` | 0.6800 ± 0.0030 | 0.5098 ± 0.0023 | 0.4465 ± 0.0020 | 0.4207 ± 0.0029 | −0.0483 ± 0.0003 | 10.3% | −0.0354 ± 0.0002 | −0.0295 ± 0.0001 | −0.0187 ± 0.0053 |
| `t_slow` | 0.6957 ± 0.0028 | 0.5157 ± 0.0023 | 0.4484 ± 0.0020 | 0.4364 ± 0.0027 | −0.0327 ± 0.0005 | 7.0% | −0.0294 ± 0.0002 | −0.0276 ± 0.0001 | −0.0360 ± 0.0053 |
| `t_tail` | 0.6995 ± 0.0032 | 0.5321 ± 0.0023 | 0.4685 ± 0.0021 | 0.4402 ± 0.0031 | −0.0288 ± 0.0001 | 6.1% | −0.0130 ± 0.0000 | −0.0075 ± 0.0001 | −0.0066 ± 0.0053 |
| `t_comb80` | 0.6268 ± 0.0024 | 0.4722 ± 0.0022 | 0.4146 ± 0.0020 | 0.3675 ± 0.0023 | **−0.1015 ± 0.0009** | **21.6%** | −0.0729 ± 0.0004 | −0.0614 ± 0.0001 | +0.1799 ± 0.0053 |
| `t_comb82` | 0.6309 ± 0.0024 | 0.4753 ± 0.0022 | 0.4172 ± 0.0020 | 0.3716 ± 0.0023 | −0.0974 ± 0.0009 | 20.8% | −0.0698 ± 0.0004 | −0.0587 ± 0.0001 | +0.1687 ± 0.0053 |

Per-seed paired-bootstrap 95% CIs on Δgap are about ±0.001 to ±0.003, and the seed SD on Δgap is ≤ 0.001. Every change is clearly significant.

Per-PMT-total MSE at nominal (nu / `nonu_dark`): CL 0.3288 ± 0.0050 / 0.5710 ± 0.0059; C + XF 0.6394 ± 0.0217 / 0.8265 ± 0.0111.


![T5 target test: change in the MSE gap to `nonu_dark` for each target variant (§4.3). Mean over 3 seeds; error bars = seed SD.](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1l1_T5_dgap.pdf)


![T5 target test: MSE per event group for nominal and each target variant (§4.3). Dashed lines: MSE(`nonu_dark`) per model.](/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures/a1l1_T5_mse.pdf)

### 4.4 T5 findings

1. **The gap shrinks under every correction.** It shrinks most under the total scale (10–12%), then slow light (5–7%) and the tail (6%). The combined correction closes 18% (CL) and 22% (C + XF). The parts sum to slightly more than the combination (21% vs 18% for CL).
2. **The bias shows the direction.** At nominal the head under-predicts ν-event light (bias −0.04 CL, −0.11 C + XF).
   - For C + XF, the 0.80 scale alone brings the bias to −0.01, so its head's total-light prediction for ν events is close to data-like.
   - For CL, the scale already overshoots (+0.07).
   - The full combined correction overshoots for both models (+0.18 / +0.26). Applying all three corrections over-corrects these heads.
3. **The `out_av` and `nonu_flash` groups** (which also have simulated beam-flash light) move the same way, by smaller amounts.
4. **Interpretation:** light mismodelling explains about 10–20% of the ν / no-ν T5 gap, as a lower bound. The rest reflects ν events being brighter with more structured light, not mismodelling.
5. **Caveat on the reference:** the `nonu_dark` events are very dim (P_b < 20 PE), and the head over-predicts their total light (bias +0.27 / +0.44). Their per-PMT-total MSE (0.57 / 0.83) is higher than for ν events (0.33 / 0.64). The all-feature gap therefore reflects differences in light *shape* across 16 time bins, not only total yield.

## 5. Caveats

- **The light comparison is not like-for-like.** Real flashes are cosmic muons and simulated flashes are ν events, and ophits are not labelled sim/real. The corrections act on all ophits in each event's beam-flash windows. Real in-time cosmics are about 1–2% of events.
- **"No-neutrino" events are overlays with a sub-threshold ν.** Nearly all of them have some light in the flash window, so the real-light reference is dim events (P_b < 20 PE), not events without any ν light.
- **The 3D U and V width factors may still be provisional,** as the 2D ones were. To be confirmed with Sunny.
- **Two T1 definitions disagree in size.** Round 1 reported only the charge-weighted F1. The paper's per-event median shows larger width drops: CL −0.33 vs −0.29.
- **Inference runs in bf16.** Repeated passes differ by a few bf16 ulps per output. This is irrelevant at the effect sizes here (≥ 1e-3).

## 6. Cost and reproducibility

**Jobs:**

| Job | What | Walltime | Node-hours |
|---|---|---|---|
| 7837342 | width round 2 (smoke + 6 runs + summary) | 34 min × 2 nodes | ≈ 1.1 |
| 7838867 | light, failed at smoke check (check too strict; fixed) | 20 min × 2 nodes | ≈ 0.7 |
| 7839956 | light (reused smoke run + 6 runs + summary) | 21 min × 2 nodes | ≈ 0.7 |

Total about 2.5 node-hours, in the debug queue.

**Code** (jpeterso copy; uncommitted):
- Evaluation code:
  - `qlfm/eval_perturb.py`: 3D tables, plane masks, `diff_vd`, `--save-preds`, per-plane bootstrap, light variants, T5 target test.
  - `qlfm/paper_metrics.py`: adds `t1_by_plane`.
  - `qlfm/train_multitask.py`: an optional `hit_extra` argument in `predict()`.
- Tests: `tests/test_a1r2.py`, `tests/test_a1_light.py`.
- Job scripts: `scripts/a1r2_*`, `scripts/a1l1_*`, `scripts/pbs/a1r2_all.pbs`, `scripts/pbs/a1l1_all.pbs`.
- Rerun with `bash scripts/a1r2_submit.sh` or `bash scripts/a1l1_submit.sh`. Both are resumable and skip finished runs.

**Inputs:** `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/s3/mt/qlfm_M_{CL,C_xf}/finetune/f0.01_s{0,1,2}`.


## Appendix A. Per-seed 95% confidence intervals

Each bracket is the paired event-bootstrap 95% CI (1000 resamples) of the delta vs nominal, for seeds 0, 1, 2 in order.

### A.1 Width: T1 charge-weighted and T2

**qlfm-M CL**

| Variant | T1 charge-wtd (s0; s1; s2) | T2 (s0; s1; s2) |
|---|---|---|
| `w_meas` | [-0.207, -0.195]; [-0.217, -0.205]; [-0.230, -0.217] | [-0.033, -0.026]; [-0.043, -0.035]; [-0.036, -0.029] |
| `w_meas3d` | [-0.277, -0.263]; [-0.297, -0.282]; [-0.310, -0.295] | [-0.055, -0.045]; [-0.070, -0.059]; [-0.057, -0.047] |
| `w_meas3d_Y` | [-0.159, -0.149]; [-0.169, -0.160]; [-0.190, -0.180] | [-0.019, -0.014]; [-0.034, -0.027]; [-0.025, -0.020] |
| `w_meas3d_UV` | [-0.086, -0.082]; [-0.081, -0.077]; [-0.079, -0.075] | [-0.002, -0.001]; [-0.004, -0.002]; [-0.002, -0.001] |
| `w_diff` | [-0.422, -0.404]; [-0.450, -0.432]; [-0.458, -0.440] | [-0.105, -0.091]; [-0.130, -0.114]; [-0.107, -0.092] |
| `w_diff_vd` | [-0.394, -0.377]; [-0.420, -0.403]; [-0.430, -0.412] | [-0.094, -0.081]; [-0.118, -0.102]; [-0.097, -0.083] |

**qlfm-M C + XF**

| Variant | T1 charge-wtd (s0; s1; s2) | T2 (s0; s1; s2) |
|---|---|---|
| `w_meas` | [-0.032, -0.030]; [-0.028, -0.025]; [-0.030, -0.028] | [-0.003, -0.002]; [-0.003, -0.002]; [-0.003, -0.002] |
| `w_meas3d` | [-0.042, -0.039]; [-0.036, -0.033]; [-0.040, -0.037] | [-0.004, -0.003]; [-0.004, -0.003]; [-0.005, -0.003] |
| `w_meas3d_Y` | [-0.026, -0.024]; [-0.023, -0.020]; [-0.025, -0.023] | [-0.003, -0.002]; [-0.002, -0.002]; [-0.003, -0.002] |
| `w_meas3d_UV` | [-0.016, -0.014]; [-0.014, -0.013]; [-0.015, -0.013] | [-0.001, -0.000]; [-0.001, -0.001]; [-0.001, -0.000] |
| `w_diff` | [-0.064, -0.059]; [-0.056, -0.051]; [-0.061, -0.056] | [-0.007, -0.005]; [-0.006, -0.005]; [-0.007, -0.005] |
| `w_diff_vd` | [-0.057, -0.053]; [-0.050, -0.046]; [-0.054, -0.050] | [-0.006, -0.004]; [-0.006, -0.004]; [-0.007, -0.005] |

### A.2 Light: input variants, per-event-median T1 and bright-subset T2

**qlfm-M CL**

| Variant | T1 evt-median (s0; s1; s2) | T2 bright (s0; s1; s2) |
|---|---|---|
| `l_scale80` | [-0.002, -0.001]; [-0.001, -0.000]; [-0.001, -0.000] | [-0.000, +0.003]; [-0.001, +0.000]; [-0.000, +0.000] |
| `l_scale82` | [-0.002, -0.000]; [-0.001, -0.000]; [-0.001, -0.000] | [-0.000, +0.003]; [-0.001, +0.000]; [-0.000, +0.000] |
| `l_slow` | [-0.001, -0.000]; [-0.001, +0.000]; [-0.001, -0.000] | [+0.000, +0.004]; [-0.000, +0.001]; [-0.000, +0.001] |
| `l_tail` | [-0.001, +0.000]; [-0.001, +0.000]; [-0.000, +0.000] | [-0.000, +0.000]; [-0.003, +0.001]; [-0.000, +0.000] |
| `l_comb80` | [-0.003, -0.002]; [-0.002, -0.001]; [-0.002, -0.001] | [-0.000, +0.004]; [-0.001, +0.001]; [-0.000, +0.001] |
| `l_comb82` | [-0.003, -0.002]; [-0.002, -0.001]; [-0.002, -0.001] | [-0.000, +0.004]; [-0.001, +0.001]; [-0.000, +0.000] |
| `l_prompt` | [-0.007, -0.005]; [-0.007, -0.005]; [-0.006, -0.004] | [+0.001, +0.007]; [+0.002, +0.009]; [+0.001, +0.006] |

**qlfm-M C + XF**

| Variant | T1 evt-median (s0; s1; s2) | T2 bright (s0; s1; s2) |
|---|---|---|
| `l_scale80` | [-0.008, -0.005]; [-0.008, -0.005]; [-0.009, -0.006] | [-0.001, +0.001]; [-0.001, +0.002]; [-0.001, +0.001] |
| `l_scale82` | [-0.006, -0.004]; [-0.008, -0.004]; [-0.008, -0.005] | [-0.000, +0.001]; [-0.000, +0.001]; [-0.001, +0.001] |
| `l_slow` | [-0.007, -0.004]; [-0.008, -0.004]; [-0.008, -0.005] | [-0.000, +0.002]; [-0.000, +0.002]; [-0.000, +0.001] |
| `l_tail` | [-0.001, +0.001]; [-0.001, +0.001]; [-0.001, +0.000] | [-0.001, -0.000]; [-0.000, +0.000]; [-0.001, -0.000] |
| `l_comb80` | [-0.017, -0.012]; [-0.018, -0.014]; [-0.019, -0.015] | [-0.001, +0.002]; [-0.001, +0.002]; [-0.002, +0.001] |
| `l_comb82` | [-0.015, -0.011]; [-0.016, -0.012]; [-0.018, -0.014] | [-0.001, +0.002]; [-0.001, +0.002]; [-0.002, +0.001] |
| `l_prompt` | [-0.106, -0.095]; [-0.100, -0.091]; [-0.142, -0.130] | [-0.010, -0.001]; [-0.013, -0.003]; [-0.017, -0.007] |

### A.3 Light: T2 by native P_b class

**qlfm-M CL**

| Variant | P_b < 600 (s0; s1; s2) | P_b >= 600 (s0; s1; s2) |
|---|---|---|
| `l_slow` | [-0.000, +0.001]; [-0.001, -0.000]; [-0.000, +0.000] | [+0.000, +0.008]; [+0.000, +0.001]; [+0.000, +0.001] |
| `l_tail` | [-0.000, +0.000]; [-0.000, +0.000]; [-0.000, +0.000] | [-0.000, +0.000]; [-0.007, +0.001]; [-0.000, +0.000] |
| `l_comb80` | [-0.001, +0.001]; [-0.002, -0.000]; [-0.000, +0.001] | [-0.000, +0.009]; [+0.000, +0.001]; [-0.000, +0.001] |
| `l_comb82` | [-0.000, +0.001]; [-0.002, -0.000]; [-0.000, +0.001] | [-0.000, +0.009]; [+0.000, +0.001]; [-0.000, +0.001] |
| `l_prompt` | [-0.003, +0.001]; [+0.000, +0.006]; [-0.002, +0.001] | [+0.001, +0.010]; [+0.001, +0.005]; [+0.001, +0.011] |

**qlfm-M C + XF**

| Variant | P_b < 600 (s0; s1; s2) | P_b >= 600 (s0; s1; s2) |
|---|---|---|
| `l_slow` | [-0.001, +0.000]; [-0.000, +0.000]; [-0.000, +0.001] | [-0.001, +0.002]; [-0.000, +0.003]; [-0.001, +0.002] |
| `l_tail` | [-0.000, +0.000]; [-0.000, +0.000]; [-0.000, +0.000] | [-0.002, -0.000]; [-0.001, +0.000]; [-0.001, -0.000] |
| `l_comb80` | [-0.002, +0.000]; [-0.001, +0.000]; [-0.002, +0.000] | [-0.003, +0.001]; [-0.001, +0.004]; [-0.004, +0.001] |
| `l_comb82` | [-0.002, +0.000]; [-0.001, +0.000]; [-0.002, +0.000] | [-0.003, +0.001]; [-0.001, +0.004]; [-0.003, +0.001] |
| `l_prompt` | [-0.017, -0.010]; [-0.015, -0.008]; [-0.018, -0.010] | [-0.014, -0.002]; [-0.012, +0.003]; [-0.016, -0.004] |

### A.4 T5: change in gap (nu minus nonu_dark)

| Variant | CL (s0; s1; s2) | C + XF (s0; s1; s2) |
|---|---|---|
| `t_scale80` | [-0.048, -0.046]; [-0.047, -0.046]; [-0.047, -0.046] | [-0.054, -0.053]; [-0.055, -0.053]; [-0.054, -0.053] |
| `t_scale82` | [-0.043, -0.042]; [-0.043, -0.042]; [-0.043, -0.042] | [-0.049, -0.048]; [-0.049, -0.048]; [-0.049, -0.047] |
| `t_slow` | [-0.024, -0.023]; [-0.024, -0.023]; [-0.024, -0.023] | [-0.033, -0.032]; [-0.034, -0.032]; [-0.033, -0.032] |
| `t_tail` | [-0.028, -0.025]; [-0.028, -0.025]; [-0.028, -0.025] | [-0.031, -0.027]; [-0.031, -0.027]; [-0.030, -0.027] |
| `t_comb80` | [-0.086, -0.082]; [-0.085, -0.082]; [-0.085, -0.081] | [-0.104, -0.099]; [-0.105, -0.100]; [-0.104, -0.099] |
| `t_comb82` | [-0.082, -0.079]; [-0.082, -0.078]; [-0.082, -0.078] | [-0.100, -0.095]; [-0.101, -0.096]; [-0.099, -0.095] |

The nominal row is exactly 0 by construction.

### A.5 T5: reference MSE(`nonu_dark`) per seed

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| CL | 0.2146 [0.2051, 0.2247] | 0.2160 [0.2065, 0.2259] | 0.2169 [0.2076, 0.2275] |
| C + XF | 0.2588 [0.2471, 0.2711] | 0.2595 [0.2482, 0.2719] | 0.2596 [0.2480, 0.2713] |

n = 948 in every seed and model.
