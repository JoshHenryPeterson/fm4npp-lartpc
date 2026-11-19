# A1 check, round 2: 3D width factors, per-plane breakdown, data drift velocity

Status: plan (2026-11-10). Inference only, no retraining.

## 1. Background: round 1

Round 1 was run by yuhw on 2026-11-09 (`paper-campaign.md` lines 34, 59, 129; jobs 7831006 smoke, 7831163 full; 1.8 nh).

- Code: `qlfm/eval_perturb.py`. It is identical in yuhw's and jpeterso's copies.
  - Perturbations act only on simulated-neutrino hits (`hit_nu_edep > --sim-edep`).
  - `w_meas` multiplies `hit_rms` by a per-plane factor `F_MEAS[plane]`.
    - The factor is interpolated linearly over the drift-bin centres `D_C = [388, 1165, 1941, 2718, 3494, 4271]` ticks, measured from `ANODE_TICK = 830`.
    - End values are held, so the factor is 1 for d < 388.
  - `w_diff` is the diffusion-only bracket. It uses one drift velocity, `V_DRIFT_CM_US = 0.1098`, for both the data and the sim term.
- Outputs: `yuhw/fm4npp-lartpc-work/a1/full/s3_mt_<model>_finetune_f0.01_s{0,1,2}/metrics.json`.
- Headline result: T1 here is the charge-weighted global F1 from `eval_perturb`, not the paper's per-event median.

  | Model | Nominal T1 | w_meas | w_diff | w_c95 | w_inv |
  |---|---|---|---|---|---|
  | qlfm-M CL | 0.918–0.919 | −0.20 to −0.22 | −0.41 to −0.45 | −0.16 to −0.17 | −0.06 to −0.09 |
  | qlfm-M C + XF | 0.858–0.861 | −0.027 to −0.031 | −0.054 to −0.061 | −0.015 to −0.017 | +0.008 to +0.009 |

  - Bootstrap 95% CIs are about ±0.006 (T1) and ±0.004 (T2).
  - The seed-to-seed spread (about 0.02) is larger than the CIs.
  - CL is strongly width-sensitive; C + XF is only mildly sensitive.
- Gaps that round 2 has to close:
  - Every perturbation is applied to all three planes at once.
  - The hit predictions are not saved, so T1 cannot be broken down by plane.
  - The factors come from the 2D measurement.

## 2. New input (Sunny, p.13)

The 3D factors are smaller (more width reduction) at every drift distance.

| Plane (index) | 2D cathode factor (round 1) | 3D cathode factor (new) |
|---|---|---|
| Y (2) | 0.919 | 0.903 |
| V (1) | 0.955 | 0.941 |
| U (0) | 0.895 | 0.880 |

```python
F_MEAS_3D = {2: [1.000, 0.967, 0.941, 0.926, 0.915, 0.903],   # Y
             1: [1.000, 0.980, 0.954, 0.938, 0.942, 0.941],   # V
             0: [1.000, 0.960, 0.922, 0.907, 0.892, 0.880]}   # U
```

These use the same six drift bins (`D_C`) as round 1. Confirm this with Sunny; see Section 8.

## 3. Requests

1. Apply `w_meas` with the 3D factors on all planes.
2. Apply the same factors on Y only, and on U+V only, with f = 1 on the other planes.
3. Report T1 F1 by plane for the nominal and 3D runs. This needs the hit plane saved with the hit predictions.
4. Give the data term of the `w_diff` bracket its own drift velocity, 0.1076 cm/µs (measured in data, MicroBooNE D_L paper). Keep 0.1098 for the sim term. The full-drift factor then becomes 0.882 instead of 0.875.

Scope: qlfm-M CL and qlfm-M C + XF, multi-task, 1% fraction, seeds 0, 1 and 2. That is 6 runs.

## 4. Code changes (`qlfm/eval_perturb.py`)

### 4.1 Factor tables and plane masks (requests 1, 2)

- Rename `F_MEAS` to `F_MEAS_2D` and keep `F_MEAS = F_MEAS_2D` as an alias, so the round 1 variants stay reproducible. Add `F_MEAS_3D` from Section 2.
- Extend `width_factor(plane, tick, kind)` so it takes a table key and an optional plane subset, for example `kind = "meas3d"` or `"meas3d@2"` or `"meas3d@0,1"`. Hits on planes outside the subset get f = 1.
- New `VARIANTS` entries:

| Variant | Width kind | Meaning |
|---|---|---|
| `w_meas3d` | `meas3d` | 3D factors on U, V and Y (request 1) |
| `w_meas3d_Y` | `meas3d@2` | 3D factors on Y only (request 2) |
| `w_meas3d_UV` | `meas3d@0,1` | 3D factors on U and V only (request 2) |
| `w_diff_vd` | `diff_vd` | Diffusion bracket with the data drift velocity (request 4) |

- Keep `nominal`, `w_meas` (2D) and `w_diff` (round 1 form) in the round 2 runs as continuity references.
- Optional, cheap: add `w_inv3d` (`1/meas3d`) as the matching control, if the 3D result needs the "is it any width change?" check redone.

### 4.2 Drift velocity in the data term of `w_diff` (request 4)

- Add `V_DRIFT_DATA_CM_US = 0.1076`.
- New `diff_vd` kind:
  ```python
  t = (d * US_PER_TICK).clamp(0, MAX_DRIFT_CM / V_DRIFT_CM_US)      # sim drift time, as in round 1
  f = sqrt((S0 + 2*D_DATA*t / V_DATA**2) / (S0 + 2*D_SIM*t / V_SIM**2))
  ```
  Only the velocity that converts the data term from cm² to µs² changes. The drift time `t` and its clamp stay on the sim timescale.
- Checked numerically:
  - Round 1 form at full drift: 0.8745 (the 0.875 quoted).
  - This form at full drift: 0.8823, which matches the 0.882 quoted.
  - If the data drift time were also recomputed as L/v_data, the result would be 0.886 instead. That form is not what was asked for. Note this in the write-up so the bracket definition is unambiguous.

### 4.3 Save the hit predictions with the plane (request 3)

- New flag `--save-preds VARIANT[,VARIANT]`, default off. For each listed variant, write `OUT/preds_<variant>.npz` with these columns:
  - `pred` (float16)
  - `hit_tgt` (uint8)
  - `hit_q` (float32)
  - `hit_evt` (int32)
  - `gidx`
  - `hit_plane` (uint8)
  - `hit_sim` (bool: `hit_nu_edep > sim_edep`)
- This is the column layout `paper_metrics.t1` already reads, plus `hit_plane` and `hit_sim`. Write one column at a time; `paper_metrics.t1`'s docstring puts 160M hits at about 3 GB.
- Add `paper_metrics.t1_by_plane(z, thr=0.5)`. It applies `t1()` to each plane mask and returns the per-event median F1 and the charge-weighted F1 for each plane.
- Also put the in-run per-plane charge-weighted F1, with event-bootstrap CIs and deltas against nominal, in `metrics.json` under `results.hit.<variant>.by_plane`. The headline numbers then do not depend on the npz files.
- Report the paper's T1 (per-event median F1, from the npz files) alongside the `eval_perturb` charge-weighted F1. Round 1 reported only the latter, and the two are not comparable.

### 4.4 Tests to pass before submitting

- At the `D_C` bin centres, `width_factor` returns the table values exactly for each plane. With `@` masks, it returns exactly 1 on the excluded planes.
- `diff_vd` at full drift returns 0.8823. `diff` still returns 0.8745.
- `w_meas3d_Y` and `w_meas3d_UV` change disjoint sets of hits, and together they cover exactly the hits `w_meas3d` changes.
- `nominal` reproduces the score stored in round 1 for the same run, and `w_meas` (2D) reproduces the round 1 delta, both within bootstrap noise.

## 5. Runs

Read-only inputs, from yuhw's work dir (readable by group `neutrinoGPU`):

```
/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/s3/mt/qlfm_M_CL/finetune/f0.01_s{0,1,2}
/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/s3/mt/qlfm_M_C_xf/finetune/f0.01_s{0,1,2}
```

Outputs go to `/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1r2/{smoke,full}/<run_tag>/`. Use yuhw's work dir instead if yuhw runs this round.

Command for each run:

```
python -m qlfm.eval_perturb --run RUN_DIR --tasks hit,evt_nu \
  --variants w_meas,w_meas3d,w_meas3d_Y,w_meas3d_UV,w_diff,w_diff_vd \
  --save-preds nominal,w_meas3d --n-boot 1000 --out OUT_DIR
```

Steps:

1. **Smoke run.** Run qlfm_M_CL seed 0 with `--n-boot 200` and every new variant. Check the Section 4.4 tests on real data and the size of the npz files.
2. **Full run.** Six tasks, one per run directory, through `run_queue.sh` + `pbs/queue.pbs` on 1 node (4 GPUs). No feeder is needed, since this is a single `qsub`.
3. **Per-plane T1.** Run `paper_metrics.t1_by_plane` on the npz files: 6 runs × 2 variants, CPU only.

Cost estimate: round 1 took about 16 min per run for 14 variants. Round 2 has 7 variants plus writing the predictions, so about 10–12 min per run. That gives about 0.3 nh for the full run and about 0.1 nh for the smoke run, roughly 0.5 nh with margin. Add a row to the node-hour ledger in `paper-campaign.md`.

## 6. Deliverables

1. **Main table.** Rows: qlfm-M CL and qlfm-M C + XF. Columns: nominal, `w_meas` (2D), `w_meas3d`, `w_meas3d_Y`, `w_meas3d_UV`, `w_diff`, `w_diff_vd`.
   - Show T1 and T2 deltas as the mean over 3 seeds ± the seed standard deviation, plus each seed's bootstrap 95% CI.
   - Report both T1 definitions.
2. **Per-plane T1.** Rows: plane U, V, Y. Columns: nominal and `w_meas3d`, with the delta, for each model, seed-averaged.
   - Also show `w_meas3d_Y` and `w_meas3d_UV` by plane. This shows whether perturbing one plane affects the F1 on the others through the model's cross-plane context.
3. **Short write-up.** Add it to `paper-campaign.md` under A1 and update the caveat at line 59. Answer these questions:
   - Does moving from 2D to 3D factors make the `w_meas` drop larger, and by how much compared with the seed spread?
   - Is the drop driven by Y (collection) or by U+V (induction)? Is it additive, i.e. is ΔY + ΔUV ≈ Δall?
   - Is the T1 loss concentrated on one plane?
   - Where does `w_meas3d` fall relative to the diffusion bracket now that the bracket uses `w_diff_vd` (0.882 at the cathode)?

## 7. Out of scope

- No retraining. The `train_multitask --width meas|const` path is the separate fix (b) for #9.
- No PTv3, PolarMAE or FM4NPP baselines. They do not read hit width, and round 1 already showed they are unchanged under the width-only variants.
- No charge (`q±`) variants. Round 1 showed them negligible (≤ 0.001).
- No f0.001 or frozen runs.

## 8. Open points to confirm before the full run

- Are the 3D U and V factors still provisional, as the 2D ones were?
- Are the 3D factors on the same drift bins (`D_C`) and anode reference (`ANODE_TICK = 830`) as round 1?
- Who runs it: yuhw in the campaign repo, or jpeterso here with a PR to yuhw's repo? The code is currently identical in the two copies.
- Are the npz files kept after the per-plane analysis, or deleted? They take several GB per run per variant.
