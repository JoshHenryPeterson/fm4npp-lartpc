# A1 light check, round 1: data/sim light-profile sensitivity

Status: plan (2026-11-11; T5 target test added 2026-11-12). Inference only, no retraining. This is the light counterpart of the A1 width test (`a1_check_round_2_plan.md`).

## 1. Request (summary)

Findings from the open files (00–04): real off-beam flashes were compared with simulated beam flashes, matched in prompt PE, using ophits rather than flash sums.

- Prompt light agrees within 1–3%.
- Per unit of prompt light, simulation has 1.2–2.2× more slow light at 0.5–2.5 µs after the flash.
- Bright simulated flashes (prompt ≥ 600 PE) have 2–3× more late light at 8–18 µs.
- MicroBooNE reports a PE/cm simulation/data ratio of about 1.25 (and 1.22) (arXiv:2603.23691, Sec. 4.2 and 5.2).
- The comparison is not like-for-like: real flashes are cosmic muons, simulated flashes are neutrino events. Physics (protons in ν events give more prompt light) would push the other way.

The open files do not mark which ophits are simulated. The bright light around the beam flash is almost all simulated, since real in-time cosmics occur in about 1–2% of events, so the variants act on the beam-flash light of every event.

**Per-event definitions:**
- **Beam-flash time t_b:** the 0.1 µs bin with the most ophit PE (summed over PMTs) in 3.4–5.6 µs.
- **Prompt PE P_b:** the summed ophit PE over all PMTs in [t_b − 0.25, t_b + 0.5) µs.
- **dt** = t − t_b.

**Variants**, each run separately against the native result:

| # | Name | Ophits affected | Factor |
|---|---|---|---|
| 1a | `l_scale80` | dt ∈ [−0.25, 3) µs | × 0.80 (from sim/data = 1.25) |
| 1b | `l_scale82` | dt ∈ [−0.25, 3) µs | × 0.82 (from 1.22) |
| 2 | `l_slow` | dt ∈ [0.5,1.0) / [1.0,1.5) / [1.5,2.0) / [2.0,3.0) µs | P_b < 600: × 0.83 / 0.65 / 0.61 / 0.64<br>P_b ≥ 600: × 0.82 / 0.63 / 0.48 / 0.45 |
| 3 | `l_tail` | dt ≥ 8 µs, only when P_b ≥ 600 | × 0.37 |
| 1a+2+3 | `l_comb80` | variants 1a, 2 and 3 multiplied together | |
| 1b+2+3 | `l_comb82` | variants 1b, 2 and 3 multiplied together | |
| 4 | `l_prompt` | dt ≥ 0.5 µs | removed (PE set to 0) |

That is seven variants plus `nominal`.

**Scope:**
- Models: the same native-width models as the width test, qlfm-M CL and qlfm-M C + XF, multi-task, 1%, seeds 0–2. That is 6 runs.
- Tasks: T1 and T2 under the 7 input-light variants (Sections 3.1–3.2).
- **T5 target test** (added after review, Section 3.4): the same light corrections are applied to the T5 *target* of events with simulated beam-flash light. The trained light head is unchanged and its predictions are not recomputed. The test asks whether the T5 error gap between neutrino events and no-neutrino events shrinks.
- No width or charge perturbation. Hit width stays native, so the light effect is measured on its own.

## 2. What the code does today (facts the plan relies on)

- **Ophits in the batch:** `oph_ch`, `oph_t`, `oph_pe` and `oph_evt` (`qlfm/data/dataset.py:104,111-113`).
  - They are raw per-event slices of the converted `oph_*.npy` files.
  - `t` is the ophit `peaktime` in µs on the beam-window clock (about 0–23.4 µs).
  - `pe` is stored as float16 (`qlfm/data/convert.py:94-97,169-180`).
  - There is no cached light histogram, so a perturbation of `oph_pe` reaches every consumer.
  - Beam flashes sit at about 3.6–5.2 µs (`docs/round2-log.md:83`), so the 3.4–5.6 µs t_b window is consistent with that.
- **Light tokenization:** `tokenize_light` (`qlfm/tokenize.py:113-127`) histograms ophit PE per PMT into 16 time bins with edges 0, 1, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10, 12, 15, 19, 24 µs.
  - Ophits with t < 0 or t ≥ 24 µs are dropped, and PE is clamped at 0.
  - The features are `log1p` of each bin's PE plus `log1p` of the total, so 17 features × 32 PMTs.
  - The perturbation scales each ophit before binning, so the dt windows do not need to line up with the bins.
  - Setting PE to 0 is equivalent to removing the ophit, so variant 4 zeroes `oph_pe` and all tensors stay the same length.
- **Which models read light:** both do, so no variant is a no-op.
  - **qlfm-M CL:** `late_light = 0`. It is pretrained with light (`pre/m_cl_f0.90`, `light=1`), so light is an early-fusion modality in the encoder.
  - **qlfm-M C + XF:** charge-only pretraining (`pre/m_c_f0.90`, `light=0`), with `late_light = 2` (`LateLight` + `XFusion` cross-attention in the multi-task head).
- **Tasks:**
  - **T1** = `hit`.
    - `eval_perturb` reports the charge-weighted global F1.
    - The paper's T1 is the per-event unweighted median F1 (`paper_metrics.t1`).
  - **T2** = `evt_nu`.
    - `eval_perturb` reports the all-event AUC.
    - The paper's T2 is the AUC on the bright-flash subset, defined by a PE cut in 3–6 µs: 200 PE, chosen on val (`paper-campaign.md:46`).
  - **T5** = `light`: a charge-to-light regression head (`LightHead`), 32 PMTs × 17 features, test subset of 10 000 events drawn from all events.
    - Its target is `light_target(b)` (`qlfm/train_downstream.py:126-129`): the `tokenize_light` features of the event's own ophits, i.e. log1p PE per time bin plus log1p total.
    - Its metric is the MSE over all features (`qlfm/evaluate.py:82-93`).
    - Its input is charge only. CL gets there through a separate light-off pass, because its encoder reads light (`train_multitask.py:105,143`). C + XF has no light in the backbone.
    - So the input-light variants cannot change T5, but the target can be corrected without new heads.
    - Stored test MSE for seed 0: CL 0.5282, C + XF 0.5829 (`metrics.json`, 10 000 events).
  - **Event classes** (`qlfm/data/labels.py`): every event is an overlay of a simulated ν on real off-beam cosmic data, so every event has a simulated ν somewhere. The `evt_nu` classes are:
    - 1: a visible in-AV ν;
    - 2: a ν outside the AV that still leaves charge;
    - 0: ν charge below the visibility threshold ("no-neutrino").
    - Class 0 is not guaranteed to have only real light. A ν interacting in the cryostat outside the TPC can still make simulated light (`plan-initial-round.md:302`).
- **Where perturbations go:** `perturb()` in `eval_perturb.py` returns a shallow copy of the batch with tensors replaced, and `Perturbed.forward` applies it. Replacing `oph_pe` there is safe: prediction targets and labels are read from the original batch.
  - `perturb()` currently returns early when there is no width or charge change, so the light branch must come before that return.

## 3. Code changes

### 3.1 `qlfm/eval_perturb.py`: light perturbations

1. **Constants:**
   ```python
   TB_WIN = (3.4, 5.6); TB_BIN = 0.1                 # us
   PB_WIN = (-0.25, 0.5); PB_BRIGHT = 600.0          # dt window (us), PE
   SCALE_WIN = (-0.25, 3.0)
   SLOW_EDGES = [0.5, 1.0, 1.5, 2.0, 3.0]
   SLOW_DIM = [0.83, 0.65, 0.61, 0.64]; SLOW_BRIGHT = [0.82, 0.63, 0.48, 0.45]
   TAIL_DT, TAIL_F = 8.0, 0.37
   PROMPT_DT = 0.5
   ```
2. **`beam_flash(b)`:** returns per-event `t_b`, `P_b` and `has_flash`.
   - Take the ophits with t in [3.4, 5.6), compute the bin index `floor((t − 3.4)/0.1)` (22 bins), and `scatter_add` the PE into an `[n_events, 22]` histogram.
   - The argmax bin gives t_b, defined as the bin's lower edge (Section 7).
   - P_b is the PE summed in dt ∈ [−0.25, 0.5), via `scatter_add` over `oph_evt`.
   - Events with no PE in 3.4–5.6 µs get `has_flash = False` and are left unperturbed. The fraction of such events goes into `metrics.json`.
   - Both quantities are always computed from the **native** light. In the combined variants, the 600 PE split is not affected by the 0.80/0.82 scale (Section 7).
3. **`light_factor(b, kinds)`:** returns a per-ophit multiplicative factor, which is the product of the requested components.
   - `scale80` / `scale82`: × 0.80 or 0.82 for dt ∈ [−0.25, 3).
   - `slow`: the per-window factors, choosing the dim or bright set by P_b.
   - `tail`: × 0.37 for dt ≥ 8 when P_b ≥ 600.
   - `prompt`: × 0 for dt ≥ 0.5.
4. **`VARIANTS`:** extend each entry from `(width, eps)` to `(width, eps, light)`, where `light` is a tuple of components, e.g. `l_comb80 = (None, 0.0, ("scale80", "slow", "tail"))`. All existing width and charge entries get `light = ()` and behave exactly as before.
5. **`perturb()`:** if `light` is non-empty, replace `b["oph_pe"]` with `oph_pe * light_factor(...)`. This happens before the early return, on all ophits, since sim ophits cannot be identified (Section 1).
6. **Light diagnostics in `metrics.json`:** once per run, record:
   - the fraction of events with a flash;
   - quantiles of t_b and P_b;
   - the fraction of events with P_b ≥ 600.

   For each variant, also record the summed PE before and after the perturbation in four dt windows: [−0.25, 0.5), [0.5, 3), [3, 8) and ≥ 8 µs. These are a direct check that each variant does what it should (Section 4.2).

### 3.2 Metrics

- **T1:** report the charge-weighted F1, as now, and the paper's per-event median F1.
  - To get the median without writing hit-prediction files, accumulate per-event *unweighted* TP, true and selected counts in-run (the same bincount as `paper_metrics.t1`).
  - Bootstrap the median over events with the existing resampling indices.
  - No `--save-preds` is needed for this test.
- **T2:** report the all-event AUC, as now, and the **bright-subset AUC**.
  - Define the subset from **native** light (3–6 µs PE ≥ 200, as in `paper_metrics`), so the event set is the same for every variant.
  - Also report T2 split into P_b < 600 and P_b ≥ 600, because the slow and tail corrections differ between those classes.

### 3.3 T5 target test (sim/data light in the T5 target)

**Question.** T5 errors differ between neutrino events (simulated light) and no-neutrino events (real cosmic light). If the same light corrections are applied to the T5 target of the simulated-light events, does that gap shrink? If it does, part of the T5 difference is light mismodelling. The trained head is not changed.

**Method.**
1. **Predictions:** run the light head once per run, on unperturbed input, on the T5 test subset (10 000 events). This must use the model's normal T5 path, including CL's light-off pass.
   - `eval_perturb.py:296` currently sets `model.light_off_pass = False`. For CL that would feed light into the T5 prediction. When `light` is in `--tasks`, keep the trained setting instead.
   - Check: nominal MSE must reproduce the stored test MSE (Section 4.2).
2. **Targets:** in the same batch loop, build the corrected target for each target variant as `light_target(b')`. Here `b'` is the batch with `oph_pe` scaled by the same `light_factor` as in Section 3.1.
   - The correction is applied only to **simulated-light events**: events with a beam flash (`has_flash`, from native light). All other events keep the original target.
   - The beam-flash light is almost all simulated, and real in-time cosmics occur in about 1–2% of events. So this matches the request's "same light corrections applied to the simulated-light events" without needing ophit-level sim/real labels.
3. **Target variants:** `t_scale80`, `t_scale82`, `t_slow`, `t_tail`, `t_comb80` and `t_comb82`. These are the same components as `l_*`. `l_prompt` has no target counterpart, because removing late light is a model probe, not a correction.
4. **Per-event outputs**, stored in `metrics.json` summaries, with an optional small npz for the 10 000 events:
   - the MSE over the 32 × 17 features;
   - the MSE on the per-PMT total feature;
   - the signed bias of the per-PMT total, mean (pred − target);
   - for each target variant and nominal: `evt_nu` class, `has_flash`, P_b.
   - Predictions are stored once. Each target variant adds only a 10 000 × 32 × 17 target array per run, which can be kept in memory.
5. **Groups and gap:**

   | Group | Definition | Light |
   |---|---|---|
   | `nu` | `evt_nu` class 1 (visible in-AV ν) | simulated beam flash + real cosmics |
   | `out_av` | class 2 | mostly simulated beam flash |
   | `nonu_flash` | class 0 with `has_flash` | some simulated ν light outside the TPC |
   | `nonu_dark` | class 0 without a flash, or P_b below a small threshold (Section 7) | real cosmic light only (reference) |

   - Gap = MSE(`nu`) − MSE(`nonu_dark`), reported for nominal and each target variant.
   - The same is reported for `out_av` and `nonu_flash`.
   - Use a paired event bootstrap (the same resampled events for nominal and the variant). This gives CIs on each group's MSE, on the gap, and on the **change in the gap** from nominal.
   - `nonu_dark` targets are unchanged by construction, so any change in the gap comes from the simulated-light groups.
   - **MSE(`nonu_dark`) is reported explicitly**, not only through the gap, for every run:
     - its value with bootstrap 95% CI;
     - its event count;
     - its per-PMT-total MSE and signed total bias.
     - Because it is the same under every target variant, it is reported once per run, and the summary checks that it is identical across variants.
     - The fraction of the nominal gap closed by each variant, (gap_nominal − gap_variant) / gap_nominal, is reported next to it.
6. **Interpretation:**
   - The gap can shrink only if the head's predictions for ν events are closer to data-like light than to the simulated light it was partly trained on. The signed total bias shows which way it goes:
     - a negative bias on `nu` at nominal (head predicts less light than the sim target) that moves toward 0 after correction supports light mismodelling;
     - a bias that becomes positive means the correction overshoots for this head.
   - A gap that shrinks only under `t_scale*` points at the overall light yield. One that shrinks under `t_slow`/`t_tail` points at the time profile.
   - Part of the gap is expected regardless of mismodelling. ν events are brighter and their light is concentrated in fewer bins, so the result is a lower bound on the light-mismodelling share, not a decomposition.

**Cost:** one extra light-head pass over 10 000 events per run, plus a few `tokenize_light` calls per batch. That is about a minute per run, inside the same job.

### 3.4 Tests (`tests/test_a1_light.py`, CPU, run on the login node)

The synthetic events are built with known ophits.

- **t_b:** it is the argmax bin, and it ignores bright light outside 3.4–5.6 µs.
- **P_b:** it sums exactly the PE in [t_b − 0.25, t_b + 0.5).
- **Each component's factors:**
  - every factor is correct at window edges and inside windows;
  - the P_b < 600 / ≥ 600 switch works for `slow` and `tail`;
  - the combined variants equal the product of their parts.
- **Edge cases:**
  - events without a flash are untouched;
  - `l_prompt` leaves PE > 0 only where dt < 0.5.
- **No side effects:**
  - light variants leave `hit_*` tensors bitwise unchanged;
  - width and charge variants leave `oph_*` unchanged;
  - `tests/test_a1r2.py` still passes, so the round 2 width variants are unchanged.
- **Tokenization:** after `tokenize_light`, a zero-PE ophit gives the same tokens as a removed ophit.
- **T5 target test:**
  - a corrected target differs from the original only for `has_flash` events;
  - with the factor set to 1, every target is bitwise unchanged;
  - group assignment follows the `evt_nu` class and `has_flash`;
  - the paired-bootstrap gap change is 0 when the target variant is the identity.

## 4. Runs

### 4.1 Task list

There are 6 tasks, one per run directory:

```
/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/s3/mt/qlfm_M_CL/finetune/f0.01_s{0,1,2}
/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/s3/mt/qlfm_M_C_xf/finetune/f0.01_s{0,1,2}
```

```
python -m qlfm.eval_perturb --run RUN_DIR --tasks hit,evt_nu,light \
  --variants l_scale80,l_scale82,l_slow,l_tail,l_comb80,l_comb82,l_prompt \
  --t5-targets t_scale80,t_scale82,t_slow,t_tail,t_comb80,t_comb82 \
  --n-boot 1000 --out $QLFM_WORK/a1l1/full/<run_tag>
```

### 4.2 Execution (unattended)

This follows the round 2 width setup (`scripts/pbs/a1r2_all.pbs`, `scripts/a1r2_submit.sh`, `scripts/a1r2_common.sh`). Copy and adapt them as `a1l1_*`.

- **One PBS job, so no dependency chain.** The debug queue allows only one queued job per user.
  - Stage 1 is the smoke run: qlfm_M_CL s0, `--n-boot 200`, all 7 variants. It is followed by the unit tests and `scripts/a1l1_check_smoke.py`. If anything fails, the job exits before stage 2.
  - Stage 2 runs the 6 full tasks on 2 nodes × 4 GPUs.
  - Stage 3 runs `scripts/a1l1_summarize.py`, which writes `a1l1/full/summary.md`.
- A `STATUS` file at `$QLFM_WORK/a1l1/STATUS` is updated at each stage. The job is resumable: `smoke/PASSED` skips the smoke stage, and finished tasks are skipped.
- **Queueing:** the debug queue takes one queued job per user. The width job (7837342) has finished, so the debug queue is free. If another debug job is pending, use `preemptable` (up to 20 queued per user, 1–10 nodes) with `-r y`, so PBS requeues the job after preemption and finished tasks are skipped.
- **Smoke acceptance checks** (any failure exits nonzero):
  - `nominal` T1 and T2 match the stored round 1 values (and the round 2 width nominal, if done) within 1e-3.
  - T5:
    - Nominal T5 MSE reproduces the stored test MSE (CL s0 0.5282) within 1e-3. This confirms the light-off pass and the test subset.
    - The input-light variants leave the T5 predictions unchanged.
    - The `nonu_dark` target is identical across target variants.
    - Each group has at least 100 events.
  - Flash finding:
    - At least 95% of events have a flash in 3.4–5.6 µs.
    - The t_b distribution peaks in 3.6–5.2 µs.
    - The P_b ≥ 600 fraction is reported. It must be > 0, or `l_tail` is empty.
  - The PE diagnostics match the definitions within 1e-3 relative (float16 storage):
    - `l_scale80` changes only [−0.25, 3) PE, by exactly 0.80.
    - `l_prompt` gives zero PE at dt ≥ 0.5.
    - `l_tail` changes nothing in events with P_b < 600.
- **Cost:** round 2 with the width variants took about 10–15 min per run. Here there are 8 variants including nominal, plus the T5 target test (about a minute), so expect about 15 min per run. The 6 runs on 8 GPUs fit in one wave, so stage 2 takes about 15–20 min, plus about 15 min for the smoke stage. Request 1 h on 2 nodes; expect about 1 node-hour, 2 at most. Add a row to the node-hour ledger in `paper-campaign.md`.

## 5. Deliverables (`a1l1/full/summary.md`, then into `paper-campaign.md` under A1)

1. **Main table.** Rows: models. Columns: nominal and the 7 variants.
   - Show deltas against nominal as the mean over 3 seeds ± the seed standard deviation, plus each seed's bootstrap 95% CI.
   - Metrics:
     - T1, both definitions;
     - T2, all-event and bright-subset.
2. **T5 target table.** Rows: models. Columns: nominal and the 6 target variants. Entries:
   - MSE per group (`nu`, `out_av`, `nonu_flash`, `nonu_dark`) with event counts. MSE(`nonu_dark`) gets its own column, with per-seed values and CIs and the mean ± seed sd;
   - the gap `nu` − `nonu_dark`, its change from nominal and the fraction of the gap closed, with paired-bootstrap 95% CIs, mean ± seed sd;
   - the signed per-PMT total bias for `nu`.
3. **T2 by P_b class**, under 600 and at least 600, for nominal, `l_slow`, `l_tail`, `l_comb80`, `l_comb82` and `l_prompt`.
4. **Light diagnostics:** the flash-finding fraction, t_b and P_b distributions, and the PE change per dt window for each variant.
5. **Short write-up** answering:
   - **Total scale:** does the 0.80/0.82 scale move T1/T2 beyond the seed spread, and does the gap between 0.80 and 0.82 matter?
   - **Slow light vs tail:** which of `l_slow` and `l_tail` matters, and is the combined delta about the sum of its parts?
   - **Time profile:** under `l_prompt`, if T1/T2 barely move, the model does not use the time profile beyond the prompt light. That would make the slow-light and tail mismatches irrelevant to these results.
   - **CL vs C + XF:** does early fusion (CL) depend on light differently from late fusion (C + XF)?
   - **T5 target:** does correcting the simulated-light targets shrink the ν / no-ν T5 gap, which correction does it, and does the total bias move toward 0? How much of the T5 difference is light mismodelling (as a lower bound)?

## 6. Out of scope

- No retraining with corrected light.
- No width or charge changes; those are covered by the width test.
- No PTv3, PolarMAE or FM4NPP baselines, which do not read light.
- No f0.001 or frozen runs.
- No attempt to identify which ophits are simulated.
- No T5 under the *input*-light variants: the head predicts light from charge only, so they cannot change it. T5 is covered by the target test (Section 3.3) instead.
- No retraining of the light head on corrected targets.

## 7. Open points (defaults used unless told otherwise)

1. **t_b bin reference.** Default: t_b is the lower edge of the max-PE 0.1 µs bin. The bin centre would shift every window by 0.05 µs.
2. **P_b in the combined variants.** Default: the 600 PE split uses the native (simulated) P_b, before the 0.80/0.82 scale, because the measured ratios were binned in simulated prompt PE. Applying the split after scaling would move events just above 600 PE into the dim class.
3. **Late tail at dt ≥ 8 µs:** this reaches t ≈ 12–24 µs and also scales any unrelated cosmic flashes in bright events, as the request specifies for all ophits. An optional variant `l_tail_flash` could restrict this to ophits on the PMTs lit by the beam flash. Run it only if requested.
4. **Light before the scale window:** `l_prompt` keeps light before the flash (dt < −0.25 µs) and removes only dt ≥ 0.5 µs, as written.
5. **Paper T2 bright cut** (3–6 µs PE ≥ 200) is evaluated on native light, so the subset is fixed across variants. Variant 1 lowers the PE in that window, so a cut on perturbed light would change the subset.
6. **`nonu_dark` reference.** Default: class 0 without a flash in 3.4–5.6 µs. If almost every class 0 event has some PE there (cosmic light), define it instead as P_b below a small cut chosen from the smoke-run P_b distribution, e.g. 20 PE. The reviewer described no-neutrino events as real cosmic light, but class 0 can contain simulated ν light from the cryostat, so it is split by flash.
7. **Corrected events.** Default: the target correction applies to every `has_flash` event regardless of class, because the beam-flash light is what is simulated. Correcting only class 1 is a one-line change.
