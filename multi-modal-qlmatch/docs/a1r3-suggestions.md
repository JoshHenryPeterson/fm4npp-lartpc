# Suggestions for Joshua's A1 round 3 (data/sim checks), with our latest results

From Haiwang (prepared with the Polaris Claude session), 2026-11-19.

This reviews `a1_check_round_3_plan.md` (on Eagle, `LArTPC_FM/jpeterso/fm4npp-lartpc/multi-modal-qlmatch/docs/`). It also lists results from our domain-shift study (#13) that bear on it.

- **Live status of all tests:** `docs/experiments-live.pdf`.
- **Map of every planned test:** `docs/experiment-map-2026-11.md`.

## 1. Latest results relevant to round 3

Notation:
- **S** = the published overlay (your "sim").
- **D** = simulated ν hit widths × Sunny's 3D factors (your "real" without the light correction).
- qlfm-M CL, multi-task, 1% labels, 3 seeds.
- T1 = charge-weighted F1.
- "Real beam-on" = share of Wire-Cell's ν charge that the model also tags, on the round-1 beam-on sample (#10).

| pretrain | labels | fine-tune mode | T1 in S | **T1 in D** | **real beam-on** |
|---|---|---|---|---|---|
| S | S | full | 0.919 | 0.631 ± 0.013 | 0.20 |
| S | S | first 6 of 12 blocks frozen | 0.914 | 0.586 ± 0.012 | 0.16 |
| S | S | head-first (LP-FT, 2,000 steps) | 0.919 | 0.514 ± 0.009 | 0.10 |
| S | S | frozen | 0.895 | 0.485 ± 0.002 | 0.09 |
| S | D | full ("sim→real") | – | **0.911 ± 0.001** | 0.34 |

On the overlay, all of these find 0.87–0.90 of Wire-Cell's ν charge.

1. **Your round-2 number reproduces.** Our full fine-tune drops by −0.288 in D, the same as your `w_meas3d` value.
2. **From an S backbone, every form of freezing transfers worse to D than full fine-tuning.** A per-layer probe of the S backbone (`qlfm/domain_probe.py`) explains why:
   - the width difference grows from 5% at the input to 25–33% of the feature norm by blocks 4–6;
   - a linear probe tells S from D with AUC 0.95 there;
   - so the frozen features carry the width cue.
3. **Real beam-on ranks the fine-tune modes exactly as the D test does.** D is a good proxy for *ranking* methods.
4. **D captures only part of the real shift.** The labels-in-D model recovers to 0.911 in D but reaches only 0.34 on real beam-on. That is no better than the 2D factors gave (0.35–0.39). Models without width reach about 0.6.
5. **Still running:** qlfm-M CL pretrained in D, plus 15 fine-tunes from it (frozen, 6 blocks frozen, head-first, full; and labels in D). These are your "real→sim" and "real→real" arms for CL, width only. Expected within about a day.

## 2. Suggestions for the plan

1. **Avoid duplicate CL arms.**
   - Our runs cover CL, width only, with the standard pretraining set: pretrain in D → labels in S (frozen, partial, head-first, full) and → labels in D.
   - Suggested focus for round 3:
     - your full "real" domain (width + `l_comb80`), so the comparison with ours isolates the light correction in training;
     - the C+XF arms;
     - `noVnorm`;
     - your disjoint split.
   - Our D backbone and runs are under `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/dshift/` if you want to reuse them.
2. **Add partial freezing and head-first fine-tuning to the frozen/full pair.** The S-side results show the mode matters a lot: in D, 0.485 (frozen) vs 0.631 (full). The modes cost one flag each: `train_multitask --freeze-blocks 6` and `--lp-ft 2000`.
3. **Add the real-data check to every model.** Run `python -m qlfm.eval_external --run RUN --corpus $Y/cache/ub_r1 --tasks hit,evt_nu --out DIR`, then `python -m qlfm.eval_external report DIR`.
   - About 1 GPU-minute per model, plus about 5 min of CPU for the report.
   - Run reports one or two at a time: four at once ran out of memory on a login node.
   - Result 4 shows why this matters: a D-domain gain did not carry over fully to real data.
4. **Build on `main`.** It already has:
   - `--width meas3d` in fine-tuning, evaluation and `train_pretrain`;
   - `--freeze-blocks`, `--lp-ft`;
   - `qlfm/domain_probe.py`, `qlfm/eval_external.py`.

   Please push your round-2 changes first, then add `--domain` (width + light) on top of `--width`; we both defined the 3D factors under the same name. Your `--absolute-domain` fix for the double-perturbation bug is needed by us too.
5. **Re-pretraining the S backbones (+4 nh) is optional.** Sharing unlabelled pretraining events with the labelled 1% is not leakage. The disjoint split matters more for the *D* backbones.
6. **Treat `noVnorm` as a systematic band on the definition of D.** Result 4 already says the modelled domain misses part of the real shift.
7. **Select checkpoints in the labelled domain** (sim validation for real→sim), as with real data.
8. **Operational notes from this week:**
   - qsub `-v` cannot carry commas: pass multi-sample `--corpus` lists through an argument file (`pretrain.pbs` `ARGS_FILE`).
   - A corpus with many small shards (e.g. 1,781 SBND files) hits the per-process open-file limit (16,384). Merge them with `scripts/merge_shards.py`.
   - Login nodes crashed or dropped connections several times today. Keep chains inside PBS jobs, not on the login node. `train_pretrain` now refuses a second writer on the same output directory.
   - The **LArTPC_FM** allocation (about 5k node-h, ends 2027-01-10) gives access to the prod queue (10+ nodes). All `scripts/pbs/*.pbs` now default to it.
9. **A new idea to consider jointly: self-training** (C6, `docs/domain-shift-fm.md` phase 4b).
   - Start from the small labelled sample, predict on unlabelled D (later real) events, add high-confidence predictions as labels, and repeat.
   - In D, pseudo-label purity can be checked against truth.
   - Main risk: confident errors are the width shortcut itself. It needs a D-robust starting model, or an independent filter such as Wire-Cell agreement.

## 3. SBND samples on Polaris (for your use)

**Raw HDF5:** sbnd-h5maker 0.4, all pass `check_h5`. Production summary in fm4npp-lartpc#11, comment 6888730254.
- `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data/sbnd-bnb-cv/full/*.h5`: BNB CV, 627,205 events, 1,191 files.
- `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data/sbnd-nuecc/full/*.h5`: νe CC, 268,839 events, 536 files.
- `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data/sbnd-ncdr/full/*.h5`: NC Δ radiative, 26,744 events, 54 files.
- `sbnd-bnb-cv/10k/`, `test10/` and `pilot96/` are subsets of the BNB CV full sample.

**qlfm cache**, converted and merged into 60 shards (verified identical to the per-file conversion):
- `$Y/cache/sbnd_m/{bnb,nuecc,ncdr}`, with `Y=/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work`.
- Corpus string: `$Y/cache/sbnd_m/bnb,$Y/cache/sbnd_m/nuecc,$Y/cache/sbnd_m/ncdr`.
- Splits: `$Y/cache/splits_sbnd_full.npz` (train 737,991 / val 92,589 / test 92,208; `lab_f0.01` = 7,380).

**Running qlfm on SBND:** set `QLFM_DETECTOR=sbnd`. The geometry comes from `multi-modal-qlmatch/sbnd-h5maker/sbnd_detector.json`: 2 TPCs, 120 PMTs, beam window 0–3 µs.

**SBND-pretrained checkpoints** (34k steps): `$Y/sbnd_full/pre/m_cl/fm.pt` (charge + light) and `$Y/sbnd_full/pre/m_c/fm.pt` (charge only).

**Fine-tune results:** `$Y/sbnd_full/<model>/<mode>/f0.01_s{0,1,2}/`. T1 at 1% labels:
- SBND-pretrained CL: 0.921;
- MicroBooNE transfer: 0.903;
- scratch: 0.804;
- Wire-Cell: 0.931.
