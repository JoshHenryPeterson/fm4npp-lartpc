# Tech note: A1 data/sim robustness, round 3 (V-plane normalisation and cross-domain pretraining)

Date: 2026-11-23. Part 1 is inference only; Part 2 trains 4 new backbones and 24 downstream runs.

Plans:
- `docs/a1_check_round_3_preplan.md` (requests)
- `docs/a1_check_round_3_plan.md` (implementation plan)

Raw outputs: `fm4npp-lartpc-work/a1r3/`
- `novnorm/summary.md` (Part 1)
- `summary.md`, `summary.json` (Part 2)
- `pre/`, `down/`, `eval/` (per-run outputs)
- `STATUS` (job log)

**Rerun note.** `cl_sim2real_finetune_s1` and `cl_real2sim_finetune_s0` were each run twice concurrently by a job-queue lock race (fixed afterwards). Both were rerun on 2026-11-23 and all numbers below use the rerun results. The earlier outputs are kept in `fm4npp-lartpc-work/a1r3/quarantine_race_20261123/`. Real-domain T1 charge-weighted, old → new: `cl_sim2real_finetune_s1` 0.9219 → 0.9214; `cl_real2sim_finetune_s0` 0.6484 → 0.6681.

## Summary

**Part 1: `w_meas3d_noVnorm` (V-plane table not renormalised at the anode).**
- **The un-normalised V table is harsher for both models.**
  - **CL:** T1 charge-weighted −0.421 (against −0.288 for `w_meas3d`), per-event median −0.540 (against −0.328), T2 −0.087 (against −0.055).
  - **C + XF:** T1 charge-weighted −0.058 (against −0.038), per-event median −0.075 (against −0.046), T2 −0.005 (against −0.004).
- **The extra damage is in the V plane.** V alone with the un-normalised table costs CL −0.112 in T1 (−0.025 normalised) and C + XF −0.028 (−0.006 normalised). The V-plane T1 for CL falls by −0.50 (all planes together).

**Part 2: cross-domain pretrain → downstream.** Sim is the unaltered open data. Real is the `w_meas3d` hit widths on simulated-ν hits plus `l_comb80` light. Real-domain T1 and T2 are the primary results.
- **The downstream domain sets the real-domain score for finetuned models.** CL finetuned with real downstream labels reaches T1 charge-weighted 0.9175 when pretrained on real and 0.9214 when pretrained on sim. The existing sim-only CL models score 0.621 on the same real data, and sim → sim retrained on the disjoint lists scores 0.598. C + XF is 0.881 (real → real), 0.829 (sim → real) and 0.808 (existing).
- **Pretraining domain matters more when frozen.** CL frozen, real-domain T1: real → real 0.893, sim → real 0.845, real → sim 0.763, sim → sim 0.496. Real-domain pretraining gives a frozen CL backbone most of the robustness that finetuning otherwise provides.
- **CL beats C + XF in-domain; C + XF is less sensitive to the shift.** Finetuned real → real: CL 0.9175, C + XF 0.8813. Finetuned arms with a sim downstream domain lose 0.26–0.33 in T1 (CL) and 0.05–0.10 (C + XF) when evaluated on the real domain instead of the sim domain.
- **Reproduction:** the new sim → sim runs (disjoint event lists, re-pretrained) reproduce the existing models on the sim domain within 0.007 in T1 charge-weighted and 0.003 in T2 (all events), in all four model/mode combinations.
- T2 on all events is 0.93–0.997 everywhere. The bright-subset T2 separates the arms more, but T1 is the main discriminator.

## 1. Setup

- **Models:** qlfm-M CL (light-in-encoder pretraining, `--late-light 0`) and qlfm-M C + XF (charge-only pretraining, light added late by cross-attention, `--late-light 2`).
- **Metrics:** same definitions as round 2.
  - **T1:** ν/cosmic hit F1, charge-weighted (in-run score) and per-event median (paper definition).
  - **T2:** ν-event AUC, all events and the bright subset (native 3–6 µs PE ≥ 200; this is not the paper `truth.npz` 200-PE cut).
- **Scope:** only T1 and T2 are reported. Energy and light (T5) test scores are not reported; the T5 targets stay native in the real-domain downstream runs.
- **Uncertainties:** mean ± seed SD (ddof = 1) over the seeds present. A single seed shows SD 0.0000, which is not an uncertainty. Part 2 has two downstream seeds per arm and one for controls. Per-run bootstrap CIs are in `summary.md`. Per-seed values are in Appendix A.
- **Validation:** Stage 0 gates all passed (tests, splits, pretrain smoke with real/sim throughput ratio 0.983, downstream smoke with no double perturbation, `qsub`/`qdel` from a compute node, and Gate 5: `w_meas3d` reproduces round 2 on all 6 models within 0.002 in T1 and 0.001 in T2).

## 2. Part 1: `w_meas3d_noVnorm`

### 2.1 Variants

In the round-2 `F_MEAS_3D` table, the V factors were normalised to 1.0 at the anode, but the measured V ratio never reaches 1. The new table keeps Y and U unchanged and leaves V un-normalised (0.900 at the first drift bin):

| Plane | Factors, `w_meas3d` | Factors, `w_meas3d_noVnorm` |
|---|---|---|
| Y | 1.000, 0.967, 0.941, 0.926, 0.915, 0.903 | unchanged |
| V | 1.000, 0.980, 0.954, 0.938, 0.942, 0.941 | 0.900, 0.882, 0.859, 0.844, 0.848, 0.847 |
| U | 1.000, 0.960, 0.922, 0.907, 0.892, 0.880 | unchanged |

| Variant | Definition |
|---|---|
| `w_meas3d` | round-2 3D table, all planes |
| `w_meas3d_noVnorm` | un-normalised V table, all planes |
| `w_meas3d_V` | normalised V table on V only (f = 1 on U, Y) |
| `w_meas3d_noVnorm_V` | un-normalised V table on V only |

All variants act on simulated-ν hits only, applied to the 6 existing multi-task models (seeds 0, 1, 2) at inference time.

### 2.2 Results (Δ vs nominal; nominal = absolute)

Mean ± seed SD over 3 seeds.

**qlfm-M CL**

| Variant | T1 charge-wtd | T1 evt-median | T2 all | T2 bright |
|---|---|---|---|---|
| nominal | 0.9185 ± 0.0002 | 0.9217 ± 0.0006 | 0.9932 ± 0.0028 | 0.9882 ± 0.0035 |
| `w_meas3d` | −0.2877 ± 0.0165 | −0.3280 ± 0.0199 | −0.0554 ± 0.0082 | −0.0778 ± 0.0196 |
| `w_meas3d_noVnorm` | **−0.4211 ± 0.0170** | **−0.5395 ± 0.0263** | **−0.0868 ± 0.0111** | **−0.1218 ± 0.0282** |
| `w_meas3d_V` | −0.0248 ± 0.0027 | −0.0266 ± 0.0029 | −0.0016 ± 0.0009 | −0.0025 ± 0.0011 |
| `w_meas3d_noVnorm_V` | −0.1122 ± 0.0033 | −0.1317 ± 0.0015 | −0.0042 ± 0.0023 | −0.0069 ± 0.0037 |

noVnorm − `w_meas3d` (all planes): T1 charge-weighted −0.1334 ± 0.0054, T2 −0.0314 ± 0.0032. V-only, noVnorm − normalised: T1 −0.0874 ± 0.0030, T2 −0.0025 ± 0.0014.

**qlfm-M C + XF**

| Variant | T1 charge-wtd | T1 evt-median | T2 all | T2 bright |
|---|---|---|---|---|
| nominal | 0.8593 ± 0.0017 | 0.8621 ± 0.0017 | 0.9899 ± 0.0005 | 0.9689 ± 0.0004 |
| `w_meas3d` | −0.0375 ± 0.0030 | −0.0457 ± 0.0039 | −0.0035 ± 0.0003 | −0.0090 ± 0.0006 |
| `w_meas3d_noVnorm` | **−0.0579 ± 0.0036** | **−0.0751 ± 0.0040** | **−0.0053 ± 0.0002** | **−0.0136 ± 0.0014** |
| `w_meas3d_V` | −0.0056 ± 0.0001 | −0.0075 ± 0.0003 | −0.0005 ± 0.0001 | −0.0014 ± 0.0003 |
| `w_meas3d_noVnorm_V` | −0.0276 ± 0.0005 | −0.0391 ± 0.0007 | −0.0013 ± 0.0002 | −0.0036 ± 0.0010 |

noVnorm − `w_meas3d` (all planes): T1 charge-weighted −0.0203 ± 0.0010, T2 −0.0017 ± 0.0002. V-only, noVnorm − normalised: T1 −0.0219 ± 0.0005, T2 −0.0008 ± 0.0002.

### 2.3 Per-plane T1 charge-weighted (Δ vs nominal on the plane)

**qlfm-M CL**

| Plane | Nominal | `w_meas3d` | `w_meas3d_noVnorm` | `w_meas3d_V` | `w_meas3d_noVnorm_V` |
|---|---|---|---|---|---|
| U | 0.9013 ± 0.0002 | −0.3286 ± 0.0138 | −0.4056 ± 0.0170 | −0.0098 ± 0.0011 | −0.0289 ± 0.0042 |
| V | 0.9016 ± 0.0010 | −0.2565 ± 0.0118 | −0.5026 ± 0.0129 | −0.0596 ± 0.0071 | −0.3565 ± 0.0073 |
| Y | 0.9486 ± 0.0001 | −0.2827 ± 0.0223 | −0.3691 ± 0.0224 | −0.0089 ± 0.0007 | −0.0195 ± 0.0019 |

**qlfm-M C + XF**

| Plane | Nominal | `w_meas3d` | `w_meas3d_noVnorm` | `w_meas3d_V` | `w_meas3d_noVnorm_V` |
|---|---|---|---|---|---|
| U | 0.8392 ± 0.0024 | −0.0471 ± 0.0056 | −0.0447 ± 0.0051 | +0.0007 ± 0.0001 | +0.0019 ± 0.0003 |
| V | 0.8435 ± 0.0032 | −0.0274 ± 0.0013 | −0.0948 ± 0.0047 | −0.0184 ± 0.0002 | −0.0930 ± 0.0013 |
| Y | 0.8907 ± 0.0005 | −0.0385 ± 0.0025 | −0.0384 ± 0.0017 | −0.0002 ± 0.0003 | −0.0002 ± 0.0007 |

Per-plane per-event-median T1 (paper definition), absolute for `w_meas3d_noVnorm`, with the round-2 references:

| Model | Plane | `w_meas3d_noVnorm` | nominal (round 2) | `w_meas3d` (round 2) |
|---|---|---|---|---|
| CL | U | 0.2757 ± 0.0123 | 0.8929 ± 0.0006 | 0.4230 ± 0.0147 |
| CL | V | 0.1619 ± 0.0159 | 0.8964 ± 0.0009 | 0.5990 ± 0.0138 |
| CL | Y | 0.5187 ± 0.0433 | 0.9712 ± 0.0004 | 0.6831 ± 0.0335 |
| C + XF | U | 0.7495 ± 0.0065 | 0.8191 ± 0.0021 | 0.7455 ± 0.0078 |
| C + XF | V | 0.6784 ± 0.0046 | 0.8305 ± 0.0049 | 0.7912 ± 0.0060 |
| C + XF | Y | 0.8987 ± 0.0019 | 0.9334 ± 0.0012 | 0.8977 ± 0.0023 |

### 2.4 Part 1 findings

1. **The V normalisation is not a detail.** For CL it moves the all-planes T1 loss from −0.29 to −0.42 (charge-weighted), and the per-event median from −0.33 to −0.54. For C + XF the loss grows by about half (−0.038 → −0.058).
2. **In C + XF the extra loss is confined to the V plane.** V drops by −0.095 (−0.027 normalised), while U and Y match `w_meas3d` (−0.045 and −0.038). So its plane contributions remain local, as in round 2.
3. **In CL the V change spreads to all planes.** All-planes noVnorm costs U −0.41, V −0.50, Y −0.37, against −0.33, −0.26 and −0.28 normalised. V-only noVnorm costs V −0.36, but U and Y only −0.03 and −0.02. As in round 2, CL is more sensitive to interactions between planes than to any one plane.
4. **Gate 5 passed:** `w_meas3d` reproduces the round-2 scores in all six models (|ΔT1| ≤ 0.0001, |ΔT2| ≤ 0.0001).

## 3. Part 2: cross-domain pretrain → downstream

### 3.1 Domains and runs

| Domain | Definition |
|---|---|
| sim | unaltered open data (`nominal`) |
| real | `w_meas3d` widths on simulated-ν hits and `l_comb80` on all ophits of flash events (`d_real`) |

The real domain is a modelled data-like domain, not detector data.

**Event lists.** `scripts/a1r3_make_splits.py` copies the round-1/2 splits and adds `pre_r3 = pre_f0.90 − (lab_f0.01 ∪ val ∪ test)`. Pretrain (739 597 events), downstream train (7 471), validation (2 000 per task) and test (10 000 per task) lists are pairwise disjoint and identical across runs. The disjointness was re-checked on the per-run `event_lists.npz` files (4 pretrain and 24 downstream runs).

The existing models were pretrained on `pre_f0.90`, which overlaps the downstream training set. That is why the sim backbones are re-pretrained, and why the existing models appear only as a reference row.

**Backbones (4 new, seed 0, 34 000 steps, 1 node each).**

| Backbone | Light in encoder | Domain |
|---|---|---|
| `pre/sim_cl` | yes | sim |
| `pre/real_cl` | yes | real |
| `pre/sim_c` | no | sim |
| `pre/real_c` | no | real (widths only) |

**Downstream (24 runs).** Five tasks, 1% labels, 6 000 steps, batch 32, `frozen` or `finetune`. Per model and mode: sim → real and real → sim at seeds 0 and 1 (16 arm runs), plus sim → sim and real → real at seed 0 as in-domain controls (8 runs).

**Evaluations (32).** The 24 runs plus 8 existing-model baselines (`finetune` seeds 0–2 and `frozen` seed 0, for each of CL and C + XF). All use `--absolute-domain`, so `nominal` is the sim domain and `d_real` the real domain, whichever domain the model was trained on. This avoids the double perturbation that the round-2 evaluation would apply to models trained in the real domain (checked in Gate 4).

**Reading the tables.**
- Rows are labelled pretrain → downstream. "existing (sim) → sim" is the existing yuhw model, pretrained and trained on unaltered data.
- The **primary** result is the real-domain evaluation: T1 (charge-weighted and per-event median) and T2 (all events and bright subset).
- The sim-domain evaluation and the gap (real − sim, per seed) are secondary.
- C + XF pretraining sees only the width shift, because its backbone is charge-only. Light enters only in its downstream head.
- Controls and baselines have fewer seeds. Differences between arms smaller than the round-2 seed SD (≈0.002 in T1 for the existing models) should not be read.

### 3.2 CL, finetune

**Primary: real-domain scores**

| pretrain → downstream | n seeds | T1 charge-wtd | T1 evt-median | T2 all | T2 bright |
|---|---|---|---|---|---|
| real → real | 1 | 0.9175 ± 0.0000 | 0.9203 ± 0.0000 | 0.9970 ± 0.0000 | 0.9934 ± 0.0000 |
| sim → real | 2 | 0.9214 ± 0.0000 | 0.9241 ± 0.0001 | 0.9961 ± 0.0000 | 0.9939 ± 0.0013 |
| real → sim | 2 | 0.6658 ± 0.0032 | 0.6219 ± 0.0015 | 0.9658 ± 0.0073 | 0.9583 ± 0.0016 |
| sim → sim | 1 | 0.5981 ± 0.0000 | 0.5213 ± 0.0000 | 0.9293 ± 0.0000 | 0.9002 ± 0.0000 |
| existing (sim) → sim | 3 | 0.6210 ± 0.0128 | 0.5709 ± 0.0165 | 0.9400 ± 0.0116 | 0.9135 ± 0.0222 |

**Secondary: sim-domain scores and domain gap**

| pretrain → downstream | T1 cw (sim) | T1 med (sim) | T2 all (sim) | T2 bright (sim) | gap T1 cw | gap T1 med | gap T2 all | gap T2 bright |
|---|---|---|---|---|---|---|---|---|
| real → real | 0.6827 ± 0.0000 | 0.6748 ± 0.0000 | 0.9651 ± 0.0000 | 0.9482 ± 0.0000 | +0.2348 ± 0.0000 | +0.2455 ± 0.0000 | +0.0318 ± 0.0000 | +0.0451 ± 0.0000 |
| sim → real | 0.7151 ± 0.0153 | 0.7352 ± 0.0240 | 0.9651 ± 0.0091 | 0.9566 ± 0.0041 | +0.2063 ± 0.0154 | +0.1890 ± 0.0241 | +0.0310 ± 0.0092 | +0.0374 ± 0.0054 |
| real → sim | 0.9220 ± 0.0013 | 0.9265 ± 0.0023 | 0.9976 ± 0.0004 | 0.9956 ± 0.0006 | -0.2562 ± 0.0045 | -0.3046 ± 0.0008 | -0.0318 ± 0.0077 | -0.0374 ± 0.0023 |
| sim → sim | 0.9252 ± 0.0000 | 0.9290 ± 0.0000 | 0.9947 ± 0.0000 | 0.9911 ± 0.0000 | -0.3270 ± 0.0000 | -0.4077 ± 0.0000 | -0.0654 ± 0.0000 | -0.0909 ± 0.0000 |
| existing (sim) → sim | 0.9185 ± 0.0002 | 0.9217 ± 0.0006 | 0.9932 ± 0.0028 | 0.9882 ± 0.0035 | -0.2975 ± 0.0128 | -0.3508 ± 0.0170 | -0.0532 ± 0.0087 | -0.0746 ± 0.0193 |

**Decomposition of the real-domain change in T1 charge-weighted (Δ vs the sim-domain score of the same run)**

| pretrain → downstream | width only (`w_meas3d`) | light only (`l_comb80`) | `d_real` | `d_real_noVnorm` (absolute T1 cw) |
|---|---|---|---|---|
| real → real | +0.2346 ± 0.0000 | -0.0052 ± 0.0000 | +0.2348 ± 0.0000 | 0.8102 ± 0.0000 |
| sim → real | +0.2062 ± 0.0157 | -0.0019 ± 0.0011 | +0.2063 ± 0.0154 | 0.8274 ± 0.0023 |
| real → sim | -0.2421 ± 0.0005 | -0.0010 ± 0.0004 | -0.2562 ± 0.0045 | 0.5157 ± 0.0103 |
| sim → sim | -0.3210 ± 0.0000 | -0.0007 ± 0.0000 | -0.3270 ± 0.0000 | 0.4681 ± 0.0000 |
| existing (sim) → sim | -0.2877 ± 0.0164 | -0.0010 ± 0.0002 | -0.2975 ± 0.0128 | 0.4861 ± 0.0116 |

**Per-plane T1 per-event median, real domain**

| pretrain → downstream | U | V | Y |
|---|---|---|---|
| real → real | 0.9010 ± 0.0000 | 0.8952 ± 0.0000 | 0.9695 ± 0.0000 |
| sim → real | 0.9033 ± 0.0001 | 0.9043 ± 0.0019 | 0.9734 ± 0.0001 |
| real → sim | 0.4104 ± 0.0051 | 0.6765 ± 0.0075 | 0.7028 ± 0.0123 |
| sim → sim | 0.2671 ± 0.0000 | 0.6198 ± 0.0000 | 0.5660 ± 0.0000 |
| existing (sim) → sim | 0.3914 ± 0.0099 | 0.5698 ± 0.0117 | 0.6585 ± 0.0288 |

### 3.3 CL, frozen

**Primary: real-domain scores**

| pretrain → downstream | n seeds | T1 charge-wtd | T1 evt-median | T2 all | T2 bright |
|---|---|---|---|---|---|
| real → real | 1 | 0.8934 ± 0.0000 | 0.8920 ± 0.0000 | 0.9964 ± 0.0000 | 0.9882 ± 0.0000 |
| sim → real | 2 | 0.8448 ± 0.0008 | 0.8340 ± 0.0014 | 0.9886 ± 0.0008 | 0.9603 ± 0.0039 |
| real → sim | 2 | 0.7626 ± 0.0045 | 0.7175 ± 0.0053 | 0.9928 ± 0.0009 | 0.9709 ± 0.0040 |
| sim → sim | 1 | 0.4958 ± 0.0000 | 0.2434 ± 0.0000 | 0.9818 ± 0.0000 | 0.9471 ± 0.0000 |
| existing (sim) → sim | 1 | 0.4736 ± 0.0000 | 0.2431 ± 0.0000 | 0.9770 ± 0.0000 | 0.9363 ± 0.0000 |

**Secondary: sim-domain scores and domain gap**

| pretrain → downstream | T1 cw (sim) | T1 med (sim) | T2 all (sim) | T2 bright (sim) | gap T1 cw | gap T1 med | gap T2 all | gap T2 bright |
|---|---|---|---|---|---|---|---|---|
| real → real | 0.5175 ± 0.0000 | 0.3878 ± 0.0000 | 0.9785 ± 0.0000 | 0.9322 ± 0.0000 | +0.3759 ± 0.0000 | +0.5042 ± 0.0000 | +0.0180 ± 0.0000 | +0.0559 ± 0.0000 |
| sim → real | 0.7669 ± 0.0072 | 0.7314 ± 0.0109 | 0.9867 ± 0.0007 | 0.9514 ± 0.0039 | +0.0779 ± 0.0064 | +0.1026 ± 0.0096 | +0.0019 ± 0.0001 | +0.0088 ± 0.0001 |
| real → sim | 0.8681 ± 0.0005 | 0.8681 ± 0.0001 | 0.9880 ± 0.0010 | 0.9553 ± 0.0042 | -0.1055 ± 0.0041 | -0.1506 ± 0.0053 | +0.0048 ± 0.0000 | +0.0155 ± 0.0002 |
| sim → sim | 0.8932 ± 0.0000 | 0.8936 ± 0.0000 | 0.9963 ± 0.0000 | 0.9855 ± 0.0000 | -0.3974 ± 0.0000 | -0.6502 ± 0.0000 | -0.0145 ± 0.0000 | -0.0385 ± 0.0000 |
| existing (sim) → sim | 0.8944 ± 0.0000 | 0.8939 ± 0.0000 | 0.9971 ± 0.0000 | 0.9892 ± 0.0000 | -0.4208 ± 0.0000 | -0.6508 ± 0.0000 | -0.0201 ± 0.0000 | -0.0529 ± 0.0000 |

**Decomposition**

| pretrain → downstream | width only (`w_meas3d`) | light only (`l_comb80`) | `d_real` | `d_real_noVnorm` (absolute T1 cw) |
|---|---|---|---|---|
| real → real | +0.3755 ± 0.0000 | +0.0058 ± 0.0000 | +0.3759 ± 0.0000 | 0.7291 ± 0.0000 |
| sim → real | +0.0785 ± 0.0066 | -0.0039 ± 0.0009 | +0.0779 ± 0.0064 | 0.7178 ± 0.0011 |
| real → sim | -0.0985 ± 0.0039 | -0.0013 ± 0.0001 | -0.1055 ± 0.0041 | 0.5525 ± 0.0026 |
| sim → sim | -0.3918 ± 0.0000 | -0.0000 ± 0.0000 | -0.3974 ± 0.0000 | 0.3441 ± 0.0000 |
| existing (sim) → sim | -0.4117 ± 0.0000 | -0.0005 ± 0.0000 | -0.4208 ± 0.0000 | 0.2782 ± 0.0000 |

**Per-plane T1 per-event median, real domain**

| pretrain → downstream | U | V | Y |
|---|---|---|---|
| real → real | 0.8624 ± 0.0000 | 0.8454 ± 0.0000 | 0.9590 ± 0.0000 |
| sim → real | 0.7610 ± 0.0013 | 0.7882 ± 0.0002 | 0.9342 ± 0.0005 |
| real → sim | 0.6780 ± 0.0107 | 0.6958 ± 0.0012 | 0.8007 ± 0.0041 |
| sim → sim | 0.0664 ± 0.0000 | 0.3003 ± 0.0000 | 0.2053 ± 0.0000 |
| existing (sim) → sim | 0.1008 ± 0.0000 | 0.2341 ± 0.0000 | 0.2456 ± 0.0000 |

### 3.4 C + XF, finetune

**Primary: real-domain scores**

| pretrain → downstream | n seeds | T1 charge-wtd | T1 evt-median | T2 all | T2 bright |
|---|---|---|---|---|---|
| real → real | 1 | 0.8813 ± 0.0000 | 0.8880 ± 0.0000 | 0.9833 ± 0.0000 | 0.9633 ± 0.0000 |
| sim → real | 2 | 0.8294 ± 0.0064 | 0.8223 ± 0.0096 | 0.9856 ± 0.0012 | 0.9568 ± 0.0051 |
| real → sim | 2 | 0.8076 ± 0.0077 | 0.7980 ± 0.0155 | 0.9803 ± 0.0008 | 0.9521 ± 0.0022 |
| sim → sim | 1 | 0.7667 ± 0.0000 | 0.7298 ± 0.0000 | 0.9847 ± 0.0000 | 0.9565 ± 0.0000 |
| existing (sim) → sim | 3 | 0.8084 ± 0.0019 | 0.7901 ± 0.0028 | 0.9865 ± 0.0005 | 0.9604 ± 0.0009 |

**Secondary: sim-domain scores and domain gap**

| pretrain → downstream | T1 cw (sim) | T1 med (sim) | T2 all (sim) | T2 bright (sim) | gap T1 cw | gap T1 med | gap T2 all | gap T2 bright |
|---|---|---|---|---|---|---|---|---|
| real → real | 0.8603 ± 0.0000 | 0.8698 ± 0.0000 | 0.9744 ± 0.0000 | 0.9509 ± 0.0000 | +0.0210 ± 0.0000 | +0.0182 ± 0.0000 | +0.0090 ± 0.0000 | +0.0125 ± 0.0000 |
| sim → real | 0.8223 ± 0.0055 | 0.8238 ± 0.0069 | 0.9817 ± 0.0012 | 0.9470 ± 0.0052 | +0.0071 ± 0.0009 | -0.0015 ± 0.0027 | +0.0039 ± 0.0000 | +0.0097 ± 0.0001 |
| real → sim | 0.8999 ± 0.0001 | 0.9097 ± 0.0003 | 0.9905 ± 0.0008 | 0.9742 ± 0.0018 | -0.0923 ± 0.0078 | -0.1116 ± 0.0152 | -0.0102 ± 0.0000 | -0.0221 ± 0.0005 |
| sim → sim | 0.8654 ± 0.0000 | 0.8678 ± 0.0000 | 0.9921 ± 0.0000 | 0.9743 ± 0.0000 | -0.0987 ± 0.0000 | -0.1380 ± 0.0000 | -0.0074 ± 0.0000 | -0.0178 ± 0.0000 |
| existing (sim) → sim | 0.8593 ± 0.0017 | 0.8623 ± 0.0014 | 0.9899 ± 0.0005 | 0.9689 ± 0.0004 | -0.0509 ± 0.0022 | -0.0722 ± 0.0024 | -0.0034 ± 0.0005 | -0.0084 ± 0.0011 |

**Decomposition**

| pretrain → downstream | width only (`w_meas3d`) | light only (`l_comb80`) | `d_real` | `d_real_noVnorm` (absolute T1 cw) |
|---|---|---|---|---|
| real → real | +0.0181 ± 0.0000 | -0.0001 ± 0.0000 | +0.0210 ± 0.0000 | 0.8608 ± 0.0000 |
| sim → real | +0.0025 ± 0.0005 | +0.0035 ± 0.0004 | +0.0071 ± 0.0009 | 0.8235 ± 0.0082 |
| real → sim | -0.0816 ± 0.0038 | -0.0044 ± 0.0010 | -0.0923 ± 0.0078 | 0.7477 ± 0.0097 |
| sim → sim | -0.0817 ± 0.0000 | -0.0094 ± 0.0000 | -0.0987 ± 0.0000 | 0.7116 ± 0.0000 |
| existing (sim) → sim | -0.0375 ± 0.0030 | -0.0099 ± 0.0003 | -0.0509 ± 0.0022 | 0.7875 ± 0.0013 |

**Per-plane T1 per-event median, real domain**

| pretrain → downstream | U | V | Y |
|---|---|---|---|
| real → real | 0.8561 ± 0.0000 | 0.8631 ± 0.0000 | 0.9479 ± 0.0000 |
| sim → real | 0.7732 ± 0.0166 | 0.7845 ± 0.0109 | 0.9091 ± 0.0056 |
| real → sim | 0.7102 ± 0.0155 | 0.8220 ± 0.0030 | 0.8710 ± 0.0158 |
| sim → sim | 0.6061 ± 0.0000 | 0.7259 ± 0.0000 | 0.8292 ± 0.0000 |
| existing (sim) → sim | 0.7146 ± 0.0055 | 0.7638 ± 0.0040 | 0.8742 ± 0.0020 |

### 3.5 C + XF, frozen

**Primary: real-domain scores**

| pretrain → downstream | n seeds | T1 charge-wtd | T1 evt-median | T2 all | T2 bright |
|---|---|---|---|---|---|
| real → real | 1 | 0.7413 ± 0.0000 | 0.6903 ± 0.0000 | 0.9845 ± 0.0000 | 0.9484 ± 0.0000 |
| sim → real | 2 | 0.6835 ± 0.0006 | 0.6100 ± 0.0003 | 0.9821 ± 0.0009 | 0.9340 ± 0.0027 |
| real → sim | 2 | 0.6983 ± 0.0076 | 0.6217 ± 0.0150 | 0.9840 ± 0.0020 | 0.9517 ± 0.0016 |
| sim → sim | 1 | 0.6385 ± 0.0000 | 0.5361 ± 0.0000 | 0.9814 ± 0.0000 | 0.9367 ± 0.0000 |
| existing (sim) → sim | 1 | 0.6486 ± 0.0000 | 0.5506 ± 0.0000 | 0.9843 ± 0.0000 | 0.9461 ± 0.0000 |

**Secondary: sim-domain scores and domain gap**

| pretrain → downstream | T1 cw (sim) | T1 med (sim) | T2 all (sim) | T2 bright (sim) | gap T1 cw | gap T1 med | gap T2 all | gap T2 bright |
|---|---|---|---|---|---|---|---|---|
| real → real | 0.7393 ± 0.0000 | 0.6989 ± 0.0000 | 0.9805 ± 0.0000 | 0.9388 ± 0.0000 | +0.0020 ± 0.0000 | -0.0086 ± 0.0000 | +0.0040 ± 0.0000 | +0.0096 ± 0.0000 |
| sim → real | 0.6857 ± 0.0019 | 0.6226 ± 0.0088 | 0.9783 ± 0.0016 | 0.9211 ± 0.0050 | -0.0022 ± 0.0025 | -0.0125 ± 0.0092 | +0.0038 ± 0.0007 | +0.0129 ± 0.0023 |
| real → sim | 0.7604 ± 0.0021 | 0.7297 ± 0.0070 | 0.9860 ± 0.0010 | 0.9513 ± 0.0009 | -0.0621 ± 0.0056 | -0.1080 ± 0.0080 | -0.0019 ± 0.0010 | +0.0003 ± 0.0025 |
| sim → sim | 0.7136 ± 0.0000 | 0.6630 ± 0.0000 | 0.9811 ± 0.0000 | 0.9319 ± 0.0000 | -0.0751 ± 0.0000 | -0.1269 ± 0.0000 | +0.0003 ± 0.0000 | +0.0049 ± 0.0000 |
| existing (sim) → sim | 0.7207 ± 0.0000 | 0.6627 ± 0.0000 | 0.9839 ± 0.0000 | 0.9433 ± 0.0000 | -0.0722 ± 0.0000 | -0.1120 ± 0.0000 | +0.0004 ± 0.0000 | +0.0028 ± 0.0000 |

**Decomposition**

| pretrain → downstream | width only (`w_meas3d`) | light only (`l_comb80`) | `d_real` | `d_real_noVnorm` (absolute T1 cw) |
|---|---|---|---|---|
| real → real | -0.0024 ± 0.0000 | +0.0039 ± 0.0000 | +0.0020 ± 0.0000 | 0.7329 ± 0.0000 |
| sim → real | -0.0058 ± 0.0003 | +0.0031 ± 0.0020 | -0.0022 ± 0.0025 | 0.6814 ± 0.0022 |
| real → sim | -0.0419 ± 0.0009 | -0.0148 ± 0.0047 | -0.0621 ± 0.0056 | 0.6724 ± 0.0071 |
| sim → sim | -0.0502 ± 0.0000 | -0.0219 ± 0.0000 | -0.0751 ± 0.0000 | 0.6147 ± 0.0000 |
| existing (sim) → sim | -0.0466 ± 0.0000 | -0.0218 ± 0.0000 | -0.0722 ± 0.0000 | 0.6254 ± 0.0000 |

**Per-plane T1 per-event median, real domain**

| pretrain → downstream | U | V | Y |
|---|---|---|---|
| real → real | 0.5949 ± 0.0000 | 0.6199 ± 0.0000 | 0.8152 ± 0.0000 |
| sim → real | 0.4650 ± 0.0004 | 0.4775 ± 0.0104 | 0.7702 ± 0.0040 |
| real → sim | 0.4885 ± 0.0147 | 0.5682 ± 0.0153 | 0.7417 ± 0.0164 |
| sim → sim | 0.3506 ± 0.0000 | 0.4320 ± 0.0000 | 0.6852 ± 0.0000 |
| existing (sim) → sim | 0.3733 ± 0.0000 | 0.4588 ± 0.0000 | 0.6872 ± 0.0000 |

### 3.6 Part 2 findings

All comparisons are on the real domain unless stated otherwise.

1. **sim → real vs real → real (does pretraining domain matter when the downstream domain is real?).**
   - **CL finetune:** the two are within 0.005. T1 charge-weighted 0.9214 against 0.9175 (single-seed control), and bright T2 0.9939 against 0.9934.
   - **CL frozen:** real pretraining helps, 0.893 against 0.845 (−0.049).
   - **C + XF:** sim pretraining costs 0.052 (finetune, 0.829 against 0.881) and 0.058 (frozen, 0.684 against 0.741). C + XF pretraining sees only the width shift, so this is the effect of a width-only domain mismatch in the backbone.
   - The decomposition tables show the real-domain gain over the sim domain is almost entirely from the widths.
2. **real → sim vs sim → sim (does pretraining domain matter when the downstream domain is sim?).** Both arms train downstream on sim. Real pretraining raises the real-domain T1 in every arm.
   - **CL frozen:** real-domain T1 0.763 against 0.496 (+0.27). Its sim-domain T1 is 0.868 against 0.893.
   - **CL finetune:** 0.666 against 0.598 (+0.07); sim-domain T1 0.922 against 0.925.
   - **C + XF:** real-domain 0.808 against 0.767 (finetune) and 0.698 against 0.639 (frozen). In the sim domain, C + XF real → sim is better than sim → sim by 0.035 (finetune) and 0.047 (frozen).
   - Domain-robust features from real pretraining thus carry over to downstream heads trained on sim, but not as far as training downstream on real: no real → sim arm reaches the sim → real or real → real level.
3. **Frozen vs finetune.**
   - Finetuning raises the real-domain T1 in every arm where the downstream domain is real, by 0.077 (CL sim → real, 0.845 → 0.921), 0.024 (CL real → real), 0.146 (C + XF sim → real) and 0.140 (C + XF real → real).
   - For real → sim, finetuning lowers CL (0.763 → 0.666) and leaves C + XF higher (0.698 → 0.808). Frozen CL real → sim reaches 0.763 on the real domain with no real downstream data, so a good part of the real-domain performance is carried by the pretrained features.
4. **CL vs C + XF.** CL is better when the downstream domain is real: finetuned real → real 0.9175 against 0.8813, sim → real 0.9214 against 0.8294. C + XF is less sensitive to the domain: existing → real costs 0.05 (against 0.30 for CL), so it ends higher than CL for sim-trained models (0.808 against 0.621).
5. **Where the real-domain loss comes from.** The width correction accounts for almost all of it. The light correction (`l_comb80`) costs at most 0.022 in T1 (C + XF frozen, sim-downstream arms) and 0.006 or less for CL, as in the light round.
6. **`d_real_noVnorm` (link to Part 1).** Using the un-normalised V table in the real domain lowers every arm's T1 charge-weighted, by 0.09–0.21 for CL and 0.002–0.06 for C + XF. Even CL real → real falls from 0.9175 to 0.8102 (finetune) and from 0.8934 to 0.7291 (frozen). The real-domain scores above therefore depend on the V normalisation choice. For finetuned CL the two real-downstream arms stay well above the sim-downstream ones (0.81–0.83 against 0.47–0.52).
7. **Reproduction (new sim → sim vs existing models, sim domain).**

   | Model | Mode | Existing T1 cw | New sim → sim T1 cw | Existing T2 all | New T2 all |
   |---|---|---|---|---|---|
   | CL | finetune | 0.9185 ± 0.0002 | 0.9252 | 0.9932 ± 0.0028 | 0.9947 |
   | CL | frozen | 0.8944 | 0.8932 | 0.9971 | 0.9963 |
   | C + XF | finetune | 0.8593 ± 0.0017 | 0.8654 | 0.9899 ± 0.0005 | 0.9921 |
   | C + XF | frozen | 0.7207 | 0.7136 | 0.9839 | 0.9811 |

   The disjoint pretrain/downstream split and re-pretraining change the sim-domain T1 by 0.001 to 0.007 and T2 by at most 0.003. The new sim → sim runs are single seeds, so the comparison is against the round-2 seed SD of about 0.002. These checks support the sim → sim control as a like-for-like partner of the real arms.

## 4. Caveats

- **"Real" is a modelled data-like domain,** built from the round-2 width factors and the `l_comb80` light correction. It is not detector data. Widths are corrected on simulated-ν hits only; light on all ophits of flash events (sim ophits are not labelled).
- **C + XF pretraining sees only the width part of the shift** (charge-only backbone). Its light correction acts only downstream and at evaluation.
- **One pretraining seed per backbone.** Seed spread is downstream only, over two seeds per arm and one for controls. Quote bootstrap CIs, and do not read differences below about 0.01 in T1 without them.
- **Rerun of two runs.** A job-queue lock race ran `cl_sim2real_finetune_s1` and `cl_real2sim_finetune_s0` twice concurrently each, interleaving their logs. Both were rerun on 2026-11-23 and the tables use the rerun results; the quarantined outputs are in `a1r3/quarantine_race_20261123/`. Real-domain T1 charge-weighted, old → new: 0.9219 → 0.9214 (`cl_sim2real_finetune_s1`) and 0.6484 → 0.6681 (`cl_real2sim_finetune_s0`).
- **Energy and light test scores are not reported.** Only T1 and T2 are in scope, and the T5 targets stay native in the real-domain downstream runs.
- **T2 bright** uses the native 3–6 µs PE ≥ 200 subset, not the paper `truth.npz` cut.
- **Inference runs in bf16**, as in round 2 (irrelevant at these effect sizes).
- **Stage 1a launch bug.** Only 3 of the 4 pretrains were launched in the first stage-1 job (fixed; the fourth ran in stage 1b). It affected the schedule, not the results.

## 5. Compute and reproducibility

**Jobs** (Polaris, 4 × A100 per node, account `LArTPC_FM`). Walltime runs from the first start to completion, as logged in `a1r3/STATUS`, with requeues included.

| Stage | Job | Queue | Nodes | Walltime (h) | Node-hours | Notes |
|---|---|---|---|---|---|---|
| 0 | 7862003 | debug | 2 | 0.35 | 0.70 | tests, splits, smoke, gates, Part 1 (6 runs) |
| 1a | 7862258 | preemptable | 4 | 2.07 | 8.27 | 3 of 4 pretrains launched (launch bug, now fixed) |
| 1b | 7869665 | preemptable | 4 | 4.69 | 18.74 | one remaining pretrain (`real_c`), with requeues |
| 2 | 7873432 | preemptable | 6 | 2.36 | 14.17 | 24 downstream runs, one requeue |
| 3 | 7875850 | preemptable | 4 | 1.28 | 5.13 | 32 evaluations and summary |
| rerun, stage 2 | 7880513 | debug | 1 | 0.81 | 0.80 | the 2 raced downstream runs; 22/24 already done, still 22/24 at its end |
| rerun, stage 2 retry | 7880705 | debug | 1 | 0.30 | 0.30 | finished the 2 raced runs (24/24) |
| rerun, stage 3 | 7880812 | debug | 1 | 0.18 | 0.18 | 2 re-evaluations (30/32 already done) and summary |
| | | | | **total** | **≈ 48.3** (47.0 + 1.28 rerun) | |

The plan expected ≈ 22 nh (≈ 34 nh maximum charged). The overrun comes from preemptable requeues, the first-launch bug (stage 1a charged for 4 nodes with 3 pretrains), and holding a 4-node allocation for `real_c` (a single-node run) through the stage 1b requeues.

Rerun (2026-11-23): submitted 17:09; stage 2 started 17:49 (debug queue wait 0.7 h), the retry 19:15, stage 3 20:24; summary complete 20:35. Node-hours are nodes × (end − start) from the `job started` lines to the next stage submission or COMPLETE.

Queue waits per stage: stage 0, 1.7 h; stage 1a, 36 h; stage 1b, 16 h; stage 2, 11 h; stage 3, 11 h. The campaign took about 4 days from submission (2026-11-19 03:08) to the summary (2026-11-22 18:22).

Verification against `STATUS`: the stage 0 to 3 start and end times reproduce the walltimes and node-hours above. The `STATUS` file shows stage 1b with four starts (11:17, 11:39, 14:21, 15:35 on 2026-11-21), stage 2 with two (03:12 and 04:50 on 2026-11-22), and stage 3 with two (17:06 and 17:49 on 2026-11-22). The node-hours are wall from the first start and may overstate the charge if the preempted segments were shorter than the elapsed time.

**Code** (jpeterso copy): `qlfm/eval_perturb.py` (`w_meas3d_noVnorm` variants, `--absolute-domain`, composite `d_real` variants), `scripts/a1r3_make_splits.py`, round-3 stage scripts under `scripts/pbs/a1r3_stage*.pbs`, and the summary generator that wrote `summary.md` and `summary.json`. Domain support in `train_pretrain.py` and `train_multitask.py` (`--domain sim|real`) and the per-run `event_lists.npz` outputs.

**Inputs:** existing models in `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/s3/mt/qlfm_M_{CL,C_xf}/`.

## Appendix A. Per-seed real-domain values

Seeds in the order listed.

### A.1 CL, finetune

| pretrain → downstream | seeds | T1 cw | T1 evt-median | T2 all | T2 bright |
|---|---|---|---|---|---|
| real → real | 0 | 0.9175 | 0.9203 | 0.9970 | 0.9934 |
| sim → real | 0, 1 | 0.9214, 0.9214 | 0.9242, 0.9241 | 0.9961, 0.9961 | 0.9949, 0.9930 |
| real → sim | 0, 1 | 0.6681, 0.6636 | 0.6209, 0.6230 | 0.9710, 0.9606 | 0.9594, 0.9571 |
| sim → sim | 0 | 0.5981 | 0.5213 | 0.9293 | 0.9002 |
| existing (sim) → sim | 0, 1, 2 | 0.6336, 0.6215, 0.6079 | 0.5853, 0.5744, 0.5529 | 0.9456, 0.9267, 0.9478 | 0.9360, 0.8915, 0.9131 |

### A.2 CL, frozen

| pretrain → downstream | seeds | T1 cw | T1 evt-median | T2 all | T2 bright |
|---|---|---|---|---|---|
| real → real | 0 | 0.8934 | 0.8920 | 0.9964 | 0.9882 |
| sim → real | 0, 1 | 0.8442, 0.8453 | 0.8330, 0.8349 | 0.9880, 0.9892 | 0.9575, 0.9630 |
| real → sim | 0, 1 | 0.7594, 0.7658 | 0.7138, 0.7213 | 0.9922, 0.9935 | 0.9680, 0.9737 |
| sim → sim | 0 | 0.4958 | 0.2434 | 0.9818 | 0.9471 |
| existing (sim) → sim | 0 | 0.4736 | 0.2431 | 0.9770 | 0.9363 |

### A.3 C + XF, finetune

| pretrain → downstream | seeds | T1 cw | T1 evt-median | T2 all | T2 bright |
|---|---|---|---|---|---|
| real → real | 0 | 0.8813 | 0.8880 | 0.9833 | 0.9633 |
| sim → real | 0, 1 | 0.8249, 0.8339 | 0.8155, 0.8291 | 0.9864, 0.9847 | 0.9604, 0.9532 |
| real → sim | 0, 1 | 0.8021, 0.8131 | 0.7871, 0.8090 | 0.9798, 0.9809 | 0.9505, 0.9537 |
| sim → sim | 0 | 0.7667 | 0.7298 | 0.9847 | 0.9565 |
| existing (sim) → sim | 0, 1, 2 | 0.8082, 0.8105, 0.8067 | 0.7910, 0.7924, 0.7871 | 0.9863, 0.9870, 0.9861 | 0.9612, 0.9607, 0.9594 |

### A.4 C + XF, frozen

| pretrain → downstream | seeds | T1 cw | T1 evt-median | T2 all | T2 bright |
|---|---|---|---|---|---|
| real → real | 0 | 0.7413 | 0.6903 | 0.9845 | 0.9484 |
| sim → real | 0, 1 | 0.6831, 0.6840 | 0.6098, 0.6103 | 0.9827, 0.9815 | 0.9359, 0.9320 |
| real → sim | 0, 1 | 0.6929, 0.7037 | 0.6111, 0.6323 | 0.9826, 0.9854 | 0.9506, 0.9528 |
| sim → sim | 0 | 0.6385 | 0.5361 | 0.9814 | 0.9367 |
| existing (sim) → sim | 0 | 0.6486 | 0.5506 | 0.9843 | 0.9461 |
