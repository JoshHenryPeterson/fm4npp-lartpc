---
title: a1_check_round_3_preplan.md

---

# A1 check, round 3: V-plane corrections, Domain shifts

This is a document to guide the development of a plan to execute a third round of checks related to data/simulation discrepancy in the MicroBooNE open data and the impact that it has on foundtion model performance.

## 1. Background

### File Locations

The repo holding current scripts and documentation is in this location in the Eagle filesystem: `/lus/eagle/projects/LArTPC_FM/`

The directory holding relevant documentation of past work is found in the following directory: `/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc/multi-modal-qlmatch/docs`

The following directory holds work outputs: `/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work`

Another work directory can be found here: `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work`
    
### Previous Rounds
    
Round 1 was run by yuhw on 2026-11-09 and documented in `paper-campaign.md`

Round 2 was run by jpeterso and documented in `a1_check_round2_plan.md`, `a1_light_check_round_1_plan.md`, and `technote_a1_round2.md`

## 2. Round 3 Requests

### 1. New Hit Width Correction Variant
For the charge width checks in round 2, the following corrections, labeled `F_MEAS_3D`, were used to correct the neutrino charge hit widths for each plane:

```python
F_MEAS_3D = {2: [1.000, 0.967, 0.941, 0.926, 0.915, 0.903],   # Y
             1: [1.000, 0.980, 0.954, 0.938, 0.942, 0.941],   # V
             0: [1.000, 0.960, 0.922, 0.907, 0.892, 0.880]}   # U
```

These values were normalized such that the the correction at the anode must be 1.  However, it was found that for the V plane that the correction never reached 1.  To evaluate the effect of the normalization, new checks should be done with the following corrections:

```python
F_MEAS_3D_NOVNORM = {2: [1.000, 0.967, 0.941, 0.926, 0.915, 0.903],   # Y
             1: [0.9, 0.882, 0.859, 0.844, 0.848, 0.847],   # V
             0: [1.000, 0.960, 0.922, 0.907, 0.892, 0.880]}   # U
```

For the charge checks in A1 round 2, found in section 2: Hit-width test (round 2) in `technote_a1_round2.md`, a new variant `w_meas3d_noVnorm` should be added using the `F_MEAS_3D_NOVNORM` corrections.

### 2. Cross Domain Checks

The goal of this check is to determine if the foundation model can overcome domain shift challenges.

For these checks the following should be used as the different domains:
| Domain | Definition |
|--------|------------|
| Simulation | Unaltered MicroBooNE public data |
| Real | MicroBooNE public data with `w_meas3d` variant corrections for neutrino charge hits and `l_combo80` variant corrections for light hits |

The following training variants should be performed for the T1 and T2 tasks:
* Train the qlfm-M CL and qlfm-M C+XF models with the simulation domain data for the pretraining and the real domain data for the downstream heads.
* Train the qlfm-M CL and qlfm-M C+XF models with the real domain data for the pretraining and the simulation domain data for the downstream heads.

The training variants listed above should be done once where the backbone weights are frozen before downstream training and once where the backbone weights are finetuned during downstream training.

The variants described above are summarized in the table below:
| Model | Pretraining Domain Data | Downstream Domain Data | Freeze or FineTune |
| ------ | ------- | ------ | ----- |
| qlfm-M CL | simulation | real | freeze |
| qlfm-M CL | simulation | real | finetune |
| qlfm-M C+FX | simulation | real | freeze |
| qlfm-M C+FX | simulation | real | finetune |
| qlfm-M CL | real | simulation | freeze |
| qlfm-M CL | real | simulation | finetune |
| qlfm-M C+FX | real | simulation | freeze |
| qlfm-M C+FX | real | simulation | finetune |

Once the variant models are trained their performance should be evaluated for the T1 and T2 tasks on real domain data.  The performance of the existing qlfm-M CL and qlfm-M C+XF models (those entirely trained on the unmodified MicroBooNE public data) should also be evaluated on the real domain data for T1 and T2 tasks.  Make sure to keep seperate event lists for pretraining, downstream training, and evaluation for each variant training.

If it isn't resource intensive, use a second downstream seed for the CL runs (same backbones).