# Domain-shift study for the foundation model (simulated shift, MicroBooNE overlay)

Tracking issue: #13. Related: #9 (Sunny's width check), #10 (real-data check), #12 (tracker), status slides `status-2026-11-13.pdf`.

## 1. Goal

Test the usual foundation-model (FM) recipe for a data/simulation mismatch:

1. **Pretrain** self-supervised in the domain we care about (data-like).
2. **Freeze** all or part of the backbone.
3. **Fine-tune** on labelled simulation (sim-like).
4. **Evaluate** in the data-like domain.

We also want to learn whether the representation can tell *data/sim-consistent* features from *sim-only* features, and how.

This round uses a **simulated** domain shift on the open-data overlay, where we control the shift exactly. It prepares for pretraining on real MicroBooNE data (beam-on/EXT), which needs more clean-up first (section 8).

## 2. Domains

The overlay has two kinds of hits:
- **ν hits** (signal) are simulated.
- **Cosmic hits** (background) are real data.

The known mismatch is hit width. Simulated ν hits broaden faster with drift than real charge does: data/sim width is about 0.88–0.94 at full drift, depending on the plane (#9).

| domain | ν-hit widths | signal vs background width | role |
|---|---|---|---|
| **S, sim-like** | nominal simulation (raw overlay) | very different | labelled simulation as we have it |
| **D, data-like** | simulated ν widths × Sunny's measured data/sim factor (per plane and drift) | similar, as in real data | stands in for real data |

- The transform from S to D touches only hits with a simulated ν deposit. The truth is used to *build* the domain, never as a label.
- S and D hold the **same events**, so every event exists in both domains: a paired counterfactual. Real data cannot offer this. It is the main reason to start with a simulated shift.
- **Known caveat:** D corrects only the width. Other differences remain between simulated ν and real data: the ν light is simulated, plus noise and other hit-shape details. The real-data check (#10) measures what is left.

## 3. What we have: the "reversed" study

So far we pretrained in S and adapted to D. qlfm-M CL, 1% labels, T1 hit F1.

| pretrain | fine-tune labels | test domain | fine-tuned | frozen | source |
|---|---|---|---|---|---|
| S | S | S | 0.92 | 0.89 | s3/mt |
| S | S | **D** | **≈0.71** (−0.21) | ≈0.56 (−0.33) | #9 perturbation test |
| S | D | D | 0.91 | 0.85 | #9 fix (b) |
| S | S | real beam-on (agreement with WC ν hits) | 0.18–0.24 | – | #10 |
| S | D | real beam-on | 0.35–0.39 | – | #10 |

Notes on this table:
- The table uses the 2D factors (#9). Sunny's newer 3D factors are not applied yet.
- The "test in D" rows for labels in S are the nominal score plus the #9 perturbation Δ. That Δ was measured on charge-weighted T1 F1, so these two entries are approximate.
- Labelled data from the target domain (S → D) recovers the simulation score. That only works because we can *make* labelled D events. For real data we cannot. The standard FM argument runs the other way.

## 4. The FM hypothesis, and why it is not automatic

**Hypothesis.** Pretraining on unlabelled target-domain events (D) learns features calibrated to D: for example, the width-vs-drift relation of real charge. Then:
- freezing those features keeps sim labels from re-teaching sim-only cues;
- fine-tuning only a head (or a few layers) on sim labels learns the task from the data-consistent features.

**Caution, part of what we test.** Freezing does not remove information.
- The MAE reconstructs every hit feature, width included, so a D-pretrained backbone still encodes width.
- During fine-tuning on S, the simulated ν hits are *off-manifold* for that backbone: broader than anything it saw. A head could learn "looks unusual" as the ν cue, which is the shortcut again in another form.
- Results on the robustness of linear probing vs fine-tuning point both ways. Full fine-tuning can distort pretrained features and hurt out-of-distribution accuracy, while a re-trained last layer can drop spurious features when the training data allow it (refs 1, 3). So we must measure, not assume.

**How a model can tell the two kinds of features apart.** These are the methods we test:
- **(a) Paired counterfactual sensitivity.** Run the same event in S and D. Features that change a lot are width-sensitive (sim-only cues); features that stay put are data/sim consistent. This works per layer, per dimension or per direction (principal components of the S−D difference). It needs paired events, so it exists only for a simulated shift.
- **(b) Domain probe.** Train a small classifier to tell S from D (or sim from real) from frozen features, layer by layer. Directions it uses carry domain information; directions useful for the task but not for the probe are what we want the head to use. This works without pairs, so it carries over to real data.
- **(c) Invariance during fine-tuning** with unlabelled target events. Add a penalty that makes the features of S and D (or sim and real) indistinguishable, or makes predictions agree:
  - paired consistency loss (S ↔ D);
  - domain-adversarial training (DANN, ref 4);
  - CORAL or MMD feature alignment (ref 5);
  - pivoting against a nuisance (ref 6) — here the nuisance is width, so this is a "decorrelate from width" objective.

## 5. Plan

### Phase 0: code (no compute)
- [x] **Data-like definition:** add Sunny's 3D per-plane factors (`F_MEAS_3D`, #9) as the D transform. Keep the 2D factors for comparison.
- [x] **D in pretraining:** `train_pretrain --width meas3d` applies the transform inside the MAE forward pass, as `train_multitask --width` does. This needs the per-hit ν deposit in the pretraining loader, used only to pick which hits to transform.
- [x] **Fine-tune modes** in `train_multitask`:
  - `--freeze-blocks k`: freeze the hit/light embeddings and the first k Transformer blocks; train the rest and the heads;
  - `--lp-ft N`: train the heads on the frozen backbone for N steps, then unfreeze;
  - optional `--l2sp λ`: penalise drift from the pretrained weights (ref 2). Not implemented yet.
- [ ] **`qlfm/domain_probe.py`:** paired S/D feature sensitivity per layer (method a) and a per-layer domain probe (method b).

### Phase 1: pretraining in D (about 2 node-h)
- [ ] qlfm-M CL pretrained on D: same recipe and steps as the existing S checkpoint (`pre/m_cl_f0.90`, 34k steps × 256 events). Only the domain differs.
- [ ] Optional: qlfm-M C on D (charge only; less width-dependent in #9), about 2 node-h.

### Phase 2: fine-tune matrix (about 5 node-h)
Multi-task T1–T5, 1% labels, 6,000 steps, 3 seeds. Fine-tune labels in **S** unless noted.

| pretrain | fine-tune mode | runs | note |
|---|---|---|---|
| S | frozen / freeze-blocks 6 / LP-FT / full | 3 each (some exist) | reference for each mode |
| **D** | frozen / freeze-blocks 6 / LP-FT / full | 3 each | **main question** |
| D | full, labels in D | 3 | oracle (upper bound) |
| S | full, labels in D | 3 | oracle, existing (2D factors; rerun if 3D) |

About 23 new runs.

### Phase 3: evaluation and diagnostics (about 2 node-h)
For every run:
- T1–T5 in **D** (main) and in **S** (reference);
- **shortcut sensitivity:** ΔT1 and ΔT2 when the test events switch S → D (`eval_perturb`);
- **real-data check (#10):** agreement with Wire-Cell ν hits on beam-on, and the beam-on pass rate at EXT = 2.2% (`eval_external`).

Diagnostics on the S- and D-pretrained backbones: paired sensitivity and domain-probe curves per layer.

### Phase 4: explicit feature selection, only if Phase 2/3 shows freezing alone does not recover (about 4–6 node-h)
- [ ] Paired S/D consistency loss during fine-tuning (method c, needs pairs).
- [ ] DANN or CORAL with unlabelled D events (method c, unpaired). This is the version that carries over to real data.
- [ ] Freeze or project out the most width-sensitive directions found in Phase 3, then fine-tune.

### Phase 4b: self-training with pseudo-labels (idea noted 2026-11-19, not scheduled)
Haiwang's idea:
1. Start from a small labelled sample.
2. Predict on unlabelled events of the target domain.
3. Add the high-confidence predictions as labels.
4. Retrain, and iterate.

This is self-training or pseudo-labelling (refs 9–11). It fits the case "labels only in simulation, plenty of unlabelled data": D here, real beam-on/EXT later.
- **Procedure (simulated shift first):**
  - Labels: the 1% S labels.
  - Unlabelled pool: D events from the train split.
  - Each round: train → predict on the pool → keep hits/events with p > τ (or the top x% per class, so ν and cosmic stay balanced) → retrain on S labels + D pseudo-labels.
  - About 3 rounds, compared with C2/C3 by the recovery fraction.
  - In D the truth is known, so the precision of every pseudo-label round can be measured before the method is trusted on real data.
- **Main risk: confirmation bias.**
  - The S-trained model is *confidently wrong* in exactly the way that matters. On beam-on, the S-backbone frozen model finds only 9% of Wire-Cell's ν charge, and is likely sure that real ν hits are cosmics.
  - Pseudo-labels from it would reinforce the width shortcut.
- **Mitigations:**
  - start from the most robust model (D-pretrained, or width-free);
  - class-balanced and calibrated thresholds (calibrated on D validation);
  - a teacher–student (EMA) scheme with strong augmentation, including width jitter;
  - event-level consistency (hit labels consistent with the T2 prediction);
  - on real data, agreement with Wire-Cell as a second, independent filter (co-training).
- **Cost:** about 3 node-h per 3-round loop in D.

## 6. Metrics and success criteria
- **Main:** T1 F1 in D for (pretrain D, labels S) vs (pretrain S, labels S), in the same fine-tune mode.
- **Recovery fraction:** R = (score − naive) / (oracle − naive).
  - naive = pretrain S, labels S, tested in D;
  - oracle = labels in D.
- **Shortcut sensitivity:** |ΔT1| under S → D at test, compared with the test-set 95% interval. Aim: within a few times the interval.
- **Real data:** beam-on agreement with WC ν hits. Today it is 0.18–0.39 for width-reading models and 0.56–0.63 for width-free ones; overlay level is about 0.8–0.9. Also the beam-on excess at fixed EXT rate (WC: 8.2% vs 2.2%).
- **No large loss in S.**

What would support the FM hypothesis: pretraining in D with a frozen or partly frozen backbone gives a clearly higher R and a smaller ΔT1 than pretraining in S in the same mode, and the real-data agreement moves toward the overlay level.

## 7. Budget and order
- **Phases 0–3:** about 9 node-h. **Phase 4:** about 5 node-h. **Total:** about 14 node-h (to be approved).
- **Queues:** preemptable for pretraining, debug for fine-tunes (as for SBND).
- **Order:** Phase 0 code can start now. Compute follows the full SBND run (#11), or runs in parallel on preemptable if approved.

## 8. Toward real-data pretraining
The simulated shift says *which adaptation recipe* works. The next step is pretraining on real beam-on/EXT data (#10). That needs:
- **More events.** Round 1 has about 2k real events; MicroBooNE pretraining used about 750k.
- **Clean-up of information unrelated to physics:**
  - run- and epoch-dependent detector conditions: dead or noisy channels, gain, electron lifetime, PMT gains and timing;
  - trigger and software-filter differences between beam-on and EXT;
  - so the model does not learn "which run" instead of physics.
- **Mixing:** decide how to mix real and overlay events in pretraining.

## 9. Decisions (owner, 2026-11-19)
1. **D** = Sunny's 3D factors (`--width meas3d`). The oracle runs (labels in D) are redone with them.
2. **Compute:** run alongside the full SBND run (#11) on the preemptable queue.
3. **Model:** qlfm-M CL only.

**Implementation choices:**
- freeze-blocks = 6 (embeddings + 6 of 12 blocks frozen);
- LP-FT = 2,000 head-only steps of the 6,000;
- runs with labels in S train and select in S (no labelled D, as with real data). Their test in D comes from `eval_perturb --variants w_meas3d` (T1/T2) and the real-data check from `eval_external` (#10).

Reused runs: pretrain S, full fine-tune s0–2 (`s3/mt/qlfm_M_CL/finetune`) and frozen s0 (`s3/mt/qlfm_M_CL/frozen`).

**Jobs:**
- D pretraining: 7862299.
- Pretrain-S fine-tunes (11): 7862302.
- Pretrain-D fine-tunes (15): submitted by `$QLFM_WORK/jobs/dshift_chain.sh` after the D pretraining.

Outputs: `$QLFM_WORK/dshift/{pre,ft}/`.

## References
1. Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (LP-FT), arXiv:2202.10054.
2. Li, Grandvalet, Davoine, *Explicit Inductive Bias for Transfer Learning with Convolutional Networks* (L2-SP), arXiv:1802.01483.
3. Kirichenko, Izmailov, Wilson, *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*, arXiv:2204.02937.
4. Ganin et al., *Domain-Adversarial Training of Neural Networks*, arXiv:1505.07818.
5. Sun, Saenko, *Deep CORAL: Correlation Alignment for Deep Domain Adaptation*, arXiv:1607.01719.
6. Louppe, Kagan, Cranmer, *Learning to Pivot with Adversarial Networks*, arXiv:1611.01046.
7. Lee et al., *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts*, arXiv:2210.11466.
8. Shen et al., *Connect, Not Collapse: Explaining Contrastive Learning for Unsupervised Domain Adaptation*, arXiv:2204.00570.
9. Lee, *Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks*, ICML 2013 workshop.
10. Xie et al., *Self-training with Noisy Student improves ImageNet classification*, arXiv:1911.04252.
11. Sohn et al., *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*, arXiv:2001.07685.
