# Experiment map, November 2026: every planned training and test, grouped by purpose

Status as of 2026-11-19. This covers Haiwang's work (the Polaris Claude session; issues #8–#13) and Joshua's A1 data/sim checks (rounds 2–3, `jpeterso/.../docs/a1_check_round_3_plan.md`).

Throughout:
- **S** ("sim", "sim-like") = the open-data overlay as published. Simulated ν hits are broader than the real cosmic hits around them.
- **D** ("real", "data-like") = the same events with the simulated ν hit widths scaled by Sunny's measured 3D data/sim factors. Joshua's "real" also scales the beam light (`l_comb80`).
- **T1** = ν hit tagging. **T2** = ν-event vs no-ν. **T3** = flavour. **T4** = energy. **T5** = light from charge.
- **CL** = qlfm-M with light in pretraining. **C+XF** = charge-only pretraining, light added at the head.

---

## Summary by category

| category | question it answers | items | status |
|---|---|---|---|
| **1. Measure the problem** | How much does the simulation mismatch move the scores, and where? | A1 (2D width test), A2 (Joshua's 3D, per-plane and light checks), A3 (V-plane variant), A4 (feature probes) | A1–A2 done, A3–A4 planned or queued |
| **2. Labels in the data-like domain** | Upper bound: what if we had labelled data-like events? | B1 (fix b), B2 (sim→real), B3 (real→real) | B1 done; B2–B3 planned by Joshua and queued by Haiwang |
| **3. Foundation-model paradigm** | Does pretraining in the data-like domain plus freezing stop sim labels teaching the width shortcut? | C1 (pretrain in D), C2 (real→sim, frozen/full), C3 (partial freezing, head-first), C4 (naive baseline), C5 (explicit invariance), C6 (self-training) | C3 running, rest queued or planned |
| **4. Validate on real data** | Do the gains hold on real beam-on data? | D1 (first check), D2 (all new models), D3 (round-2 data), D4 (real-data pretraining), D5 (fine-tune on WC-labelled real data), D6 (data scaling) | D1 done; D3 awaits approval |
| **5. SBND** | New detector, all-simulation testbed | E1 (10k tests), E2 (full run) | E2 running |
| **6. Paper** | What can we fairly claim? | F1 (width fairness for PTv3), F2 (ME1 headline) | proposed |

How the categories fit together:
- **The core of the plan is category 3.** Category 2 supplies its upper bound and category 4 checks it against real data.
- **Most overlap with Joshua's plan is in categories 2 and 3:** B2, B3, C1 and C2 are in both. The differences are that his data-like domain also corrects the light, and he pretrains on events kept apart from the labelled 1%. Merging them would save about 4–6 node-hours.

---

## Overview

| # | item | owner | purpose (category) | status | compute |
|---|---|---|---|---|---|
| A1 | Width/charge perturbation test (2D factors) | H | 1. Measure the problem | done (#9) | 1.6 nh |
| A2 | 3D factors, per plane, light corrections, T5 target (inference) | J | 1. Measure the problem | done (round 2) | ~1 nh |
| A3 | V-plane normalisation variant (`noVnorm`) | J | 1. Measure the problem | planned (round 3, part 1) | <1 nh |
| A4 | Feature probes: which layers carry the width cue | H | 1. Measure the problem | queued (#13) | <1 nh |
| B1 | Fine-tune with data-like or no width (fix b) | H | 2. Labels in D (reversed direction) | done (#9) | ~3 nh |
| B2 | "sim→real": pretrain S, labels in D | J + H | 2. Labels in D (reversed direction) | planned (J) / queued (H: oracle runs) | J ~4 nh, H ~1.5 nh |
| B3 | "real→real": pretrain D, labels in D | J + H | 2. Labels in D (control) | planned (J) / queued (H) | J ~1 nh, H ~1 nh |
| C1 | Pretrain in D | J + H | 3. FM paradigm | queued (H: 7862299) / planned (J) | 2 nh each |
| C2 | "real→sim": pretrain D, labels in S, frozen or full fine-tune | J + H | 3. FM paradigm | planned (J) / queued (H) | J ~4 nh, H ~2 nh |
| C3 | Partial freezing and head-first fine-tuning | H | 3. FM paradigm | running (S) / queued (D) | ~3 nh |
| C4 | Naive reference: pretrain S, labels S, test in D | J + H | 3. FM paradigm (baseline) | exists; evaluation planned | eval only |
| C5 | Explicit invariance (consistency loss, adversarial, feature removal) | H | 3. FM paradigm (if C1–C3 fail) | proposed (#13 phase 4) | ~5 nh |
| C6 | Self-training with pseudo-labels (high-confidence predictions on unlabelled D / real events become labels) | H | 3. FM paradigm | idea noted (#13 phase 4b) | ~3 nh |
| D1 | First real-data check (beam-on/EXT vs Wire-Cell) | H | 4. Validate on real data | done (#10) | <1 nh |
| D2 | Real-data check of every new model | H | 4. Validate on real data | planned (#13 phase 3; suggested for J) | <1 nh |
| D3 | MicroBooNE round 2: more beam-on/EXT, clean overlay | FNAL session | 4. Validate on real data | awaiting approval | FNAL side |
| D4 | Pretraining on real data | – | 4. Real data in training | future (after #13) | TBD |
| D5 | Fine-tune on real data with Wire-Cell labels (D-pretrained M CL; 90% train / 10% test, 10 folds) | H | 4. Real data in training | done 2026-11-20: not successful (both backbones) | ~1 nh |
| D6 | If D5 works: more WC-labelled real data, data-scaling curve | H / FNAL | 4. Real data in training | on hold (D5 not successful) | TBD |
| E1 | SBND 10k pipeline tests | H | 5. New detector | done (#11) | ~3 nh |
| E2 | SBND full: pretrain M CL and M C, 13 fine-tunes | H | 5. New detector | running (7862256) | ~11 nh |
| F1 | Fair width comparison: PTv3 with width, or no width anywhere | – | 6. Paper fairness | proposed (#12) | ~3 nh |
| F2 | Paper ME1 headline (#8) | H | 6. Paper | depends on 1–4 | – |

H = Haiwang (Polaris session), J = Joshua. "nh" = Polaris node-hours.

**Overlaps:**
- B2, B3, C1 and C2 are in both plans. They differ in the light correction (J includes `l_comb80`, H is width only) and the split (J pretrains on events disjoint from the labelled 1%; H reuses the standard pretraining set).
- One shared set of D backbones would save about 4–6 nh.

---

## Category 1. Measure the problem: how much does the simulation mismatch move the scores?

Inference only on already-trained models: perturb the inputs, re-evaluate, compare. This tells us *how big* and *where* the problem is, not how to fix it.

### A1. Width and charge perturbation test (2D factors), done (#9)
- **Purpose:** check Sunny's concern that the model scores by recognising "simulated-looking" ν hits.
- **Procedure:**
  - Scale simulated ν hit widths by the 2D data/sim factor, or charge by ±2/5%.
  - Re-evaluate T1/T2 of every trained model.
  - Controls: scale all hits, broaden instead of narrowing, apply a uniform 2% change.
- **Outcome:**
  - Charge: negligible.
  - Width: CL loses 0.21 in T1. C+XF loses 0.03. PTv3 is unaffected (it ignores width).
  - CL has learned a sharp width-vs-drift relation tuned to the over-diffused simulation.

### A2. 3D factors, per plane, light corrections, T5 target, done (Joshua, round 2)
- **Purpose:** refine A1 with Sunny's 3D factors, find which planes matter, and test light mismodelling.
- **Procedure:**
  - 3D width variants (all planes, Y only, U+V only), per-plane T1.
  - Light variants: 0.80 PE scale, slow component, late tail, combined `l_comb80`, prompt-only.
  - T5 target corrections.
- **Outcome:**
  - CL loses 0.29 in T1 with 3D factors (0.33 per-event median); U loses most. C+XF loses 0.04.
  - Light corrections: ≤0.002 for CL, 0.01 for C+XF. C+XF relies on late light, CL does not.
  - Light mismodelling explains 10–20% of the T5 ν/no-ν gap.

### A3. V-plane normalisation variant, planned (Joshua, round 3, part 1)
- **Purpose:** the V-plane factor has an unexplained 0.90 offset at the anode, normalised away in A2. This measures how much that choice matters.
- **Procedure:** inference with the V factors not renormalised (V widths 10% smaller everywhere), plus V-only variants. Six existing models.
- **Possible outcomes:**
  - A clearly larger drop than A2. That is likely, since a uniform 2% change already costs 0.01. Then the definition of D carries a real systematic, and conclusions must be checked under both definitions.
  - A small change. Then the definition of D is robust.

### A4. Feature probes, queued (#13, `qlfm.domain_probe`)
- **Purpose:** find *where* in the network the width cue lives, and whether data/sim-consistent features can be told apart from sim-only ones.
- **Procedure:** run the same events in S and D through a pretrained backbone. Per layer, measure:
  - how much the features change between S and D;
  - how well a linear probe tells S from D;
  - whether a ν/cosmic probe trained on S still works on D.

  Done for the S- and D-pretrained backbones (and optionally fine-tuned ones).
- **Possible outcomes:**
  - Width information concentrated in early layers. Then partial freezing of those layers (C3) or removing it (C5) is promising.
  - Width spread throughout. Then the representation itself must be trained to be invariant (C5).
  - The D backbone carries less domain information than the S backbone. That would be evidence that pretraining in D helps.

---

## Category 2. Labels in the data-like domain ("reversed" direction)

Train the task heads on D labels. This is possible here only because we can *create* labelled D events. With real data there are no labels, so these runs serve as **upper bounds and controls**, not as a recipe for real data.

### B1. Fine-tune with data-like or no width ("fix b"), done (#9)
- **Purpose:** first check of whether retraining with corrected widths restores performance.
- **Procedure:** S-pretrained CL and C+XF, fine-tuned and tested with D widths (2D factors) or with width removed. 1% labels, 3 seeds.
- **Outcome:**
  - With D widths, CL recovers T1 0.914 (vs 0.922 nominal).
  - With width removed, the CL lead disappears (0.829, PTv3's level).
  - On real beam-on data (D1), D-width training recovers only part of the drop.

### B2. "sim→real": pretrain S, fine-tune with labels in D, planned (J) / queued (H)
- **Purpose:** upper bound when target-domain labels exist. Joshua's version adds the light correction; H's oracle runs use 3D width only.
- **Procedure:**
  - J: CL and C+XF, frozen and fine-tuned, 2 seeds, on a disjoint split.
  - H: CL full fine-tune with 3D widths, 3 seeds (`dshift/ft/PS_oracleD`).
- **Possible outcomes:** the score near the nominal one (as in B1) sets the "oracle" for the recovery fraction in category 3. A frozen S backbone doing much worse than a fine-tuned one shows the S features themselves encode the shortcut.

### B3. "real→real": pretrain D, labels in D, planned (J) / queued (H)
- **Purpose:** the fully in-domain reference, everything in D.
- **Procedure:**
  - J: CL and C+XF, frozen and fine-tuned, 1 seed.
  - H: CL full fine-tune, 3 seeds (`PD_oracleD`).
- **Possible outcomes:**
  - If B3 is well above B2, pretraining in the target domain helps even with target labels.
  - If they are equal, only the labels matter.

---

## Category 3. The foundation-model paradigm: pretrain in D, freeze, fine-tune on sim labels

This is the realistic setting. Unlabelled data exists in the target domain; labels exist only in simulation. The question is whether pretraining in D, and how much of the backbone is frozen, keeps the sim labels from teaching the sim-only width shortcut.

### C1. Pretraining in D, queued (H: job 7862299) / planned (J)
- **Purpose:** the D-domain backbone that C2/C3/B3 start from.
- **Procedure:** same recipe and steps as the existing S backbone (34k steps, about 2 nh), with simulated ν widths scaled inside pretraining.
  - J: CL and C on a disjoint split; CL also with corrected light.
  - H: CL only, width only.
- **Possible outcomes:** pretraining loss similar to S (the domain is not harder). Feature probes (A4) show whether it encodes width differently.

### C2. "real→sim": pretrain D, labels in S, frozen or full fine-tune, planned (J) / queued (H)
- **Purpose:** the central test of the FM paradigm.
- **Procedure:** fine-tune on S labels, select checkpoints on S validation (as with real data), test in D.
  - J: CL and C+XF, frozen and fine-tuned, 2 seeds.
  - H: CL, frozen and full, 3 seeds (`PD_frozen`, `PD_full`).
- **Possible outcomes:**
  - **Frozen recovers, full does not:** the D backbone is right, but full fine-tuning re-learns the shortcut. Use frozen or partial freezing.
  - **Both recover:** pretraining in D is enough.
  - **Neither recovers:** freezing doesn't remove width information. The head learns "unusual width = ν" from the frozen features, so explicit invariance (C5) is needed.

### C3. Partial freezing and head-first fine-tuning, running (S) / queued (D) (H only)
- **Purpose:** fill the gap between "frozen" and "full fine-tune". Published results disagree on which is more robust.
- **Procedure:** from both S and D backbones, labels in S, 3 seeds:
  - freeze the embeddings + the first 6 of 12 blocks;
  - train only the heads for 2,000 steps, then everything.
- **Possible outcomes:** identifies the least-damaging adaptation. Combined with A4, it shows whether freezing exactly the layers that carry width helps.

### C4. Naive reference: pretrain S, labels S, test in D (existing models)
- **Purpose:** the "do nothing" baseline that C2/C3 must beat.
- **Procedure:** the existing paper-campaign models (CL and C+XF, frozen and fine-tuned), evaluated in D. Both plans include it.
- **Outcome:** known approximately from A1/A2 (CL T1 about 0.63–0.71 in D).
- **Main metric for category 3:** recovery fraction R = (score − naive) / (oracle − naive), with naive = C4 and oracle = B2/B3.

### C5. Explicit invariance, proposed (#13 phase 4, only if C2/C3 fall short)
- **Purpose:** make the model *unable* to use the sim-only cue, rather than hoping freezing does it.
- **Procedure**, unlabelled D events used during fine-tuning:
  - a paired S/D consistency loss;
  - domain-adversarial training or feature alignment;
  - projecting out the width-sensitive directions found in A4.
- **Possible outcomes:**
  - A recipe that also works with real data (the unpaired variants).
  - Or evidence that the remaining gap is not width, which points to D4.

### C6. Self-training with pseudo-labels, idea noted (#13 phase 4b)
- **Purpose:** use the unlabelled target-domain events directly. Start from a small labelled sample, predict on unlabelled D (later real) events, and add the high-confidence predictions as labels; iterate.
- **Procedure:**
  - 1% S labels plus a D pool, about 3 rounds of predict → threshold (class-balanced, calibrated) → retrain.
  - Pseudo-label precision is measured in D (truth known) before the method is used on real data.
  - On real data, Wire-Cell agreement serves as a second filter.
- **Possible outcomes:**
  - A practical path to real-data adaptation without real labels.
  - Or confirmation bias. The S-trained model is confidently wrong on real ν hits, which argues for starting from the D-pretrained or width-free model.

## Category 4. Validation on real data

Everything in categories 1–3 uses a *modelled* data domain. These items check it against real MicroBooNE data.

### D1. First real-data check, done (#10)
- **Purpose:** do overlay-trained models find real neutrinos?
- **Procedure:** 11 existing models on beam-on (849 events) and EXT (1,248) with Wire-Cell labels. Measured:
  - agreement with Wire-Cell ν hits;
  - beam-on pass rate at a fixed EXT rate (cosmic background held at Wire-Cell's 2.2%).
- **Outcome:**
  - Overlay: every model beats Wire-Cell and Pandora.
  - Real data: Wire-Cell wins (8.2% beam-on pass rate). CL finds 18–24% of Wire-Cell's ν charge (90% on overlay).
  - The drop follows each model's reliance on width: data-like widths 0.35–0.39, no width 0.63.

### D2. Real-data check of every new model, planned
- **Purpose:** test whether "robust in D" also means "better on real data", i.e. whether D is a good stand-in.
- **Procedure:** run `eval_external` (about 1 GPU-minute per model) on all C2/C3/B2/B3 models; suggested for Joshua's runs too.
- **Possible outcomes:**
  - D-robust models also improve on beam-on: the modelled shift is a good proxy.
  - They don't: other mismodelling (charge shape, simulated light, noise) matters, and real-data pretraining (D4) is necessary.

### D3. MicroBooNE round 2 production, awaiting approval (FNAL session, #10)
- **Purpose:** more statistics for D1/D2 (only 70 Wire-Cell-selected beam-on events today), and clean overlay events not in our training set.
- **Procedure:** produce more beam-on/EXT; overlay from public files missing from the published HDF5 (BNB) and the two unused νe files.
- **Possible outcomes:** firmer real-data numbers, and a real-data pool large enough for fine-tuning with Wire-Cell labels.

### D5. Fine-tune on real data with Wire-Cell labels, done 2026-11-20: not successful
- **Purpose:** use the Wire-Cell-labelled real data directly as a training set. Does a model fine-tuned on real events reproduce Wire-Cell's ν selection on held-out real events, where simulation-trained models found only 9–34% of Wire-Cell's ν charge?
- **Procedure:**
  - Events: the 2,097 real beam-on/EXT events with Wire-Cell output (98 Wire-Cell ν candidates).
  - Labels: T1 = hit in WC's in-time main cluster of a WC generic-ν event; T2 = WC generic ν (`train_multitask --labels wc`).
  - 10 folds, stratified by the WC decision, each 90% train / 10% test (`scripts/make_wc_folds.py`).
  - Full fine-tune of the D-pretrained qlfm-M CL (main arm) and of the S-pretrained one (control), T1 + T2 + T5.
  - Test: alignment with Wire-Cell on the held-out fold, compared with Pandora and with the simulation-trained models on the same events.
- **Possible outcomes:**
  - High alignment: a path to real-data adaptation with Wire-Cell as labeller. D6 then measures how it scales with more data.
  - Low alignment with only ~88 ν events per fold: the sample is too small. D6 is needed to say more.

### D6. Data scaling with more Wire-Cell-labelled real data, planned
- **Purpose:** if D5 works, measure how alignment grows with the number of labelled real events.
- **Procedure:** curate more WC-labelled beam-on/EXT (MicroBooNE round 2, D3). Repeat D5 at several training-set sizes.

### D4. Pretraining on real data, future
- **Purpose:** the end goal of the FM paradigm.
- **Procedure:** pretrain on beam-on/EXT (needs many more events and clean-up of run-dependent detector conditions), then fine-tune on simulation as in C2/C3.
- **Possible outcome:** depends on what categories 3 and 4 show. They tell us which freezing and fine-tuning recipe to use.

---

## Category 5. New detector and scale: SBND (#11)

### E1. SBND 10k pipeline tests, done
- **Purpose:** check that the SBND files contain everything qlfm needs, and that MicroBooNE pretraining transfers.
- **Outcome:**
  - The pipeline passes end to end.
  - MicroBooNE-pretrained CL fine-tuned on SBND: T1 0.85 vs 0.60–0.70 from scratch.
  - Wire-Cell ν clusters: 0.91.

### E2. SBND full sample, running
- **Purpose:** a full-statistics SBND foundation model and an all-simulation testbed (no overlay, so no sim/real mixing inside events).
- **Procedure:**
  - Pretrain M CL (running, job 7862256) and M C (queued) on 738k events.
  - 13 fine-tunes at 1% labels, T1–T5:
    - SBND-pretrained CL fine-tuned ×3, frozen ×1;
    - C+XF ×3;
    - MicroBooNE-pretrained ×3;
    - scratch ×3.
- **Possible outcomes:**
  - Whether an SBND-pretrained model beats MicroBooNE transfer and scratch.
  - Whether it beats Wire-Cell on T1 (the 10k test had Wire-Cell ahead).
  - T3 becomes a real 3-class test with the νe CC sample.

---

## Category 6. Paper fairness and claims

### F1. Fair width comparison, proposed (#12)
- **Purpose:** only qlfm reads hit width, so "qlfm beats PTv3" partly measures width use.
- **Procedure:** add width as an input to PTv3 (and others), or remove width from all models.
- **Possible outcomes:** a fair architecture comparison. If PTv3 with width gains as much as qlfm, the architecture claim weakens. Partly answered already: qlfm without width ≈ PTv3 (B1).

### F2. Paper ME1 headline (#8)
- **Purpose:** decide what the paper claims.
- **Depends on:** whether the T1/T2 "light helps" result holds in D (category 3) and on real data (category 4). The choice slide in `status-2026-11-13.pdf` lists the options:
  - A: open data only, simulated shift, with a caveat;
  - B: plus real data with Wire-Cell labels;
  - C: SBND simulation only.

---

## Decisions pending
1. **Merge the overlapping parts of the two plans** (B2, B3, C1, C2): one shared set of D backbones, a common split convention, and a common evaluation (including D2).
2. **D definition:** width only (H), or width + light (J). The light effect is ≤0.002 for CL, so this matters mainly for C+XF.
3. **Pretraining split for the D backbones:** standard set (H, matches the existing S backbone) or disjoint from the labelled 1% (J, cleaner, `pre_r3`).
4. **MicroBooNE round-2 production (D3), and F1.**
