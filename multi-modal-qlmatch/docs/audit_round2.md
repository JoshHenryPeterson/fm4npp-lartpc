# qlfm architecture critique

This document reviews the design choices in the qlfm model. It covers the data, tokenization, backbone, pretraining, downstream heads, and evaluation. It focuses on decisions that are wrong or weak and on places where the code, the scripts, and the docs disagree, and it suggests a fix for each.

- **Background:** read `docs/qlfm_architecture_summary.md` first for how the model works.
- **Prior audit:** `docs/audit-round1.md` covers some of the same ground. This critique cites it where findings overlap and does not repeat its smaller items.
- **Method:** the review is based on reading the code and docs only; nothing was run. Severity is a judgment of how much each issue could change the reported conclusions.
- **References:** files are named by module and function rather than line number.

---

## Summary of priorities

| # | Issue | Severity | Cost to fix or test |
|---|---|---|---|
| 1 | Downstream step budget is confounded with label fraction; code defaults ≠ the runs actually used | **High** (for the label-scaling claims) | Low |
| 2 | Comparisons between qlfm (charge + light) and the references are not like-for-like | **High** (for the "best FM" claim) | Low to medium |
| 3 | U/V/Y planes share a single "wire-cm" coordinate; there is no cross-plane geometry | Medium | Medium |
| 4 | T1 per-hit resolution is limited by the 16×64 cells and the hard token→hit lookup | Medium | Medium |
| 5 | Per-batch target standardization in the MAE; the light loss is on a different scale | Medium | Low |
| 6 | Pretraining LR and compute were not tuned symmetrically (MAE vs SD vs references) | Medium | Medium |
| 7 | T2 drops the hard "gap" events; label thresholds are inconsistent | Medium | Low |
| 8 | bf16 logits feed high-AUC metrics; single seeds; no confidence intervals | Medium | Low |
| 9 | Smaller code-hygiene issues (final norm, drop-path, sampler, defaults, token cap) | Low | Low |

**Design choice, not an issue:** the MAE decoder gets each masked cell's charge-weighted centroid as its position (§1.1). It is listed here so readers know it was deliberate.

---

## 1. Pretraining objective

### 1.1 Design choice: the decoder gets the masked cell's centroid

This is a deliberate design choice, not a defect. It follows the Point-MAE / PoLAr-MAE convention that a patch's position is its centre on the data. Round 1 chose it to fix an MAE that could not beat the mean baseline (`docs/round1-log.md`; `docs/technote-round1.md` §3.3).

**What the code does.**
- `MAE.forward` (`models/fm.py`) adds `dec_pos_c(fm.charge.pos_embed(ct))` to every decoder token, masked or visible.
- With `pos_mode="centroid"`, that embedding includes the cell's charge-weighted centroid (`tok_cpos`) and its offset within the cell (`tok_crel`). Both are computed from the cell's own hits.

**What this implies for interpreting the results.**
- Because the centroid is the first moment of the target charge distribution, the MAE is asked to predict *how much* charge is in the cell and its local shape, given *where* the charge sits. It is not asked to predict where tracks are.
- The round-1 improvement from 0.97–0.99 to 0.61 × the mean baseline therefore partly reflects this extra input. The count and charge stats (0.52) are a cleaner measure of contextual prediction.
- Keep this framing in the paper's description of the objective.

**Optional ablation (only if the stronger objective is wanted).**
- Give masked decoder tokens only cell-centre positions, or predict `tok_crel` as an extra target.
- Compare downstream results with the current choice.

### 1.2 Per-batch target standardization — Medium

**What the code does.**
- `target_norm=True` standardizes each of the 34 charge-target dimensions over the masked tokens *of the local GPU batch* (`tgt - mean) / (std + 1e-3)`).
- Most sub-grid dimensions are zero most of the time, so the rare non-zero corner bins get large weights.

**Why it matters.**
- The loss scale depends on the batch, so losses are not comparable between batches or between train and validation.
- With very few masked tokens the statistics are unstable; the audit already notes the NaN case with one token.

**Fix.** Compute a fixed mean and std per dimension once from the training set and store them as buffers. Alternatively, standardize with a running estimate.

### 1.3 The light loss weight is arbitrary — Medium

**What the code does.**
- `loss_c` is on standardized targets, so it is O(1).
- `loss_l` is raw MSE on mostly-zero `log1p(PE)` bins.
- They are summed as `loss_c + 1.0 · loss_l`. About 20% of events contribute no light loss, because their light masking mode is "none".

**Why it matters.** No sweep of `light_weight` is recorded, so the balance between the two modalities was never tested.

**Fix.** Standardize the light targets the same way as the charge targets, with fixed statistics. Then sweep `light_weight` ∈ {0.3, 1, 3} and report both downstream T5 and the event-task results.

### 1.4 Strip masking gives an uneven realised mask ratio — Low-Medium

**What the code does.** `charge_mask` draws Bernoulli(0.3) per 256-tick strip per event, then adds random token masking.

**Why it matters.**
- Overlay cosmics spread charge over most of the window, so the mask ratio averaged over the whole event is reasonable.
- The neutrino interaction occupies only a few strips, though. It is therefore often either entirely masked or entirely visible.
- That gives a high-variance training signal on exactly the region the downstream tasks care about.

**Fix.** Mask a fixed number of strips per event, i.e. a quota, not independent coin flips. Optionally guarantee that each event keeps part of its highest-charge region visible.

---

## 2. Data, tokenization, and embeddings

### 2.1 One wire coordinate for three differently oriented planes — Medium

**What the code does.**
- The wire coordinate is `wire · 0.3 cm` on every plane, measured from wire 0 of each plane. U and V are at ±60° and Y is vertical, so the same number means a different direction on each plane.
- The only per-plane information is an additive `Embedding(3, d)`.
- The drift coordinate is the only axis the planes share.

**Why it matters.**
- The encoder cannot directly check that three 2D views are consistent, which is the main physics advantage of a multi-plane LArTPC.
- It has to learn from data a geometry that is known exactly.

**Fixes.**
- Map each (plane, wire) to its known projected coordinate, using the wire angle and offset, so that Y-plane wire-cm ≈ z.
- Alternatively, encode the wire *line* in (y, z) for each token, e.g. Fourier features of the wire's position along its normal direction plus its angle.
- Either way, U, V, and Y tokens would then share a frame with each other and with the PMT (y, z) positions.

### 2.2 Cell size and T1 per-hit resolution — Medium

**What the code does.**
- A cell is 16 wires × 64 ticks, i.e. 4.8 cm × 3.5 cm (slightly anisotropic) and 32 µs.
- T1 gives every hit the *same* contextual feature: `z[hit2tok]` is a hard lookup with no interpolation.
- Within-cell discrimination comes only from `h_hit`, a 4→64→128 MLP. It sees one hit's in-cell offsets, charge, and width, and never the neighbouring hits.

**Why it matters.** A ν track and a crossing cosmic in the same cell cannot be separated except through `h_hit`. This sets an upper bound on T1 that the backbone cannot overcome.

**Fixes.**
- Replace the hard lookup with feature propagation, e.g. inverse-distance interpolation from the k nearest token centroids on the same plane. Alternatively, add a light per-hit refinement layer: hits cross-attend to the tokens in the neighbouring cells.
- Ablate the cell size (e.g. 8×32, 4×16) for T1 specifically. Report the number of tokens per event and the throughput cost.

### 2.3 The token cap: random, nondeterministic, and different between stages — Low-Medium

**What the code does.**
- Events above the cap lose randomly chosen cells, with no priority.
- The cap is 4096 in pretraining and in T5 zero-shot, but 8192 in `train_downstream.py` (`--max-tokens` default). The audit already notes this mismatch.
- `FM.forward` calls `tokenize` with `generator=None`, so the dropped cells are re-drawn on every evaluation.
- Dropped hits get logit −20 and **remain in the T1 metric as predicted negatives**.

**Why it matters.**
- It affects ~0.2% of BNB events according to `round1-log.md`, measured on 3 shards. It was never measured on the νe sample, which has busier topologies.
- Random dropping can remove ν cells as easily as cosmic cells.

**Fixes.**
- Use one cap value everywhere.
- When dropping, keep the highest-charge cells, or the cells closest to the flash.
- Seed the generator at evaluation time.
- Log `frac_hits_dropped` per run and per backbone; PoLAr-MAE's unmatched hits are treated the same way.

### 2.4 Labels and task definitions — Medium

**Thresholds are inconsistent.**
- "Visible" means ≥30 neutrino hits on the **collection plane**, while "no ν" means ≤5 neutrino hits over **all planes**.
- Two different definitions of a ν hit are used: edep > 0 for visibility, and > 0.05 MeV for T1. The audit already notes this.
- None of these thresholds has a sensitivity study.

**T2 is idealized.**
- In-AV events in the 6–29-hit gap are dropped from both training and test, so T2 tests only the easy, well-separated cases.
- In an overlay sample every event contains a neutrino. Class 0 therefore means "the ν left almost no charge", not "a cosmic-only event", and the docs should say so.

**`nu_vtx` vs `nu_vtx_corr`.**
- The labels use the true vertex. That is correct for "interacted in the AV".
- With a 3 cm FV margin and space-charge distortions of up to ~10 cm, though, the observed charge may not be contained.
- The choice should be documented, and its effect on T3/T4 checked.

**Class priors depend on the sample mix.**
- The νe sample is enriched in νe, so T3 class priors and T4 energy spectra depend on the bnb : nue mix.
- Report the priors, and report T3 at physical priors.

**Fixes.**
- Report T2 *including* the gap events, either as a fourth class or as a calibration curve.
- Vary the 30- and 5-hit thresholds.
- Unify the ν-hit definition.

### 2.5 PMT geometry map is unvalidated — Low

`geometry.PMT_YZ` comes from external repos and has no test.

**Fix.** In νe CC events, check that per-PMT PE correlates with the true vertex z. A permuted channel map would show up immediately.

### 2.6 Minor data issues — Low

- **Oversized time constants.** The time grid is sized for 8192 ticks, but only 6400 are stored (in `tokenize`, `fm.py` `n_cells`, and strip masking). This only over-allocates, but derive the constants from `geometry.N_TICKS`. (This corrects the caveat in the architecture summary: it is harmless.)
- **Ophit drop rate is not measured.** The ~1% out-of-window ophit rate is a code comment, not a measurement. Record the rate in `meta.json`. Note that it also removes pre- and post-window light that carries cosmic information.
- **Too many Fourier frequencies.** The `ChargeEmbed` and `LightEmbed` constructors default to 16 Fourier frequencies. The finest wavelength is then far below the detector resolution, which adds noise. The config default of 10 is fine, but make the constructor defaults match it.
- **Silent clipping.** Negative integrals and negative times are clipped silently. Add counters.

---

## 3. Backbone

### 3.1 Position enters once, additively, with no relative geometry — Medium (ablation)

**What the code does.**
- Position is added only at the input.
- The Transformer has no RoPE or relative bias, so attention between two tokens can only use their geometric relationship if the first layers preserve it.

**Why it matters.** Point-cloud MAEs (Point-MAE, PoLAr-MAE) re-inject position at every layer. 2D/3D RoPE on (drift, projected wire) is a cheap way to give attention relative geometry.

**Fix.** Ablate re-injecting position in every block, and ablate 2D RoPE on (drift-cm, projected-cm).

### 3.2 Normalization inconsistencies — Low

**What the code does.**
- The encoder has `final_norm=False`, but the MAE decoder is built with `final_norm=True`.
- `enc2dec` therefore consumes un-normalized residual-stream features.
- `mask_c` and `mask_l` are initialized at std 0.02, so the decoder's input scales are mismatched.
- Downstream heads apply their own LayerNorm, so the effect there is small.

**Fix.** Apply a LayerNorm before `enc2dec`, either always or as an ablation.

### 3.3 Drop-path is dead code and is wrong if enabled — Low

**What the code does.**
- `Block._dp` draws **one** random number per batch, so it drops the residual branch for every event at once.
- `FMCfg.drop_path=0` and there is no CLI flag, so it is never used.
- Meanwhile the reference backbones train with drop-path 0.25–0.3, which is another asymmetry.

**Fix.** Implement it per event, with a mask indexed by the event of each packed token, and expose `--drop-path`.

### 3.4 Model size used in docs vs code — Low

**What the code does.**
- The `FMCfg` default is d = 256, 8 layers (6.6M parameters).
- The headline qlfm-M is d = 384, 12 layers (21M), set only by script flags.
- The round-1 ablations (position mode, target norm, E3/E4) were run on the 6.6M model.

**Fix.** Add named presets (`--preset S|M`) and record them in `metrics.json`. Re-confirm the key ablations at M.

---

## 4. Fairness of the comparisons

### 4.1 Asymmetric tuning and compute — Medium

**qlfm got more tuning.**
- qlfm's design was tuned in round 1: position mode, target norm, size. The references are explicitly run with untuned recipes (`technote-round2.md`).
- Compute was matched by wall-clock node-hours: qlfm-M had ~2 and the references ~4. Wall-clock matching favours MAE, whose 60% token drop makes steps cheap.

**The pretraining LR is low and was never swept.**
- The LR is 1.5e-4 per 256 events, i.e. 1.9e-5 at batch 32 on one GPU.
- qlfm-SD inherited this LR. It saw 2.0M events against 8.7M for MAE, so "SD underperforms MAE" is confounded. `technote-round2.md` already notes this LR is ~3× low.

**Fixes.**
- Report events seen and FLOPs next to node-hours.
- Run a 2-point LR sweep for MAE and re-run SD at its own recipe's LR with equal events, or soften the claim.
- State plainly that the tuning was asymmetric.

### 4.2 Early fusion vs charge-only references — High for the headline claim

**What the code does.**
- For qlfm-CL, `QLFMBackbone` returns `z = cat[charge, light]`, so AttnPool on T2–T4 sees light tokens. Panda, PoLAr-MAE, and FM4NPP see charge only unless `--late-light` is set.
- With `--late-light 1`, qlfm-CL sees light twice: its own tokens plus `LateLight`.

**Why it matters.** "qlfm beats Panda" on T2–T4 mixes better representations with extra input. The round-2/3 late-light and charge-only rows partly address this, but the headline tables should be like-for-like.

**Fixes.**
- Make "qlfm-CL vs reference + `--late-light 2`" the primary comparison.
- Also report qlfm-CL probed with `use_light=False`, as T5 already does, to isolate the contribution of light.

### 4.3 T1 per-hit input differs between backbones — Medium

**What the code does.**
- The `native` 128-d `h_hit` exists only for qlfm; the references have `hit_dim = 0`.
- Panda's per-hit features come from a 0.3 cm grid, while qlfm's come from 16-wire cells. The resolutions are very different.

**Fix.** Use `--hit-feat raw` (the same `RawHit` MLP for every backbone) as the main T1 comparison, and report `native` as a qlfm-specific extra.

### 4.4 The "linear" mode is not a linear probe for T1/T5 — Low

**What the code does.**
- `mode=linear` only swaps in `MeanLinear` for event tasks. T1 and T5 keep the full `HitHead` / `LightHead` (at LR 3e-3).
- `MeanLinear` averages over up to 4096 charge tokens plus 32 light tokens, so light contributes about 1% of the mean.

**Fix.** Rename the mode (e.g. `meanlinear_evt`), and average charge and light separately before concatenating them. Document that T1 and T5 ignore the mode.

---

## 5. Training protocol and evaluation

### 5.1 Step budget is confounded with label fraction; defaults don't match the runs — High

**What the code does.**
- `steps = clip(30 · N_train / bs, 400, 15000)`.
- Small label fractions get hundreds of epochs at the 400-step floor, while large fractions are capped at under 1–2 epochs.
- With `eval_every=300`, a 400-step run picks its "best" checkpoint from only two evaluations.

**Code defaults are not what the runs used.**

| Setting | Code default | Round 2/3 scripts | Round 4 scripts |
|---|---|---|---|
| Step cap | `--max-steps 15000` | 3000 | per-fraction `STEPS` / `FT_STEPS` tables |
| Layer decay | `--layer-decay 0.75` | 0.9 | 0.9 |

The summary doc and anyone reading the code defaults will get the wrong picture. The `layer_groups` docstring is also inaccurate: the heads get decay⁰, the last block decay¹, and the embeddings decayⁿ, i.e. ≈0.02 at 0.75 and 13 layers.

**Fixes.**
- Use a fixed step budget per label fraction, the same for every backbone, and evaluate ~10–20 times per run.
- Publish the step table in the technote.
- Change the code defaults to the values actually used (0.9, and the r4 tables), or make the scripts the single source of truth.
- Fix the docstring.

### 5.2 Precision of logits and metrics — Medium

**What the code does.**
- Heads run under bf16 autocast, so logits have ~3 significant digits.
- The headline T2 AUC is 0.986–0.999, and `eff_at_rej99` uses quantiles of the background score.

**Why it matters.** Ties and quantization in exactly that regime can produce spurious differences in the third decimal place.

**Fix.** Run the heads in fp32 (`autocast(enabled=False)` around `head`, with `z.float()`), at least at evaluation time.

### 5.3 Loss/metric mismatches — Low-Medium

**T1.**
- The loss is weighted by log1p(q), but the selection metric is charge-weighted F1 at a **fixed 0.5 threshold**.
- The Pandora comparison then depends on how well calibrated the threshold is.
- Report the F1 at the best threshold on validation, and the AUC.

**T3.**
- The best checkpoint is chosen by macro-F1 on 3000 validation events, and the νe class there is small.
- Select by log-loss or macro-AUC instead.

**T4.**
- Training uses Huber on log E, but the score is the 16–84% half-width of E_pred/E_true − 1, which is asymmetric. Bias is not part of the score.
- Report bias next to resolution, and consider selecting on the log-space width.

**T5.**
- The loss counts the total-PE channel twice, but zero-shot was never trained with that weighting.
- Compare zero-shot and supervised on `mse_shape` and `mse_total` separately.

**Class weights.** At 0.1% labels a class can be missing from the pool. `max(cnt, 1)` then gives it a huge weight; guard against this.

### 5.4 T5 zero-shot runs on inputs it never saw in pretraining — Low-Medium

**What the code does.**
- Zero-shot encodes all charge tokens with **no** light tokens in the encoder sequence.
- In pretraining, the all-masked-light mode did the same thing (masked light tokens are dropped from the encoder), so this part matches.
- Unlike pretraining, though, zero-shot sees **0% masked charge**, whereas the encoder was trained only with ~40% of the charge visible.

**Fix.** Report zero-shot with the pretraining charge mask applied (several draws, averaged) as well as fully visible. Add a supervised `LightHead` on the frozen CL encoder as a third arm.

### 5.5 Statistics — Medium

**What the code does.**
- Most table entries are a single seed.
- The 20k-event test set gives small sampling error but no measure of run-to-run variance.

**Why it matters.** Differences below ~0.02 cannot be interpreted.

**Fixes.**
- Run 3 seeds for headline entries at ≤1% labels.
- Compute bootstrap CIs from the stored `preds_test.npz`.
- Seed the DataLoader workers and the head initialization explicitly.

---

## 6. Engineering and robustness — Low

- **Pretraining sampler.** Each rank draws its own permutation of the full dataset, so ranks overlap within an epoch, and the seed includes `step` on resume. Use a strided, `DistributedSampler`-style index. Also, `find_unused_parameters=True` costs speed and is needed only for `--light 0` and SD; set it only for those.
- **Checkpoint loading.**
  - When a checkpoint lacks `pos_mode` / `pos_freq`, `qlfm_backbone.build` falls back to `"fourier"` / 16, while `FMCfg` defaults to `"centroid"` / 10. The strict `load_state_dict` will catch a mismatch, but a missing key should raise a clear error.
  - Downstream `--pos-mode`, `--pos-freq`, and `--cell-*` flags are silently ignored when a checkpoint is given; warn.
- **Downstream loading.**
  - `load_state_dict(strict=False)` when loading the best checkpoint can hide key-name bugs. Assert that no head keys are missing.
  - If `--time-limit` is hit before the first evaluation, `best.pt` doesn't exist and the test step crashes.
- **Weight decay.** The pretraining exclusion matches `"pos"` by name only, so plane, PMT, and modality embeddings are decayed. Downstream non-finetune modes decay biases and norms too.
- **Pretraining extras.** No EMA of the weights is kept for MAE, and the SD "val loss" depends on the schedule, so it is not meaningful.

---

## 7. Recommended next steps (cheapest first)

1. Align the code defaults with the runs (layer decay, step tables, token cap). Fix the `layer_groups` docstring. Log `frac_hits_dropped`, events seen, and arguments into `metrics.json`.
2. Run the heads in fp32. Add bootstrap CIs and 3 seeds for the headline entries.
3. Fixed MAE target statistics (§1.2) and a light-weight sweep (§1.3). Optionally, the decoder-position ablation (§1.1).
4. Like-for-like light comparisons (§4.2) and `raw` hit features for T1 (§4.3).
5. Projected per-plane geometry (§2.1) shared with the PMT (y, z).
6. T1 feature propagation or a cell-size ablation (§2.2), and T2 evaluated with the gap events included (§2.4).
