# Code audit, round 1: `fm4npp-lartpc`

**Scope.** Every Python, shell and PBS file under `multi-modal-qlmatch/` (about 6.9k lines): the `qlfm` package, the reference backbones in `qlfm/ref`, the pimm configs, and `scripts/`. No code was changed.

**How findings were checked.** Every finding comes from reading the code. Where possible, a finding was also checked against the saved run outputs under `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work`, which is the work directory that `scripts/env.sh` points at. Those checks are marked **[verified on outputs]**.

**Severity levels.**
- **High:** changes numbers already reported in the tech notes.
- **Medium:** can bias a comparison or a monitoring metric, or contradicts a claim made in the notes.
- **Low:** robustness problems, edge cases and hygiene.

Issues already listed as caveats in `docs/technote-round2.md` (the Panda head-only fine-tune, five time-limited runs, Pandora scored on a 20k-event superset) are not repeated here, except where the code makes the problem harder to detect.

---

## Summary

| # | Severity | Finding | Where |
|---|---|---|---|
| 1 | High | FM4NPP "learned layer mix" adapter is never trained in frozen/linear runs | `train_downstream.py:101`, `ref/fm4npp.py:174` |
| 2 | High | Time-limited downstream runs write a normal-looking `metrics.json`; 22 of the 192 round-1 (E3) runs were truncated, and this is not documented | `train_downstream.py:288`, `make_tasks.py`, `make_tasks_r2.py`, `run_queue.sh:30` |
| 3 | Medium | Pretraining validation set is index-ordered: first 2,048 val events, BNB only | `train_pretrain.py:129` |
| 4 | Medium | Round-4 large-light cut is tuned on the test set it is then used to split | `scripts/r4_light_stats.py:83` |
| 5 | Low | Token cap differs between pretraining (4096), downstream (8192) and T5 zero-shot (4096) | `train_downstream.py:64`, `t5_zeroshot.py:37` |
| 6 | Medium | Panda downstream preprocessing differs from its pretraining views (no centre shift; batch-global grid origin) | `ref/pimm_backbone.py:86-100` vs `pimm_configs/panda_uboone.py:88-97` |
| 7 | Medium (design) | MAE decoder sees each masked token's own charge-weighted centroid and in-cell offset | `models/fm.py:148`, `models/encoder.py:65-68` |
| 8 | Low | Conditionally present loss keys are averaged over all steps; `all_reduce` over dicts whose keys can differ between ranks | `train_pretrain.py:109-116, 210-217` |
| 9 | Low | `best.pt` may never be written, so the final `torch.load` crashes | `train_downstream.py:284-296` |
| 10 | Low | Visibility labels count hits with edep > 0; T1 hit labels use edep > 0.05 MeV | `data/convert.py:139`, `data/labels.py:16` |
| 11 | Low | Fine-tune layer decay puts FM4NPP's final RMSNorm in the lowest-lr group | `ref/fm4npp.py:169-172` |
| 12 | Low | Converter trusts `hit_id` == row index and PMT channel < 32 without enforcing it | `data/convert.py:109,120,160,170` |
| 13 | Low | Smaller robustness and hygiene items | see section 13 |

---

## 1. FM4NPP adapter (`adapter_mix`, `adapter_norm`) is never trained in frozen/linear modes — High [verified on outputs]

**What happens.**
- `DownstreamModel.forward` runs the whole backbone under `torch.no_grad()` when the mode is `frozen` or `linear` ([train_downstream.py:101-103](../qlfm/train_downstream.py#L101-L103)).
- `FM4NPPBackbone.forward` computes the softmax layer mix and the LayerNorm inside that same call ([ref/fm4npp.py:177-179](../qlfm/ref/fm4npp.py#L177-L179)).
- The adapter parameters are marked `requires_grad=True` ([train_downstream.py:216](../qlfm/train_downstream.py#L216)) and are passed to AdamW, but they never receive a gradient. AdamW skips any parameter whose `.grad` is `None`.

**Evidence.** `r2p/fm4npp/evt_nu/frozen/f0.01_s0/best.pt` holds `backbone.adapter_mix` = 12 × 0.0, `adapter_norm.weight` = all 1.0 and `adapter_norm.bias` = all 0.0. These are exactly the initial values. The matching `finetune/f0.01_s0/best.pt` has moved values, so the parameters can train when there is no `no_grad` around them.

**Impact.**
- Every FM4NPP frozen and linear result used a fixed uniform average of the 12 Mamba layer outputs. This covers round 2 R2P (`frozen`, `frozen_nolight`), round 3 (`frozen`, `frozen_xf`, `frozen_raw`) and FM4NPP m4.
- [technote-round2.md:76](technote-round2.md) says FM4NPP uses "a learned mix of all 12 layers". That is true only for the fine-tune runs.
- The direction of the effect on scores is unknown. A uniform mix plausibly understates FM4NPP, since the last layers of a next-neighbour predictor are often not the most linearly useful ones.

**Fix (minimal).** Use `no_grad` only when the backbone has nothing trainable:

```python
# DownstreamModel.forward
frozen_all = self.mode in ("frozen", "linear") and not any(p.requires_grad for p in self.backbone.parameters())
with torch.no_grad() if frozen_all else contextlib.nullcontext():
    o = self.backbone(b)
```

With the Mamba weights at `requires_grad=False`, autograd only tracks the adapter ops, so the memory cost stays small. Then re-run the FM4NPP frozen cells.

---

## 2. Time-limited downstream runs look like finished runs — High [verified on outputs]

**What happens.**
- `run_queue.sh` passes `--time-limit $((left-5))` ([run_queue.sh:30](../scripts/run_queue.sh#L30)).
- Without `--resume 1`, `train_downstream` stops at the time limit, tests the best checkpoint so far, and writes `metrics.json` ([train_downstream.py:288-302](../qlfm/train_downstream.py#L288-L302)).
- `run_queue.sh` and `remaining_tasks.sh` then treat the task as done.
- The round-1 and round-2 task generators (`make_tasks.py`, `make_tasks_r2.py`) do not add `--resume 1`. Rounds 3 and 4 do.
- `metrics.json` records the *planned* `steps` and `best_step`, but not the step actually reached. A truncated run can only be found by inspecting `val_log`.

The `--label-frac`/`--accum` regex in `run_queue.sh:19` tries to keep long runs out of short slots. It is only a heuristic, and it misses `--label-frac 1` at the end of a line because it requires a trailing space.

**Scan of saved runs.** A run counts as truncated if the last step in `val_log` is below the planned `steps`.

A full run always evaluates at `step == steps`, so this test has no false positives.

| Run set | Truncated / scanned | Documented? |
|---|---|---|
| `e3`: round-1 E3, the 192 runs behind `technote-round1.md` §5.1 and `figures/e3_summary.md` | **22 / 192** | **No.** The note says steps are "30 epochs … clipped to 400–15,000" |
| `r2p`: round-2 R2P | 5 / 232 | Yes: exactly the five runs listed in `technote-round2.md` |
| `r3`: round 3 (`--resume 1`) | 0 / 160 | n/a |
| `r4`, `e4`, `e5` | not scanned: the glob did not match the `r4` layout, and sshfs timed out before `e4`/`e5` | `r4` uses `--resume 1`; `e4`/`e5` were generated by `make_tasks.py`, so they are at risk |

The truncated E3 runs (last evaluated step / planned steps) are listed below. They are concentrated in exactly the long, large-budget cells that the label-efficiency curves rely on:

```
energy  finetune_C  f0.01_s0  2100/2502     evt_nu  frozen_C    f1_s0    9600/15000
energy  finetune_CL f0.01_s0  2100/2502     evt_nu  frozen_CL   f1_s0    8400/15000
energy  frozen_C    f0.1_s0  10800/15000    evt_nu  scratch_C   f0.01_s0 6300/6692
energy  frozen_CL   f0.1_s0   9900/15000    evt_nu  scratch_C   f0.01_s1 5100/6692
energy  frozen_CL   f1_s0     5400/15000    evt_nu  scratch_CL  f0.01_s0 6000/6692
evt_nu  finetune_C  f0.01_s0  5100/6692     evt_nu  scratch_CL  f0.01_s1 5100/6692
evt_nu  finetune_CL f0.01_s0  5100/6692     flavor  frozen_C    f0.1_s0 11100/15000
evt_nu  frozen_C    f0.01_s2  4800/6692     flavor  frozen_C    f1_s0    8400/15000
hit     finetune_CL f0.01_s1  3000/3169     flavor  frozen_CL   f0.1_s0 10800/15000
hit     frozen_C    f0.1_s0   9900/15000    flavor  frozen_CL   f1_s0    6300/15000
hit     frozen_C    f1_s0     5100/15000    hit     frozen_CL   f0.1_s0  9900/15000
```

**Impact.**
- Several of the 100% and 10% label points ran for only 34–72% of their planned steps, e.g. `energy frozen_CL f1` and `flavor frozen_CL f1`.
- C and CL are often cut by different amounts, e.g. `flavor f1`: C at 8,400 steps vs CL at 6,300. That directly affects the C-vs-CL comparison (the note's Q2).
- At 1% labels the truncated seeds feed seed means and spreads.
- These cells should be re-run with `--resume 1`, or marked in the round-1 tables and figure.

The scan script is at `/tmp/audit_tools/scan_trunc2.py` on the auditing machine; it is short enough to re-create from this description.

**Fix.**
- Add `final_step=step` and `truncated=step < steps` to the `metrics.json` dict.
- Have `summarize*.py` flag or drop truncated runs.
- Default `--resume` to 1, or add it in every task generator.

---

## 3. Pretraining validation uses the first 2,048 val indices — Medium

[train_pretrain.py:129](../qlfm/train_pretrain.py#L129): `val_idx = sp["val"][: args.n_val]`.

**Why it is biased.** Split indices are sorted, and the corpus is concatenated BNB then νe. So the pretraining validation loss is computed on events from the first few BNB shards only, with no νe-sample events.

`train_downstream.py:206` already fixes this same bias for downstream ("fixed random (not index-ordered) val/test subsets"), and the work directory contains `archive_v1_e3_biasedval`. The pretraining script was not updated.

**Impact.** Monitoring only. Pretraining saves the last checkpoint, not the best, so model selection is not affected. Any val-loss curve or comparison quoted in the notes is on a non-representative subset.

**Fix.** Use the same `pick(...)` as downstream, e.g. `np.sort(rng(11).permutation(sp["val"])[:n_val])`.

---

## 4. Round-4 large-light cut is chosen on the test set — Medium

**What happens.**
- [r4_light_stats.py:83-88](../scripts/r4_light_stats.py#L83-L88) chooses the PE cut that maximises balanced accuracy on `te = sp["test"][elig]`.
- [r4_posthoc.py](../scripts/r4_posthoc.py) then reports T1 and T2 on the pass/fail subsamples of the same test events.

**Impact.** The cut is snapped to a coarse grid of six values (10…500 PE), so the optimisation bias is small. It is still test-set tuning.

**Fix.** Choose the cut on `sp["val"]`, or state in the note that it was tuned on test.

---

## 5. Inconsistent per-event token cap — Low (was Medium; see impact)

**The three caps.**
- Pretraining: `--max-tokens 4096` ([train_pretrain.py:40](../qlfm/train_pretrain.py#L40)).
- Downstream: default `--max-tokens 8192` ([train_downstream.py:64](../qlfm/train_downstream.py#L64)). `qlfm_backbone.build` takes `max_tokens` from the downstream args, not from the checkpoint ([ref/qlfm_backbone.py:36](../qlfm/ref/qlfm_backbone.py#L36)).
- T5 zero-shot: takes the checkpoint's 4096 ([t5_zeroshot.py:37](../qlfm/t5_zeroshot.py#L37)).

**Consequences.**
- Frozen encoders see longer sequences downstream than they ever saw in pretraining, with a different random subsample.
- The T5 "zero-shot vs supervised" comparison is not on identical inputs, although `t5_zeroshot.py` says "Same test subset, outputs and metrics".
- In T1, hits in dropped tokens get a fixed logit of −20 ([heads.py:60](../qlfm/models/heads.py#L60)), so each cap gives a different number of forced misses.

**Impact: small [verified on outputs].** `audit_bnb/audit.json` (first 3 BNB shards, 16×64 cells) gives:
- Median 1,974 tokens per event; 99th percentile 3,357; maximum 7,600.
- `frac_events_over_4096_tokens` = 0.0021.

So the 4096 cap affects about 0.2% of events, and the 8192 cap affects none. Downgraded to Low. Still worth aligning the caps, or stating the difference, so that "identical inputs" claims are literally true.

---

## 6. Panda: downstream inputs differ from its pretraining views — Medium (needs checking)

**Pretraining** ([panda_uboone.py:87-97](../qlfm/ref/pimm_configs/panda_uboone.py#L87-L97)):
- Every view gets `CenterShift(axes=("x","y"))`, random in-plane rotation, flips and `RandomScale`.
- `global_coord` (the centred, rotated coordinates) is part of the **input feature** (`global_feat_keys=("global_coord","global_energy")`).
- `GridSample` computes the grid per cloud.

**Downstream** ([pimm_backbone.py:84-100](../qlfm/ref/pimm_backbone.py#L84-L100)):
- The input feature is the un-centred normalised coordinate.
- The voxel grid origin is the minimum over the **whole batch** (`coord.min(0)`), not per cloud. Voxel boundaries therefore shift with batch composition, so the same event can voxelise differently in different batches.
- The plane offset (10 cm → about 33 grid units) appears in `grid_coord` z. In pretraining that axis is always 0 after per-sample min subtraction.

**Impact.** Panda's frozen features may be evaluated off-distribution, which would understate Panda in the comparison.

**Fix.**
- Centre x, y per cloud before building `feat`, matching `CenterShift`.
- Compute the grid origin per cloud.
- Subtract the per-cloud z minimum from `grid_coord`.
- Then re-check one frozen cell to measure the effect.

---

## 7. MAE: masked tokens' positional input carries information from their own hits — Medium (design)

**What happens.**
- With the default `pos_mode="centroid"`, `ChargeEmbed.pos_embed` uses `tok_cpos` (the charge-weighted centroid of the token's hits) and `tok_crel` (the centroid's offset within the 16-wire × 64-tick cell) ([encoder.py:65-68](../qlfm/models/encoder.py#L65-L68)).
- The decoder adds this embedding to the mask tokens ([fm.py:148](../qlfm/models/fm.py#L148)).
- The MAE target is the 4×8 sub-grid charge map of that same cell, plus `tok_stat` ([fm.py:165](../qlfm/models/fm.py#L165)).

So for a token with one or two hits, the decoder is told where inside the cell the charge is, which largely gives away the sub-grid target. Point-MAE also gives the decoder patch centres, so this is a known design choice, not a coding bug.

**Implications.**
- The charge reconstruction loss is easier than it looks. Comparing MAE losses between `pos_mode`s (fourier/grid vs centroid) is not like-for-like.
- If not already done, consider an ablation where the decoder gets only the cell centre (`tok_pos`).

---

## 8. Loss bookkeeping in pretraining — Low

**Biased averages.**
- `out["loss_l"]` exists only when some light token is masked, and `out["loss_l_all"]` only when some event is in the all-PMT-masked mode ([fm.py:172-181](../qlfm/models/fm.py#L172-L181)).
- Both the training log ([train_pretrain.py:210-217](../qlfm/train_pretrain.py#L210-L217)) and `validate` ([train_pretrain.py:109-115](../qlfm/train_pretrain.py#L109-L115)) divide the summed value by the total number of steps or batches. That biases these averages low.
- At the default batch of 32 the bias is negligible, since P(no all-masked event) = 0.6^32. With small `--bs` or `--n-val`, as in the smoke runs, it is not.

**Possible hang.** `validate` all-reduces one tensor per dict key. If ranks end up with different key sets, the collectives are mismatched and can hang or pair up the wrong values. Same rarity as above.

**Fix.** Count per key, and all-reduce a fixed list of keys, filling missing ones with NaN or 0 together with a count.

---

## 9. `best.pt` may never exist — Low

[train_downstream.py:284-297](../qlfm/train_downstream.py#L284-L297) writes `best.pt` only when `m["score"] > best`. It is missing in two cases:
- No evaluation happens before the time limit, i.e. fewer than `eval_every`=300 steps when not resuming.
- Every val score is NaN. For example, `roc_auc_score` raises on a single-class val set, and the energy resolution can be NaN.

In both cases `torch.load(ck)` raises `FileNotFoundError` after the full training time.

**Fix.** Evaluate once at the time-limit exit, and treat NaN as "worse than any finite value".

---

## 10. Two different "neutrino hit" definitions — Low

- Event visibility (`nu_nhits`) counts hits with MC deposit **> 0** ([convert.py:139](../qlfm/data/convert.py#L139)). This feeds T2's "visible" (≥ 30 Y-plane hits) and "no-ν" (≤ 5 hits) classes and T1 event eligibility.
- The T1 per-hit target uses **> 0.05 MeV** ([labels.py:16](../qlfm/data/labels.py#L16), [train_downstream.py:132](../qlfm/train_downstream.py#L132)).

**Impact.** An event can count as "visible" while some of the hits that made it visible are labelled cosmic in T1. It can also be in T2 class 0 ("essentially no neutrino charge") while having up to 5 hits with tiny deposits. This is probably intentional, but it is not documented in `labels.py`, which is meant to be the "single source of truth".

**Fix.** Either compute `nu_nhits` with the same threshold, or document the difference.

---

## 11. Layer-wise lr decay puts FM4NPP's final norm at the bottom — Low

`FM4NPPBackbone.layer_id` gives `net.norm.weight` (the final RMSNorm, applied after the last layer) id 0, so it gets the smallest learning rate, `decay^13` ([fm4npp.py:169-172](../qlfm/ref/fm4npp.py#L169-L172)). `net.out` (the pretraining head) also gets id 0, but it is unused downstream.

PoLAr-MAE's `layer_id` sends unmatched `m.encoder.*` names to `n_layers-1`, which is the more sensible choice.

**Fix.** Map `norm.` to `n_layers`.

---

## 12. Converter assumptions that are trusted, not enforced — Low [verified on outputs]

- **`hit_id` == row index.** Per-hit truth and Pandora membership are scattered by `hit_id`, assuming it equals the row index within the event ([convert.py:120](../qlfm/data/convert.py#L120), [convert.py:160](../qlfm/data/convert.py#L160); also [pandora_features.py:92](../qlfm/data/pandora_features.py#L92)). The converter only *counts* violations (`hit_id_mismatch`). **Checked:** all 126 converted shards report `hit_id_mismatch: 0`, so current outputs are fine. But a future dataset that breaks the assumption would convert silently with wrong labels. Make it an `assert`.
- **PMT channel < 32.** `oph_ch` is stored as `uint8` and used as `evt * 32 + ch` in `tokenize_light` ([tokenize.py:122](../qlfm/tokenize.py#L122)). A channel ≥ 32 would silently add its PE to the next event's PMT histogram, or index out of range. **Checked:** `nue_NoWire_00_p0` (3.05M ophits) has channels 0–31 only. Add a check in `convert.py`.
- **Negative hit times.** These are silently clipped to tick 0 ([convert.py:113](../qlfm/data/convert.py#L113)). Worth a counter in `meta.json`, like `hit_id_mismatch`.

---

## 13. Smaller items — Low

**Robustness**
- **Empty particle table.** In `convert.py:131`, if a chunk has edep rows but no particle rows, `np.clip(..., 0, -1)` followed by indexing an empty array raises. This is unlikely with the open data.
- **Target normalisation.** `MAE` `target_norm` divides by `tgt_c.std(0)`, which is NaN when at most one token is masked in the batch ([fm.py:167](../qlfm/models/fm.py#L167)). This only matters for tiny batches.
- **Stale locks in `run_queue.sh`.** A lock directory (`$out.lock`) left by a node failure or a killed job blocks that task in every later job, with no message. `remaining_tasks.sh` still lists the task, but the workers skip it.

**Training mechanics**
- **Budget mode lr jump.** With `--budget-h`, `args.steps` changes at step 200, so `lr_at` recomputes the warmup length mid-run and the lr can jump from warmup straight into cosine decay ([train_pretrain.py:188-196](../qlfm/train_pretrain.py#L188-L196)).
- **Gradient accumulation at epoch end.** A partial accumulation (`micro > 0`) is thrown away when the epoch ends, because `micro` resets to 0 and the next `zero_grad` drops it. This wastes compute but is otherwise harmless.
- **Stochastic depth** drops a residual branch for the whole packed batch at once, not per event ([encoder.py:102-106](../qlfm/models/encoder.py#L102-L106)). It is labelled as intentional, but it is much noisier than the usual per-sample version. It only matters if `drop_path > 0`, and the default is 0.
- **Sinkhorn normalisation** in `models/sd.py` is per GPU, with no all-reduce across ranks as in DINO/Panda. This is a design difference worth noting in the SD description.

**Light encoder**
- **`LightEmbed` frequencies.** It uses `Fourier(n_freq=16)`: frequencies up to 2¹⁵·π applied to PMT coordinates scaled to about [0, 1]. The top octaves are effectively aliased noise for 32 fixed positions. The model can ignore them, but `n_freq≈6` would be enough.
- **`bucketize` boundaries** (`right=False`) put a time exactly on a bin edge into the lower bin, while the window test uses `t >= 0`. This is harmless.

**Docs and comments**
- `MambaGPT` docstring says the head is `Linear(d, 3k)` and returns `[B, L, 3k]`. The code uses `klen * 2` (2-D offsets) ([fm4npp.py:5,58,61](../qlfm/ref/fm4npp.py#L58)).
- `ChargeTokens.tok_time` is documented as "mean tick" but is the cell-centre tick ([tokenize.py:37,77](../qlfm/tokenize.py#L77)).

**Labels**
- **ν_τ CC.** `labels.py` maps ν_τ CC to class 0 ("numu CC"). This is negligible for BNB, but worth a comment.

---

## Checked and found correct

These were checked specifically, so a later audit can skip them:

**Data and splits**
- HDF5 offsets and `read_range` event indexing.
- Part-relative offsets in `convert.py`.
- Lexsort "first row per event" selections: max energy-fraction particle, max `nu_score` slice, max-PE flash.
- Key packing and overflow in `tokenize_charge` and FM4NPP `serialize` (all within int64).
- (run, subrun)-hashed splits are leakage-safe, and the nested subsets really are nested.
- Downstream val/test picks use the same permutation seeds as the Pandora references and T5 zero-shot. Same-event checks pass in `r4_posthoc.py` and `r2_select_events.py`.

**Models**
- `pack` / `unpack` permutation logic in `FM.encode` and the MAE decoder.
- `hit2tok` remapping after the token cap.
- FM4NPP `hit2tok` through padding.
- PoLAr-MAE nearest-centre mapping.
- `XFusion` and `LightHead` padding and masking.
- T5 light-task backbones have their own light input switched off (`use_light=False`), so the target does not leak into the input.
- Panda fine-tuning now really trains the backbone (`requires_grad_(True)`).

**Training infrastructure**
- Multi-run pretraining gives each run its own `MASTER_ADDR`/`MASTER_PORT`, so there are no rendezvous collisions.
- `meta.json` is written last, so `convert_node.sh` retries interrupted conversions correctly.
