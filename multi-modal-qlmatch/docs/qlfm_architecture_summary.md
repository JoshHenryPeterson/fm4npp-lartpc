# qlfm model architecture summary

This summarizes the native **qlfm** model in this repo: a charge + light foundation model for MicroBooNE LArTPC data. Code references point to files, classes, and functions under `qlfm/`. Defaults are the code defaults unless stated otherwise.

```
 HDF5 open samples ──convert──▶ .npy column cache ──collate (packed, no padding)──▶ batch
                                                                                   │
           ┌──────────────── tokenize_charge ───────────────┐   ┌── tokenize_light ──┐
           │ occupied 16-wire × 64-tick cells (≤4096/event)  │   │ 32 PMTs × 17 feats │
           └──────── ChargeEmbed (mini-PointNet) + pos ──────┘   └ LightEmbed + pos ──┘
                           + modality[0]                              + modality[1]
                                     └──────── concat per event ────────┘
                                                     │
                         Pre-LN Transformer encoder (FlashAttention-2 varlen,
                         block-diagonal per event; d=256/8 layers or d=384/12 layers)
                                                     │
                          zc [N_charge_tok, d]  ,  zl [B·32, d]  ,  h_hit [N_hits, 128]
                    ┌────────────────────┬───────────┴─────────┬──────────────────────┐
              pretraining:         T1 HitHead            T2–T4 AttnPool        T5 LightHead
              MAE decoder          (per-hit, via          (4-query attentive    (32 PMT queries
              (3 layers, d=192)     hit2tok lookup)        pooling + MLP)        cross-attend charge)
```

> Note: "qlmatch" is the repo name only. The code has no explicit charge-light matching or contrastive loss. Charge and light are fused early as tokens in a single sequence, and the pretraining objective is a multi-modal masked autoencoder (MAE).

---

## 1. Data inputs

### Source
- MicroBooNE open samples (CC BY 4.0), in hep_hpc / ph5concat HDF5 format:
  - Zenodo 8370883: inclusive BNB "NoWire" overlay.
  - Zenodo 7261921: νe.
- `qlfm/data/h5reader.py` (`OpenSampleFile`) reads per-event (start, count) offsets from `event_id.seq_cnt`.
- `qlfm/data/convert.py` writes a per-shard, column-wise `.npy` cache (`ev_*`, `hit_*`, `oph_*`, `meta.json`).
  - Truth and baseline columns go in separate files, so pretraining never reads them.

### Modalities

| Modality | Source table | Fields used | Stored as |
|---|---|---|---|
| **Charge**: TPC wire hits (2D per plane; not 3D spacepoints) | `hit_table` | `local_plane`, `local_wire`, `local_time`, `integral`, `rms` | plane uint8, wire int16, time uint16 (1/8 tick), integral and rms float16 |
| **Light**: PMT optical hits | `ophit_table` | `hit_channel`, `peaktime` (µs, beam-window clock), `width`, `pe` | ch uint8, t float32, width and pe float16 |

Detector constants are in `qlfm/geometry.py`:
- TPC: 3 planes with 2400 / 2400 / 3456 wires and 0.3 cm wire pitch.
- Time: 0.5 µs ticks with the trigger at tick 800, and 0.0549 cm per tick of drift.
- Light: 32 PMTs with a (y, z) position map, read over a 23.4 µs beam window.
- Volumes: the active volume is x ∈ [0, 256.35], y ∈ [−116.5, 116.5], z ∈ [0, 1036.8] cm. The fiducial volume uses a 3 cm margin.

### Batching
- `qlfm/data/dataset.py`: `Corpus` memory-maps the shards. It can combine several samples (e.g. `bnb,nue`).
- `collate` packs variable-length events **by concatenation with no padding**:
  - Builds `hit_evt` / `oph_evt` event-index vectors.
  - Converts the stored time back to ticks.
  - Stacks the per-event `ev_*` fields.
  - Adds per-hit truth only when the dataset was built with `truth=True`.

### Labels and splits
- Per-hit truth:
  - `nu_edep`: neutrino energy deposit (MeV).
  - `cat`: semantic category.
  - `pnd_nu`: whether the hit is in Pandora's ν slice. This is a baseline, not an input.
- Per-event truth: ν PDG, CC/NC, energy, vertex, and so on. `qlfm/data/labels.py` turns these into the downstream task labels (Section 5).
- Splits (`qlfm/data/splits.py`):
  - A hash of `run:subrun` assigns events to train (~90%), val (5%), and test (5%), so all events from one run/subrun stay together.
  - Label fractions are nested subsets of train (`nested_subset`, 0.1% to 100%).

### Normalization
Normalization is done inside the tokenizer, not stored in the cache:
- Charge: `log1p(q/200)` and `rms/5`.
- Light: `log1p(PE)`.

---

## 2. Tokenization (`qlfm/tokenize.py`, `TokCfg`)

Tokenization runs on the GPU on the packed batch. It uses a fixed wire × time grid (no FPS/kNN grouping and no voxelization), so the cell size can be changed without re-preprocessing.

### Charge tokens (`tokenize_charge`)
- **One token = one occupied (event, plane, wire-cell, time-cell) cell.**
  - The default cell is **16 wires × 64 ticks** (`cell_wires`, `cell_ticks`).
  - Occupied cells are found with `torch.unique` on a combined key, so the token count varies per event.
- **Cap:** `max_tokens = 4096` per event. Tokens above the cap are randomly dropped, and their hits get `hit2tok = −1`.
- **Points per token:** variable (all hits in the cell).
- **Per-hit features (4):**
  - In-cell wire offset and in-cell tick offset (each in [−0.5, 0.5]).
  - `log1p(q/200)`.
  - `rms/5`.
- **Per-token outputs:**
  - Hit-to-token index `hit2tok`.
  - Per-token stats (`log n_hits`, `log1p(Σq/200)`).
  - Charge-weighted centroid.
  - MAE target `tok_target`: a **4 × 8 sub-grid** (wire × tick) of `log1p(Σq/200)` inside the cell. With the 2 token stats added, the target has 34 values.

### Light tokens (`tokenize_light`)
- **Exactly 32 tokens per event, one per PMT**, always present (including PMTs with no light).
- **Features (17):** `log1p(PE)` in 16 time bins, plus `log1p(total PE)`.
  - The bin edges (µs) are 0, 1, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10, 12, 15, 19, 24. Bins are finest around the expected beam flash at about 3–6 µs.
- About 1% of ophits fall outside the window and are dropped.

---

## 3. How tokens enter the backbone (`qlfm/models/encoder.py`, `qlfm/models/fm.py`)

### Charge embedding (`ChargeEmbed`, a mini-PointNet)
1. `hit_mlp`: MLP 4 → 64 → 128 (GELU), applied to each hit. This per-hit 128-d vector is also exported as `h_hit` for the T1 head.
2. `pointnet_pool`: concatenates max-pool and mean-pool over the hits in the cell (256-d), then appends the 2 token stats (258-d).
3. `proj`: MLP 258 → d → d.
4. Positional embedding is added (`pos_embed`). The default `pos_mode="centroid"` sums:
   - A plane embedding (`Embedding(3, d)`).
   - Fourier features (10 frequencies; scales 300 and 1100 cm) of the charge-weighted centroid, in drift-cm ((tick − 800) · 0.0549) and wire-cm (wire · 0.3, per plane; no 3D reconstruction).
   - An MLP (2 → 64 → d) of the centroid offset within the cell.

   The alternative modes `fourier`, `grid`, `both` and `cellrel` use the cell centre or learned cell-index embeddings instead. Checkpoints that lack a `pos_mode` entry load as `fourier`.

### Light embedding (`LightEmbed`)
The light token is the sum of three terms:
- `proj`: MLP 17 → d → d.
- Fourier features of the PMT position (y + 65, z).
- A learned PMT-ID embedding (`Embedding(32, d)`).

### Fusion and packing
- A learned **modality embedding** (`Embedding(2, d)`) is added: index 0 for charge, 1 for light.
- `FM.encode` **concatenates charge and light tokens into one sequence per event**. This is early fusion with full joint self-attention.
- `pack()` sorts tokens by event and builds `cu_seqlens`. Attention uses `flash_attn_varlen_qkvpacked_func`, so it is block-diagonal per event and needs no padding.
- There are **no CLS or register tokens**. Pooling happens in the heads.
- Light can be turned off (`use_light=False` / `--light 0`).
- During MAE pretraining, masked tokens are **dropped**: the encoder only sees visible tokens.

### Token sequence order

Order is set in three places:
- `tokenize_charge` sorts the charge cells.
- `tokenize_light` lays out the PMT tokens.
- `pack` in `models/encoder.py`, called from `FM.encode`, joins and groups them.

**Charge tokens: raster order, one plane after another.** Each hit gets an integer cell key, and `torch.unique` returns the keys sorted:

```
key = ((evt · 3 + plane) · n_wc + wire_cell) · n_tc + time_cell
      n_wc = 3456 // 16 + 1 = 217 wire cells,  n_tc = 8192 // 64 + 1 = 129 time cells
```

So within one event, the charge tokens are ordered:
1. **By plane:** all plane-0 tokens, then plane 1, then plane 2. These are `local_plane` 0, 1, 2, i.e. MicroBooNE U (induction), V (induction), and Y (collection, 3456 wires).
2. **Within a plane, by wire cell**, ascending. Each wire cell is a column 16 wires wide.
3. **Within a wire cell, by time cell**, ascending (64-tick steps in drift time).

This is a column-major raster of each plane's wire × time image. Some notes on this order:
- **Empty cells are skipped.** The sequence only contains occupied cells, so its length and the gaps in it depend on the event.
- **The token cap keeps the order.** It drops tokens with a boolean mask, so the remaining tokens stay sorted.
- **The planes are separate 2D views.** Each plane is a separate block of tokens. Nothing in the code matches cells across planes or builds 3D points.
- **Wire coordinates mean different things on each plane.** A plane's wire-cm coordinate (wire × 0.3 cm) is measured in that plane's own wire direction. The U and V wires are at ±60°, and the Y wires are vertical.

**Light tokens: always 32 per event, in PMT channel order.** The light block for an event holds PMT OpChannels 0, 1, …, 31, with every PMT present even if it has no hits. The channel number is the hardware OpChannel ID, not a geometric ordering. In `geometry.PMT_YZ`, channel 0 is near z ≈ 950 cm and channel 31 near z ≈ 90 cm, so the order runs roughly from high z to low z.

**Combining the two blocks.** `FM.encode` builds `cat([charge tokens, light tokens])`. `pack` then applies a *stable* argsort on the event index, so each event's tokens are contiguous and keep their charge-then-light order:

```
batch sequence = [ ev0: U cells… | V cells… | Y cells… | PMT0 … PMT31 ]
                 [ ev1: U cells… | V cells… | Y cells… | PMT0 … PMT31 ] …
cu_seqlens marks the event boundaries.
```

Sequence lengths:
- At most 4096 charge tokens + 32 light tokens per event.
- During MAE pretraining the encoder gets only the visible subset, in the same order.
- The decoder gets the full sequence, with mask tokens put back in their original positions. It has the same layout.
- After encoding, `inv` puts the outputs back into the separate `zc` (charge) and `zl` (light) arrays.

**The order does not affect the model's output.** The model never uses a token's position in the sequence:
- There is no sequence-index or positional embedding, RoPE, or relative position bias.
- `flash_attn_varlen_qkvpacked_func` runs with its default non-causal, fully bidirectional attention.
- MLPs and LayerNorm act on each token separately.

So, within an event, the encoder is permutation-equivariant: reordering the tokens just reorders the outputs. The raster order only matters for bookkeeping, such as the `hit2tok` indices and splitting the outputs back into `zc` and `zl`. It does not work like the serialized order in FM4NPP or PTv3. All structure comes from what is *added to* each token:

| Information | Charge token | Light token |
|---|---|---|
| Which view | plane embedding (`Embedding(3, d)`) | — |
| Where | Fourier(charge-weighted centroid: drift-cm, wire-cm) + in-cell offset MLP | Fourier(PMT y, z) + PMT-ID embedding (`Embedding(32, d)`) |
| Which modality | `modality[0]` | `modality[1]` |

**Comparison: order matters for FM4NPP and PTv3, not for the qlfm Transformer.**

| Backbone | Sequence order | Does order matter? |
|---|---|---|
| **qlfm** (Transformer) | Plane → wire cell → time cell raster, then 32 PMTs | **No.** Full bidirectional attention within each event, with no sequence-position encoding. |
| **FM4NPP** (Mamba2, `ref/fm4npp.py`) | One token per hit. Ordered by drift slab (8) → plane → wire box (8) → wire → tick | **Yes.** The scan is causal, so each hit only sees earlier hits. The pretraining targets are the k = 30 nearest *later* hits on the same plane, so the order also defines the targets. |
| **Panda / PTv3** (`ref/pimm_configs/panda_uboone.py`) | Space-filling curves: Hilbert, Hilbert-trans, Z, Z-trans, randomly assigned to layers (`shuffle_orders=True`) | **Yes.** Attention only runs inside consecutive 256-point chunks along the curve, so the curve decides which points attend to each other. The sparse-conv xCPE in each block does not depend on order. |

How the three planes are tied together:
- **Shared drift coordinate.** The drift coordinate, (tick − 800) · 0.0549 cm, means the same thing on all three planes. Attention can use it to link tokens across planes; it is the only position coordinate the planes share.
- **Cross-plane masking.** The MAE's strip masking hides the same 256-tick drift-time window on all three planes at once. This makes the model fill in a masked region from the other tokens of the event, not from a matching unmasked strip on another plane.
- **No direct charge–light alignment.** Light tokens carry a beam-window time profile and a (y, z) PMT position. There is no explicit drift or time alignment between charge and light tokens; the model has to learn their relationship through joint attention.

---

## 4. Backbone architecture (`Encoder`, `Block`, `FMCfg`)

| | qlfm default (`FMCfg`) | qlfm-M (the round-2+ headline model) |
|---|---|---|
| Type | Pre-LN Transformer | same |
| Width d | 256 | 384 |
| Depth | 8 | 12 |
| Heads | 8 (32-d each) | 6 (64-d each) |
| MLP | ratio 4, GELU | same |
| Params | ≈6.6M | ≈21M |

The qlfm-M settings come from `docs/technote-round1.md` (E6) and the `--d 384 --depth 12 --heads 6` arguments in `scripts/make_tasks_r4.py`.

Details shared by both configurations:
- LayerNorm, with attention in bf16 through FlashAttention-2 (varlen).
- Optional linearly scaled drop-path (default 0), no dropout, and no relative position bias.
- **No final LayerNorm** (`final_norm=False`, following PoLAr-MAE). Heads apply their own LayerNorm.

### Pretraining objective: multi-modal MAE (`MAE`, `MAECfg`)
- **Charge masking:** 60% of tokens. Half of that budget is cross-plane **drift-time strips** 256 ticks wide, shared across planes in an event; the rest are random tokens.
- **Light masking**, chosen per event:
  - 40%: mask all 32 PMTs (forces charge → light prediction).
  - 40%: mask a random 50% of PMTs.
  - 20%: mask none.
- **Decoder:**
  - `enc2dec` Linear d → 192, then learned `mask_c` / `mask_l` tokens and a projected copy of the encoder positional embedding.
  - *Design choice:* masked charge tokens keep their full position embedding, including the cell's own charge-weighted centroid and in-cell offset (`pos_mode="centroid"`). This follows the Point-MAE / PoLAr-MAE convention of placing patch centres on the data. The decoder therefore predicts the amount and shape of charge given where it sits, not where tracks are. See `qlfm_architecture_critique.md` §1.1.
  - 3 Transformer blocks (d = 192, 6 heads) over the visible and mask tokens of both modalities, followed by LayerNorm.
- **Heads and losses:**
  - `head_c` → 34 outputs (4 × 8 charge sub-grid + 2 token stats). The targets are standardized per batch, and the loss is MSE on masked charge tokens only.
  - `head_l` → 17 outputs. The loss is MSE on masked PMT tokens only.
  - Total loss = `loss_c + 1.0·loss_l`.
- **Optimizer:**
  - AdamW with betas (0.9, 0.95) and weight decay 0.05 (no decay on bias, norm, and position parameters).
  - LR 1.5e-4 per 256 events, scaled by the global batch size.
  - 5% warmup followed by cosine decay, gradient clip 1.0, bf16.
  - Only the encoder (`fm.pt`) is saved for downstream use.

### Variants and reference backbones
These variants are mainly for comparison:
- **qlfm-SD** (`models/sd.py`): the same FM encoder trained with Panda/DINO-style self-distillation:
  - EMA teacher.
  - 2 global and 6 local views.
  - 4096 prototypes with Sinkhorn.
- **External references** behind the same interface (`qlfm/ref/`):
  - FM4NPP (Mamba2).
  - Panda (PTv3 + Sonata).
  - PoLAr-MAE (ViT-S).

---

## 5. From backbone output to downstream heads

### Backbone outputs
- `FM.forward` returns `zc` [N_charge_tok, d], `zl` [B·32, d], `h_hit` [N_hits, 128], and the tokenizer outputs. Tokens come back in their original order.
- `QLFMBackbone` (`qlfm/ref/qlfm_backbone.py`) exposes the common interface used by `train_downstream.py`:
  ```
  z       = cat[zc, zl]                    # charge first so hit2tok indices stay valid
  evt     = cat[charge tok_evt, light evt] # event index per token
  h_hit   = per-hit 128-d PointNet features
  hit2tok = hit → charge-token index (−1 if dropped by the token cap)
  ```
- Event-level heads therefore pool over both the charge and the light tokens.

### Token → hit propagation
This is a **hard lookup**: each hit takes the feature of the cell token it belongs to (`z[hit2tok]`). There is no kNN interpolation. The head concatenates this token feature with the hit's own `h_hit` embedding, and hits with `hit2tok < 0` get logit −20.

### Training modes (`train_downstream.py --mode`)

| Mode | Backbone | Trainable params | Default LR |
|---|---|---|---|
| `scratch` | random init | all | 3e-4 |
| `frozen` | `no_grad`, eval mode | head only | 1e-3 |
| `finetune` | pretrained | all, layer-wise LR decay (default 0.75) | 2e-4 |
| `linear` | frozen | `MeanLinear` for event tasks; T1/T5 keep their usual heads | 3e-3 |

Other settings:
- **Label fraction:** `--label-frac`, a nested pool from 0.1% to 100% of train (default 1%).
- **Late-light fusion:** `--late-light`:
  - 0: none.
  - 1: `LateLight` appends 32 PMT tokens at the head.
  - 2: `XFusion` cross-attention.
- **T1 per-hit input:** `--hit-feat`:
  - `native`: `h_hit`.
  - `raw`: a `RawHit` MLP 7 → 64 → 64.
  - `none`: no per-hit input.

---

## 6. Downstream heads (`qlfm/models/heads.py`)

### Tasks (`qlfm/data/labels.py`)

| Task | Description | Output | Head | Loss | Primary metric |
|---|---|---|---|---|---|
| **T1 `hit`** | per-hit ν vs cosmic/noise (ν if edep > 0.05 MeV); visible in-AV ν events | 1 logit / hit | `HitHead` | BCE, weighted by log1p(q) | charge-weighted F1 |
| **T2 `evt_nu`** | 3 classes: no ν (≤5 ν hits) / visible in-AV ν (≥30 collection-plane hits) / out-of-AV ν with charge | 3 | `AttnPool` | class-weighted CE | AUC (in-AV ν vs no ν) |
| **T3 `flavor`** | νμ CC / νe CC / NC, for in-FV visible events | 3 | `AttnPool` | class-weighted CE | macro-F1 |
| **T4 `energy`** | regress log(E_ν/GeV), for in-FV visible CC events | 1 | `AttnPool` | Huber (δ = 0.5) | −resolution (half of the 16–84% width of E_pred/E_true − 1) |
| **T5 `light`** | predict the PMT light pattern from charge only | 32 × 17 | `LightHead` (or zero-shot MAE decoder) | MSE (+ extra MSE on total PE) | −MSE |

### Head architectures
- **`AttnPool`** (T2–T4):
  1. LayerNorm(z), then a KV projection d → 2d.
  2. **4 learned queries**, 8 heads, with a per-event segment softmax (`scatter_max` / `scatter_add`) over all of the event's tokens (charge + light).
  3. The 4 pooled vectors are concatenated (4d) → MLP [4d, 256, n_out] (GELU).
- **`MeanLinear`** (linear probe): LayerNorm, then a per-event mean, then one Linear.
- **`HitHead`** (T1):
  - Input is concat[`h_hit` (128), LayerNorm(z)[hit2tok] (d), optional per-event late-light vector (d)].
  - It runs through MLP [in, 128, 128, 1] (GELU) and outputs a per-hit logit.
- **`LightHead`** (T5, supervised):
  1. Projects the charge tokens to 256.
  2. **32 learned PMT queries** run 2 layers of [cross-attention to the event's charge tokens (padding-masked) + MLP ratio 2], with 8 heads.
  3. LayerNorm → Linear(256 → 17), giving [n_events, 32, 17].
  - For T5 the encoder runs on charge only (`use_light=False`, late-light forced off).
- **T5 zero-shot** (`qlfm/t5_zeroshot.py`):
  - Uses no new head. It reuses the pretrained MAE decoder.
  - All charge tokens are visible and all 32 light tokens are replaced by `mask_l` + position embedding.
  - `head_l` predicts the light pattern. This requires a charge + light MAE checkpoint.
- **Optional fusion modules:**
  - `LateLight`: 32 PMT tokens built with a `LightEmbed`-style embedding plus a modality vector, appended to the pooled set.
  - `XFusion`: projects to 256, then 2 layers of charge-token → PMT-token cross-attention + MLP.

### Fine-tuning optimization
- AdamW with weight decay 0.05 (in `finetune` mode, no decay on 1-D parameters).
- 5% warmup followed by cosine decay, gradient clip 1.0, bf16, batch 32 (`--accum` for heavy backbones).
- Steps = clip(30 epochs, 400, 15000).
- Evaluation every 300 steps. The best-val checkpoint is evaluated on the test set (20k events by default; seed 12).

---

## Caveats
- The layer-wise LR decay default is 0.75 in code, but `docs/technote-round2.md` reports 0.9 for its fine-tune runs. Each run's `metrics.json` stores the arguments it was run with.
- The tokenizer's time grid is sized for 8192 ticks, but the stored readout is 6400 ticks.
