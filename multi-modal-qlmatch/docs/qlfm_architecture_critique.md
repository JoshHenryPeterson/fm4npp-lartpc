# qlfm architecture critique: the large design decisions

This note reviews the big architectural choices in qlfm, not its bugs or evaluation hygiene. `docs/audit_round2.md` covers confounds, small inconsistencies and code quality, and this note does not repeat it. Where an audit finding bears on an architecture decision, I link to it and move on.

Scope: backbone, tokenization (charge, light, planes), charge-light fusion, positional encoding, pretraining objective, downstream heads, and the overall picture. For each I give what was chosen, how it compares with the literature, what it means for this physics problem, and what I would try instead.

**Conventions.**
- Claims about PTv3, Sonata, Panda (v1 and v2), PoLAr-MAE, Point-MAE, MAE, I-JEPA, V-JEPA, VideoMAE, NuGraph2 and attentive probing were checked against the primary papers (see [References](#references)). Claims about FM4NPP, SPINE, Wire-Cell, MicroBooNE DL reco and the Mamba variants rest partly on abstracts or search summaries and are worded accordingly. Claims about other methods (MinkowskiEngine, DINO, Swin and so on) remain from general knowledge. I quote no benchmark numbers for external methods.
- Numbers about qlfm come from `docs/technote-round1.md`, `docs/technote-round2.md` and `docs/round1-log.md`.
- File references use `qlfm/...` paths.

**Context that shapes fairness.** This is a prototype. Compute was limited (about 38 node-hours on Polaris for all of round 2, about 2 node-hours for the headline qlfm-M pretraining). The data is about 831k MicroBooNE open-data events. The model is small (21M parameters at most). A choice that looks naive next to a 91M-parameter PTv3 can be right at this scale, and I try to say so.

---

## Summary table

| Decision | Typical / unusual | Verdict | Recommended change | Expected impact |
|---|---|---|---|---|
| Plain global-attention Transformer, d=384, depth 12 (`models/encoder.py`) | Typical for MAE/ViT. Unusual for LArTPC, where sparse CNNs and PTv3 dominate | Good for a first prototype. Fine at about 2k tokens/event | Keep. Add a 2–3 stage hierarchy only if the cell size shrinks or if per-hit work is the goal | Medium, later |
| Fixed 16×64 wire-time cell tokens, per-hit PointNet (`tokenize.py`, `ChargeEmbed`) | Unusual (grid patches with PointNet inside). Close to PoLAr-MAE in spirit, with grid grouping instead of FPS/NMS | Reasonable and cheap. Its main cost is a hard ceiling on hit-level resolution | Smaller cells with a hierarchical or windowed backbone, or a per-hit refinement stage | High for T1, low for T2–T4 |
| Three planes as three independent 2D token sets (`tokenize_charge`, `ChargeEmbed.plane`) | Unusual among ML reco. SPINE works on 3D images. MicroBooNE DL reco and Wire-Cell build 3D spacepoints from the planes [10, 17-19]. SparseSSNet runs per-plane 2D [20]. NuGraph2 keeps per-plane graphs but adds learned cross-plane message passing through nexus nodes [12] | Gives up the detector's main geometric constraint | Add cross-plane geometry: wire-line coordinates in a common (y, z) frame, plus a drift-time-matched cross-plane attention bias | High for hit and topology tasks |
| Early fusion: 32 PMT tokens in the same sequence (`FM.encode`) | Unusual but sensible (similar to single-stream multimodal encoders such as MultiMAE [15]). Little precedent in LArTPC (closest: Y-Net [14]) | Strongest single choice. Supported by the T2 results | Keep. Add PMT-to-drift-time alignment cues. Try a few registers for light | Medium |
| Light token = 16-bin time profile + PMT position (`tokenize_light`, `LightEmbed`) | Reasonable. Hand-binned | Good for a first version | Binning is fine. Add a charge-weighted drift-time to flash-time relation as an input | Low-Medium |
| Centroid + Fourier positional encoding, additive, absolute (`ChargeEmbed.pos_embed`) | Typical for point-MAE. Absolute Fourier may be weaker than the modern choices (untested here) | Adequate. Not the bottleneck | Try RoPE or a relative bias in (drift, wire) space. Low priority | Low-Medium |
| MAE with 60% masking, drop masked tokens, small decoder, standardized regression targets (`MAE`) | Typical (60% masking as in Point-MAE and PoLAr-MAE [4, 5]; image MAE uses 75% [23]). Unusual on cell-histogram targets | Right first choice. The results back it | Keep MAE as the base. Add a light-prediction head or contrastive charge-light term. Re-run SD with tuned LR before dismissing it | Medium |
| Heads: 4-query attentive pool, hard-lookup per-hit head, frozen or fine-tune (`models/heads.py`) | Typical (attentive probe [24, 25]). Hard lookup is less common than interpolation or a UNet-style decoder | Event heads are good. The T1 head caps what the backbone can show | Per-hit head with local attention or interpolation, plus a fine-tune track as the headline | High for T1 |
| No hierarchy, no CLS, single scale (whole model) | Unusual against PTv3, SPINE UResNet, Sonata | Acceptable for event-level tasks. Weak for dense prediction | Add a decoder-side U-Net only if per-hit becomes the headline | Medium |

---

## 1. Backbone

### 1.1 What was chosen

A pre-LN Transformer with global bidirectional attention inside each event. See `Encoder`, `Block` and `pack` in `qlfm/models/encoder.py`.
- FlashAttention-2 varlen: a block-diagonal mask per event, no padding.
- Defaults: d=256, depth 8, 8 heads (6.6M). The headline qlfm-M uses d=384, depth 12, 6 heads (21M).
- Single scale: every layer sees the same tokens at the same resolution.
- No CLS or register tokens, no relative position bias, no final norm (`final_norm=False`).
- The sequence is about 2k charge tokens per event (median), 3.4k at the 99th percentile, capped at 4096, plus 32 light tokens.

### 1.2 How this compares

- **MAE / ViT recipe: typical.** This is the standard MAE encoder. Dropping masked tokens in the encoder is the ViT-MAE trick. PoLAr-MAE uses the same kind of backbone (ViT-S on about 400 group tokens) [5, 23].
- **LArTPC practice: unusual.** SPINE / lartpc_mlreco3d uses sparse CNNs (MinkowskiEngine UResNet), followed by graph networks for clustering and particle identification [10]. Panda uses PTv3 with Sonata-style self-distillation [1, 2, 11]. Both are hierarchical, with local operators. A flat global Transformer over cells is not how the field has done it.
- **PTv3 / Sonata: different design philosophy.** They serialize points along space-filling curves, attend inside windows (256 points in Panda v1, per `ref/pimm_configs/panda_uboone.py` and the Panda paper [2]; Panda v2 uses 1024 [3]), use sparse-conv positional encoding (xCPE), and pool through several stages [1]. They are built for large-scale 3D scenes (PTv3 widens the attention receptive field from 16 to 1024 points).
- **SSM / Mamba: a different trade-off.** The round-1 `docs/ssm-survey-round1.pdf` concludes that SSMs are not a drop-in replacement today. The FM4NPP result in round 2 is not a clean SSM-vs-Transformer comparison, because the tokenization, size and ordering all differ. Mamba pays off only at per-hit sequence lengths (about 14.5k tokens/event). Vanilla Mamba scans causally, so it needs a fixed token order. Bidirectional or multi-directional scan variants exist (e.g. Mamba-based MAEs [7, 8]), but they add engineering, and the survey's concern about masking still applies. FM4NPP itself uses a Mamba-2 backbone over serialized TPC spacepoints [9].

### 1.3 Strengths and weaknesses here

Strengths:
- **Token count fits global attention.** Two thousand tokens, with only 40% visible during pretraining, is cheap on an A100. Global attention lets a ν track in one corner of a plane relate to a cosmic across the TPC, and to the flash.
- **It is the natural place to put early fusion.** Charge and light are the same kind of object (tokens) in one attention graph. A sparse CNN cannot do this cleanly, because the PMTs have no grid.
- **It scales in a way the data shows.** S to M (6.6M to 21M) improves every probe, and data scaling reappears at 21M (`docs/technote-round1.md`, E6). This is the strongest evidence that the backbone is a sound base.
- **The results are good.** qlfm-M beats Panda at matched pretraining node-hours on all four tasks (`docs/technote-round2.md`). That is despite Panda having far more architectural machinery. The caveat from `audit_round2.md` §4.2 still applies: qlfm sees light inside the backbone.

Weaknesses:
- **No locality prior.** Wire-time images are locally smooth, with tracks as thin lines. CNNs and windowed attention get locality for free. A global Transformer has to learn it from 831k events. This is likely why the model needs about 8.7M events seen to do well, and why it is capacity-limited at 6.6M.
- **Single scale.** Event-level tasks (T2–T4) want a global summary. Dense per-hit tasks (T1) want fine resolution plus context. A flat model at cell resolution does the first well and the second only through the head (Section 6).
- **Quadratic cost caps the token count.** Going to per-hit tokens, or to cells 4× smaller, costs roughly 16× in attention. The design is locked to coarse cells unless the backbone changes.
- **Size relative to data.** The model is small for 831k events, and the round-1 scaling result says capacity, not data, was binding. At 21M it is still below Panda's 91M and PTv3-class models. There is likely more to gain from width and depth. The limit was compute (2 node-hours for qlfm-M), not design.

### 1.4 Alternatives

| Alternative | Likely better? | Reasoning |
|---|---|---|
| **PTv3 / Panda-style serialized windowed attention** | Not for qlfm's current tokens. Yes if tokens become per-hit | Windows exist to make per-point attention tractable. At 2k cell tokens they add nothing but locality. Panda's own score here (strong charge-only T2) suggests locality helps the charge-only path, but it does not use light natively. |
| **Sparse CNN UResNet (MinkowskiEngine / SCN)** | Better for T1-type dense tasks. Worse as a general multimodal backbone | Strong locality and a ready-made decoder, and it is the field standard. It cannot take PMT tokens without a bolt-on, and MAE on sparse CNNs needs care to avoid mask leakage (general knowledge). A hybrid is more attractive: a sparse-conv stem on hits that produces tokens, then the Transformer. |
| **Hierarchical Transformer (Swin / PTv3-like stages)** | Probably better at smaller cell sizes | Cells of 8×32 would give about 4× the tokens. A 2–3 stage hierarchy would pool back to about 2k tokens in the deep layers. The cost is real engineering, and MAE with masking on hierarchies needs masking-aware merging. |
| **Mamba / Mamba-ND** | Not likely better here | Advantage is linear scaling at per-hit length, which qlfm does not need at 2k tokens. A plain causal scan is a poor fit to MAE masking (the survey makes the same point). Bidirectional variants exist [7, 8]. Revisit only if the per-hit-token route is taken. |
| **Wider and deeper plain Transformer (d=512–768, depth 12–16)** | Probably yes, cheaply | The scaling evidence says capacity is binding. It costs the least engineering. |

**Verdict.** Keep the backbone for now. It is well matched to the token count and to early fusion, and the evidence supports it. The cheapest real upgrade is scale. The most useful structural upgrade is a hierarchy or local mixing, but only if the tokenization changes (Section 2).

---

## 2. Tokenization of charge

### 2.1 What was chosen

`tokenize_charge` in `qlfm/tokenize.py`: one token per occupied (event, plane, wire-cell, time-cell) cell of 16 wires × 64 ticks. `ChargeEmbed` runs a per-hit MLP (4 → 64 → 128), then max+mean pooling over the cell's hits, then a projection to d. The MAE target is a 4×8 sub-grid histogram of charge inside the cell, plus two statistics. Hits are mapped back to tokens with `hit2tok`.

### 2.2 How this compares

- **Closest relative: Point-MAE / PoLAr-MAE.** Both build groups of points and embed each group with a mini-PointNet. Point-MAE picks centres by FPS and groups by kNN [4]. PoLAr-MAE picks centres with C-NMS (candidates from FPS or the whole cloud) and groups by ball query, capping groups at 32 points [5]. qlfm replaces that with a fixed grid.
- **Fixed grid versus FPS.** FPS and C-NMS adapt to density: more groups where there is data. A grid does not, but is deterministic, GPU-fast, needs no neighbour search, and has no group overlap. PoLAr-MAE introduces C-NMS because FPS with kNN or ball query gives either many ungrouped points or heavy patch overlap on LArTPC clouds with variable density. C-NMS gives a tunable overlap, and in the paper's own timings it is faster than FPS+kNN and slower than FPS+ball query [5]. For 2D wire-time hit images, a grid is a natural choice. The round-2 notes record that PoLAr-MAE needed a larger ball radius and up to 1,024 groups to cover a plane.
- **Versus ViT patches.** The grid is a ViT patch grid, made sparse by keeping only occupied cells. That is a sound adaptation. The unusual part is the per-hit MLP and pooling inside the patch, which keeps the exact hit positions and charges.
- **Versus voxels / serialization.** Panda (built on PTv3 and Sonata) works on 3D voxelized deposits at a 3 mm pitch, one point per occupied voxel [2]. Those points go through serialized windows of 256 points and several pooling stages. Resolution is much finer, and cost is controlled by windowed attention and pooling.
- **Versus per-hit tokens.** FM4NPP serializes TPC hits with a hierarchical raster scan and runs a Mamba SSM over them [9]. I have not confirmed its exact tokenization. SPINE works on 3D images (voxels) with sparse convolutions [10].
- **Versus learned or adaptive patching.** Rare in this field. In general vision, adaptive patching (merging tokens by content) exists but is not standard.

### 2.3 Strengths and weaknesses here

Strengths:
- **Cheap and stable.** About 2k tokens per event, tokenized on the GPU with a `torch.unique`. This is what made pretraining on 8.7M events affordable, against 10–30× more cost per step for the point-cloud references (`docs/round2-log.md`).
- **The 4×8 histogram target is well chosen.** It is a dense, fixed-size, permutation-free target for a variable-size set. This avoids Chamfer loss, which PoLAr-MAE needs because its targets are point sets. For sparse, line-like 2D images, a histogram of log-charge is a sensible reconstruction target.
- **The token carries intra-cell structure.** Pooling max and mean over MLP-embedded hits keeps fine detail that a plain patch mean would lose.
- **Wire-time geometry is native.** Cells are aligned with how the detector reads out. There is no 3D reconstruction and no ambiguity from matching.

Weaknesses:
- **Resolution is fixed and coarse.** A cell is about 4.8 cm in wire by 3.5 cm in drift. A MIP track and a nearby delta ray, or a ν track crossing a cosmic, can share a cell. `audit_round2.md` §2.2 covers the immediate T1 consequence. The architectural point is that the whole system is built around this one scale, so changing it means retraining and revisiting the token cap.
- **Mixed content in one token.** A cell may hold a cosmic muon, a ν vertex, or noise. The pooled embedding averages across these, so the label for the token is not well-defined. The T1 result of 0.89 frozen, and 0.947 fine-tuned at 1% labels, suggests the per-hit MLP `h_hit` carries much of the load (round-2 table, `docs/technote-round2.md` §3.4), not the contextual token.
- **Grid-edge artefacts.** The same track gets a different tokenization if it sits across a cell boundary. A random per-event grid offset at training time would cheaply remove this sensitivity, and is a standard augmentation in patch models. I did not find it in `tokenize.py`.
- **Token count varies with occupancy, not with information.** A long straight cosmic uses many cells, and a dense shower uses few cells that hold a lot. There is no mechanism to spend more tokens on the interesting region.

### 2.4 Hit-level tasks

This is where the tokenization hurts most. T1 uses `z[hit2tok]` plus `h_hit` (Section 6). Everything that distinguishes two hits in the same cell comes from a 4-feature per-hit MLP that sees no neighbours. The backbone's context is the same for every hit in a cell. So hit-level quality is bounded by the grid, however good the backbone is.

### 2.5 The three planes

Each plane is tokenized separately and tagged with `Embedding(3, d)`. Cells are indexed by (plane, wire cell, time cell), and nothing links cells across planes except the shared drift coordinate. `audit_round2.md` §2.1 notes the wire-coordinate inconsistency. The architectural point is larger.

- **What the field does.** MicroBooNE reconstruction (Pandora, Wire-Cell, the DL-based chain) uses the fact that a 3D point projects to one wire on each of three planes. MicroBooNE's DL chain (LArMatch) runs a sparse U-Net on the three images, proposes candidate 3D spacepoints non-ML and scores them with an MLP [19]. Wire-Cell builds 3D tomographically from the redundancy of all three planes [17, 18]. SPINE takes a 3D image [10], and SparseSSNet does per-plane 2D semantic segmentation [20]. Panda in this repo treats each plane as a 2D cloud with an offset in a fake third axis (`plane·10 cm`, per `docs/technote-round2.md` §2.2), which also discards cross-plane geometry.
- **Precedents for keeping planes separate but linked.** NuGraph2 keeps per-plane hit subgraphs and adds learned cross-plane message passing through "nexus" nodes tied to spacepoints [12]. The Heterogeneous Point Set Transformer does something similar with intra-view and inter-view attention, though on simulated multi-view data from a pixel detector, not real wire planes [13]. Both show that a per-plane design can still share information across planes.
- **qlfm's choice is consistent with Panda's,** and with the rest of the repo's "no 3D reconstruction" principle, which keeps the model robust to ghost and ambiguity problems. That is a defensible prototype decision.
- **Cost.** Each event is effectively three images with a shared drift axis and an inferred relationship. The model can use drift time as a handle (the MAE strip masking is tied to it), but it must learn the wire-angle geometry from data. This is the biggest piece of known physics left on the table.

### 2.6 Alternatives

| Alternative | Likely better? | Reasoning |
|---|---|---|
| **FPS + kNN grouping (Point-MAE style)** | No | More expensive, adds neighbour-search cost, and has overlapping groups. It buys density adaptivity, which is moderate on 2D images. It would not fix hit-level resolution. |
| **Smaller cells (8×32 or 4×16)** with the current backbone | Partly | Improves T1 and reduces mixed-content cells. It multiplies tokens about 4–16×, so it needs the token cap and attention cost addressed. The audit already suggests an ablation. |
| **Multi-scale tokens: coarse cells for context, fine cells near the ν candidate** | Probably better | Spends tokens where the information is. Requires a hierarchy or a two-pass scheme. |
| **Per-hit tokens + sparse windowed attention (PTv3-like)** | Better for T1. Costs engineering and the light-fusion simplicity | Removes the resolution ceiling. Combine with a cell-level global stream to keep global context cheap. |
| **Sparse-conv stem then cell tokens** | Likely a good compromise | A small submanifold sparse conv over hits gives every hit neighbour-aware features before pooling. That gives `h_hit` real context for almost no token-count cost. |
| **Grid-offset augmentation** | Yes, cheap | Randomizes cell boundaries at train time. A few lines. |
| **Cross-plane tokens** (3D-aware: tokens at matched (y, z, drift) from wire-line intersections) | Higher payoff, higher risk | Gives the model exact 3D structure. Ghosts and matching ambiguity are real. A softer version (a shared (y, z) frame in the positional encoding) gets most of the benefit with no matching. |

**Verdict.** The cell tokenization is a reasonable prototype choice, and it is the reason the project could pretrain on millions of events. Its limits show on hit-level tasks and on plane handling. The most valuable cheap fixes are a sparse-conv or local-attention stem (per-hit context) and a shared-frame positional encoding (cross-plane geometry).

---

## 3. Light tokenization and charge-light fusion

### 3.1 What was chosen

- **Light tokens** (`tokenize_light`): exactly 32 tokens per event, one per PMT. The feature vector is `log1p(PE)` in 16 time bins (finest around 3–6 µs) plus total PE. `LightEmbed` adds a Fourier embedding of the PMT (y, z) and a learned PMT-ID embedding.
- **Fusion** (`FM.encode`): early fusion. Charge and light tokens are concatenated into one sequence per event with a modality embedding, then joint self-attention for all layers.
- **Masking:** light is fully masked in 40% of events and half masked in another 40%, so the model must predict light from charge.
- **Alternatives built:** `LateLight` and `XFusion` in `qlfm/models/heads.py`.

### 3.2 How this compares

- **Early fusion in one stream** is common in recent multimodal models (single-stream encoders, e.g. MultiMAE [15], which feeds visible patches of several modalities to one shared encoder, and single-stream vision-language Transformers). CAV-MAE [16] sits between early and late fusion: separate encoders plus a joint layer. Cross-attention fusion (two towers plus cross-attention) and late fusion (separate encoders, combine features) are the other main families.
- **In LArTPC work,** light is typically used in a late, hand-crafted way: flash matching for Pandora-based selections [21] and Wire-Cell's charge-light matching [18], and PE or flash features in downstream classifiers. To my knowledge, learned fusion of PMT time profiles with wire charge has little precedent. The closest I found is Y-Net (a student technical report, not peer reviewed), a sparse U-ResNet with a separate light encoder that merges with charge features in latent space [14]. SPINE, PoLAr-MAE and Panda (charge-only) do not use light. The survey notes none of the references use light.

### 3.3 Strengths and weaknesses here

Strengths:
- **It works, and the evidence is the best in the repo.** T2 AUC goes to 0.999 with early fusion. Adding the same late-fused light to other backbones lifts them from 0.61–0.91 charge-only to 0.96–0.98 (`docs/technote-round2.md` §1 and §3.1). Light is the dominant cue for neutrino versus cosmic [22], and early fusion extracts more of it than late fusion.
- **The physics fits.** Light tells you when a beam-spill interaction happened, and where roughly (PE pattern over PMTs). That is what lets the model decide which charge is in-time. Joint attention allows charge tokens to condition on that.
- **Only 32 tokens, so it is cheap.** Light adds under 2% to the sequence.
- **The light-masking schedule is a good idea.** Forcing charge-to-light prediction creates a self-supervised charge-light association, and it makes T5 zero-shot possible.
- **PMT position is encoded sensibly.** Fourier(y, z) plus a learned ID covers both geometry and per-channel gain.

Weaknesses:
- **No explicit time alignment between the two modalities.** Charge tokens carry drift time, which is relative to an unknown interaction time. Light tokens carry absolute time in the beam window. The link (drift distance = (tick − trigger) · velocity relative to the true t0) is physical but must be learned. The architecture gives attention no hook for it. This is the key physical relationship in charge-light matching, and the model has to discover it.
- **Light is 32 tokens among about 2,000.** Attention may under-weight a small group, though the T2 results suggest it does not in practice.
- **Light time profile is hand-binned.** The edges are tuned around the beam flash. This is reasonable. It does mean cosmic out-of-time light is coarsely resolved, and cosmic rejection depends partly on that.
- **No PMT-pair relationships.** Each PMT is its own token, with geometry only through the positional embedding. That is fine. A graph or neighbour-aware prior would add little at 32 sensors.
- **Early fusion confounds evaluation.** See `audit_round2.md` §4.2. This is an evaluation problem, not an architecture one.

### 3.4 Alternatives

| Alternative | Likely better? | Reasoning |
|---|---|---|
| **Cross-attention fusion** (charge queries, PMT keys, in a few layers) | Probably not better than joint self-attention | With 32 PMT tokens, joint attention costs almost nothing, and it lets light attend back to charge. Cross-attention is cheaper only at large sizes. The repo's `XFusion` is the late version of this, and it is weaker. |
| **Late fusion** | Worse | Round 2 shows this directly: late light lifts the other backbones, but qlfm's early fusion goes further. |
| **Explicit charge-light matching loss (contrastive, CLIP-style)** | Worth adding as an auxiliary loss, not as a replacement | MAE already forces light prediction from charge. A contrastive term between a pooled charge embedding and the light embedding would add pressure toward matching. It does not fit well with the many cosmics per event, where there is not one matching flash. A flash-to-slice assignment target would be a more natural version. |
| **Flash-time-aware charge positional encoding** | Yes, cheap and physical | Provide each charge token with a "candidate x position" relative to the dominant flash time, as an extra feature or an attention bias. This gives the model a direct handle on the physics. |
| **Light register / summary tokens** | Maybe | A few extra light tokens (e.g. total flash, earliest flash) as a compact summary. Cheap to try. |

**Verdict.** Early fusion in one stream is the strongest architectural decision in the project, and I would keep it. The most promising addition is a physics-informed alignment cue between drift time and flash time, not a different fusion topology.

---

## 4. Positional encoding

### 4.1 What was chosen

`ChargeEmbed.pos_embed` with `pos_mode="centroid"`: the sum of a plane embedding, `Fourier` features of the charge-weighted centroid (drift cm, wire cm) with scales 300 and 1100 cm, and an MLP of the centroid's offset within the cell. It is added to the token once at the input. Light uses Fourier of the PMT (y, z) plus a PMT-ID embedding. No RoPE, no relative bias, no sequence position.

### 4.2 How this compares

- **Typical:** Fourier or MLP absolute position, added once. Point-MAE, PoLAr-MAE and most point-cloud ViTs do this. Placing the position at a data-derived location, not at a grid centre, follows Point-MAE and PoLAr-MAE. Both use a learned MLP of the patch-centre coordinates [4, 5], where qlfm uses Fourier features of a charge-weighted centroid.
- **Newer practice:** PTv3 uses sparse conv for implicit relative position (xCPE) [1]. Panda v2 moves to a LitePT backbone with 3D rotary positional embeddings [3]. Relative or rotary schemes are now common in general vision Transformers too.

### 4.3 Strengths and weaknesses here

Strengths:
- **Simple and adequate.** The model has a smooth coordinate-based position at multiple scales. The in-cell offset gives sub-cell detail.
- **Tying position to the data (centroid) is a deliberate choice.** The decoder knows where charge is and predicts how much and what shape. That is a valid MAE task, but it means the model never has to predict where tracks are. See Section 7, where this affects what the MAE learns.

Weaknesses:
- **Absolute and once only.** Tracks are defined by relative geometry: direction, adjacency, parallel lines. Absolute Fourier features must be decoded into relative relationships by the MLPs. RoPE or a relative bias builds it into every attention layer.
- **Fourier with low-frequency scales** (300 and 1100 cm, 10 frequencies) can limit fine relative resolution. The in-cell MLP compensates inside one cell, not across cells.
- **Cross-plane position is inconsistent.** See Section 2.5.
- **No distance bias.** Attention has no prior toward near neighbours. This is part of why a global Transformer needs a lot of data.

### 4.4 Alternatives

- **2D RoPE in (drift, wire-cm) per plane**, with the same drift axis shared across planes. I think this is the most likely small improvement in the backbone, but it is not a priority. Expected effect: better sample efficiency and better local structure.
- **Relative distance bias** (a learned bias on (Δdrift, Δwire)). FlashAttention-2 supports ALiBi slopes but not an arbitrary additive bias tensor [6], and ALiBi slopes are tied to the 1D token index, so a 2D learned bias would need a kernel that supports arbitrary additive biases. RoPE is the practical choice with FlashAttention-2.
- **Keep absolute Fourier, but with a shared (y, z, drift) frame across planes.** This is the cross-plane fix from Section 2.5, and probably larger than RoPE.

**Verdict.** Adequate. The upgrade with the most value is the cross-plane geometric frame, not RoPE.

---

## 5. Pretraining objective

### 5.1 What was chosen

`MAE` / `MAECfg` in `qlfm/models/fm.py`:
- 60% of charge tokens masked, half as 256-tick drift strips shared across planes, half random. Masked tokens are dropped from the encoder.
- Light masking: all 32 PMTs (40%), half (40%), none (20%).
- Decoder: 3 blocks, d=192, over visible tokens and mask tokens. Masked tokens keep their full positional embedding, including the centroid.
- Loss: MSE on a 34-d standardized target for charge, and MSE on 17-d light features.
- Alternative tried: `SelfDistill` in `qlfm/models/sd.py` (Panda-like EMA teacher, 2 global + 6 local views, 4096 prototypes, Sinkhorn). It underperformed.

### 5.2 How this compares

- **MAE with high masking, dropped tokens, light decoder: typical.** That is the standard MAE recipe [23], followed by PoLAr-MAE for points [5].
- **Regression onto a cell histogram: reasonable and close to MAE's pixel regression.** MAE found per-patch-normalized pixel targets better than raw pixels [23]. qlfm's standardization is per-feature across the dataset, which is a different choice. PoLAr-MAE uses Chamfer distance on masked point patches plus an auxiliary L2 per-point energy head [5], so qlfm's regression onto a charge histogram is simpler but in the same spirit as the energy head.
- **Strip masking across planes: a thoughtful, physics-aware choice.** Tube masking is used in VideoMAE because video is temporally redundant: a masked patch can be copied from its temporal neighbours [26]. Strip masking across planes plays a similar role here. Note that for images MAE found random masking better than block masking at 75% [23], so the strip share (50%) is a design choice that should be ablated.
- **Teacher-student (DINO, iBOT, Sonata, Panda) is the other main family.** Sonata argues that point-cloud pretraining is prone to a "geometric shortcut": representations collapse to low-level cues such as normals and height. It also argues that generative targets anchor features to predefined cues, and that coordinates leak spatial information past the mask. It counters this with self-distillation on masked and local views, plus jitter on masked points [11]. Panda follows this recipe [2], and argues explicitly against PoLAr-MAE-style reconstruction on a coarse, fixed-radius tokenization, which it says limits high-data performance. qlfm uses a similarly coarse cell with reconstruction, so that criticism applies in part. Panda also omits a [cls] token, because LArTPC events contain causally unrelated interactions [2].
- **JEPA (I-JEPA, V-JEPA)** predicts in representation space from context, with no pixel decoder. It aims to avoid spending capacity on low-level detail, though the V-JEPA paper only shows a frozen-evaluation advantage on video [25]. Both use an EMA target encoder [25, 27]. It is not used in this repo.

### 5.3 Strengths and weaknesses here

Strengths:
- **MAE is the right first choice for this data.** Wire images are sparse, noise-dominated in places, with fine structure. Reconstruction gives a dense, stable signal. It needs no augmentations, no teacher, and no careful hyperparameter tuning. The result is the best-performing model in the repo, so the empirical case is strong.
- **Charge-to-light prediction is a real pretext task with physical content.** It forces the model to learn what a given charge deposit looks like in the PMTs. T5 zero-shot reuses the pretrained decoder for this.
- **Dropping masked tokens makes pretraining 2–3× cheaper.** This is what made 8.7M events possible.

Weaknesses:
- **Reconstruction of charge amounts can reward low-level texture.** Predicting the 4×8 histogram given the centroid position can be solved partly from neighbours' charge. This may under-incentivize topology and particle-type understanding. The frozen flavor result (0.71–0.76 macro-F1) shows the features are useful but not near the ceiling. Pandora + BDT is competitive at 0.1% labels.
- **Giving the decoder the masked token's centroid (as in Point-MAE [4]) makes the task easier.** A deliberate choice, but it means the model does not learn "where do tracks go". It is not necessarily bad, but it is the main reason to suspect the MAE under-trains global structure. Masking the position too (predict occupancy and centroid) would be a harder and arguably more informative task.
- **Light is reconstructed in log space with MSE.** The 1.0 weight is arbitrary (`audit_round2.md` §1.3). A flash is a sharp temporal peak, so MSE on binned log-PE may favour the bulk shape.
- **SD underperforming is not evidence against self-distillation.** The round-2 notes attribute it to an inherited low learning rate (about 1.9e-5), so it is closer to an untested hypothesis than a verdict.
- **No cross-view or augmentation-based invariance is learned.** MAE learns equivariance-ish features. Downstream tasks such as flavor depend on invariance to nuisance factors (position in the detector, accompanying cosmics). The T2–T4 heads must learn that from labels.

### 5.4 Alternatives

| Alternative | Likely better? | Reasoning |
|---|---|---|
| **MAE as is** | Baseline | Strong result, low cost. |
| **MAE with harder targets** (decoder does not get the centroid, or predicts occupancy and centroid too) | Possibly | Forces global geometry. Risk: the task is harder and noisier. Run as an ablation. |
| **MAE + DINO-style or JEPA-style latent loss** (multi-task) | Possibly the best long-term | Combine local reconstruction with a global, invariant objective. This is where Sonata and Panda went. It needs a tuned teacher (EMA, temperature, LR). |
| **Re-run SD with the correct learning rate** | Worth one run | The only reason to drop it was likely an LR confound. A fair test costs about 2 node-hours. |
| **JEPA (predict masked-token latents from visible tokens)** | Plausible, untested | Avoids spending capacity on noise-level detail. With strips masked across all planes, the latent target is quite well defined. It would need an EMA target encoder. Not a priority. |
| **Contrastive charge-light matching** | As an auxiliary term | Gives a direct signal for the physics link. Hard in multi-flash events (see Section 3.4). |
| **Event-level auxiliary targets** (pretrain with Pandora-derived labels such as slice count, nu_score) | Possibly high value, but it changes the claim | Reconstruction features are the strongest low-label prior (BDT competitive at 0.1%). Using them as weak labels in pretraining could close that gap. It makes the model not purely self-supervised, though the repo already keeps truth and baseline columns separate on purpose. |

**Verdict.** MAE is a good fit, and the evidence supports it. The recommended improvements are additive: a harder or more global component, not a replacement. I would hold off on JEPA until the simpler tweaks are tried.

---

## 6. Downstream heads

### 6.1 What was chosen

In `qlfm/models/heads.py` and `qlfm/train_downstream.py`:
- **Event heads (T2–T4):** `AttnPool`, 4 learned queries with segment softmax over all charge and light tokens, then a 4d → 256 → out MLP.
- **Linear probe:** `MeanLinear` (diagnostic).
- **Hit head (T1):** `HitHead` takes `h_hit` (the 128-d per-hit MLP feature) concatenated with `z[hit2tok]` (hard lookup of the cell token) and an optional light vector, then a 3-layer MLP.
- **Light head (T5):** `LightHead`, 32 PMT queries cross-attending to charge tokens.
- **Modes:** `frozen`, `finetune` (layer-wise LR decay), `scratch`, `linear`.

### 6.2 How this compares

- **Attentive probes** (a learned-query cross-attention readout over frozen tokens) are widely used to evaluate masked-modelling features (V-JEPA [25]; Psomas et al. [24]). Linear probes on MAE features understate their quality: He et al. report that linear-probe and fine-tune results are largely uncorrelated [23], and Psomas et al. attribute the gap to discriminative information being spread over patch tokens rather than concentrated in one token (MAE ViT-B: 67.7% linear vs 75.6% attentive on ImageNet-1K) [24].
- **Hard token-to-point lookup** is cruder than the common alternatives. PointNet++ decoders interpolate (kNN, inverse-distance-weighted feature propagation) or un-pool through a learned hierarchy. Point-MAE's part-segmentation head is simpler: it concatenates features from several Transformer layers and up-samples the group-center features to the points [4]. PoLAr-MAE likewise uses simple feature-upsampling heads [5]. Panda uses an optional 16M-parameter UNet-like multi-scale decoder [2]. So qlfm's hard lookup is cruder than these, but it is not the only simple choice in the literature.
- **Frozen plus fine-tune reporting** is the standard SSL protocol (MAE, PoLAr-MAE and Panda all report frozen or linear, and full fine-tune results [2, 5, 23]).

### 6.3 Strengths and weaknesses here

Strengths:
- **Event heads are well designed.** An attentive probe with four queries lets different queries specialize (charge topology, light). It is small and cheap and gives the backbone a fair chance. The results show this.
- **Frozen and fine-tune with layer-wise decay is the right pair.** The label-efficiency curves (0.1% to 100%) are informative for this problem.
- **Heads are identical across backbones in round 2,** which is a sensible design for comparison.

Weaknesses:
- **The T1 head limits what the backbone can show.** It does not interpolate or look at neighbours. Per-hit discrimination inside a cell comes from the 4-feature `h_hit` MLP, which is part of the tokenizer and is trained during pretraining only through the pooled token. The gap between frozen (0.892) and fine-tuned (0.947) T1 at 1% labels suggests the frozen representation does not hold everything the head needs. That is consistent with the hit MLP and the token being under-specified for hits.
- **`h_hit` has an unclear status.** It is part of the model, not the head, but it is a per-hit feature built from a 4-d input, so it behaves like raw-feature input. This makes frozen T1 a mix of backbone quality and a hand-fed per-hit baseline. (`audit_round2.md` §4.3 covers the comparison issue. The architectural issue is that the interface lets weak backbones be rescued by `h_hit`, and strong ones be capped by it.)
- **The frozen headline hides a fine-tune gap for others.** For qlfm the frozen and fine-tuned results are close on T2–T4, which is good. For the reference backbones, fine-tuning behaved differently, and the Panda fine-tune was head-only until re-run. This means that "frozen" may be the fairest column, but heads were not designed to show fine-tune quality across backbones.
- **Four queries with a large MLP** (4d to 256) is a small amount of structure. Fine for classification, but energy regression depends on summing charge, which attention (a weighted average) does poorly at without a count or sum. The tokens carry `log n_hits` and `log1p(Σq)` stats, which helps.
- **No dense head beyond T1.** Segmentation of particle type or instance clustering, the actual reconstruction tasks of SPINE, would need a real decoder.

### 6.4 Alternatives

| Alternative | Likely better? | Reasoning |
|---|---|---|
| **Interpolated hit head** (kNN over token centroids on the same plane, distance-weighted) | Yes, cheap | Gives each hit a position-aware blend of neighbouring contexts. Removes the stair-step artefact at cell edges. |
| **Hit cross-attention to nearby tokens** | Likely better | A small per-hit transformer layer: each hit attends to tokens in the surrounding 3×3 cells. More expensive, much more expressive. Likely the largest T1 gain per unit of work. |
| **Sparse-conv refinement of per-hit features** | Likely better | Uses neighbouring hits directly. Costs a small new module in the head or tokenizer. |
| **Unfrozen last N blocks** (partial fine-tune) | Possibly | A middle ground between frozen and full fine-tune that keeps the probe cheap. |
| **Sum-plus-attention pooling** for energy | Minor | Add a sum-pooled charge feature to the pooled vector. |
| **Report a standard linear-probe row in every table** | Yes | `MeanLinear` exists but T1 and T5 do not have a true linear probe. |

**Verdict.** The event heads are good and give the backbone a fair chance. The T1 head is the weak spot, and fixing it is the highest-value head-side change. I would also headline a fine-tune row, since the frozen row is bounded by the head for hit tasks.

---

## 7. Cross-cutting observations

**What the architecture is optimized for.** It is optimized for event-level decisions at coarse resolution with light, and for cheap pretraining at scale. It is much less suited to per-hit or per-particle reconstruction, which is what SPINE and Panda (in its published uses) are built for [2, 10]. The headline results match: great at T2, strong at T3 and T4, and T1 good only because of the per-hit MLP and fine-tuning.

**The MAE's blind spot is geometry across planes.** The planes, wire angles, and time-to-position relationships are known exactly. qlfm learns them from data, spending capacity and pretraining signal on what could be supplied. This links tokenization (Section 2.5), position (Section 4) and objective (Section 5).

**Scale is the cheapest knob.** Round 1 shows data scaling reappears at 21M. The model is still small compared with the references, and the node-hours were small.

---

## 8. Typical versus unusual

| Choice | Status |
|---|---|
| Flat Transformer with global attention | Typical for MAE, unusual for LArTPC |
| Dropping masked tokens, light decoder | Typical |
| Fixed-grid patches with PointNet inside | Unusual combination, sensible |
| Histogram regression target | Unusual, sensible |
| Per-plane 2D tokens, no 3D reconstruction | Shared with Panda here. Unusual against SPINE. NuGraph2 links per-plane graphs across planes [12] |
| Early fusion of PMT tokens | Unusual in LArTPC (I found no published precedent beyond Y-Net's latent fusion [14]), found in multimodal learning [15] |
| Centroid-based Fourier position, additive | Typical |
| Attentive-probe event heads | Widely used [24, 25] |
| Hard lookup per-hit head | Cruder than the interpolating or decoder-based heads in PointNet++, PoLAr-MAE and Panda. Weak here (qlfm audit) |
| Strip masking across planes | Thoughtful, domain-specific |

---

## 9. Top changes, ranked by expected impact

1. **Give the model the detector geometry across planes.** Map every (plane, wire) to a shared (y, z) frame using the known wire angles, and use it, with drift, in the position encoding. If it is worth the effort, add a cross-plane attention bias for tokens at matching drift time. This uses exact known physics, applies to all tasks, and costs little compute. Risk: low for the encoding, moderate for the attention bias. *Expected impact: high on topology and flavor (T3, T4), medium on T1.*

2. **Fix the hit-level path.** Replace the hard `z[hit2tok]` lookup with an interpolated or locally attended lookup, and give `h_hit` neighbour context (a sparse-conv stem or per-hit local attention). The tokenizer stays as it is. This lifts T1, and makes the frozen T1 number reflect the backbone. *Expected impact: high on T1, none on T2–T4.*

3. **Scale the backbone (width, depth, pretraining steps) before redesigning it.** The round-1 data suggests capacity binds. A d=512–768, depth 12–16 model on the same tokens is a low-risk test. A hierarchy or smaller cells are the follow-up if T1 is the goal. *Expected impact: medium-high, low engineering cost, compute-bound.*

4. **Strengthen the pretraining objective with a harder or more global component.** In order: (a) one ablation where the decoder does not receive the centroid of masked tokens, (b) a fair SD re-run with a tuned learning rate, (c) a latent-space (JEPA or DINO-style) term added to MAE. *Expected impact: medium, uncertain. The ablation is cheap and informative.*

5. **Add a physics-informed charge-light alignment cue.** Supply each charge token with its implied interaction time relative to the dominant flash (via drift velocity), as a feature or a bias, and optionally add a few light summary tokens. Early fusion already works, and this gives it the link it has to discover now. *Expected impact: medium on T2 and T3, and on cosmic rejection in harder events.*

Lower priority: RoPE or relative position bias, grid-offset augmentation (cheap, do it anyway), Mamba or PTv3 backbone swaps, contrastive charge-light loss.

## 10. What I would not change

- The early-fusion design and the light masking schedule.
- MAE with token dropping as the base objective. The evidence supports it, and it is cheap.
- The attentive-probe event heads and the frozen-plus-fine-tune protocol.
- The fixed cell tokenization as the default, until the items above show it limits event-level tasks. For T2–T4 it does not seem to.

---

## References

[1] Wu et al. "Point Transformer V3: Simpler, Faster, Stronger." CVPR 2024. arXiv:2312.10035. https://arxiv.org/abs/2312.10035
[2] Young & Terao. "Panda: Self-distillation of Reusable Sensor-level Representations for High Energy Physics." arXiv:2512.01324. https://arxiv.org/abs/2512.01324
[3] "Panda Diplomacy: Foundation Model Pre-training across Particle Imaging Detectors for High Energy and Nuclear Physics." arXiv:2609.00611. https://arxiv.org/abs/2609.00611 (abstract-level check only)
[4] Pang et al. "Masked Autoencoders for Point Cloud Self-supervised Learning" (Point-MAE). ECCV 2022. arXiv:2203.06604. https://arxiv.org/abs/2203.06604
[5] Young et al. "Particle Trajectory Representation Learning with Masked Point Modeling" (PoLAr-MAE). arXiv:2502.02558, 2025. https://arxiv.org/abs/2502.02558
[6] FlashAttention repository (Dao-AILab), changelog, release 2.4 (ALiBi). https://github.com/Dao-AILab/flash-attention
[7] E-ViM3 (Mamba-based video masked autoencoder). arXiv:2503.20258. https://arxiv.org/abs/2503.20258
[8] MaskMamba. arXiv:2409.19937. https://arxiv.org/abs/2409.19937
[9] Park et al. "FM4NPP: A Scaling Foundation Model for Nuclear and Particle Physics." arXiv:2508.14087. https://arxiv.org/abs/2508.14087 (partly from search summaries)
[10] "Scalable, End-to-End, Deep-Learning-Based Data Reconstruction Chain for Particle Imaging Detectors" (SPINE). arXiv:2102.01033. https://arxiv.org/abs/2102.01033 ; code: https://github.com/DeepLearnPhysics/spine
[11] Wu et al. "Sonata: Self-Supervised Learning of Reliable Point Representations." arXiv:2503.16429, 2025. https://arxiv.org/abs/2503.16429
[12] Hewes et al. "Graph Neural Network for Neutrino Physics Event Reconstruction" (NuGraph2). Phys. Rev. D 110, 032008 (2024). arXiv:2403.11872. https://arxiv.org/abs/2403.11872
[13] Heterogeneous Point Set Transformer (HPST). arXiv:2504.08182, 2025. https://arxiv.org/abs/2504.08182
[14] Newsom. "Towards Light Charge Association in Liquid Argon Time Projection Chambers" (Y-Net). UC Berkeley EECS Tech Report UCB/EECS-2021-52, 2021. https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-52.html
[15] Bachmann et al. "MultiMAE: Multi-modal Multi-task Masked Autoencoders." ECCV 2022. arXiv:2204.01678. https://arxiv.org/abs/2204.01678
[16] Gong et al. "Contrastive Audio-Visual Masked Autoencoder" (CAV-MAE). ICLR 2023. arXiv:2210.07839. https://arxiv.org/abs/2210.07839
[17] Qian et al. "Three-dimensional Imaging for Large LArTPCs." arXiv:1803.04850. https://arxiv.org/abs/1803.04850
[18] MicroBooNE Wire-Cell selection paper. arXiv:2011.01375. https://arxiv.org/abs/2011.01375 (search summary only)
[19] MicroBooNE LArMatch note. MICROBOONE-NOTE-1082-PUB. https://microboone.fnal.gov/wp-content/uploads/MICROBOONE-NOTE-1082-PUB.pdf (search summary only)
[20] MicroBooNE. SparseSSNet semantic segmentation. Phys. Rev. D 103, 052012. arXiv:2012.08513. https://arxiv.org/abs/2012.08513
[21] Villanueva. "Flash-matching for the Pandora based electron neutrino selection in MicroBooNE." arXiv:1805.04629. https://arxiv.org/abs/1805.04629
[22] Caratelli et al. "Neutrino identification with scintillation light in MicroBooNE." https://www.osti.gov/pages/biblio/1606419
[23] He, Chen, Xie, Li, Dollar, Girshick. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022. arXiv:2111.06377. https://arxiv.org/abs/2111.06377
[24] Psomas et al. "Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency." arXiv:2506.10178. https://arxiv.org/abs/2506.10178
[25] Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas. "Revisiting Feature Prediction for Learning Visual Representations from Video" (V-JEPA). arXiv:2404.08471. https://arxiv.org/abs/2404.08471
[26] Tong, Song, Wang, Wang. "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training." NeurIPS 2022. arXiv:2203.12602. https://arxiv.org/abs/2203.12602
[27] Assran et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (I-JEPA). ICCV 2023. arXiv:2301.08243. https://arxiv.org/abs/2301.08243
