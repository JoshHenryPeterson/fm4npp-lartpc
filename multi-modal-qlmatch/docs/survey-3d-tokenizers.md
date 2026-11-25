# 3D information and better tokenizers for qlfm: survey and plan

*2026-11-23. Issue #15. Compiled from our results on MicroBooNE (#8, #9, #10) and SBND (#11, #14), our code, and a web literature search. In the references, "[unverified]" marks claims seen only in talks or posters, and "[2026, not re-checked]" marks recent arXiv entries found by the search that I have not opened myself.*

## 0. Why

Two observations from the SBND campaign (deck: `docs/sbnd-realdata-campaign.pdf`) raise four questions.

**Observation 1: C+XF is catching up with CL.** C+XF is charge-only pretraining with strong late light fusion; CL has light in pretraining (early fusion).
- **Q1:** is early fusion necessary?

**Observation 2: on T1 (hit-level ν/cosmic), qlfm is still at or below Wire-Cell on SBND.** Wire-Cell reasons in 3D and matches charge to light in 3D. qlfm sees 2D hits per plane, and must learn the 2D-to-3D geometry and the charge-to-light physics implicitly.
- **Q2:** how do we bring 3D information in? We can use Wire-Cell's 3D reconstruction if needed.
- **Q3:** can a CNN-based charge tokenizer (for example an autoencoder) replace the current per-cell MLP pooling and capture more geometry?
- **Q4:** can an OpFlash-based light encoder (multi-PMT, same time window, starting with the in-beam flash only) replace the per-PMT time-profile tokens?

## 1. Where we stand

### 1.1 Early fusion (CL) vs late fusion (C+XF): not settled, confounded by hit width

| | CL (light in pretraining) | C+XF (charge-only pretraining, strong late light) |
|---|---|---|
| MicroBooNE sim, T1 F1, 1% labels, nominal widths | **0.922** | 0.862 |
| MicroBooNE sim, T1 F1, width removed (`w_const`) | 0.829 | 0.809 |
| MicroBooNE real data, share of WC ν charge also tagged | 0.18-0.24 | **0.60** |
| SBND sim, T1 / T2 AUC / T3 / T4 res / T5 MSE | **0.921** / **0.996** / **0.77** / **0.22** / **0.62** | 0.896 / 0.992 / 0.73 / 0.23 / 0.78 |
| SBND data, WC ν charge tagged, simulation fine-tune | 0.49-0.53 | **0.64-0.68** |
| SBND data, same, data-like-width fine-tune (E6) | **0.70-0.71** | 0.66-0.68 |
| SBND data, same, real-data fine-tune with WC labels (E7) | 0.75 | 0.74 |

**Reading.**
- **On simulation, CL still leads on every task**, most on T5 (light from charge) and T1.
- **On MicroBooNE, most of CL's T1 lead disappears with width removed** (0.922 vs 0.862 becomes 0.829 vs 0.809). CL had learned to read hit width more than C+XF did, and width is exactly the feature that differs between data and simulation (#9, E3, E5).
- **"C+XF is catching up" is mostly CL losing on data through width.** Once CL is fine-tuned with data-like widths (E6), it is ahead again on SBND data.
- **Q1 needs a controlled comparison, not a conclusion:** CL vs C+XF pretrained under the same width treatment (§3, P1).

**Caveats.**
- **Light differs too.** CL uses light in pretraining but no late light at fine-tune (`late_light 0`); C+XF uses `XFusion`. They differ in both the light path and the head.
- **The T5 gap may not be about fusion.** CL's T5 advantage comes from its light-pretraining objective, which C+XF never sees.

### 1.2 T1 vs Wire-Cell

| | Wire-Cell | qlfm (best) |
|---|---|---|
| MicroBooNE overlay, charge-weighted T1 F1 | 0.82 / 0.88 (bnb; label = main & generic-ν / main cluster) | **0.92** |
| SBND simulation, T1 F1 vs truth, 10k test events | **0.931** | 0.921 (SBND-pre CL, 1% labels) |
| SBND real data | reference, no truth | agreement with WC 0.51-0.75, depending on fine-tune |

- **On MicroBooNE, qlfm beats Wire-Cell's ν label.** But that label (main cluster plus generic-ν selection) is strict, so the comparison is not like-for-like with SBND's.
- **On SBND simulation, Wire-Cell's footprint label is slightly ahead** (0.931 vs 0.921), even though the footprint label is broad: every cluster of any beam-matched candidate.
- **On SBND data we have no truth.** The large disagreement with Wire-Cell (0.5-0.75) is partly a simulation-to-data gap (E4-E7), not only a 2D-vs-3D gap.

What Wire-Cell has that we lack:
1. **3D blobs:** tomographic tiling of the 3 planes per time slice, with charge solved under a sparsity prior.
2. **3D clusters.**
3. **Per-cluster predicted light** from a photon library, compared with the measured PMT pattern (charge-light matching).
4. **Cosmic taggers built on those** (through-going muon, stopping muon, light mismatch).

Items 1-3 are what Q2 asks about.

### 1.3 What the current tokenizers keep and discard

**Charge** (`qlfm/tokenize.py: tokenize_charge`, `models/encoder.py`):
- **Grouping:** GausHits are grouped into cells of 16 wires × 64 ticks, per (TPC, plane).
- **Per-hit features:** position inside the cell, log charge, RMS.
- **Pooling:** a small MLP per hit, then max+mean pooling (PointNet-style) into one token.
- **Token embedding:** drift x and the plane's own wire coordinate, plus a plane embedding.
- **Missing:**
  - (a) any 3D coordinate: the three planes' tokens are linked only through attention;
  - (b) explicit local shape inside a cell beyond what pooled per-hit features carry: track direction, kinks, adjacency between hits.
- **MAE target:** 4×8 sub-grid charge inside the cell.

**Light** (`tokenize_light`, `LightEmbed`):
- **Tokens:** one per PMT (32 MicroBooNE, 120 SBND), carrying log(1 + PE) in 16 time bins on the beam clock plus log(1 + total PE).
- **Embedding:** MLP(features) + Fourier(y, z) + learned PMT id.
- **Missing:**
  - (a) flash grouping: which PMTs fired together, and at what time;
  - (b) x: facing PMTs in SBND's two TPCs share (y, z);
  - (c) Fourier scales retuned for SBND.
- **Cache:** the brightest in-window flash per event (`ev_fl_pmt` per-PMT PE, `fl_t`, `fl_pe`) is already stored for MicroBooNE and SBND. The full flash tables (`opflash_table`, `opflashsumpe_table`, with y/z centre and width) are in every HDF5 file.

## 2. Survey

### 2.1 Bringing in 3D

| approach | what it is | reference | 3D source for us | cost / risk |
|---|---|---|---|---|
| **A. Wire-Cell blobs / 3D points as a third token stream** | one token per blob (or per group of 3D points), with (x, y, z) and charge; cross-attention between blob tokens and the 2D hit tokens of the wires each blob spans | Wire-Cell imaging (Qian et al. 2018); MicroBooNE Wire-Cell selection (2021) | MicroBooNE round-1: `TC` tree in the stored nusel files (every imaged cell: cluster, time slice, U/V/W wire indices, charges; #10 doc §4.1b). SBND: Wire-Cell `tracking-pr.root`; the full list of 3D trees has to be confirmed (§4). Open data: none | needs an `sbnd-h5maker` / round-1 extension (a `wc_blob_table`); couples the model to Wire-Cell's 3D, including its ghosts |
| **B. "Nexus" 3D nodes linking planes** | NuGraph2's design: 2D hit nodes per plane plus 3D nodes that connect hits across planes; 3D nodes from space points | NuGraph2 (Hewes et al. 2024) | Wire-Cell 3D points (as A), or our own wire-triplet candidates (D) | architecture change in the encoder (cross-plane edges or attention masks) |
| **C. Auxiliary 3D head (2D-to-3D lifting by supervision)** | predict each charge token's or hit's 3D position (or the blob it belongs to); the planes then have to agree in a shared 3D frame | NuGraph3 per-hit 3D position head [unverified, 2026 talk]; LArMatch (MicroBooNE note 1082) | simulation truth (deposit positions; SBND HDF5 projects G4 deposits onto hits). Data: Wire-Cell 3D points as pseudo-labels | cheap: one head plus a loss; keeps 2D inputs; works on data only through pseudo-labels |
| **D. Learned wire-triplet 3D points** | enumerate the geometrically valid (U, V, Y) wire triplets per time slice, score them with a network (keep real, drop ghosts) | LArMatch (MicroBooNE); SPINE deghosting on ICARUS/SBND [unverified, slides] | our own, from the hits we already have | the most work; independent of Wire-Cell; many ghost candidates |
| **E. 3D charge to predicted light (learned Q/L matching)** | from 3D charge (tokens or blobs), predict the per-PMT PE pattern with a learned optical model, and compare with the measured flash, as a pretraining target or a matching score | SIREN photon-propagation surrogate and differentiable OpT0Finder (Lei et al. 2022); Wire-Cell Q/L matching | needs 3D (A, C or D) plus PMT geometry; the target is the measured light we already have | medium; it is the physics step Wire-Cell has and we lack |
| **F. Native 3D encoders** | if 3D points exist, use a proven 3D self-supervised recipe | PoLAr-MAE (Young et al. 2025); Panda (Young & Terao 2025); SPINE sparse CNNs (Dominé & Terao 2020) | as A or D | the larger change: a separate 3D backbone; Panda's distillation beat MAE in label efficiency |

**No published LArTPC work fuses wire planes through 3D queries** (a BEVFormer-style cross-attention or a tri-plane representation); this looks open. NuGraph2's nexus nodes are the nearest LArTPC precedent.

### 2.2 Charge tokenizer

| option | idea | reference | fit for us |
|---|---|---|---|
| **current** | per-cell MLP plus max/mean pooling (PointNet / Deep Sets) | Qi et al. 2017; Zaheer et al. 2017 | simple; loses sub-cell shape |
| **sparse 2D CNN stem** | a few submanifold sparse-convolution layers on the wire-time image of each plane, then pooling per cell; "convolutional stems help transformers" | SparseSSNet on MicroBooNE wire planes (2021); Early Convolutions (Xiao et al. 2021); spconv / MinkowskiEngine | **best first try.** `spconv` 2.3.8 imports in the PTv3/Panda venv on Polaris (torch 2.10), but not in the conda environment qlfm normally uses: run P2 through the existing `@pimm` task prefix, or install `spconv` for qlfm. Round 3 found Panda (sparse-conv embedding, PTv3) the better charge-only encoder |
| **CNN autoencoder tokenizer** | pretrain a sparse conv autoencoder on cells, then freeze it as the tokenizer (as the user suggests) | VQ-VAE; Point-BERT dVAE tokenizer | possible second stage; a two-stage pipeline; a jointly trained stem is simpler |
| **content-adaptive patches** | farthest-point sampling + kNN or ball groups instead of a fixed grid | Point-MAE; PoLAr-MAE | fixes cells that cut tracks; PoLAr-MAE warns of a sub-token failure mode |
| **discrete tokens (VQ)** | codebook tokens and masked-token prediction | OmniJet-α (2024); Point-BERT | useful if the MAE charge target is noisy; not a first priority |

### 2.3 Light tokenizer

| option | idea | reference / precedent | fit for us |
|---|---|---|---|
| **current** | one token per PMT, 16-bin time profile | -- | keeps timing; no flash grouping; no x |
| **OpFlash token(s)** | one token per flash: the per-PMT PE vector (120 or 32) + flash time + total PE (+ y/z centre and width), encoded by an MLP or a set encoder over PMTs | Wire-Cell and OpT0Finder both match per-PMT PE per flash, not waveforms | **user's proposal; start with the in-beam (brightest in-window) flash.** Already in the cache (`ev_fl_pmt`, `fl_t`, `fl_pe`), so no reconversion. Loses out-of-flash light and time structure |
| **PMTs inside the flash** | keep per-PMT tokens, but the feature is that PMT's PE in the in-beam flash (+ its PMT hit time) | Wire-Cell per-PMT flash PE | a middle ground: same token layout, flash-level content |
| **geometry fixes** | add x (or a TPC-side embedding) to the PMT position, retune the Fourier scales for SBND, PMT coordinates as continuous features | GraphNeT (PMT/DOM as nodes with position); Set Transformer | cheap; also helps cross-detector transfer |
| **waveform-level** | ns-level pulse features near the spill | SBND PDS paper (2024) | not now: the HDF5 holds optical hits, not waveforms |

No published LArTPC study compares waveform-level and flash-level ML light representations, and no peer-reviewed joint charge-light network exists for LArTPCs. qlfm's multimodal pretraining appears to be new.

## 3. Plan (proposed, needs the owner's go-ahead)

Ordered by cost; each step has a decision criterion. The test bench is the SBND simulation (T1-T5) plus the real-data agreement and event-level metrics of E4-E7, all on held-out files.

**P0: light tokenizer, cheap (fine-tune only, about 2-3 node-h).**
- **Testbed:** the late-light path of C+XF. The encoder stays charge-only, so a new light encoder needs no re-pretraining.
- **Variants, each with 3 seeds and the E2 recipe:**
  - (a) current per-PMT profile tokens;
  - (b) one in-beam OpFlash token;
  - (c) per-PMT flash-PE tokens;
  - (d) (a) with x and SBND-tuned position scales.
- **Measure:** T1/T2/T5 on simulation, and agreement with Wire-Cell on data.
- **Decision:** if (b) or (c) matches (a) on simulation and is closer on data, the flash representation goes into P1's pretraining.

**P1: is early fusion necessary? (two pretrainings, about 4-5 node-h).**
- **Run:** pretrain CL and C (charge-only) under the same treatment, with widths randomised per plane during pretraining (or data-like), then fine-tune both with the E2 recipe; C gets XF late light.
- **Measure:** T1-T5 on simulation, and E4-style agreement on data.
- **Decision:**
  - if CL ≈ C+XF on simulation and data, early fusion is not needed and later work can use C+XF;
  - if CL still leads on T1/T5, keep early fusion and carry the P0 light encoder into pretraining.

**P2: sparse-CNN charge stem (one or two pretrainings, about 3-5 node-h).**
- **Change:** replace the per-cell MLP pooling with a 2-3 layer submanifold sparse-conv stem per (TPC, plane), on hits rasterised at wire × 4-tick resolution, pooled to the same 16 × 64 cells. The token layout, MAE target and the rest of the model stay the same.
- **Compare:** at equal steps against the current tokenizer, using the better pretraining variant from P1.
- **Option:** pretrain the stem as an autoencoder first and freeze it (the CNN-AE variant), only if the jointly trained stem helps.

**P3: 3D, in two steps.**
- **P3a, auxiliary 3D head (C), about 3-4 node-h:**
  - add a head predicting each charge token's 3D position: the charge-weighted mean of the truth deposit positions of its hits, on simulation;
  - train it during fine-tuning first, then in pretraining;
  - Wire-Cell 3D points can supply pseudo-labels on data later.
  - Cheapest way to force cross-plane consistency without new inputs.
- **P3b, 3D tokens (A, then E):**
  - extract Wire-Cell blobs or 3D points into the HDF5 (`wc_blob_table`) for the SBND simulation and data, and for MicroBooNE round 1;
  - add blob tokens with cross-attention to the 2D tokens;
  - then a learned charge-to-light head on the 3D tokens (a SIREN-like optical surrogate) as a Q/L matching objective.
  - Needs the Aurora and FNAL sessions for extraction and the owner's approval for the reprocessing.

**Not now:** learned triplet 3D (D) and a separate 3D backbone (F). Revisit if P3a/P3b show that 3D is the missing piece.

**Budget:** P0-P3a total about 15 node-h on `LArTPC_FM`. P3b depends on the extraction.

## 4. Open items to confirm before P3

1. **SBND `tracking-pr.root` contents:** which trees hold 3D points or blobs with charge, and their cluster or candidate IDs. `sbnd-h5maker` reads only `T_tagger`, `T_kine`, `T_proj_data` and `T_bundle` today. (Ask `prod_aurora` / `porting_sbndbuild03`.)
2. **MicroBooNE round-1 `TC` tree:** blob → 3D position conversion (wire-index triplet plus time slice to (x, y, z)) and charge per blob.
3. **Truth deposit positions per hit in the SBND simulation HDF5** (the `truth_projection` attribute describes the projection; positions may need a new column).
4. **The list of the PR cosmic tagger's ten tests** (also asked on #14).

## 5. References

**Wire-Cell.**
- *Three-dimensional Imaging for Large LArTPCs*, Qian, Zhang, Viren, Diwan, JINST 13 P05032 (2018), arXiv:1803.04850.
- *Neutrino Event Selection in the MicroBooNE LArTPC using Wire-Cell 3-D Imaging, Clustering, and Charge-Light Matching*, MicroBooNE, JINST 16 P06043 (2021), arXiv:2011.01375.
- *Cosmic Ray Background Rejection with Wire-Cell LArTPC Event Reconstruction in MicroBooNE*, MicroBooNE, Phys. Rev. Applied 15, 064071 (2021), arXiv:2101.05076.
- *Wire-Cell 3D Pattern Recognition Techniques for Neutrino Event Reconstruction in Large LArTPCs*, MicroBooNE, JINST 17 P01037 (2022), arXiv:2110.13961.
- *Augmented Signal Processing in LArTPCs with a Deep Neural Network*, Yu et al., JINST 16 P01036 (2021), arXiv:2007.12743.

**Other LArTPC reconstruction and 3D ML.**
- Pandora (MicroBooNE), EPJC 78:82 (2018), arXiv:1708.03135.
- Pandora deep-learning vertexing (DUNE), EPJC 85, 697 (2025), arXiv:2502.06637.
- NuGraph2, Hewes et al., PRD 110, 032008 (2024), arXiv:2403.11872.
- NuGraph3: talk only [unverified]. Domain-adaptive NuGraph3, arXiv:2611.10057 [2026, not re-checked].
- Sparse UResNet, Dominé & Terao, PRD 102, 012005 (2020), arXiv:1903.05663.
- PPN, PRD 104, 032004 (2021), arXiv:2006.14745.
- GrapPA, PRD 104, 072004 (2021), arXiv:2007.01335.
- Full chain, arXiv:2102.01033.
- PILArNet, arXiv:2006.01993.
- PoLAr-MAE, Young, Jwa, Terao, arXiv:2502.02558.
- Panda, Young & Terao, arXiv:2512.01324.
- Panda Diplomacy, arXiv:2609.00611 [2026, not re-checked].
- FM4NPP, Park et al., ICLR 2026, arXiv:2508.14087.
- LArMatch, MICROBOONE-NOTE-1082-PUB (2020).
- Wire vs pixel readout, arXiv:1912.10133.

**Charge-light matching and optical surrogates.**
- OpT0Finder: github.com/DeepLearnPhysics/OpT0Finder.
- SIREN photon surrogate and differentiable flash matching, Lei et al., arXiv:2211.01505.
- SBND photon detection system, EPJC 84 (2024), arXiv:2406.07514.
- ICARUS light and cosmic rejection, arXiv:2312.05684.
- MicroBooNE light simulation and calibration, arXiv:2603.23691 [2026, not re-checked].

**Tokenizers.**
- ViT, arXiv:2010.11929.
- Early Convolutions Help Transformers See Better, arXiv:2106.14881.
- PointNet, arXiv:1612.00593.
- Deep Sets, arXiv:1703.06114.
- Point-BERT, arXiv:2111.14819.
- Point-MAE, arXiv:2203.06604.
- VQ-VAE, arXiv:1711.00937.
- OmniJet-α, arXiv:2403.05618.
- Submanifold sparse convolution, arXiv:1711.10275.
- MinkowskiEngine, arXiv:1904.08755.
- Point Transformer V3, arXiv:2312.10035.
- MicroBooNE CNNs: arXiv:1611.05531, arXiv:1808.07269.
- SparseSSNet, PRD 103, 052012 (2021), arXiv:2012.08513.
- GraphNeT, arXiv:2210.12194.
- Set Transformer, arXiv:1810.00825.
- BEVFormer, arXiv:2203.17270.
- EG3D tri-plane, arXiv:2112.07945.
