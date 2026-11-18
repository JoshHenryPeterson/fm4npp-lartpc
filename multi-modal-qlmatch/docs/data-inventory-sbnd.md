# Data inventory: MicroBooNE open data → what qlfm uses → what an SBND sample needs

Purpose: first step towards an SBND sample for the qlfm charge + light foundation model. This note lists
(1) what the MicroBooNE open-data files contain, (2) which parts the qlfm pipeline actually reads, and for what,
and (3) the minimum an SBND sample must provide. SBND does not need to reproduce the MicroBooNE files; it needs
the columns marked **input** and **label** below, plus the detector constants in §4.

Code references: `qlfm/data/h5reader.py` (reader), `qlfm/data/convert.py` (HDF5 → cache),
`qlfm/data/dataset.py` (what a batch contains), `qlfm/tokenize.py` (model inputs), `qlfm/data/labels.py` (tasks),
`qlfm/geometry.py` (detector constants), `qlfm/data/pandora_features.py` (baselines).
Related: `docs/uboone-data-curation.md` (#10) covers how the MicroBooNE HDF5 is produced (the `HDF5Maker`
analyzer on reco2 artroot) and where the real-data inputs live.

## 1. The MicroBooNE open data we use

| Sample | Zenodo | Files used | Events | Hits | Optical hits |
|---|---|---|---|---|---|
| BNB inclusive overlay, NoWire | 10.5281/zenodo.8370883 | 18 (`BNB_All_NoWire_00–17.h5`, 183 GB) | 753,467 | 11.9 G | 270 M |
| νe-CC overlay, NoWire | 10.5281/zenodo.7261921 | 18 of 20 (`nue_NoWire_*`, 30 GB) | 77,340 | 1.30 G | 37.9 M |
| **Merged corpus** | | | **830,807** | 13.2 G | 308 M |

- MCC9.1, Run 1. "NoWire" = reconstructed hits only, no raw wire waveforms.
- **Overlay:** the neutrino interaction (and its light) is simulated and added on top of a real off-beam
  (EXT, unbiased) cosmic readout. Cosmic hits therefore carry no MC truth, and every "no-ν" example is real data.
  See §5 for why this matters.
- Median 15.4k hits per event; about 370 optical hits per event.
- Cache after conversion: 147 GB (bnb) + 17 GB (nue), column-wise `.npy`, memory-mapped.
- Splits: hashed on (run, subrun), 90/5/5 % train/val/test (`qlfm/data/splits.py`), so events sharing a cosmic
  frame stay in one partition. Pretraining and label subsets are nested fractions of train.

### 1.1 File format

HDF5 in the hep_hpc / ph5concat layout. Each table is a group with one `(N, k)` dataset per column, plus
`event_id` `(N, 3)` = (run, subrun, event) and `event_id.seq_cnt` = (event sequence number, row count). Rows of a
table are contiguous per event, so per-event slices are offset + count. Per-hit tables are linked by `hit_id`
(the row index within the event), particles by `g4_id`, flashes by `flash_id`, Pandora objects by `slice_id` /
`pfp_id`.

### 1.2 Tables, columns and what we use them for

Usage key:
- **input**: fed to the model.
- **label**: builds a downstream target or eligibility.
- **baseline**: only for the Pandora / light reference baselines.
- **audit**: diagnostics, event displays, metric definitions.
- **–**: not read.

**`event_table`** (one row per event; MC truth of the neutrino)

| Column | Shape | Use | Notes |
|---|---|---|---|
| `event_id` | 3 | splits | (run, subrun, event) |
| `nu_pdg` | 1 | **label** | T3 flavour |
| `is_cc` | 1 | **label** | T3, T4 |
| `nu_energy` | 1 | **label** | T4 target, GeV |
| `nu_vtx` | 3 | **label** | AV / FV containment (cm) |
| `nu_vtx_wire_time` | 1 | audit | ν time in ticks (flash-lag check) |
| `lep_energy`, `nu_dir`, `nu_vtx_corr`, `nu_vtx_wire_pos` | | – | cached or available, unused |

**`hit_table`** (one row per reconstructed TPC hit; Gaussian hit finder)

| Column | Use | Notes |
|---|---|---|
| `local_plane` | **input** | 0 = U, 1 = V, 2 = Y (collection) |
| `local_wire` | **input** | wire index in the plane |
| `local_time` | **input** | peak time, ticks (0.5 µs); cached at 1/8-tick precision |
| `integral` | **input** | charge, ADC × tick |
| `rms` | **input** | pulse width in ticks. Only qlfm reads it, and it is the sim/data-sensitive variable (#9) |
| `hit_id` | join key | must equal the row index within the event (asserted) |
| `tpc` | – | always 0 in MicroBooNE; **needed for SBND** (§4) |

**`edep_table`** (MC energy deposits matched to hits; only simulated hits have rows)

| Column | Use | Notes |
|---|---|---|
| `hit_id`, `energy` | **label** | per-hit ν deposit (MeV), summed over particles. T1 target: > 0.05 MeV. Event visibility counts (T2 classes, T1/T3/T4 eligibility): hits with any deposit, per plane |
| `g4_id`, `energy_fraction` | cached | link to `particle_table`; dominant particle category per hit (cached as `hit_cat`, unused by the current tasks) |

**`particle_table`** (MC particles): only `g4_id` and `category` are read (for `hit_cat`). The rest
(`g4_pdg`, momentum, start/end position and process, `parent_id`, ...) is unused.

**`ophit_table`** (one row per PMT optical hit)

| Column | Use | Notes |
|---|---|---|
| `hit_channel` | **input** | PMT OpChannel 0–31 (asserted < 32) |
| `peaktime` | **input** | µs on the beam-gate clock. Light tokens use 0–24 µs; the beam flash lies at 3–6 µs |
| `pe` | **input** | photoelectrons |
| `width` | cached | unused |
| `amplitude`, `area`, `sumpe_id`, `wire_pos`, `hit_id` | – | |

**`opflash_table`, `opflashsumpe_table`** (reconstructed flashes, per-PMT PE of each flash): **baseline / audit**
only. The largest flash per event (time, total PE, 32-PMT PE) feeds the Pandora + light BDT baseline and the
light-statistics audit. The model itself never sees flashes, only optical hits.

**`pandoraPrimary_table`, `pandoraHit_table`, `pandoraPfp_table`** (Pandora reco): **baseline** only.
- The "Pandora ν slice" is the slice with the highest `nu_score`. Its hits give the T1 baseline (`hit_pnd_nu`), and its score gives the T2 baseline.
- Per-event slice / PFP / vertex summaries feed the Pandora + BDT baseline (`qlfm/data/pandora_features.py`).

## 2. What the model pipeline actually reads

### 2.1 Model inputs (pretraining and downstream; no truth)

Per event, exactly two point sets:

| Modality | Per-element fields | Tokenisation (`qlfm/tokenize.py`) |
|---|---|---|
| TPC charge | plane, wire, time (ticks), integral, rms | Hits are grouped into cells of 16 wires × 64 ticks per plane; one token per occupied cell, capped at 4,096 tokens per event. Per-hit features: in-cell (wire, time) offset, log1p(q/200), rms/5. Token position: drift x = (tick − trigger tick) × cm/tick, wire coordinate = wire × pitch |
| PMT light | channel, peak time (µs), PE | One token per PMT (32 per event): log1p(PE) in 16 time bins over 0–24 µs (fine around the beam flash), plus log1p(total PE). PMT (y, z) position enters as a positional embedding |

Nothing else enters the model: no wire waveforms, no 3D space points, no Pandora objects, no flashes.

The reference models read subsets of the same hits:
- PTv3 / Panda / PolarMAE: wire cm, drift cm, plane offset, and integral. They do not use rms.
- FM4NPP: normalised (wire, tick, plane) and log integral.

### 2.2 Labels (downstream tasks, `qlfm/data/labels.py`)

| Task | Needs |
|---|---|
| T1 per-hit ν vs cosmic | per-hit MC ν energy deposit; event eligibility (visible in-AV ν: ≥ 30 collection-plane ν hits) |
| T2 event ν / no-ν / out-of-AV ν | `nu_vtx` (in AV?), per-plane count of hits with ν deposit |
| T3 flavour (νμCC / νeCC / NC) | `nu_pdg`, `is_cc`, FV (AV shrunk by 3 cm) |
| T4 energy | `nu_energy`, `is_cc`, FV |
| T5 light from charge | no label array: the target is the event's own optical hits |

Paper-metric extras (`qlfm/paper_metrics.py`):
- T2 bright-flash subset: beam-window PE from optical hits.
- T4 clean subset: ν share of collection-plane charge, and visible fraction = ν deposit on collection hits / E_ν.

### 2.3 Baselines and audits (nice to have, not required)

- **Pandora:** slice `nu_score`, slice hit membership, flash-match score, PFP track/shower scores, vertices.
- **Reconstructed flashes:** time, total PE, per-PMT PE.

## 3. Detector constants baked into the code (MicroBooNE values)

These are what must be re-specified for SBND (`qlfm/geometry.py`, `qlfm/tokenize.py`, `qlfm/eval_perturb.py`):

| Quantity | MicroBooNE | Where |
|---|---|---|
| TPCs | 1 | implicit everywhere |
| Planes, wires per plane | U 2400, V 2400, Y 3456; pitch 0.3 cm | `N_WIRES`, `WIRE_PITCH_CM`, tokenizer cell grid (`3456 // cell_wires`) |
| Readout window | 6400 ticks × 0.5 µs; trigger at tick 800 | `N_TICKS`, `TRIG_TICK`; tokenizer time grid sized for 8192 ticks |
| Drift | 256.35 cm at 0.1098 cm/µs (0.0549 cm/tick); cathode ≈ tick 5470 | `TICK_TO_CM`, `CATHODE_TICK` |
| Anode tick for the width study | 830 | `eval_perturb.ANODE_TICK` (Sunny's convention) |
| Active / fiducial volume | x 0–256.35, y ±116.5, z 0–1036.8 cm; FV margin 3 cm | `AV`, `FV_MARGIN_CM` |
| Photodetectors | 32 PMTs, fixed (y, z) map by OpChannel | `N_PMT`, `PMT_YZ`; light token count = 32 per event |
| Light window | 0–24 µs (23.4 µs beam-gate readout); beam flash 3–6 µs | `TokCfg.light_edges`, `r4_light_stats.WIN_LO/HI` |
| Hit-width / charge scales | rms/5 ticks, q/200 ADC·tick | `TokCfg.rms_scale`, `q_scale` |

## 4. What an SBND sample needs

### 4.1 Required (to pretrain and run T1–T5)

Per event:
- **Event id:** (run, subrun, event), for the leakage-safe splits.
- **ν truth:** pdg, CC/NC, energy (GeV), interaction vertex (cm). Also the ν interaction time, for the beam-window check.

Per TPC hit (Gaussian hit finder output, all planes):
- **tpc**, **plane**, **wire**, **peak time** (ticks; state the tick length and the trigger tick).
- **integral** (ADC·tick) and **rms** (ticks).
- **MC energy deposited by the neutrino interaction** (MeV), summed over particles. It is 0 or absent for cosmic and noise hits. This is the only per-hit truth the tasks need.
- `hit_id` must equal the row index within the event, or else provide a direct per-hit column instead of a separate deposit table.

Per optical hit (PMTs):
- **channel**, **peak time** (µs on the beam-gate clock, with the beam spill time stated), **PE**.

Detector description (once, not per event):
- Wire counts and pitch per plane.
- Drift velocity, tick length and trigger tick.
- TPC boundaries / active volume.
- PMT channel → (x, y, z) map.

### 4.2 Recommended

- **Per-hit dominant MC particle** (pdg or category) and track id. Enables semantic segmentation tasks beyond ν/cosmic.
- **Cosmic truth when cosmics are simulated** (CORSIKA): per-hit cosmic deposit. This lets the "no-ν" side be simulated too (see §5).
- **Reconstructed flashes** (time, total PE, per-PMT PE): a cheap light baseline and audit.
- **Pandora slice `nu_score` and slice hit membership:** the reference baseline for T1/T2.

### 4.3 Not needed

- Raw wire waveforms and optical waveforms.
- Full particle tables beyond the dominant-particle id.
- Pandora PFP details.
- Anything in `particle_table` besides `g4_id`/category.

Format suggestion: write the same table/column names as the MicroBooNE files (`event_table`, `hit_table` with
`tpc`, `edep_table`, `ophit_table`, optional `opflash*` / `pandora*`), in the same `event_id.seq_cnt` layout.
Then `qlfm/data/h5reader.py` and `qlfm/data/convert.py` work almost unchanged. The MicroBooNE files come from
the `HDF5Maker` art analyzer ([uboone/hdf5maker](https://github.com/uboone/hdf5maker), branch `opensamples`; see
`docs/uboone-data-curation.md`). An SBND port of it that writes only the §4.1 columns is a natural route. A flat per-event-offset HDF5 or
Parquet is equally fine if that is easier on the SBND side; the converter is about 200 lines.

### 4.4 SBND detector differences the code must absorb

These are to be confirmed against the SBND geometry; the list is for planning.

- **Two TPCs with a central cathode.** Drift runs in ±x, about 2 m each way. Hits need the `tpc` index, and drift x must be signed per TPC (x = ±(tick − trigger) × v). The tokenizer cell key must include the TPC: today it is (event, plane, wire cell, time cell).
- **Wire planes.** Each TPC has U, V and Y planes with different wire counts from MicroBooNE. The tokenizer's wire-cell and time-cell grid sizes are hard-coded to MicroBooNE maxima and must come from the geometry.
- **Readout window and drift time.** SBND's drift time and readout window differ. The time grid, the anode/trigger tick and the width-study drift bins must be re-derived.
- **Photon detection.** SBND has about 120 PMTs, plus X-ARAPUCAs, behind the anode planes on both sides. The light tokenizer assumes 32 PMTs (`N_PMT`), one token per PMT, and a fixed map. These must become geometry-driven, with a per-detector (x, y, z) positional embedding. Start with PMTs only; X-ARAPUCAs are an option.
- **Light time binning.** The 16 bins are tuned to the MicroBooNE beam-gate clock (beam flash at 3–6 µs). Re-tune them around the SBND beam spill time.
- **Normalisations.** `q_scale` and `rms_scale` should be re-checked against SBND hit distributions. A per-plane histogram of integral and rms is enough.

## 5. Lessons from MicroBooNE to carry over

- **Do not mix "simulated ν" with "real-data negatives" in training if avoidable.** In the open overlay every ν hit is simulated and every cosmic hit is real data. The simulation over-diffuses (D_L 6.40 vs 3.74 cm²/s), so ν hits are wider than data hits at long drift.
  - qlfm-M CL reads this width difference: its T1 F1 drops 0.92 → 0.71 when the simulated widths are rescaled to data, with no retraining (#9).
  - Fine-tuning on data-like widths restores it (0.914); removing width entirely gives 0.829.
  - For SBND, prefer a fully simulated sample (simulated ν + simulated cosmics, same detector simulation) for training, so positives and negatives share one simulation. Keep an overlay or real-data sample for a sim/data check.
- **Record the detector-simulation configuration** (diffusion, electron lifetime, noise model, light yield) and keep the same reconstructed-hit producer for data and simulation. Then a width / charge comparison like Sunny's (cosmic muons in data vs simulation, matched hit by hit in drift and 3D direction) can be repeated.
- **Keep (run, subrun)** for leakage-safe splits. If the cosmics are simulated, the event's random seed or file is the equivalent grouping key.
- **Scale.** The MicroBooNE corpus is 831k events. In round 4, T5 results saturated beyond about 10% of the pretraining split at the current model size and step budget. A first SBND sample of about 100k events is enough to pretrain and to run the label-efficiency study. Adding a νe-enriched sample (as with MicroBooNE's νe overlay, about 9% of the corpus) keeps T3 balanced.
