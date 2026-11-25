# P3: 3D Wire-Cell blobs and new tokenizers in qlfm (combined first)

*Tracking: issue #16. Background and survey: `docs/survey-3d-tokenizers.md` (#15). Started 2026-11-24.*

## Decision (owner, 2026-11-24)

- **Combined step first.** Go straight to the combined step: new tokenizers + 3D (P3a + P3b).
- **Ablations later.** The stand-alone P0 (light tokenizer), P1 (early vs late fusion) and P2 (CNN charge stem) become ablations.
- **New SBND HDF5 files with 3D blobs** are made by `prod_aurora` (simulation, Aurora) and `porting_sbndbuild03` (real data, FermiGrid). They follow how `TensorSetLabeler` dumps blobs and blob track IDs for the NuGraph HDF5 ([larwirecell `sbnd-wc-flatcaf`: `larwirecell/aiml/TensorSetLabeler.cxx`](https://github.com/HaiwangYu/larwirecell/blob/sbnd-wc-flatcaf/larwirecell/aiml/TensorSetLabeler.cxx)). A particle-level truth table is added too.
- **Order:**
  1. a test sample of about 1k events from `prod_aurora`;
  2. if it works on Polaris, full reprocessing of the SBND simulation and data;
  3. meanwhile, `fmlar_polaris` builds the qlfm side and iterates with `prod_aurora` on the test files.

## 1. HDF5 additions (`sbnd-h5maker` 0.6), requested spec

Same layout as now: hep_hpc tables, one row per object, `event_id` + `event_id.seq_cnt`. Units as in 0.5: cm, ns, GeV, ticks of 0.5 µs. Columns marked *(sim)* are filled in simulation and set to −1 / 0 in data.

### `wc_blob_table`: one row per Wire-Cell blob, all clusters (not only ν candidates)

| column | content |
|---|---|
| `blob_id` | index within the event |
| `cluster_id` | Wire-Cell cluster ident (same numbering as `wc_hit_table.wc_cluster_id`) |
| `tpc` (or `apa`, `face`) | the TPC the blob lives in |
| `slice_index_min`, `slice_index_max` | time-slice range, with the tick convention (ticks per slice, offset) as attributes |
| `u_wire_index_min/max`, `v_wire_index_min/max`, `w_wire_index_min/max` | Wire-Cell wire-index ranges per plane, with the mapping to our `local_wire` as attributes (cf. `wc_channel_offsets`) |
| `center_x`, `center_y`, `center_z` | blob centre (cm). State whether x is t0-corrected for matched clusters, and whether SCE is applied |
| `charge` | solved blob charge (electrons) |
| `wc_nu_index` | ν-candidate index if the blob's cluster belongs to a candidate, else −1 |
| `trackid` *(sim)* | dominant G4 track ID by projected deposit charge, as in `TensorSetLabeler`; −1 if none |
| `truth_q`, `truth_q_dom`, `trackid2`, `truth_q2` *(sim)* | total accepted deposit weight, the dominant track's share, the second track and its share (issue-41 detail) |

### `wc_cluster_table`: one row per Wire-Cell cluster

`cluster_id`, `n_blobs`, `charge`, `flag_main_cluster`, `cluster_t0`, `flag_STM`, `flag_TGM`, `flag_FC`, `lm_flag`, `matched_flash_id`, `wc_nu_index`.

These are for labels and analysis only. **qlfm will not use the cluster flags as model inputs**, so they don't leak Wire-Cell decisions into the model.

### `wc_hit_blob_table`: hit ↔ blob association, many-to-many

`hit_id` (row in `hit_table`), `blob_id`, and optionally `weight`. A hit belongs to a blob if its (plane, wire, tick) falls inside the blob's wire range on that plane and its slice range. Ghost blobs can share hits.

### `track_table`: one row per G4 track *(sim)*

As `TensorSetLabeler`'s `truth_per_track`: `trackid`, `pdg`, `mother_trackid`, `mother_pdg`, `status`, `start_x/y/z/t`, `start_px/py/pz/E`, `end_x/y/z/t`, `end_px/py/pz/E`, `nu_idx`, `process` (G4 creation-process code).

Also add the per-track blob census if cheap: `n_blobs`, `q_blobs`, `q_dom_blobs`. It feeds particle-level labels later (PID, momentum, Michel, …). `trackid` must use the same convention as `edep_table.g4_id` and `particle_table.g4_id`, or the mapping must be stated.

### Per-hit truth 3D position *(sim)*

In `edep_table` (or a new `hit_truth_table`): the charge-weighted mean deposit position (x, y, z) of each hit's projected deposits, and its spread.

This is the target for P3a, the auxiliary head that predicts each token's 3D position.

### Optional (estimate the size on the test sample first)

Wire-Cell's blob-sampled 3D points (`BlobSampler` "3d" point cloud: x, y, z, q, `blob_id`). This would allow 3D point tokens later.

## 2. Production steps

| step | who | what | status |
|---|---|---|---|
| 1 | prod_aurora | implement the 0.6 tables (Wire-Cell clustering re-run with the labeler); test sample of about 1k BNB CV events (and, if easy, the 67 selected data events) on Eagle; `check_h5` with blob checks | requested 2026-11-24 |
| 2 | fmlar_polaris | convert the test sample, check blobs against hits (projection, charge, truth track purity), run P3a/P3b smoke tests, feed back | waiting for 1 |
| 3 | prod_aurora | full simulation reprocessing (BNB CV, νe CC, NC; about 923k events) | after the test passes |
| 4 | porting_sbndbuild03 | full real-data reprocessing (beam-on 90k, off-beam 50k), same recipe; Wire-Cell clustering re-run on FermiGrid | after the test passes |

## 3. qlfm side (Polaris), what gets built

All behind options, so the current models are unchanged.

1. **Converter and cache.**
   - Blob columns per event: centre, charge, wire and slice ranges, TPC, `cluster_id`, truth track and share.
   - Hit-blob association arrays.
   - `track_table` columns.
   - Per-hit truth xyz.
2. **Charge tokenizer: sparse-CNN stem** (`--charge-stem cnn`).
   - A few submanifold sparse-conv layers per (TPC, plane) on hits rasterised at wire × 4-tick, pooled to the existing 16 × 64 cells.
   - Same token layout and MAE target. Runs in the `pimm` venv, which has `spconv`.
3. **Light tokenizer: OpFlash** (`--light-tok flash`).
   - One token for the in-beam (brightest in-window) flash: per-PMT PE vector, flash time, total PE.
   - Option: per-PMT tokens carrying the flash PE.
   - PMT position embedding with x (TPC side) and SBND scales.
4. **3D blob tokens (P3b)** (`--blobs`).
   - One token per blob: centre xyz (Fourier), log charge, extent (wire and slice ranges as cm), TPC.
   - Blob ↔ charge-token attention through the hit-blob association: a blob attends to the 2D tokens containing its hits, and vice versa (NuGraph "nexus" idea).
   - Masked-blob reconstruction in pretraining.
   - **Charge-to-light (Q/L) head:** predict per-PMT PE of the in-beam flash from the blob tokens, compared with the measured flash. This is the learned version of Wire-Cell's charge-light matching.
5. **Auxiliary 3D head (P3a)** (`--aux3d`).
   - Predict each charge token's 3D position: the charge-weighted per-hit truth xyz.
   - On data, the blob-centre pseudo-target from the hit-blob association, optional.
6. **Labels from blobs.**
   - Blob-level ν/cosmic from `trackid` → `track_table.nu_idx`.
   - Particle-level labels (PID, Michel) later.

**Evaluation:** the T1-T5 test bench on SBND simulation, plus the E4-E7 real-data metrics on held-out files, against the current models and Wire-Cell.

## 4. Open questions

- Whether the blob centre x is t0-corrected (matters for cosmics out of time), and whether SCE is applied.
- Data-side cost: the Wire-Cell clustering re-run on FermiGrid for 2,875 pairs.
- Size of the optional sampled 3D points.

## 5. Log

- **2026-11-24:** plan written; request sent to `prod_aurora` and `porting_sbndbuild03`.
- **2026-11-24:** both sessions found a cheaper route (from the Wire-Cell step-1 outputs).
  - The labeler's blob scalars, track IDs, the track table and the 3D points already exist in the step-1 tars.
  - Only step 2's cluster splitting is rerun, so cluster IDs match `wc_hit_table`.
  - `prod_aurora` test job 9122698; data side ~250-400 CPU-h on FermiGrid.
- **2026-11-24:** qlfm side implemented (options off by default; an existing SBND checkpoint still loads strictly).
  - Components: `--charge-stem cnn`, `--pos3d`, `--light-tok flash|pmt_flash`, `--pmt-pos v2`, `--blobs`, `--aux3d-weight`; converter `convert_blobs` (0.6 tables) and `merge_shards` offsets.
  - Smoke-tested on GPU on a 64-event fake-blob cache (`scripts/make_fake_blob_cache.py`): pretraining (all losses), the flash variant, and fine-tuning from the checkpoint and from scratch.
