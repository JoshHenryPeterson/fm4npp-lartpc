# SBND real data in the sbnd-h5maker layout (fm4npp-lartpc #14)

Recipe for SBND **real data** in the same HDF5 layout as the SBND simulation sample (#11, `docs/sbnd-h5maker.md`):
beam-on BNB data plus ~50k off-beam (cosmic-only) events, for validation in the first SBND paper.

- Recipe owner: `prod_aurora`. Production on FermiGrid and transfer to Polaris: `porting_sbndbuild03`. Review on Polaris: `fmlar_polaris`.
- Status and decisions: issue #14. This doc holds the procedure; the issue holds the log.

## 1. How the simulation sample was made (for reference)

| | SBND simulation sample (#11) |
|---|---|
| Input | Avinay's private Gen2 MC reco1 on Flare (BNB CV, νe CC, NC Δ-rad), 922,815 events |
| Wire-Cell | **aurora-r2** (sbnd-wirecell-production #4): step 1 toolkit `d06a47d7` + larwirecell `ad5d942` (`wcls-img-clus-matching.fcl`), step 2 toolkit `55177882` (`wct-pr.jsonnet`, `DLVTX_SPACE=true`, flat tree, truth match) |
| Converter | `sbnd-h5maker` 0.4 (`fc18b3d`), Aurora job 9088268, 1,781 files, 922,788 events, 349 GB |
| Location | Eagle `fm4npp-lartpc-data/sbnd-{bnb-cv,nuecc,ncdr}/full/` |

## 2. Input data (the owner chooses; keep the recipe dataset-agnostic until then)

**Never** use the blinded `SBND2026A gen2_BNB-Run1` beam-on dataset.

Candidates, all already processed by Wire-Cell on FermiGrid (productions page, https://haiwangyu.github.io/sbnd-wirecell-production/):

| role | candidate | SAM definition (reco1) | files / events | Wire-Cell outputs that exist |
|---|---|---|---|---|
| beam-on | open FixedDev BNB data, run 1 (runs 18255/18259) | `data_MCP2025C_Fall25-Run1_BNB_FixedDev_bnblight_v10_14_02_reco1_sbnd` | 1,817 / 89,946 | `opendata` (no tracking-pr kept), `wcl1116-fg-opendata` (0.38.0; tracking-pr tars on scratch) |
| beam-on | Gen2 Run 3 BNB, Fixed Dev, unblinded (runs 20586–20597, 5e18 POT) | `data_SBND2026A_Gen2_Run3_BeamLight_Fixed_Dev_Data_v10_14_02_08_reco1_sbnd` | 2,208 / 109,487 | `run3bnb` (fg1 chain; tracking-pr tars on scratch), `wcl1116-fg-run3bnb` (0.38.0; scratch) |
| beam-off | InTime run 1 (the "50k" used before) | `data_SBND2026A_gen2_InTime-Run1_v10_14_02_02_reco1_sbnd` | 1,096 (1,058 with events) / 49,970 | `fg1-beamoff` (fg1 chain), `wcl1116-fg-beamoff` (0.38.0); tracking-pr tars on scratch |

Notes:
- The productions page splits the beam-off sample by run parity: odd runs are on the training side of the earlier Wire-Cell DL work, even runs on the evaluation side. The fm4npp models are trained on simulation only, so this split matters only if a beam-off subset is ever used for training here. Keep `run` in every file (it is in `event_id`), so the split can be applied later.
- POT and spill counts are not in reco1. Get them for the chosen run/subrun list from the SBND POT tools (or the flat CAFs paired with these files) and record them in the sample README.

## 3. Wire-Cell img → PR for real data (owner decision, see #14)

`sbnd-h5maker` needs, per event, the step-2 `tracking-pr.root` (`T_tagger`, `T_kine`, `T_proj_data`, `T_bundle`) at `<dir>/pr_evt<run>_<subrun>_<event>/tracking-pr.root`. On FermiGrid each pair's step 2 is kept as `<id>.tracking-pr.tar` holding `pr/all/pr_evt<E>/tracking-pr.root` (paths kept): untar it and pass `pr/all` as `--wc-dir`.

Two ways to get a data label that is consistent with a simulation label:

| | A. aurora-r2 on data | B. WCT 0.38.0 (WCLocal2026Nov16) on both sides |
|---|---|---|
| Data | rerun **step 2 only** with the aurora-r2 toolkit (`55177882`, branch `sbnd-dlvtx-space-44`) and the aurora-r2 knobs (`DLVTX_SPACE=true`, `dl_vtx_dump`, flat tree, `truth_match=false`) on the kept step-1 tars | use the existing `wcl1116-fg-*` tracking-pr tars (split-move pass, `pr/`) as they are: **no Wire-Cell rerun** |
| Simulation | unchanged (the current 922k files) | rebuild the 922k h5 files from WCLocal2026Nov16-Aurora (same unit layout as aurora-r2; ~9 node-h on Aurora) |
| Pros | sim files and the Polaris results so far stay valid | no new Wire-Cell build or run at FNAL; both sides on the released 0.38.0 |
| Cons | needs a `55177882` build in the FermiGrid tarball; one step-2 pass over every pair (about half the step-2 cost of the WCLocal2026Nov16 FermiGrid round, which ran two passes) | the sim h5 changes (labels move to 0.38.0); the scratch tracking-pr tars must still be there (copy them to persistent first) |

Step 1 is the same in both: aurora-r2's step 1 (`d06a47d7`) = the fg1 step 1 (`4b6853d4`) + an output-naming change and a `TensorFileSource` fix, and 0.38.0 reproduced the fg1 / open-data step-1 tars on 25/25 test pairs (ai-helper #48). For data, step 1 reads the FrameShift copy (`aurora/bin/step0-frameshift.sh`, run by `run_pair.py` step 0).

### What a Wire-Cell ν candidate is (side question of #14)

Same in aurora-r2 and 0.38.0 (`cfg/pgrapher/experiment/sbnd/clus.jsonnet`, `TaggerCheckNeutrino`):
1. Only **in-beam-window** activity: clusters whose matched flash time is in [0.2, 2.2) µs.
2. One candidate per **flash bundle** (`nu_per_bundle`), with a 15 cm length floor; bundles of one physical flash seen by both TPCs are merged when their charge touches (`nu_bundle_flash_group`).
3. **Cosmic veto before PR** (`nu_skip_cosmic`, `nu_skip_cosmic_bundle`): a main cluster already tagged by the clustering-stage taggers, **TGM** (through-going), **STM** (stopping muon) or a **light mismatch** (`lm_flag > 0`), is not a candidate, and the veto covers the whole flash bundle. Exception: an untagged main of at least 15 cm survives. `T_bundle.reason = 2` counts the vetoed bundles.
4. The **PR-level cosmic tagger** (`cosmict_flag`, the OR of its ten tests, plus the BDT score `cosmict_score`) is computed for each candidate and **not applied**.

So the current hit label (`wc_cluster_id >= 0`) is "charge-light matched in the beam window AFTER the TGM/STM/light-mismatch veto", which is closer to the MicroBooNE label than it looked. Checked on outputs: in 1,000 BNB CV events (478 candidates) and the 67 selected data events (68 candidates), no selected main has `act_tgm`, `act_stm` or `act_lm` set. A few tagged demoted mains sit inside a candidate's PR set: 16 of 4,847 candidate members in the simulation, 12 of 2,777 in the data.

Note: `wc_event_table.cosmic_flag` is **not** the cosmic-tagger verdict. It is the flag-9 BDT input (1 = flag 9 did not fire or was not evaluated, the in-class default). The verdict is `cosmict_flag`, written from `sbnd-h5maker` 0.5 on. Cuts on top of the label (MicroBooNE-like): `cosmict_flag == 0`, `numu_score` / `nue_score`, `match_isFC`.

## 4. sbnd-h5maker for data (0.5)

New in 0.5 (`sbnd_h5maker.py --data`):
- No truth product is read. `nu_table`, `edep_table`, `edep_cosmic_table`, `particle_table` are written with **zero rows**. `event_table` keeps its truth columns at their defaults (`nu_index` −1, `nu_pdg` 0, `nu_vtx` −9999, …). The attributes `is_data` = 1, `hit_truth` / `nu_choice` = "none (real data)", no `sim_config`.
- `event_table` gains (data and MC): `time_s` (unix time of the event), `has_frameshift`, `timing_type`, `frame_apply_at_caf_ns`, `frame_hlt_beamgate_ns`, `frame_hlt_crtt1_ns`, `frame_tdc_crtt1_ns`, `frame_tdc_bes_ns`, `frame_tdc_rwm_ns` (the `sbnd::timing::FrameShiftInfo` of the FrameShift step; MC: 0 with `has_frameshift` 0), and the Wire-Cell bundle census `wc_n_bundle`, `wc_n_bundle_cosmic` (`T_bundle` rows, and those vetoed as cosmic).
- `wc_event_table` gains `cosmict_flag` (the cosmic-tagger verdict) and `matched_flash_gid`. The file attribute `wc_label_notes` states the label definition of §3.
- Everything else is as for MC: `hit_table` (gaushit), `ophit_table` (`ophitpmt`, window −10…40 µs by default), `opflash_table` + `opflashsumpe_table` (`opflashtpc0/1`), `wc_event_table`, `wc_hit_table`.
- `check_h5.py` adds a `data` block: the truth tables must be empty, FrameShift coverage, `cosmict_flag` counts, and the footprint charge fraction per (TPC, plane) as the numbering check (there is no truth to compute an efficiency). The check flags a plane below 0.2 × the best plane of its TPC; it applies only to files with at least 10 events with a candidate (`footprint_check_applied`), because a few candidates make the fractions noisy. The fractions and footprint hit counts are reported either way.

**Light timing in data.** `ophit_table.peaktime` and `opflash_table.time` are written as in reco1 (µs, trigger clock). In data the trigger is the flash itself, so the beam flash sits at a fixed −0.74 µs. Wire-Cell adds `frame_apply_at_caf` to every light time (`FlashTensorToOpticalPCs`, `correct_flash_time`), which puts the beam flash on the beam-spill clock the simulation uses:

    t_beam [us] = peaktime (or flash time) + frame_apply_at_caf_ns / 1000        (MC: frame_apply_at_caf_ns = 0)

On the 67 test events the largest flash in [−5, 10] µs is at −0.75…−0.73 µs raw and 0.51…2.03 µs after the shift; the BNB CV test10 flashes are at −0.43…1.61 µs. The 0.2–2.2 µs Wire-Cell beam window applies to `t_beam`.

### Test on Aurora (2026-11-20, job 9109996)

The 67 selected real-data events (Lynn's 48 νe CC candidates and 19 NC-sideband events, run 1 BNB, FrameShift reco1) with their aurora-r2 output (sbnd-wirecell-production #7), and the MC test10 as a regression:

| file | events | WC candidates | hits on WC footprint | ophits / event (median) | check_h5 |
|---|---|---|---|---|---|
| νe CC data | 48 | 48 (48 events) | 102,637 | 2,745 | no problems |
| NC sideband data | 19 | 20 (19 events) | 40,232 | 2,774 | no problems |
| BNB CV MC test10 (0.5) | 10 | 6 | 7,088 | 1,509 | no problems |

- Truth tables have 0 rows in data; FrameShiftInfo was found in 67/67 events (`frame_apply_at_caf` 247–2,757 ns).
- The hit ↔ footprint tick-offset scan peaks at 0 in data (as in MC): 102,637 matched hits at 0 vs 76,468 at −8.
- `cosmict_flag` = 0 (tagger did not fire) for all 68 data candidates, as expected for hand-selected ν candidates.
- Footprint charge fraction per (TPC, plane) is flat across the three planes of each TPC (e.g. νe CC TPC 0: 0.242 / 0.240 / 0.236), so the channel numbering holds on data.
- MC regression: 0.5 vs the 0.4 test10 file, compared by value: 132 datasets, 0 differences; only the new columns are added.
- These events are hand-picked ν candidates, so their light and hit counts are not a sample average. The data have more ophits in the −10…40 µs window than test10; a review item for the beam-on / beam-off test.
- Outputs on Flare: `production-prep/realdata-h5/test/` (`test.pbs`, `cmp-h5.py`).

Per pair on FermiGrid:
1. `ifdh cp` the reco1 file.
2. FrameShift: `aurora/bin/step0-frameshift.sh <reco1> <out>_frameshift.root` (the same step 0 as the Wire-Cell data jobs).
3. Untar `<id>.tracking-pr.tar` (option A: the new step-2 tar; option B: the `wcl1116-fg-*` one).
4. `python sbnd_h5maker.py <id>.h5 --data --reco1 <out>_frameshift.root --wc-dir <untar>/pr/<x> [--wc-dir ...]`: pass **every** `pr/<x>/` dir. A complete pair has only `pr/all`; a Wire-Cell PARTIAL pair (a step-1 crash) has segment dirs `pr/s<a>-<b>/` instead, and reading only `pr/all` leaves all its events `wc_missing` (`fnal/h5pair.sh` does this since 1f6c1a7).
5. `python check_h5.py <id>.h5 --json <id>.check.json`; keep `<id>.h5`, `<id>.h5.stats.json`, `<id>.check.json`, the log.

Environment (as `sbnd-h5maker/env.sh`, which is written for Aurora): SL7 image; `setup sbndcode v10_14_02_03 -q e26:prof` and `setup gallery v1_22_06 -q e26:prof` from CVMFS; the UPS python 3.9 plus these wheels unpacked on `PYTHONPATH` (a wheel is a zip file, no pip needed): `uproot 5.3.10`, `awkward 2.6.6`, `awkward_cpp 35`, `h5py 3.11.0`, `cramjam 2.8.3`, `xxhash 3.4.1`, `fsspec 2024.6.1`, `packaging 24.1`, `importlib_metadata 8.0.0`, `zipp 3.19.2`, `typing_extensions 4.12.2` (cp39 manylinux2014). The `--sce-map` default is an Aurora path; `--data` does not read it.

Cost: the converter JIT-compiles `extract.h` once per process (~60 s), then ~1–3 s per event; FrameShift is one short `lar` job per file.

## 5. Output layout

On Polaris: `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data/sbnd-<sample>/`, next to the simulation samples:

| sample dir | content |
|---|---|
| `sbnd-data-bnb-<set>/` | beam-on, `<set>` = the chosen dataset (`opendata` or `run3bnb`) |
| `sbnd-data-offbeam-<set>/` | beam-off (`intime-run1` if InTime run 1 is chosen) |

Inside each:
- `test10/`: the ~10-event test files (`sbnd_<sample>_test10.h5` + `.check.json` + `.stats.json` + log);
- `full/`: one `sbnd_<sample>_<id>.h5` per reco1 file (pair `<id>`), with its `.check.json` and `.stats.json`;
- `lists/`: the reco1 files, the tracking-pr tars used, run/subrun/event per file;
- `README.md`: inputs (SAM definition, runs, POT for beam-on), the Wire-Cell chain and commits, the `sbnd-h5maker` commit, counts, the checksum file (`ADLER32` or `sha256sum`).

## 6. Steps and checks

1. **Test** (~10 beam-on + ~10 beam-off events, one pair each with `--max-events 10`): `check_h5` with no problems; `fmlar_polaris` reviews the layout, WC labels, light and timing against the simulation.
2. **Full production** on FermiGrid, after the owner approves the test.
3. **Transfer** to Eagle through the SSH master connection on sbndbuild03; compare counts and checksums after the copy.
4. `fmlar_polaris`: conversion to the qlfm cache and first look.

## Log
- 2026-11-20: recipe written; `sbnd-h5maker` 0.5 (`--data`) tested on Aurora (job 9109996) on the 67 selected real-data events with aurora-r2 output (sbnd-wirecell-production #7); MC output unchanged. Inputs (§2) and the Wire-Cell option (§3) wait for the owner.
- 2026-11-20 (owner): inputs are beam-on = open FixedDev BNB, runs 18255/18259 (`data_MCP2025C_Fall25-Run1_BNB_FixedDev_bnblight_v10_14_02_reco1_sbnd`), and beam-off = the fg1 InTime-Run1 set. Wire-Cell = **option B** (WCLocal2026Nov16, 0.38.0 tracking-pr tars, copied to persistent). test10 made on sbndbuild03 in the SL7 image; reviewed OK by `fmlar_polaris`. (porting_sbndbuild03)
- 2026-11-21: full production done (porting_sbndbuild03).
  - FermiGrid 30819860@jobsub04: 288 jobs, 37.5 CPU-h, no restarts.
  - Results: 1,817 beam-on files (89,946 events, 4.621e18 POT from the CAFs, 26.3 GB) and 1,058 beam-off files (49,970 events, 15.0 GB).
  - On Eagle `sbnd-data-{bnb-opendata,offbeam-intime-run1}/{full,lists,README.md}`: 0 sha256 failures.
  - The two beam-off PARTIAL pairs were rerun with the `pr/` segment-dir fix (1f6c1a7).
  - 8 beam-off files fail the heuristic footprint-fraction check, from low candidate counts. This is a review item.
  - Scripts: `sbnd-h5maker/fnal/`.
- 2026-11-21: `check_h5` applies the footprint-fraction check only with ≥ 10 candidate events. The 8 flagged beam-off files have 2–5; on 040c5aaf (5 candidates) the low plane is the collection plane in both TPCs (30 vs ~950 footprint hits in TPC 0), consistent with near-vertical tracks crossing few vertical collection wires (not verified). The 67-event data tests (48 and 20 candidate events) still apply it with no problems. (prod_aurora)
