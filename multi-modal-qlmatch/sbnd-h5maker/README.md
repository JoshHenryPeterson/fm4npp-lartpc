# sbnd-h5maker: SBND reco1 + Wire-Cell → HDF5 in the MicroBooNE open-data layout

Converts Avinay's SBND Gen2 reco1 MC, plus our Wire-Cell img→PR output (aurora-r2,
sbnd-wirecell-production #4), into HDF5 files with the table and column names of the MicroBooNE
open data (`HDF5Maker`, uboone/hdf5maker branch `opensamples`; zenodo 8370883).

- What qlfm needs: `docs/data-inventory-sbnd.md`.
- Progress: `docs/sbnd-h5maker.md` and issue #11.

| File | Purpose |
|---|---|
| `sbnd_h5maker.py` | Driver, run inside the SL7 container. gallery + PyROOT read the art products; uproot reads the Wire-Cell `tracking-pr.root`; h5py writes the output. |
| `extract.h` | C++ helper, JIT-compiled by cling (about 60 s per process). It fills flat per-event vectors and implements the hit-truth projection. |
| `env.sh` | Container setup: sbndcode v10_14_02_03, gallery v1_22_06, and h5py/uproot wheels for the UPS python 3.9. |
| `make_detector_json.py` → `sbnd_detector.json` | Detector constants, the linear wire-plane model (channel from (y, z)) and the OpDet positions. |
| `check_h5.py` | Layout checks and summary distributions, including the WC label efficiency and purity per (tpc, plane). Needs only h5py + numpy. |
| `relabel_wc.py` | Rewrites `wc_hit_table` of an existing file from the Wire-Cell footprints (h5py + uproot, no ROOT). Used to fix 0.3 files. |
| `pbs/run.pbs` | Aurora PBS job: one converter process over a list of aurora-r2 unit directories. |

## Run (Aurora)

```bash
qsub -v "OUT=/path/out.h5,UNITS=units.txt,MAXN=10" pbs/run.pbs   # UNITS: one aurora-r2 unit dir per line
```

- An aurora-r2 unit directory holds `reco1_files.json` (its reco1 files) and `pr/all/pr_evt<r>_<s>_<e>/tracking-pr.root`.
- The h5py/uproot wheels (py3.9, manylinux2014) are unpacked into `$Y/tools/sl7-py39-h5/site`. A wheel is a zip file, so no pip is needed.
- `$Y/tools/sl7-py39-h5/hostpy.sh check_h5.py out.h5` runs the check on a login node.

## Layout

Every table is a group with one `(N, k)` dataset per column, plus:
- `event_id`: `(N, 3)` = (run, subrun, event);
- `event_id.seq_cnt`: [event sequence number, row count] for each event that has rows.

Rows are contiguous per event, so `qlfm/data/h5reader.py` reads the files unchanged.

| Table | Rows | Columns (MicroBooNE names first) |
|---|---|---|
| `event_table` | 1 per event | `is_cc, nu_pdg, nu_energy` [GeV], `lep_energy, nu_dir(3), nu_vtx(3)` [cm], `nu_time_ns, lep_pdg, nu_mode, nu_int_type`, all for the event's chosen neutrino; `nu_index` (which `nu_table` row, −1 = none), `n_nu, beam_gate_start_ns, beam_gate_width_ns, wc_status` (1 = Wire-Cell output present), `file_index` (into the `reco1_files` attribute), `n_mcp_no_truth` |
| `nu_table` | 1 per neutrino MCTruth | `nu_index` (position in the `generator` collection), `nu_pdg, is_cc, nu_energy, lep_energy, lep_pdg, nu_vtx(3), nu_dir(3), nu_time_ns, nu_mode, nu_int_type, in_av, edep_collection` [MeV on collection-plane hits], `nhits_collection` (collection hits whose largest contributor is this neutrino) |
| `hit_table` | gaushit | `hit_id` (= row in the event), `integral, rms, tpc, local_plane, local_wire, local_time` [ticks from readout start], `channel, peak_amplitude` |
| `edep_table` | (hit, particle) | `hit_id, g4_id, energy` [MeV], `energy_fraction`, `is_max_ide, num_electrons, origin, nu_index` (the particle's neutrino). **Beam-neutrino particles only** (MicroBooNE semantics: the per-hit ν deposit is the sum over the hit's rows). |
| `edep_cosmic_table` | (hit, particle) | Same columns, for every other origin. 2 = CORSIKA cosmic; 0 = no MCTruth link, in which case `g4_id` is negative. |
| `particle_table` | particles with deposits + ancestors | `g4_id, g4_pdg, parent_id, category` (HDF5Maker categories 0–8), `instance, origin, nu_index` (−1 = not from a beam neutrino), `momentum, start_position(3), end_position(3), start_time, start_process, end_process` |
| `ophit_table` | PMT ophits (`ophitpmt`) | `hit_id, hit_channel` (OpChannel 0–311, PMTs only), `peaktime` [µs, trigger clock], `peaktime_abs, width, area, amplitude, pe`. Kept only in the window in the `ophit_window_us` attribute (default −10…40 µs). |
| `opflash_table` | `opflashtpc0` + `opflashtpc1` | `flash_id, tpc, time, time_width, y_center, y_width, z_center, z_width, totalpe` (all flashes) |
| `opflashsumpe_table` | (flash, channel with PE > 0) | `sumpe_id, flash_id, pmt_channel, sumpe` |
| `wc_event_table` | 1 per Wire-Cell ν candidate | `nu_index, cluster_id` (main), `has_vertex, neutrino_type, reco_vtx(3), reco_enu` [MeV, `T_kine`], `flash_time_us, flash_pe, flash_tpc, match_isFC, cosmic_flag, numu_score, nue_score, cosmict_score, n_clusters_in_pr` |
| `wc_hit_table` | 1 per hit (same rows as `hit_table`) | `hit_id, wc_cluster_id` (cluster of a WC ν candidate whose 2D footprint contains the hit, else −1), `wc_nu_index, wc_main` (in the candidate's main cluster), `wc_in_enu` (cluster counted in the WC energy) |

The file attributes hold:
- `detector`: the `sbnd_detector.json` text;
- `units`: the units of every time, energy and position column. Event-level times are ns on the G4 clock; ophit and flash times are µs on the trigger clock;
- `nu_choice`: the rule for the event-level neutrino (below);
- `sim_config`: the detector-simulation and hit-finder settings (diffusion D_L/D_T, lifetime, field response, noise model, PMT efficiencies, hit-finder module);
- `reco1_files`, `wc_dirs`, `labels`;
- `hit_truth` and `truth_projection`: how the per-hit truth was made;
- `ophit_window_us` and `wc_tick_offset`.

## Several neutrinos per spill

A BNB CV spill can hold more than one neutrino interaction (`n_nu` = 2 in 4 of the first 10 events). Every neutrino has a `nu_table` row, and every ν deposit carries its `nu_index`.

`event_table` keeps one neutrino, so the MicroBooNE-style T2/T3/T4 code works unchanged:
- it is the neutrino with the largest `edep_collection`, the deposited energy of its particles on collection-plane hits;
- ties (for example, nothing visible) go to the neutrino with its vertex in the active volume, then to the lowest `nu_index`;
- `event_table.nu_index` says which neutrino it is.

## Per-hit truth: projected Geant4 deposits

The Gen2 reco1 MC has no per-hit truth:
- the `gaushitTruthMatch` assns are empty;
- the `simtpc2d:simpleSC` SimChannels exist but hold no IDEs.

The converter rebuilds the BackTracker quantities from the Geant4 deposits (`ionandscint:priorSCE`):
1. Take the deposit midpoint.
2. Apply the forward space-charge shift (`SCEoffsets_SBND_E500_dualmap_CV_voxelTH3.root`, `TrueFwd_Displacement_*`, the map Wire-Cell's truth labeler uses).
3. Drift to each wire plane (x = ±201.45/201.75/202.05 cm, 0.1563 cm/µs) and map to a channel with the linear plane model.
4. Compute tick = 410 + 2·(t_G4 [µs] + drift [µs]) + dt[plane]. Fitted on BNB CV: dt = +5 (U), +2 (V), −2 (Y) ticks.
5. Sum per particle the deposits on the hit's channel within PeakTime ± 1·RMS. This is the BackTracker window (`HitTimeRMS` 1).

Particles that larg4 dropped are folded into their kept ancestor through `ParticleAncestryMap`. Diffusion and the field-response shape are not modelled; the hit window absorbs them.

On the first 10 BNB CV events:
- 2.5% of hits get no deposit;
- 91% of the projected ν energy on the collection planes lands on a hit;
- a ±2 RMS window double-counts (more matched than projected energy).

If a sample has filled assns or SimChannels, the converter uses those instead (`hit_truth` attribute: 1 = assns, 2 = SimChannel BackTracker, 3 = projection).

## Wire-Cell labels

They come from the per-event `tracking-pr.root` of aurora-r2:
- `T_tagger`: one entry per ν candidate, with `act_cluster_id / act_in_pr / act_in_enu` for the clusters in the candidate;
- `T_kine`: `kine_reco_Enu`;
- `T_proj_data`: the 2D footprint (channel, 4-tick time slice) of every cluster in a ν candidate.

A gaushit is on a cluster when (WC channel, floor(PeakTime / 4)) is in that cluster's footprint.

Wire-Cell numbers the channels plane-major, [U0 U1][V0 V1][Y0 Y1], while LArSoft numbers them TPC-major, [U0 V0 Y0][U1 V1 Y1]. The wire order inside each block is the same, so WC channel = LArSoft channel + an offset per (tpc, plane): (0, +1984, +3968) for TPC 0 and (−3654, −1670, 0) for TPC 1. The offsets are in the `wc_channel_offsets` attribute. Versions before 0.4 matched on the LArSoft channel, so only U in TPC 0 and Y in TPC 1 got labels. The tick offset is 0: a scan over ±3400 ticks peaks at 0. The main cluster wins when footprints overlap.
