# MicroBooNE data curation: input tier, Wire-Cell overlap, real-data inputs

Status 2026-11-11. Tracking issue: [#10](https://github.com/HaiwangYu/fm4npp-lartpc/issues/10).

So far the project has trained on the MicroBooNE open-data HDF5 (Zenodo
[8370883](https://zenodo.org/records/8370883), inclusive BNB overlay, NoWire; and
[7261921](https://zenodo.org/records/7261921), the nue sample). This note covers three questions:

1. From which data tier can HDF5 like ours be made?
2. Does the open dataset overlap the existing Wire-Cell (WC) final outputs ("checkout" files),
   so the two can be compared event by event?
3. For real data (BNB beam-on and beam-off EXT), where are the input files (samweb)?

Josh's setup for rebuilding `hdf5maker` and converting the public artroot on the grid is in
[JoshHenryPeterson/fm4npp-lartpc: multi-modal-qlmatch/docs/microboone_artroot_to_hdf5.md](https://github.com/JoshHenryPeterson/fm4npp-lartpc/blob/main/multi-modal-qlmatch/docs/microboone_artroot_to_hdf5.md).
This note does not repeat it.

Tools used: `samweb` from CVMFS
(`/cvmfs/fermilab.opensciencegrid.org/products/common/prd/sam_web_client/v3_6/NULL/bin/samweb -e uboone`,
read-only queries need no proxy), `uproot`, and `scripts/artroot_event_ids.py`, which dumps
(run, subrun, event) from artroot without art.

## 1. Input tier for the HDF5

**Answer: reco2 artroot.** The public HDF5 was made by the `HDF5Maker` analyzer
([uboone/hdf5maker](https://github.com/uboone/hdf5maker), branch `opensamples`) run on the
MCC9.1 run1 overlay reco2 files. MCC9 reco2 files keep the reco1 products, including the
`butcher` wires, so one reco2 file holds every product HDF5Maker reads.

The public artroot is on dCache disk (readable without credentials over xrootd, and directly
on the GPVMs):

| Directory under `/pnfs/uboone/persistent/PublicAccess/` | Files |
|---|---|
| `prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run1_reco2_reco2` | 24,436 |
| `prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run1_reco2_reco2_nfiles3400` | 3,400 |
| `prodgenie_bnb_intrinsice_nue_uboone_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2` | 2,151 |

The samweb definition with the same name as the first directory has 24,419 files, 17 fewer
than the public list. The difference was not checked.

Products HDF5Maker reads (labels from `HDF5Maker.fcl`), as found in one public file:

| HDF5Maker input | Product in the reco2 file | Producing art process (stage) |
|---|---|---|
| HitLabel `gaushit` | `recob::Hits_gaushit` | `OverlayStage1a` (reco1) |
| WireLabel `butcher` (WithWire only) | `recob::Wires_butcher` | `OverlayStage1a` (reco1) |
| OpHitLabel `ophitBeam` | `recob::OpHits_ophitBeam` | `OverlayStage1a` and `OverlayStage1Optical` |
| OpFlashLabel `simpleFlashBeam` | `recob::OpFlashs_simpleFlashBeam` | `OverlayStage1a` and `OverlayStage1Optical` |
| HitTruthLabel `gaushitTruthMatch` | hit-MCParticle assns | `OverlayRecoStage1b` |
| PandoraLabel `pandora` | PFParticles, slices, clusters | `OverlayRecoStage2` (reco2) |
| T0Label `flashmatch` | `anab::T0s_flashmatch` | `OverlayRecoStage2` (reco2) |
| TruthLabel `generator`, `largeant`, `mcreco` | MCTruth, MCParticles, MCShower/MCTrack | gen / g4 |

Notes:
- `ophitBeam` and `simpleFlashBeam` exist from two processes in MC (`OverlayStage1a`,
  `OverlayStage1Optical`) and three in data (`DataOpticalFilter`, `DataRecoStage1Test`,
  `DataStage1Optical`). `getByLabel` without a process name returns the most recent process.
  The art `ProcessHistory` puts the optical re-reconstruction last in both: `OverlayStage1Optical`
  after `OverlayStage1a`, and `DataStage1Optical` after `DataRecoStage1Test`. So HDF5Maker reads
  the `*Stage1Optical` light in both data and MC, which is consistent. Pin the process name in the
  data job anyway.
- reco1 files alone would lack `pandora` and `flashmatch`. HDF5Maker only needs those for the
  Pandora tables and T0, so the core hit/flash tables could be made from reco1.
- The public HDF5 has 753,467 events, against 1,046,139 in the artroot (72.0%). A full ID dump of
  all 24,436 artroot files (165 have no events) and of the `event_table` of all 18 HDF5 files shows:
  - every HDF5 event is in the artroot, with no duplicates;
  - **whole files are in or out**: 17,468 artroot files are fully in the HDF5, 6,803 are fully
    missing, none partly (and likewise 54,392 subruns fully in, 21,459 fully out);
  - the missing files are spread evenly over the list (69-75% kept in every block of 1,000), have
    the same mean events per file (43.0 vs 43.1) and the same `PhysicsRun`/`BeamOffRun` mix.

  So the loss looks like random whole-file loss (failed jobs, or a random subsample), not an event
  selection. Converting the full list would add about 293k events.
- The nue HDF5 (Zenodo 7261921, 20 files, 89,339 events) holds every event of its 2,151 artroot
  files (2,133 non-empty).

## 2. Overlap with the Wire-Cell checkout files

### 2.1 What the checkout files contain
`/pnfs/uboone/persistent/users/jjo/processed_chekcout_rootfiles/` has `bnb/{run1,run2,run3,run4,run4a-d,run5}/{cv,data,systs}`
and `numi*`. Each file has one directory, `wcpselection`, with the trees
`T_eval`, `T_PFeval`, `T_BDTvars`, `T_KINEvars`, `T_pot`.
- `T_eval` has `run/subrun/event`, flash/match info, the `stm_*` cosmic tagging, and for MC
  the truth (`truth_nuEnergy`, `truth_vtx*`, `truth_isCC`, ...) and weights.
- `T_pot` has one entry per subrun (`runNo`, `subRunNo`, POT, spills).
- The files are not skimmed: `match_found` is true for 78.5% of the events in the run1 ν
  overlay, and events failing preselection are kept.
- **They hold no input-level information** (no hits, wires or flashes, and no input file names).
  The inputs have to be traced by event ID and samweb. For data, the reco2 file names end in
  `..._reco1_postwc_...`, so the reco2 and the WC chain share the same reco1 parents. The
  celltree definitions are in samweb (e.g. `data_bnb_mcc9.1_v08_00_00_25_reco2_C1_celltree`,
  `data_extbnb_mcc9.1_v08_00_00_25_reco2_C1_celltree`; `..._reco2_C1_celltree` has 0 files).

### 2.2 Open-data overlay vs WC run1 ν overlay (event level)
I dumped event IDs from a random 400 of the 24,436 public artroot files (15,709 events, all
IDs unique) and matched (run, subrun, event) against `bnb/run1/cv`:

| WC file | WC events | Public events found in WC |
|---|---|---|
| `checkout_prodgenie_bnb_nu_overlay_run1.root` | 581,941 | **52.7%** |
| `checkout_prodgenie_bnb_nu_overlay_run1_PF.root` | 581,941 | 52.7% (same events) |
| `checkout_prodgenie_bnb_nu_overlay_LowE_run1.root` | 20,364 | 0% |

The overlap is by whole subruns:
- Of the 2,036 sampled subruns, 1,055 are in the WC `T_pot`. Of those, 1,026 have every event
  matched and 29 have most of them.
- No subrun outside `T_pot` has any match.
- Where checked, the event count per subrun is identical on both sides.

So these are the same events, not chance ID collisions.

**Full count** (all 24,436 artroot files, all 18 HDF5 files, `checkout_prodgenie_bnb_nu_overlay_run1.root`):

| Event set | Events | Also in WC | Also in published HDF5 |
|---|---|---|---|
| Public artroot | 1,046,139 | 557,568 (53.3%) | 753,467 (72.0%) |
| Published HDF5 (what we train on) | 753,467 | **402,231 (53.4%)** | - |
| WC `T_eval` | 581,941 | - | 402,231 (69.1%) |

- 557,568 of the 581,941 WC events (95.8%) are in the public artroot. The other 24,373 are not.
- Of the WC events in the artroot, 155,337 sit in the files missing from the published HDF5.
  Converting the full list ourselves would raise the WC-comparable set from 402k to 558k events.
- Truth check on the 402,231 shared events: HDF5 `nu_energy` (GeV) × 1000 equals WC
  `truth_nuEnergy` (MeV) for all of them (median difference 0), and `nu_pdg` and `is_cc` agree
  100%. The matched events are the same simulated interactions.

Intrinsic nue (`checkout_prodgenie_bnb_intrinsic_nue_overlay_run1{,_PF}.root`, 68,203 events):
29,337 shared with the public nue HDF5 (32.8% of its 89,339 events, 43.0% of WC). Truth agrees
100%. The `_LowE` WC samples share no events with the public files.

How the HDF5 IDs were read: `event_table` straight from Zenodo with HTTP range requests
(`h5py.File(fsspec.filesystem("http").open(url, block_size=4<<20))`, about 35 s per file; run one
file at a time, since parallel requests get HTTP 429).

The overlay event IDs carry the run number of the overlaid EXT data (e.g. run 7419). The
public file names start with the EXT file name.

### 2.3 Real data vs WC run1 data (subrun level)
WC run1 data checkout files (`bnb/run1/data`):

| File | Events | Runs | Subruns in `T_pot` |
|---|---|---|---|
| `checkout_data_bnb_run1_PF.root` | 544,769 | 4952-6998 | 85,064 |
| `checkout_data_bnb_run1_all.root` | 545,083 | 4952-6998 | 85,111 |
| `checkout_data_extbnb_run1_PF.root` | 633,290 | 4952-7770 | 91,566 |
| `checkout_data_bnb_run1_5e19.root` | 177,957 | 5121-5946 | 22,750 |
| `checkout_data_bnb_run1_opendata_Aug8.root` | 139,054 | 5121-5946 | 19,332 |

`checkout_data_bnb_run1_opendata_Aug8.root` is BNB beam-on data from the same run range as the
5e19 sample. It is most likely the unblinded ("open") run1 subset, **not** the Zenodo public
release, which is MC overlay only. The POT branches in the data `T_pot` are not filled
(data POT comes from the beam database).

For 40 random files of each samweb reco2 definition, the share of their subruns found in the WC
`T_pot`:

| samweb definition | Subruns in WC |
|---|---|
| `data_bnb_mcc9.1_v08_00_00_25_reco2_C1_beam_good_reco2` vs `checkout_data_bnb_run1_PF` | 564/712 (79%) |
| `data_extbnb_mcc9.1_v08_00_00_25_reco2_C1_all_reco2` vs `checkout_data_extbnb_run1_PF` | 1280/2208 (58%) |
| `data_extbnb_mcc9.1_v08_00_00_25_reco2_C2_all_reco2` vs `checkout_data_extbnb_run1_PF` | 955/1923 (50%) |

In the other direction, 12 random subruns of `checkout_data_bnb_run1_all` are all in C1
`beam_good`, and 12 random subruns of `checkout_data_extbnb_run1_PF` are all in EXT C1 (9) or
C2 (3). So the WC run1 data is a subset of these definitions.

For real data, (run, subrun, event) identifies an event uniquely, so a direct WC comparison is
possible for any reco2 data we convert. Event-level matching needs IDs read from the reco2
files, which are on tape (§3).

## 3. Inputs for new real-data HDF5

### 3.1 samweb definitions (MCC9.1, `v08_00_00_25`, reco2)
Beam-on: `data_bnb_mcc9.1_v08_00_00_25_reco2_<epoch>_beam_good_reco2`.
Beam-off EXT: `data_extbnb_mcc9.1_v08_00_00_25_reco2_<epoch>_all_reco2`.
The epochs are C1, C2, D1, D2, E1, E2, F, G1, G2, G2a. C1 and C2 are run1, based on the run
ranges above. The mapping of the other epochs to runs 2 and 3 still needs confirming.

| Definition | Files | Events | Size |
|---|---|---|---|
| `data_bnb_mcc9.1_v08_00_00_25_reco2_C1_beam_good_reco2` | 5,864 | 638,039 | 3.44 TB |
| `data_extbnb_mcc9.1_v08_00_00_25_reco2_C1_all_reco2` | 1,905 | 441,403 | 2.38 TB |
| `data_extbnb_mcc9.1_v08_00_00_25_reco2_C2_all_reco2` | 1,138 | 398,046 | 2.05 TB |

All epochs (BNB `beam_good` / EXT `all`; first run is that of the first file listed):

| Epoch | First run | BNB files / events / size | EXT files / events / size |
|---|---|---|---|
| C1 | 5127 | 5,864 / 638,039 / 3.4 TB | 1,905 / 441,403 / 2.4 TB |
| C2 | - | (no beam_good def) | 1,138 / 398,046 / 2.0 TB |
| D1 | 8336 | 307 / 6,733 / 37 GB | 34 / 6,084 / 33 GB |
| D2 | 8957 | 4,908 / 701,858 / 3.8 TB | 6,995 / 1,088,484 / 5.9 TB |
| E1 | 11203 | 1,688 / 277,157 / 1.5 TB | 1,777 / 365,471 / 2.0 TB |
| E2 | - | 0 files | 1,632 / 407,974 / 2.2 TB |
| F | 13907 | 927 / 149,481 / 0.8 TB | 698 / 183,667 / 1.0 TB |
| G1 | 15701 | 6,288 / 786,256 / 4.4 TB | 8,430 / 1,628,176 / 9.0 TB |
| G2, G2a | 17567, 18926 | 42 and 68 files, 5 and 9 events | 764 / 253,145; 1,362 / 388,434 |

The event counts match the WC data checkout files run by run: run2 WC 975,911 vs D2+E1 978,991 (WC
runs 8406-11947), and run3 WC 900,334 vs F+G1 935,737 (WC runs 13697-17566). This supports
D* and E* = run2 and F and G1 = run3.

Products: one C1 `beam_good` file that happened to be on disk (`ONLINE_AND_NEARLINE`; the EXT
files checked were `NEARLINE`) has `gaushit` and `butcher` (`DataRecoStage1Test`), `pandora` and
`flashmatch` (`DataRecoStage2`). `ophitBeam` and `simpleFlashBeam` each exist from **three**
processes (`DataOpticalFilter`, `DataRecoStage1Test`, `DataStage1Optical`), so a data fcl should
name the process, chosen to match what the overlay HDF5 used.

The beam-good definition is `file_type data and ub_project.name data_bnb_mcc9.1_v08_00_00_25 and data_tier reconstructed and ub_project.stage reco2_C1 and data_stream beam_good and defname: goodruns_mcc9_run1_hardcoded`,
so the run1 good-run list is already applied. File names show the chain
`..._bnb_..._merged_..._optfilter_..._reco1_postwc_...` (software optical filter, reco1, WC stage).

Location: tape only (`enstore:/pnfs/uboone/data/uboone/reconstructed/prod_v08_00_00_25/...`).
The files need to be prestaged (`samweb prestage-dataset --defname=...`) or read through a SAM
project in grid jobs. Prestaging 3-6 TB per definition needs a word with the data-management
people first.

Other related definitions exist (e.g. `data_bnb_optfilter_C1_5e19_goodruns_mcc9.1_reco2`,
`_high_lifetime`/`_low_lifetime` splits, filtered streams such as `_nucc_reco2` and
`_pi0_reco2`, `_celltree`). List them with
`samweb -e uboone list-definitions --defname='data_bnb_mcc9.1_v08_00_00_25_reco2_%'`.

### 3.2 HDF5Maker changes needed for data
`HDF5Maker_module.cc` (opensamples) assumes MC:
- With `EventInfo: "nu"` it throws if there is no MCTruth. Use `EventInfo: "none"`.
- It always reads `mcreco` MCShower/MCTrack (`getValidHandle`, which throws on data), and it
  always builds the backtracker / `ParticleInventoryService` (`initBacktrackingParticleVec`,
  `pi->Rebuild(e)`). These need to be skipped when `e.isRealData()`.
- Set `HitTruthLabel: ""` and `UseMap: false`. Check what the energy-deposit and particle
  tables do with no truth.
- `hdf5maker_uB_public_job.fcl` sets `DetectorClocksService.TrigModuleName: "triggersim"`;
  for data this should be `"daq"`. Check the remaining data-specific services, such as
  space-charge and calibration settings.
- The build is against uboonecode `v08_00_00_54` while data reco2 is `v08_00_00_25`. Reading
  should be fine, but test it on one file.

A data job fcl plus a small `isRealData()` guard in the module should be enough. Test it on one
prestaged C1 beam-good file and one EXT file.

## 4. Wire-Cell neutrino selection at hit level

Goal for the first round (decided 2026-11-11): four samples of about 1000 events each.
- BNB ν overlay and intrinsic ν_e overlay: events shared by the open data and the WC checkout.
- Beam-on BNB and beam-off EXT data.

Each gets the WC neutrino selection at event level and, if possible, at hit level.

### 4.1 Where WC hit-level information lives
- The **checkout files are pruned**: only the five event-level trees, in every run (run1, run3,
  run4b, run5 checked). The `T_PFeval` reco particles (`reco_startXYZT`/`reco_endXYZT`) are
  the finest WC info there.
- The WC chain for run1 is reco1 → celltree → WCP imaging, matching and nu selection (outside art)
  → the **`nusel*_WCP_*.root`** file → port into art (**reco1.5**,
  `ubreco/WcpPortedReco/ProducePort/job/run_slimmed_port_{overlay,data}.fcl`) → wirecell reco2 →
  checkout. The nusel files are what the port reads as `./merge.root`, and they are kept in samweb
  as data tier `celltree`.
- From the port modules (`PortedHits_module.cc`, `PortedSpacePoints_module.cc`), the nusel files
  contain:
  - `T_port_2d`: run, subrun, event, `channel` (0-8255), `start_tick` (4-tick bins, offset 2400),
    `charge`, `charge_error`, **`main_flag`**. This is 2D, hit-like deconvolved charge with the
    neutrino-candidate (in-beam, flash-matched) cluster flag.
  - `T_port_3d`: `x,y,z,q,nq`, `time_slice`, `ch_u/ch_v/ch_w`, **`main_flag`**. These are 3D space
    points with their wire/time projection already attached.
  - `T_eval` (and the other selection trees).
- The art products in reco1.5 (`portedThresholdhit`: `recob::Hit`; `portedSpacePointsThreshold`:
  `recob::SpacePoint`) **drop `main_flag`** and keep every cluster, so the nu/cosmic split is lost
  there. Use the nusel ROOT files directly. (Newer productions also port `Trec`/`Tcluster` space
  points with cluster IDs, and `WireCellAnaTree` has `SaveT*SpacePoints` options, but the run1
  productions were run without them.)
- Proposed hit label: a gaushit is WC-ν if its (plane, wire, peak tick) falls in a `T_port_2d` bin
  with `main_flag == 1` **and** the event passes the WC generic ν selection in `T_eval`.
  Otherwise it is WC-cosmic. The tick convention (offset 2400, 4-tick bins) still has to be checked
  on a real file.
- What `T_port_2d`/`T_port_3d` hold, from the WCP source
  ([BNLIF/wire-cell `uboone_nusel_app/apps/prod-wire-cell-matching-nusel.cxx`](https://github.com/BNLIF/wire-cell/blob/master/uboone_nusel_app/apps/prod-wire-cell-matching-nusel.cxx)):
  only the TPC-light bundle matched to the **in-time (beam) flash**. That is its main cluster
  (`main_flag = 1`) plus the bundle's other clusters (`main_flag = 0`). Cosmic clusters outside
  that bundle are not stored, and an event with no in-time match has no entries.
  `start_tick = 4 * time_slice` (the port subtracts 2400 to get LArSoft ticks). Each 3D point
  carries `ch_u/ch_v/ch_w` and `time_slice` from `convert_3Dpoint_time_ch`. For a ν/cosmic hit
  label that is enough: anything not in the main cluster is WC-cosmic.

### 4.1b What the stored nusel files actually contain (checked 2026-11-11)
**The stored `nusel*_WCP_*.root` files do NOT contain `T_port_2d`/`T_port_3d`.** Those exist only
inside the grid job. The stored files have `TC` (every imaged cell: `cluster_id`, `time_slice`,
`wire_index_u/v/w`, `wire_charge_*`), `TDC` (dead cells), `T_match` (cluster bundle to flash,
`event_type` bits: 0 in-time, 2 FC, 3 TGM), `T_flash`, `Trun` (with MC truth), and `T_eval` (flash
match and truth only; the `stm_*` cosmic tagging and the BDT scores are only in the checkout
files). That is more than `T_port_2d` gives: a 2D footprint for **every** WC cluster, not just the
in-time bundle.

`scripts/wc_hit_labels.py` rebuilds the labels. The in-time main cluster is chosen as in WCP
(in-time `T_match` rows, non-TGM first, max measured PE). Hits are matched by (plane, wire,
`time_slice = (local_time + 2400) // 4`). The tick offset 2400 is confirmed: the share of hits on a
WC cluster jumps from about 1% to 95% at 2400. The event-level generic ν selection comes from the
checkout. Check on one BNB overlay file (98 events, 1.6M hits, truth-ν hit = hit with an
`edep_table` entry):

| WC ν label | truth-ν hit efficiency | purity | cosmic hits mislabelled |
|---|---|---|---|
| in-time main cluster | 0.823 | 0.970 | 0.10% |
| main cluster & generic ν selection | 0.658 | 0.984 | 0.04% |

94.7% of all gaushits lie on some WC cluster footprint.

### 4.2 samweb definitions of the WC outputs (all on tape, NEARLINE)

| Sample | nusel (`celltree` tier) definition | Files | Events | Size |
|---|---|---|---|---|
| BNB ν overlay run1 | `prodgenie_bnb_nu_uboone_overlay_mcc9.0_run1_wcp_reco1.5_v08_00_00_29_reco1.5_celltree` | 25,065 | 1,072,287 | 4.05 TB |
| ν_e overlay run1 | `prodgenie_bnb_intrinsic_nue_uboone_overlay_mcc9.0_run1_wcp_reco1.5_v08_00_00_29_reco1.5_celltree` | 4,713 | 197,911 | 0.99 TB |
| Beam-on C1 | `data_bnb_mcc9.1_wcp_port_TEST_v08_00_00_28_reco1.5_C1_celltree` | 64,556 | 727,390 | 2.62 TB |
| EXT C1 | `data_extbnb_mcc9.1_wcp_port_TEST_v08_00_00_28_reco1.5_C1_celltree` | 30,006 | 471,872 | 1.64 TB |
| EXT C2 | `data_extbnb_mcc9.1_wcp_port_TEST_v08_00_00_28_reco1.5_C2_celltree` | 29,252 | 516,981 | 1.82 TB |

The matching `..._reco1.5` definitions hold the ported art files. To find the WC file for a given
reco2 file: its samweb parent (the reco1 file), then that parent's `nusel*_WCP_*` child whose
`ub_project.name` is the production above. Checked on a public ν_e file.

### 4.3 ν_e overlap
For 60 random public ν_e files (2,364 events), 1,085 (45.9%) are in
`checkout_prodgenie_bnb_intrinsic_nue_overlay_run1.root`, with 96 of 219 subruns in its `T_pot`.

## 5. Open items
- [x] Full event-ID dump of all 24,436 public artroot files: 557,568 events shared with WC run1
      overlay (§2.2).
- [x] Event IDs from the open-data HDF5 `event_table` (read from Zenodo): 402,231 HDF5 events in
      WC, truth agrees 100%. The 72% is random whole-file loss (§1).
- [ ] Save the shared (run, subrun, event) lists (402k BNB, 29k nue) next to the data cache, as
      the WC-comparison subset. Optionally convert the 6,803 missing files (+155k WC-shared events).
- [ ] Ask MicroBooNE (microboone_info@fnal.gov) why 6,803 files are missing from the public HDF5.
- [x] Which process's `ophitBeam`/`simpleFlashBeam` HDF5Maker reads: `*Stage1Optical` in both data
      and MC (§1 notes).
- [x] Round-1 samples (`datasets/uboone_round1/`, 4 x ~1000 events): staging 201 files (30.7 GB)
      through samweb definition `yuhw_fm4npp_round1_stage` (agreed with the production team).
      Done (2026-11-12): all four samples converted, WC-labelled and merged into
      `/pnfs/uboone/persistent/users/yuhw/fm4npp_round1/<sample>.h5`. Summary table and caveats are in
      `datasets/uboone_round1/README.md`.
- [x] Epoch-to-run mapping: D*/E* = run2, F/G1 = run3, from the event counts against WC (§3.1).
- [ ] Pick the real-data sample size (all of run1? a subset with matching WC events?).
- [x] Products in a data reco2 file: all present (one C1 beam-good file, §3.1). Still to do:
      check one EXT file once prestaged.
- [ ] Patch HDF5Maker for data, pin the optical process names, and run on one beam-good and one
      EXT file.
- [ ] POT / trigger counts for beam-on and EXT normalization if needed.
- [ ] Clarify "light triggered beam off": is it the standard EXT stream (external trigger plus
      the software PMT trigger / optical filter), or something else?
