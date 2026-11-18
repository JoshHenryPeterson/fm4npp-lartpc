# SBND HDF5 sample for qlfm: progress log (#11)

Goal: an SBND sample in the MicroBooNE open-data HDF5 layout (zenodo 8370883, `HDF5Maker`). It is built from:
- Avinay's Gen2 reco1 MC (BNB CV; later also νe CC and NC);
- our Wire-Cell img→PR chain output, aurora-r2 (sbnd-wirecell-production #4).

The content follows `data-inventory-sbnd.md` §4. Code: `sbnd-h5maker/` (README = table layout and methods).

Produced on ALCF Aurora:
- data: `/lus/flare/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data`;
- copied for Polaris to `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data`.

The Polaris session `fm4npp-lartpc@polaris` checks each sample before the next step.

## Owner decisions (2026-11-12)

- **Wire-Cell content:** event level (ν candidate, flash, scores, reco vertex and energy) plus hit level (WC ν-candidate cluster footprint per gaushit). This is the SBND stand-in for the Pandora baseline.
- **MC truth:** the MicroBooNE edep layout (hit → particle energy and fraction, particle table with category), plus cosmic truth.
- **Per-hit truth:** the reco1 MC has none, so it is built by projecting the 3D Geant4 deposits (below).
- **Sample sizes:** ~10 BNB CV events first, then 10k BNB CV, then a cost estimate for ~1M (CV + νe CC + NC). Nothing larger runs before the owner approves.

## Inputs

- **Reco1:** `/lus/flare/projects/neutrinoGPU/abhat/sbnd/gen2-wct0319-bnb-1m-restartable-20261013-shard0{1..3}` (+ batch01–09).
  - sbndcode v10_14_02_03; `prodgenie_corsika_proton_rockbox0p1_sbnd`, so cosmics are simulated (CORSIKA).
  - 1–3 events per reco1 file.
- **Wire-Cell:** aurora-r2 unit directories `production-prep/dlspace/bnb/node<k>/<unit>/`, with `reco1_files.json` and `pr/all/pr_evt<r>_<s>_<e>/tracking-pr.root`.
  - 11,888 BNB units, 627,114 events with WC output.

## What the reco1 files hold

| Product | Status | Used for |
|---|---|---|
| `gaushit` | yes | `hit_table` |
| `gaushitTruthMatch` (hit → MCParticle, BackTrackerHitMatchingData) | **empty** in every event | — |
| `simtpc2d:simpleSC` SimChannels | one per channel (11,276), **no IDEs** | — |
| `ionandscint:priorSCE` SimEnergyDeposits | yes (~12 MB/event) | per-hit truth by projection |
| `largeant` MCParticles + MCParticle→MCTruth assns + ParticleAncestryMap | yes (every particle has an MCTruth) | `particle_table`, origin ν / cosmic |
| `generator` MCTruth, BeamGateInfo | yes | `event_table` |
| `mcreco` MCShower / MCTrack | yes | HDF5Maker particle categories |
| `ophitpmt`, `opflashtpc0/1` | yes; PMT readout ±1.5 ms, ~36k ophits/event | `ophit_table` (−10…40 µs window), `opflash*` (all) |

## Per-hit truth by deposit projection

Each deposit midpoint is processed as follows:
1. Shift it by the forward SCE map (`SCEoffsets_SBND_E500_dualmap_CV_voxelTH3.root`, `TrueFwd_*`), as the Wire-Cell truth labeler does.
2. Drift it to each wire plane at 0.1563 cm/µs.
3. Map it to a channel with the linear wire-plane model from the WCT wire geometry (`sbnd-wires-geometry-v0206`). Every plane is linear to < 0.02 cm, and the channel model agrees with the `(tpc, plane)` of all hits.
4. Compute tick = 410 + 2·(t_G4 + t_drift)[µs] + dt[plane].
5. Sum the deposits per particle within the hit's PeakTime ± 1 RMS, the larsim BackTracker window.

Calibration on the first 3 BNB CV events, maximising the hits that get a deposit:
- dt = +5 (U), +2 (V), −2 (Y) ticks;
- the peaks are broad (±2 ticks within 0.2%).

| 10 BNB CV events | dt = 0 | dt = (5, 2, −2), ±1 RMS | dt = (5, 2, −2), ±2 RMS |
|---|---|---|---|
| Hits with no deposit | 45,587 (30.6%) | 3,788 (2.5%) | 3,109 (2.1%) |
| ν energy on collection hits / projected ν energy (collection) | 89.9% | 91.5% | 101.5% (double counting) |
| WC ν-cluster hit purity vs truth ν hits | 0.78 | 0.87 | 0.89 |

The default is ±1 RMS.

## Status

| Date | Step | Result |
|---|---|---|
| 2026-11-12 | Converter, detector JSON, check script | Done. Debug-queue jobs 9079466–9080493 (development). |
| 2026-11-12 | 10 BNB CV events (job 9080512) | `sbnd-bnb-cv/test10/sbnd_bnb_cv_test10.h5`: 4.7 MB, `check_h5.py`: no layout problems. Copied to Eagle; sent to `fm4npp-lartpc@polaris` for checking. |
| 2026-11-12 | Review 1 by fm4npp-lartpc_polaris | Approved for 10k once A and B are in; done in 0.3 (4077488). test10 rebuilt (job 9080764). |
| 2026-11-12 | 10k BNB CV (job 9080765, capacity, 1 node, 24 processes) | 10,754 events, 21 files, 3.9 GB, 21 min wall. Copied to Eagle; fine-tune test requested from fm4npp-lartpc_polaris. |
| 2026-11-12 | Review 2 (fine-tune test) | Format and truth fine; WC hit labels matched on only 2 of 6 (tpc, plane). Fixed in 0.4; test10 and 10k relabelled in place with `relabel_wc.py` and copied to Eagle again. |
| 2026-11-14 | 96-process pilot (job 9088015) and Polaris full-pipeline test | Both passed ([pilot](#96-process-pilot-2026-11-14); Polaris [comment 6872306589](https://github.com/HaiwangYu/fm4npp-lartpc/issues/11#issuecomment-6872306589)). |
| 2026-11-14 | Full production (job 9088268) | 922,788 events, 1,781 files, 349 GB, ~8.7 node-h. `check_h5.py` (job 9088503): no problems. Copied to Eagle 2026-11-15. See [Full production](#full-production-2026-11-14). |

### test10 summary (`check_h5.py`)

- **Events:** 10 (one aurora-r2 unit), all `wc_status` = 1. ν: 10 νμ, 7 CC, 5 with the vertex in the AV; 4 events have 2 ν MCTruths.
- **Hits:** 148,939 (2.5k–25k per event; cosmics dominate). `local_wire` up to 1972 (U/V) and 1669 (Y); ticks 1–3425 (readout 3,427 ticks).
- **Truth:** 7,534 hits with a ν deposit > 0.05 MeV and 137,587 with a cosmic deposit; 1,059 particles (472 ν, 587 cosmic).
- **Light:** 15,113 ophits in −10…40 µs (155–2,971 per event); 97 PMT channels; beam flash at 0–4 µs. 409 flashes over the full readout.
- **Wire-Cell:** 6 ν candidates in 6 events. WC ν-cluster hits have purity 0.87 against truth ν hits. Efficiency is 0.29 over all events (5 of 10 ν vertices are outside the AV).

## 10k BNB CV (2026-11-12)

`sbnd-bnb-cv/10k/sbnd_bnb_cv_10k_000..020.h5`:
- 205 aurora-r2 units, evenly spaced over the 11,872 OK units of the 1M production (`units.txt`); 10,754 events.
- Run with `pbs/run-many.pbs`: 21 parts of 10 units, 24 processes on one capacity node.
- Every part rc 0, 11–21 min, 1.5–2.1 GB RSS.
- 2.14 process-seconds per event, including the ~50 s compile per process.

| Quantity | 10k BNB CV |
|---|---|
| Events / with WC output | 10,754 / 10,754 |
| Neutrinos (`nu_table`) / event-level ν vertex in AV / CC | 19,402 / 4,807 / 8,032 |
| Hits / per event | 132.4M / 12.3k |
| Hits with ν deposit > 0.05 MeV / cosmic / none | 6.41M / 122.7M / 3.16M (2.4%) |
| Projected ν energy on collection planes landing on hits | 92.2% |
| PMT ophits in −10…40 µs / per event | 19.4M / 1.8k |
| WC ν candidates (events) | 5,155 (5,055) |
| WC ν-cluster hit purity vs truth (all / main cluster) | 0.89 / 0.93 |
| Size | 3.89 GB = 362 kB/event |

## Estimate for the full sample (made before the run)

The full sample is the aurora-r2 production: 627,205 BNB CV + 268,865 νe CC + 26,745 NC Δ-radiative = 922,815 events.

| | Basis | Full sample |
|---|---|---|
| CPU | 2.14 process-s/event | ~550 process-hours |
| Node-hours, measured packing (24 processes/node) | 0.35 node-h per 10.75k events | ~30 node-h |
| Node-hours, 64 processes/node (RSS 2 GB, 102 cores; not yet measured) | if Lustre keeps up | ~12 node-h |
| Disk on Flare | 362 kB/event (BNB; νe CC events have more ν hits) | ~0.33 TB (±30%), ~1,850 files of ~500 events |
| Copy on Eagle | same | ~0.33 TB; rsync ran at ~130 MB/s, so ~45 min |

Suggested plan: 8 capacity nodes × 64 processes, about 2–4 h wall, after one 64-process pilot node to confirm the packing.

## Review 1 (fm4npp-lartpc_polaris, 2026-11-12)

The Polaris session approved the 10k run once two changes were in.
- h5reader reads every table unchanged. A scratch SBND variant of convert.py converts all 10 events, and the batch path works.
- The ν labels are usable for T1/T2.
- The PMT map and the −10…40 µs light window are fine.

Requested changes, all done in sbnd-h5maker 0.3:
- **A (required): multiple neutrinos per spill.** Added `nu_table`, `nu_index` in `edep_table` and `particle_table`, and the event-level choice rule (attribute `nu_choice`). Event 0 of test10 now takes the in-AV neutrino (54 collection hits) instead of the out-of-cryostat one at x = 390.
- **B (required): units.** The event times are now `nu_time_ns`, `beam_gate_start_ns` and `beam_gate_width_ns`, and a `units` attribute lists the units of every column.
- **C (nice to have): simulation configuration.** The detsim and hit-finder settings are in `sbnd_detector.json` "simulation" and the `sim_config` attribute.

## 96-process pilot (2026-11-14)

Job 9088015 (debug, 1 node, 96 processes):
- Input: 96 BNB CV aurora-r2 units not in the 10k (`sbnd-bnb-cv/pilot96/units.txt`), one unit per process, `--max-events 11`, so 1,056 events.
- Result: all 96 rc 0, 4.7 min wall. `check_h5.py` finds no problems, and all events have WC output.
- WC per-(tpc, plane) efficiency: mean 0.83–0.88 over files.

| Per process | 24 processes/node (10k) | 96 processes/node (pilot) |
|---|---|---|
| Compile (cling) | 47 s | 94 s |
| First event | 11 s | 20 s |
| Later events | 1.90 s (median) | 2.50 s (median) |
| CPU busy | 72–78% | 41% (includes the start-up) |
| Max RSS | 2.1 GB (~500 events) | 1.0 GB (11 events) |
| Node throughput | 12.6 events/s | 38 events/s (×3.0) |

Each process is ~30% slower at 96, but the node does 3× the work. Memory is not a limit: 96 × 2.1 GB = 0.2 TB of the node's 1 TB.

### Revised estimate for the full 922,815 events

- **Node-hours:** steady state 922,815 × 2.50 s / 96 = 6.7 node-h. Start-up adds ~0.6 node-h (~1,850 parts of ~500 events, 115 s each). Tail effects add ~10%. Total **~8 node-h (7–10)**, about 2 h wall on 4 nodes.
- **Disk:** the pilot gave 409 kB/event (the 10k gave 362). So **~0.35 TB (±30%)** on Flare, plus the same on Eagle.
- **Not covered:** the steady phase lasted only ~3 min, and Lustre load from several nodes at once is untested. The first production job will be watched.

`check_h5.py`'s WC check flagged 3 pilot files in which WC missed the neutrino in one TPC: all three planes were low together, with high purity. The check now flags a plane only when it is far below another plane of the same TPC, which is the signature of a channel-numbering bug.

## Review 2: fine-tune test (fm4npp-lartpc_polaris, 2026-11-12)

Issue comments: Polaris findings 6840433723; Aurora fix 6833078440. From 2026-11-13, both sides post a #11 comment after each iteration.

qlfm runs on SBND (`QLFM_DETECTOR=sbnd`, fm4npp-lartpc eea8fe1). The 10k was split 80/10/10, with 6 runs of 3,000 steps on one A100.
- The MicroBooNE-pretrained model, fine-tuned, beats scratch on every task: T1 charge-weighted hit F1 0.85 vs 0.60–0.70, T5 light MSE 0.66 vs 0.97. A frozen encoder does much worse on T1 (0.49).
- No run is converged at 3,000 steps, and scratch is the furthest from it.
- The format, per-ν truth, `_ns` units and `sim_config` work as delivered.

Requested:
- **(a) Required: WC hit labels.** They matched on U in TPC 0 and Y in TPC 1 only. Cause: Wire-Cell numbers channels plane-major ([U0 U1][V0 V1][Y0 Y1]), LArSoft TPC-major. A per-(tpc, plane) channel-shift scan of the footprints against the hits gives WC = LArSoft + (0, +1984, +3968) in TPC 0 and (−3654, −1670, 0) in TPC 1.
  - Fixed in 0.4 (`wc_channel_offsets` attribute).
  - `check_h5.py` now reports WC efficiency and purity per (tpc, plane) and flags a plane below 0.2.
  - The existing files were relabelled with `relabel_wc.py`; only `wc_hit_table` changed.
- **(b) Strongly wanted: a νe CC sample**, so that T3 is a real 3-class task (BNB CV has ~30 νe CC per 10k). This waits for the owner's approval.

WC labels after the fix:

| | test10 before | test10 after | 10k after |
|---|---|---|---|
| WC ν-cluster hits: efficiency / purity vs truth ν hits (> 0.05 MeV, all events) | 0.29 / 0.87 | 0.85 / 0.91 | 0.66 / 0.90 |
| Main cluster: efficiency / purity | 0.17 / 0.88 | 0.55 / 0.93 | 0.55 / 0.93 |
| Per (tpc, plane), charge-weighted, events with a WC candidate: efficiency | 0.96, 0, 0 / 0.002, 0, 0.90 | 0.90–0.96 | 0.89–0.91 (purity 0.92–0.94) |

The 10k efficiency counts the ν hits of every event, including those where WC has no ν candidate.

Checks after the fix:
- A fresh 0.4 conversion of test10 (job 9081816, rc 0) matches the relabelled 0.3 file in every dataset.
- Polaris re-scored T1 with the fixed labels on the same 425 test events (ν vertex in the AV): WC ν clusters reach charge-weighted F1 0.912 (efficiency 0.892, purity 0.933) and per-event median 0.919. Fine-tuned qlfm reaches 0.852 / 0.836.
- So WC leads by ~0.06 on T1 after 3,000 qlfm steps on 3.6k hit-labelled events. The per-event means are close (0.79 vs 0.77) because WC has no candidate in some events.

## Full production (2026-11-14)

The run used `pbs/make_parts.py`, `pbs/run-multi.pbs` and `pbs/node-many.sh`: job 9088268, capacity queue, 4 nodes × 96 processes, 06:05–08:15 UTC. Totals: 2 h 10 min wall, ~8.7 node-h, close to the ~8 estimated. All 1,781 parts (~500 events each) exited 0, and every expected event is present.

| Sample | Events | Files | Size | kB/event | Hits/event | Hits on WC ν clusters | ν energy on collection hits |
|---|---|---|---|---|---|---|---|
| BNB CV | 627,205 | 1,191 | 228.1 GB | 364 | 12.4k | 3.5% | 92.3% |
| νe CC | 268,839 | 536 | 111.1 GB | 413 | 13.5k | 11.4% | 89.7% |
| NC Δ-radiative | 26,744 | 54 | 10.0 GB | 374 | 12.5k | 4.8% | 89.2% |
| **Total** | **922,788** | **1,781** | **349 GB** | | | | |

- **Event count:** 27 fewer than the 922,815 in the estimate. `make_parts.py` takes only units whose status is OK or PARTIAL.
- **Smoke test:** before the run, νe CC and NC Δ-rad were tested on 12 events each (job 9088218): WC efficiency 0.88, no problems.
- **Checks:** `check_h5.py` ran on every file (job 9088503, debug queue, 5 min with 64 in parallel). 0 files have problems, including the per-(tpc, plane) WC check. The JSON reports are saved next to each file as `*.h5.check.json`.
- **Aurora location:** `/lus/flare/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data/sbnd-{bnb-cv,nuecc,ncdr}/full/`. Run bookkeeping is in `full-run/` (`tasks.tsv`, `work/node*.{log,sampler}`).
- **Polaris copy:** `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data/sbnd-{bnb-cv,nuecc,ncdr}/full/`, made by rsync on 2026-11-15, 02:51–03:45 UTC. File counts and byte totals match Aurora.
- **Eagle space:** the neutrinoGPU project on Eagle has only ~14 TB of its 3 PB quota left.
- **Overlap:** `sbnd-bnb-cv/{test10,10k,pilot96}` are subsets of the BNB CV full sample, so keep evaluation events out of training.
- **Next:** fmlar_polaris was asked for a full-scale pre-train + fine-tune.
- **Issue #11 comments:** [production summary 6888730254](https://github.com/HaiwangYu/fm4npp-lartpc/issues/11#issuecomment-6888730254).

## Notes for the qlfm side

Differences from the MicroBooNE files:
- **Two TPCs:** `tpc` is 0 for x < 0 and 1 for x > 0, and the drift sign differs per TPC. Wires: 1,984 U / 1,984 V / 1,670 Y per TPC, 0.3 cm pitch. Readout 3,427 ticks of 0.5 µs; trigger at tick 410.
- **PMTs:** 120 PMTs, OpChannel 0–311, mixed with X-ARAPUCA channel numbers. There is no `hit_channel < 32` assumption to keep. Channel → (x, y, z, type) is in the `detector` attribute.
- **Cosmic truth:** `edep_table` keeps MicroBooNE semantics (ν only); the cosmic deposits are in `edep_cosmic_table`.
- **WC labels** replace the Pandora tables.
