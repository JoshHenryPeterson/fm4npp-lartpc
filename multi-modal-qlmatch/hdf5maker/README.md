# HDF5Maker (uboone/hdf5maker, branch opensamples) built in the uboone area, with a data mode

Build area: `/exp/uboone/app/users/yuhw/hdf5maker_build` (uboonecode `v08_00_00_54 e17:prof`,
larsoft `v08_05_00_17`, built in the SL7 container on a uboone GPVM, 2026-11-11). Josh's
build recipe (`microboone_artroot_to_hdf5.md` in his fork) is the reference.

| File | Purpose |
|---|---|
| `sl7.sh` | runs a command (or a shell, with no args) in the SL7 dev container with `/cvmfs,/exp,/nashome,/pnfs/uboone` bound |
| `build.sh` | `mrb newDev`, clone hdf5maker, apply `patches/*.patch` on a fresh clone, `mrb i` |
| `setup_env.sh` | source inside the container: uboonecode + local hdf5maker + `fcl/` on `FHICL_FILE_PATH` |
| `patches/0001-data-mode.patch` | new FHiCL flag `IsData` (default false). When true, skips the backtracker, `mcreco`, ParticleInventory, and the `edep_table`/`particle_table` |
| `fcl/hdf5maker_uB_data-nowire_job.fcl` | real data: `IsData: true`, `EventInfo: none`, NoWire, `TrigModuleName: daq`, light pinned to `*::DataStage1Optical` |

```bash
bash sl7.sh "bash $PWD/build.sh"                                  # build (once)
bash sl7.sh "source $PWD/setup_env.sh; lar -c hdf5maker_uB_data-nowire_job.fcl -s <data reco2>.root"
bash sl7.sh "source $PWD/setup_env.sh; lar -c hdf5maker_uB_public-nowire_job.fcl -s <overlay reco2>.root"
```

Why a FHiCL flag rather than `e.isRealData()`: MicroBooNE **overlay events report
`isRealData() == true`**, because the event auxiliary comes from the overlaid cosmic data. Guarding on it
silently drops all MC truth (seen in the first test).

Tests (2026-11-11), with outputs in the session scratch:
- **Beam-on C1 reco2 (run 5364, 20 events):** rc 0. 15.6k hits/event, 1.1 flashes/event,
  371 ophits/event, 30 Pandora slices/event. `event_table` has `event_id` only; edep and particle
  tables are empty.
- **Public BNB overlay file (10 events), standard public NoWire job:** rc 0. 15.4k hits/event,
  10,943 edep rows, 877 particles. Same table schema as the public HDF5.
- Not yet done: a byte-level comparison of our overlay output with the published Zenodo HDF5 for
  the same events.
