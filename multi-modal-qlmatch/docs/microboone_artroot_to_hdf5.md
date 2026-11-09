# MicroBooNE open data: converting artroot to HDF5

Status as of 2026-11-09. This is a record of how we set up the software and the grid workflow for converting the MicroBooNE open-data artroot files to HDF5, so the work can be reproduced. The emphasis is on setting up the software and building `hdf5maker`. All paths are GPVM paths (`/exp/...`, `/pnfs/...`).

## 1. Background

### 1.1 Data and sources
- MicroBooNE public datasets page: https://microboone.fnal.gov/documents-publications/public-datasets/
- Documentation repo: https://github.com/uboone/OpenSamples (local clone: `/exp/dune/app/users/jpeterso/foundation_model_code/OpenSamples`)
- Paper: arXiv:2309.15362

The sample used is "Inclusive, NoWire": a BNB inclusive overlay (simulated BNB neutrino interactions overlaid on cosmic data). Zenodo DOI: 10.5281/zenodo.8370883.

| Format | Files | Size | Events |
|---|---|---|---|
| Published HDF5 (`BNB_All_NoWire_00..17.h5`, ~10-11 GB each) | 18 | 195 GB | 753,467 |
| Artroot | 24,436 | 6.4 TB | 1,046,139 |

The published HDF5 covers only about 72% of the artroot events. The reason is not documented.

### 1.2 Why convert ourselves
We wanted the artroot to HDF5 conversion reproduced with the same code MicroBooNE used. None of the public sources (README, website, Zenodo, hdf5maker repo, paper) describe grid submission. The grid workflow in section 5 is our own.

### 1.3 Artroot URL list
Zenodo hosts `public-artroot-bnb-all.list` (24,436 lines), one xroot URL per file, for example:

```
xroot://fndca1.fnal.gov:1095//pnfs/fnal.gov/usr/uboone/persistent/PublicAccess/prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run1_reco2_reco2/<file>.root
```

The files are readable over xrootd with no credentials. Download the list with:

```bash
curl -L -o public-artroot-bnb-all.list https://zenodo.org/records/8370883/files/public-artroot-bnb-all.list
```

### 1.4 The converter: HDF5Maker
The converter is the `HDF5Maker` art module in https://github.com/uboone/hdf5maker, branch `opensamples`. This is the code MicroBooNE used to make the public HDF5 (adapted from numl). It writes HDF5 using `hep_hpc`.

Job configurations (FHiCL):
- `hdf5maker_uB_public_job.fcl` (WithWire). Includes `reco_uboone_data_mcc9_8.fcl` and `HDF5Maker.fcl`; `process_name` is `trueenergy`; sets the space-charge services and `DetectorClocksService.TrigModuleName "triggersim"`.
- `hdf5maker_uB_public-nowire_job.fcl` (the one we use). It is just:

```
#include "hdf5maker_uB_public_job.fcl"
physics.analyzers.hdf5maker.WireLabel: ""
physics.analyzers.hdf5maker.OutputName: "NeutrinoML_Open_NoWire"
```

Labels in `HDF5Maker.fcl`: TruthLabel `generator`, HitLabel `gaushit`, HitTruthLabel `gaushitTruthMatch`, EventInfo `nu`, WireLabel `butcher`, PandoraLabel `pandora`, OpHitLabel `ophitBeam`, OpFlashLabel `simpleFlashBeam`, T0Label `flashmatch`, UseMap `true`, FilesBySubrun `false`.

With `FilesBySubrun false`, each `lar` job writes ONE file named `<OutputName>_ts<microseconds>.h5` in the current directory, plus a `reco_hist.root` from TFileService.

Output tables (`event_table`, `hit_table`, `edep_table`, `particle_table`, `opflash_table`, ...) are documented in `OpenSamples/file-content-hdf5.md`. The `event_id.seq_cnt` datasets were added in post-processing by ph5concat (section 6).

## 2. Software setup and building hdf5maker

### 2.1 Concepts for non-Fermilab readers
- **UPS** is Fermilab's package manager. It provides the `setup` command, which puts a versioned package and its dependencies on your environment.
- **uboonecode** is the MicroBooNE software release, built on **LArSoft** (the shared LAr TPC framework), which uses the **art** event-processing framework. The `lar` executable runs art jobs.
- **mrb** ("multi-repository build") is the tool for building local packages against a UPS release.
- **jobsub_lite** submits jobs to FermiGrid (section 5).

### 2.2 Release
- uboonecode `v08_00_00_54`, qualifiers `e17:prof` (e17 = GCC 7.3; prof = optimized with debug symbols).
- It pulls in LArSoft `v08_05_00_17`, art, ROOT, hep_hpc, etc.
- It is built for SL7. The DUNE GPVM is AlmaLinux 9, so everything must run inside an SL7 Apptainer container.

### 2.3 Enter the SL7 container
From a fresh GPVM login:

```bash
/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer shell --shell=/bin/bash \
  -B /cvmfs,/exp,/nashome,/pnfs/dune,/opt,/run/user,/etc/hostname,/etc/hosts,/etc/krb5.conf \
  --ipc --pid /cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest
```

### 2.4 Build recipe
This follows the OpenSamples README. Run it inside the container. Our development area is `/exp/dune/app/users/jpeterso/foundation_model_code/uboone_open_data_code` (the README used the name `ubtest`).

```bash
cd /exp/dune/app/users/jpeterso/foundation_model_code
mkdir uboone_open_data_code && cd uboone_open_data_code
source /cvmfs/uboone.opensciencegrid.org/products/setup_uboone_mcc9.sh
setup uboonecode v08_00_00_54 -q e17:prof
mrb newDev
source localProducts_larsoft_v08_05_00_17_e17_prof/setup
cd srcs
git clone -b opensamples https://github.com/uboone/hdf5maker.git
mrb updateDepsCM
mrbsetenv
mrb i
```

What each step does:

| Step | Purpose |
|---|---|
| `source .../setup_uboone_mcc9.sh` | Initializes UPS and adds the MicroBooNE, LArSoft and Fermilab-common CVMFS product areas to `$PRODUCTS`. Must be sourced. |
| `setup uboonecode ...` | Loads the release and all dependencies by setting `PATH`, `LD_LIBRARY_PATH`, `FHICL_FILE_PATH`, `CET_PLUGIN_PATH`, etc. This is how `lar` and `reco_uboone_data_mcc9_8.fcl` are found. |
| `mrb newDev` | Creates the mrb development area for the LArSoft version currently set up: `srcs/`, `build_slf7.x86_64/`, and `localProducts_larsoft_v08_05_00_17_e17_prof/` (with a `setup` script). The directory name comes from the LArSoft version and qualifiers. |
| `source localProducts.../setup` | Sets the `MRB_*` variables (`MRB_TOP`, `MRB_SOURCE`, `MRB_BUILDDIR`, `MRB_INSTALL`) and prepends the local products directory to `$PRODUCTS`, so locally built products take priority. Requires UPS and mrb to be set up already. |
| `git clone -b opensamples` | Gets the module, the job `.fcl` files, and `ups/product_deps` (declares `hdf5maker v09_25_00`, which depends on `hep_hpc v0_11_00`). |
| `mrb updateDepsCM` | Regenerates `srcs/CMakeLists.txt` to include all packages in `srcs/`. Needed because we used a plain `git clone` instead of `mrb gitCheckout`. |
| `mrbsetenv` | Sets up the build environment (reads each package's `product_deps` and sets up dependencies such as `hep_hpc` and `cetbuildtools`). |
| `mrb i` | `mrb install`: runs cmake and make in `build_slf7.x86_64/` and installs into `localProducts_...`. Add `-j N` for a parallel build. |

### 2.5 What the build produces
```
uboone_open_data_code/
  srcs/hdf5maker/                                  # git clone
  build_slf7.x86_64/                               # ~40 MB, only needed to compile
  localProducts_larsoft_v08_05_00_17_e17_prof/     # ~14 MB
    hdf5maker/v09_25_00/
      slf7.x86_64.e17.prof/lib/libhdf5maker_HDF5Maker_HDF5Maker_module.so
      fcl/   HDF5Maker.fcl, hdf5maker_uB_public_job.fcl, hdf5maker_uB_public-nowire_job.fcl
      ups/hdf5maker.table
```
The UPS product is declared with flavor `Linux64bit+5.14-2.17` and qualifiers `e17:prof`.

### 2.6 Caveat: localProducts is not relocatable
The `localProducts_.../setup` script hardcodes `MRB_TOP` and related paths as `/exp/dune/app/users/jpeterso/foundation_model_code/uboone_open_data_code`. The UPS database itself is relocatable (its `PROD_DIR` is relative). Grid jobs therefore copy `localProducts` to scratch and `sed`-replace that path (section 5.3).

### 2.7 Day-to-day environment: setup_env.sh
`setup_env.sh` lives in the build area. It must be sourced, not executed, and must be used inside the SL7 container.

```bash
# installed hdf5maker, for running lar
source /exp/dune/app/users/jpeterso/foundation_model_code/uboone_open_data_code/setup_env.sh

# build environment, for editing HDF5Maker_module.cc and re-running mrb i
source /exp/dune/app/users/jpeterso/foundation_model_code/uboone_open_data_code/setup_env.sh --dev
```

What it does:
1. Checks it is running in SL7 (otherwise prints the container command from 2.3) and that `/cvmfs` is available.
2. Sources `setup_uboone_mcc9.sh` and sets up uboonecode.
3. Sources the `localProducts` setup.
4. Sources `$MRB_DIR/bin/setup_local_products` (the `mrbslp` alias), or with `--dev` `$MRB_DIR/bin/mrbSetEnv` (the `mrbsetenv` alias).
5. Checks that `lar` is on `PATH`.

The mapping of those two files to the `mrbslp` / `mrbsetenv` aliases was not independently verified.

### 2.8 Rebuilding after editing the module
Inside the `--dev` environment:

```bash
mrb i
```

If the build gets confused:

```bash
mrb z          # clean the build directory
mrbsetenv
mrb i
```

## 3. Running interactively
Once installed, this works from any directory because the fcl is on `FHICL_FILE_PATH`. Source `setup_env.sh` (no `--dev`) in the container first.

Ten events from the first file in the list:

```bash
lar -c hdf5maker_uB_public-nowire_job.fcl -n 10 -s "$(head -1 public-artroot-bnb-all.list)"
```

All events from a list of files:

```bash
lar -c hdf5maker_uB_public-nowire_job.fcl -n -1 -S my_list.list
```

Output: `NeutrinoML_Open_NoWire_ts<microseconds>.h5` and `reco_hist.root` in the current directory.

## 4. Chunking the URL list
To parallelize, split the 24,436 URLs into chunks of 20 with numeric names (on the GPVM):

```bash
cd /pnfs/dune/scratch/users/jpeterso/process_artroot_to_hdf5
mkdir -p xrootd_url_chunks_20
split -l 20 -d -a 4 public-artroot-bnb-all.list xrootd_url_chunks_20/chunk_
```

This gives 1,222 chunks, `chunk_0000` to `chunk_1221` (the last has 16 URLs). `chunk_NNNN` holds lines `20*N+1` to `20*N+20` of the list. Earlier 100-URL chunks in `xrootd_url_chunks/` (names `aa`..`jk`) are superseded.

## 5. Grid processing (FermiGrid via jobsub_lite)

### 5.1 Scripts
The source is kept in `/exp/dune/app/users/jpeterso/foundation_model_code/uboone_open_data_grid`. Copies MUST be placed in `/pnfs/dune/scratch/users/jpeterso/process_artroot_to_hdf5`, because the job executable has to be there; the submitter always submits the `/pnfs` copy.

### 5.2 submit_hdf5_jobs.sh
Run on the GPVM host (not in the container) after `kinit`. Use `bash` explicitly, since `/pnfs` may not keep exec bits.

It calls `jobsub_submit` with:
- `-G dune` (no `--role`)
- `--singularity-image=/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-sl7:latest`
- `--tar-file-name tardir:///exp/dune/app/users/jpeterso/foundation_model_code/uboone_open_data_code`
- `-e CHUNK_OFFSET`, `CHUNK_DIR`, `OUTPUT_DIR`, `OUTPUT_PREFIX`
- `--memory 2000MB`, `--disk 10GB`, `--expected-lifetime 2h` (default), `--maxConcurrent 200`
- usage model `DEDICATED,OPPORTUNISTIC`

It verifies that chunk names are contiguous, skips chunks whose output `.h5` already exists, and groups the remaining consecutive chunk numbers into separate submissions, each with its own `CHUNK_OFFSET`. Rerunning therefore resubmits only missing outputs; check `jobsub_q` first so you do not duplicate jobs still in the queue.

Options: `--test` (chunk_0000 only), `--range FIRST LAST`, `--lifetime`, `--memory`, `--disk`, `--max-concurrent`, `--dry-run`.

```bash
cd /pnfs/dune/scratch/users/jpeterso/process_artroot_to_hdf5
kinit
bash submit_hdf5_jobs.sh --dry-run --range 2 11
bash submit_hdf5_jobs.sh --test
```

### 5.3 run_hdf5_job.sh (the worker)
The chunk processed is `chunk_$(printf %04d $((CHUNK_OFFSET + JOBSUBJOBSECTION)))` (falls back to `PROCESS`). Steps:
1. Copy the chunk file with a token-aware ifdh (jobsub's `${JSB_TMP}/ifdh.sh`, else Spack `ifdh_env current`).
2. In a subshell:
   - set up uboonecode;
   - copy `localProducts` from the tardir tarball (`$INPUT_TAR_DIR_LOCAL`, read-only) to scratch and `sed` the `/exp` path;
   - source it, then `setup hdf5maker v09_25_00 -q e17:prof`. Fallback if the UPS flavor does not match the worker: set `LD_LIBRARY_PATH`, `CET_PLUGIN_PATH`, `FHICL_FILE_PATH` manually and `setup hep_hpc`. The fallback is untested;
   - `xrdcp` each URL to local disk (3 attempts, 30 s then 60 s waits; failures are listed in `failed_files_chunk_NNNN.txt` and skipped);
   - `lar -c hdf5maker_uB_public-nowire_job.fcl -n -1 -S local_inputs.list`;
   - delete the local inputs.
3. Rename the output to `NeutrinoML_Open_NoWire_chunk_NNNN.h5` and `ifdh cp` it to `/pnfs/dune/scratch/users/jpeterso/process_artroot_to_hdf5/hdf5_files/` (skipped-file lists go to `hdf5_files/failed_files/`).

The job exits non-zero with an `ERROR:` line on any failure.

### 5.4 Monitoring and tokens
```bash
jobsub_q -G dune
jobsub_fetchlog -G dune --jobid=<id>
```

jobsub uses JWT bearer tokens obtained via `kinit` plus htgettoken/Vault (first use prompts for CILogon). The job token had `storage.create:/scratch/users/jpeterso`, which is sufficient to write outputs.

## 6. Merging (not done yet)
The published 18 files were made by merging per-job outputs with ph5concat (https://github.com/NU-CUCIS/ph5concat), which also adds the `event_id.seq_cnt` keys:

```bash
mpiexec -n <np> ph5_concat -i list.txt -o out.h5
```

The exact flags MicroBooNE used are unknown (contact microboone_info@fnal.gov). Merging our outputs has NOT been done yet.

## 7. Problems found and how they were fixed

1. **Streaming inputs over xrootd failed and was I/O-bound.** The first test (a 100-URL chunk) failed on the 3rd file with `Fatal Root Error: TNetXNGFile::ReadBuffer [FATAL] Redirect limit has been reached`. CPU time was 15.5 s against 4383 s real for 32 events (one event took 30 min). HDF5Maker reads data products lazily, each a round trip through the `fndca1:1095` redirector. A whole-file `xrdcp` of the same files from the GPVM took a few seconds each, so the files were fine. Fix: copy inputs locally with `xrdcp` first, then run `lar`.
2. **"Notice: Unable to find valid grid or kerberos credentials".** This comes from the old ifdhc bundled with uboonecode v08, which predates tokens. It is harmless for reads. Fix: do all `/pnfs` copies with a current token-aware ifdh and keep the uboonecode environment in a subshell.
3. **`$PROCESS` is 0 in every job under `--maxConcurrent`.** jobsub_lite turns a `--maxConcurrent` submission into a DAG (`maxconcurrent.dag`, `DAGMAN_MAX_JOBS_SUBMITTED=200`) where each node is its own single-job cluster. In the first full run all 1,221 jobs processed `chunk_0001`; one wrote the output and the other 1,220 failed at copy-out with `gfal-copy error: 17 (File exists) ... overwrite is not set`. Fix: use `JOBSUBJOBSECTION` (set per DAG node, and equal to `$(Process)` in plain submissions). The surviving `NeutrinoML_Open_NoWire_chunk_0001.h5` was verified as correct: its job opened exactly lines 21-40 of the list in order, and the size matches (71,201,731 bytes).
4. **dCache will not overwrite existing files with the default token scope.** Delete any 0-byte or partial outputs before resubmitting.

## 8. Measured performance
For a 20-URL chunk (`chunk_0001`):

| Item | Value |
|---|---|
| Copy 20 inputs (1.2 GB) | 20-60 s |
| `lar` | 192 events, CPU 38.6 s vs real 39.3 s (~0.2 s/event) |
| Whole job | ~2-3 min |
| Peak RSS | ~630 MB |
| Output | 71 MB |

Events per artroot file vary (this chunk has ~10 per file against ~43 on average), so job lengths vary.

## 9. Status and next steps

Done:
- Build area, `setup_env.sh`, 20-URL chunks, grid scripts.
- Outputs exist for `chunk_0000` (test) and `chunk_0001`.
- The `JOBSUBJOBSECTION` fix is implemented and verified by local simulation, but not yet run on the grid.

Next:
1. Copy the updated scripts to `/pnfs/dune/scratch/users/jpeterso/process_artroot_to_hdf5`.
2. Run a small test and confirm 10 distinct chunks in the logs:

   ```bash
   bash submit_hdf5_jobs.sh --range 2 11 --max-concurrent 5
   grep -h '^Chunk:' *.out | sort | uniq -c
   ```
3. Run the rest:

   ```bash
   bash submit_hdf5_jobs.sh --lifetime 1h
   ```
4. Rerun the submitter to pick up failures (check `jobsub_q` first).
5. Optionally merge with ph5concat.

Open questions:
- Why the published HDF5 has fewer events than the artroot.
- Which ph5concat flags MicroBooNE used.
- The UPS flavor fallback in the worker is untested.

## 10. License and citation
The data are released under CC-BY 4.0. Suggested acknowledgement:

> We acknowledge the MicroBooNE Collaboration for making publicly available the data sets [data set DOIs] employed in this work. These data sets consist of simulated neutrino interactions from the Booster Neutrino Beamline overlaid on top of cosmic data collected with the MicroBooNE detector [2017 JINST 12 P02017].
