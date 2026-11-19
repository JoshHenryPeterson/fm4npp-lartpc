# Make the fork runnable from jpeterso's directory

> **Part I is the plan as written before implementation (kept unchanged).**
> Part II, at the end, records what was actually done.

# Part I: Plan

Fork: `/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc`
Upstream: `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc`
Audit date: 2026-11-10. Nothing has been changed yet.

## 1. What the audit found

**Yes, the repo hardcodes paths, and as-is the scripts would still run yuhw's checkout and write into yuhw's
space.** Every hardcoded path has the same prefix, `/lus/eagle/projects/neutrinoGPU/yuhw/`, and there are
three distinct roots:

| Root | Purpose | Our access (we are in group `neutrinoGPU`) | Action |
|---|---|---|---|
| `.../yuhw/fm4npp-lartpc/multi-modal-qlmatch` (`QLFM_ROOT`) | code checkout | read | **Must change.** Otherwise the scripts run yuhw's code, not ours. |
| `.../yuhw/fm4npp-lartpc-data` (`QLFM_DATA`) | raw HDF5 input (`bnb_nowire/`, `nue_nowire/`) | read | Can keep (read-only). |
| `.../yuhw/fm4npp-lartpc-work` (`QLFM_WORK`, `W`) | `cache/` converted data, `pre/` checkpoints, `tools/` (pimm checkout and venvs), job inbox `jobs/box`, all run outputs (`r2p/`, `r4/`, `paper/`, ...) | read, **no write** | **Must change** for anything we write. We can keep reading `cache/`, `pre/` and `tools/`. |

Other values tied to the environment:
- **`#PBS -A neutrinoGPU::WireCellDev`** appears in all 15 `scripts/pbs/*.pbs` files. **Decision: change
  it to `#PBS -A LArTPC_FM`.**
  - `neutrinoGPU` is the project and `WireCellDev` is a restricted suballocation of it, in the form
    `project::subname`.
  - `LArTPC_FM` (allocation 17991) has a single suballocation with no subname, so the plain project name is
    enough. It has 5,000 node-hours, valid until 2027-01-10.
  - PBS directives cannot use shell variables, so every `.pbs` file needs the literal. `qsub -A ...` on the
    command line still overrides it.
- `module load conda/2025-09-25` and `/soft/...` python, `TMPDIR=/tmp`, `filesystems=home:eagle`,
  `system=polaris` and `-q debug` are all system-wide. **Keep as is.**

### Files affected (39 code files; the 8 doc/tex/csv files are historical records, so leave them)

**Shell / PBS** (`multi-modal-qlmatch/scripts/`)
- `env.sh:5-7` is the central definition of `QLFM_ROOT`, `QLFM_DATA` and `QLFM_WORK`. It is the only real
  central setting, but other files source it **by absolute yuhw path**.
- These source yuhw's env.sh by absolute path: `run_queue.sh:10`, `convert_node.sh:3`, `box_run.sh:6`,
  and `pbs/{smoke,smoke_r2,smoke_r3,gates,r1_demo,pandora_feat,pandora_refs,infer_hits,pretrain,pretrain_multi}.pbs`.
- These have duplicated literals and bypass env.sh:
  - `feeder.sh:17` (`cd .../work/jobs`)
  - `convert_tasks.sh:3-4` (`D=` data, `W=` cache output)
  - `box_run.sh:3` (`BOX=`)
  - `node_agent.sh:5` (`BOX=` default)
  - `pbs/convert.pbs:9-10`
  - `pbs/hold.pbs:10-11`
  - `pbs/queue.pbs:13`
  - `pbs/pimm_pretrain.pbs:9,11` (`QLFM_WORK=`, `R=`)

**Python**
- These read the env var `QLFM_WORK` with a yuhw fallback (already OK once `QLFM_WORK` is set):
  `qlfm/paper_metrics.py:26`, `scripts/pandora_refs.py:23`, `scripts/r4_light_stats.py:26`, `scripts/r4_posthoc.py:29`.
- These read `PIMM_ROOT` with a yuhw fallback: `qlfm/ref/pimm_backbone.py:20-22`, `qlfm/ref/pimm_train.py:10`.
- These set `W = "/lus/.../yuhw/fm4npp-lartpc-work"` as a **hard constant** that cannot be overridden:
  - `scripts/make_tasks_r2.py:12` (checkpoint paths in `BACKBONES`)
  - `scripts/r1_io_figures.py:28`
  - `scripts/r1_mae_demo.py:22`
  - `scripts/r2_event_displays.py:25`
  - `scripts/r2_select_events.py:18` (also reads yuhw run results `r2p/...` and `round2/...`)
  - `scripts/r4_event_displays.py:26`
  - `scripts/r4_t5_handscan.py:32`
- **pimm configs** (hard constants, and the comment says "pimm configs cannot import modules"):
  - `qlfm/ref/pimm_configs/{panda_uboone,panda_uboone_f010,panda_l_uboone,polarmae_uboone}.py`
  - In each, the `_base_ = [".../yuhw/.../tools/pimm/configs/_base_/default_runtime.py"]` and `W = "..."` lines
    feed `corpus=` and `splits=`.
- The only Python *write* into W is `qlfm/paper_metrics.py:62`, which creates `$W/paper/truth.npz` if it is
  missing. That write would fail in yuhw's space.
- Already fine: the `__file__`-relative `sys.path` inserts (`r4_event_displays.py:21`, `r4_t5_handscan.py:25`,
  `summarize_r4_t5.py:22`), `qlfm/dist.py`, and all other output dirs (they are CLI arguments).

## 2. Strategy

Keep **one write root** that belongs to us. Reach yuhw's large read-only assets **through symlinks inside
it**, so that every script needs only one variable (`QLFM_WORK`) and keeps working unchanged.

```
/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/      <- new QLFM_WORK (ours, writable)
    cache -> /lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/cache   (read-only symlink)
    pre   -> .../yuhw/fm4npp-lartpc-work/pre                                  (read-only symlink)
    tools -> .../yuhw/fm4npp-lartpc-work/tools   (pimm checkout + venvs; venvs have absolute shebangs, so symlink, don't copy)
    jobs/box/  logs/  paper/  r2p/  r4/ ...                                   (ours)
```

Add a separate `QLFM_REF_WORK` (yuhw's work dir) for the few scripts that analyse **yuhw's published runs**
(`r2_select_events.py`, `r4_posthoc.py`, `r4_event_displays.py`, `paper_metrics.py`'s
`r4/light/light_stats.npz`). These scripts can then point at either yuhw's results or ours.

Later, if we want independence from yuhw, we replace the symlinks with real data. Rerun `pbs/convert.pbs` to
rebuild `cache/`, and rebuild the venvs under `tools/`. No code changes would be needed.

## 3. Steps

### Step 0: Branch and set up the work dir
- [ ] `git checkout -b portable-paths`
- [ ] `mkdir -p /lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/{jobs/box,logs}`, then create the
      `cache`, `pre` and `tools` symlinks shown above.
- [ ] Optionally symlink `paper/truth.npz` from yuhw's work dir, to avoid regenerating it.
- [ ] Check the `LArTPC_FM` balance: `sbank list allocations -p LArTPC_FM`.

### Step 1: `scripts/env.sh` becomes the single source of truth
```bash
export QLFM_ROOT=${QLFM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export QLFM_DATA=${QLFM_DATA:-/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data}       # read-only
export QLFM_WORK=${QLFM_WORK:-/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work}     # ours
export QLFM_REF_WORK=${QLFM_REF_WORK:-/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work}  # yuhw runs, read-only
export PIMM_ROOT=${PIMM_ROOT:-$QLFM_WORK/tools/pimm}
export BOX=${BOX:-$QLFM_WORK/jobs/box}
export QLFM_PBS_ACCOUNT=${QLFM_PBS_ACCOUNT:-LArTPC_FM}   # used by qsub calls in feeder.sh
```
The `${VAR:-default}` form means a caller (`qsub -v`, the shell) can still override any value.

### Step 2: Shell scripts locate env.sh relative to themselves
- [ ] Replace the absolute `source .../env.sh` in `run_queue.sh` and `convert_node.sh` with
      `source "$(dirname "${BASH_SOURCE[0]}")/env.sh"`.
- [ ] Make `feeder.sh`, `convert_tasks.sh` and `node_agent.sh` source env.sh the same way. Replace their
      literals with `$QLFM_WORK/jobs`, `$QLFM_DATA`, `$QLFM_WORK/cache` and `$BOX`.
- [ ] `box_run.sh` echoes a `source .../env.sh` line into the inbox. Change it to echo
      `source $QLFM_ROOT/scripts/env.sh`, expanded at write time, and use `$BOX`.
- [ ] `feeder.sh`: add `-A $QLFM_PBS_ACCOUNT` to its `qsub` call. This keeps `LArTPC_FM` as the charged
      account even if a `.pbs` file is missed. The command-line `-A` overrides the directive.
- [ ] `feeder.sh:10`: update the comment "Account: the #PBS -A line ... (neutrinoGPU::WireCellDev)" to say
      `$QLFM_PBS_ACCOUNT` (default `LArTPC_FM`).

### Step 3: PBS scripts
PBS copies the job script to a spool dir, so `BASH_SOURCE` won't point into the repo. Use this pattern:
```bash
#PBS -A LArTPC_FM            # was neutrinoGPU::WireCellDev
QLFM_ROOT=${QLFM_ROOT:-/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc/multi-modal-qlmatch}
source $QLFM_ROOT/scripts/env.sh
cd $QLFM_ROOT
```
- [ ] In all 15 `.pbs` files, replace `#PBS -A neutrinoGPU::WireCellDev` with `#PBS -A LArTPC_FM`:
      `sed -i 's/^#PBS -A neutrinoGPU::WireCellDev$/#PBS -A LArTPC_FM/' multi-modal-qlmatch/scripts/pbs/*.pbs`
      Then check with `grep -L '^#PBS -A LArTPC_FM$' multi-modal-qlmatch/scripts/pbs/*.pbs`, which should print nothing.

The `QLFM_ROOT` line leaves one literal per file, and `qsub -v QLFM_ROOT=...` overrides it. Alternative:
`${PBS_O_WORKDIR}` if we always qsub from the repo root.
- [ ] Apply this to all 15 `.pbs` files.
- [ ] `convert.pbs`, `hold.pbs`, `queue.pbs`, `pimm_pretrain.pbs`: also drop their `R=`, `QLFM_WORK=` and
      `BOX=` literals and the absolute script paths. Use `$QLFM_ROOT/scripts/...`, `$QLFM_WORK` and `$BOX`.
- [ ] Optional: add `#PBS -o`/`-e`, or have feeder `cd $QLFM_WORK/logs` before qsub, so logs don't land in
      the cwd.

### Step 4: Python, with one shared helper
- [ ] Add `qlfm/paths.py`:
  ```python
  import os
  _YUHW = "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work"
  WORK     = os.environ.get("QLFM_WORK", _YUHW)          # read+write root (cache/, pre/, outputs)
  REF_WORK = os.environ.get("QLFM_REF_WORK", _YUHW)      # yuhw's published runs, read-only
  PIMM_ROOT = os.environ.get("PIMM_ROOT", os.path.join(WORK, "tools/pimm"))
  ```
- [ ] In the 7 scripts with hard `W = "..."`, use `from qlfm.paths import WORK as W`. Add the existing
      `sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))` idiom where `qlfm` isn't
      importable (normally `PYTHONPATH=$QLFM_ROOT` from env.sh covers it).
- [ ] In the 3 scripts with `os.environ.get("QLFM_WORK", ...)`, switch to the helper for consistency.
- [ ] Where a script reads **yuhw run results**, use `REF_WORK` or add a `--runs-root` CLI flag
      defaulting to `REF_WORK`:
  - `r2_select_events.py:24,75`
  - `r4_posthoc.py:40,148-152`
  - `r4_event_displays.py:38`
  - `paper_metrics.py:37` (light_stats)
- [ ] `paper_metrics.py:27,62`: read `truth.npz` from `WORK`, and write it there if missing. With
      `QLFM_WORK` set to our dir, this becomes our space automatically.
- [ ] `pimm_backbone.py`, `pimm_train.py`: import `PIMM_ROOT` from `qlfm.paths`, or just keep the env
      lookup and change the fallback.

### Step 5: pimm configs (needs a check)
The configs can't use the helper: per the in-file comment, pimm's config loader doesn't allow imports.
Options, in order of preference:
1. **Test whether `import os` works** in a pimm config (Pointcept-style loaders often allow it but strip
   module objects). If it works: `W = os.environ.get("QLFM_WORK", "...")` and
   `_base_ = [os.environ.get("PIMM_ROOT", W + "/tools/pimm") + "/configs/_base_/default_runtime.py"]`.
2. Otherwise, override at launch through `pimm_train.py --options data.train.corpus=... data.train.splits=...`.
   Have `pimm_train.py` inject these from `qlfm.paths` automatically. `_base_` still needs to resolve, so
   set it relative to the config file (e.g. `../../../../...`), or point it at our `tools` symlink.
3. Fallback: a plain sed to our `QLFM_WORK` path. Since `cache`/`tools` are symlinked there, one literal per
   file is enough.

### Step 6: Verify
- [ ] `grep -rIn -E 'yuhw|neutrinoGPU' multi-modal-qlmatch --include='*.py' --include='*.sh' --include='*.pbs'`
      should hit only `env.sh`, `qlfm/paths.py` (the read-only defaults) and possibly the pimm configs.
- [ ] `source scripts/env.sh && env | grep -E 'QLFM|PIMM|BOX'` should show our paths.
- [ ] Run `python -c "import qlfm.paths as p; print(p.WORK, p.REF_WORK, p.PIMM_ROOT)"`.
- [ ] Dry-run a cheap script that only reads, e.g. `scripts/r4_light_stats.py` into a temp OUT under our work dir.
- [ ] Submit `pbs/smoke.pbs` to `debug`. Check that:
  - it runs our checkout (`cd` and `PYTHONPATH`) and writes only under our `QLFM_WORK`;
  - `qstat -f <jobid> | grep Account_Name` shows `LArTPC_FM`;
  - afterwards, `sbank list jobs -p LArTPC_FM` shows the charge.
- [ ] Short `pimm_pretrain.pbs` smoke run to confirm the config `_base_`, corpus and splits resolve.

### Step 7: Upstream-friendliness (optional)
Every default stays overridable by env var, and yuhw's paths remain the read-only fallbacks. So the change
can be offered back upstream as a portability PR, and future upstream merges conflict only in `env.sh`
defaults and PBS `-A` lines.

## 4. Open questions / risks
- **Allocation budget and expiry:** `LArTPC_FM` has 5,000 node-hours and expires 2027-01-10. That is much
  sooner than `neutrinoGPU::WireCellDev`, which runs to 2027-07-01. The upstream paper campaign was budgeted
  around 2,000 node-hours (`docs/paper-campaign.md`). Plan large pretraining runs to fit before the expiry,
  and watch the balance with `sbank`.
- **Upstream merges:** future merges from yuhw will reintroduce `neutrinoGPU::WireCellDev` in `.pbs` files
  they touch. Re-run the sed/grep check from Step 3 after each merge.
- **Relying on yuhw's group-readable dirs:** if permissions or contents change upstream, the symlinked
  `cache/`, `pre/` and `tools/` break. Copy them, or regenerate with `convert.pbs` and a venv rebuild, to be independent.
- **Shared venvs in `tools/`:** we can't `pip install` into them. Build our own if we need extra packages.
- **Untraced dependencies:** some PBS/queue flows read task/plan files (`plan_*.txt`, `tasks/`) from yuhw's
  work dir. Those were not traced. Inspect each one before first use, and copy any it needs into our `QLFM_WORK`.

---

# Part II: What was actually done (2026-11-10)

The work was split between two Sonnet subagents: one for the shell/PBS scripts (Steps 1-3), one for the
Python files and pimm configs (Steps 4-5). The main session did Step 0, reviewed both sets of changes and
ran the Step 6 checks. **Nothing has been committed, and no PBS job has been submitted.**

## Differences from the plan
- **No branch.** The `git checkout -b portable-paths` step was skipped at your request. All changes are
  uncommitted in the working tree on `main` (39 modified files plus the new `qlfm/paths.py`).
- **`pre/` and `test/` are real directories, not symlinks.** Pretraining writes new checkpoints into
  `$QLFM_WORK/pre/<name>`, and `smoke.pbs` writes into `$QLFM_WORK/test`. A whole-directory read-only
  symlink would make those writes fail, so:
  - `pre/` holds one symlink per checkpoint in yuhw's `pre/`;
  - `test/` holds symlinks to yuhw's `cache_nue18` and `splits_test.npz`.
  `cache` and `tools` remain whole-directory symlinks.
- **pimm configs use `import os as __os`.** pimm's `Config._file2dict` really imports the config file and
  keeps every top-level name that does not start with `__`. A plain `import os` would leave a module object
  in the config, so the double-underscore name is used to make the loader drop it.
- **No `--runs-root` flag was added.** None of the scripts that read yuhw's run results uses argparse, so
  they read from `QLFM_REF_WORK` (set in `env.sh` or the environment) instead.
- **One more yuhw input was found after the plan was written:** `r4/light/light_stats.npz` and
  `light_cut.json`. `r4_posthoc.py` and `paper_metrics.py` now read them from `REF_WORK`.

## Step 0: work dir
`/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/` contains:
- `cache` and `tools`: symlinks to yuhw's work dir;
- `pre/` and `test/`: real directories with per-entry symlinks (see above);
- `paper/truth.npz`: a symlink to yuhw's copy;
- `jobs/box/` and `logs/`: empty directories.

## Steps 1-3: shell and PBS (`multi-modal-qlmatch/scripts/`)
- **`env.sh`:** every value now uses `${VAR:-default}`, so it can be overridden.
  - `QLFM_ROOT` is found from the script's own location.
  - `QLFM_DATA` points at yuhw's data dir (read-only).
  - `QLFM_WORK` points at the new work dir above.
  - `QLFM_REF_WORK` points at yuhw's work dir (read-only).
  - `PIMM_ROOT=$QLFM_WORK/tools/pimm`, `BOX=$QLFM_WORK/jobs/box`, `QLFM_PBS_ACCOUNT=LArTPC_FM`.
  - The module, conda, `PY`, thread and `TMPDIR` lines are unchanged.
- **`run_queue.sh`, `convert_node.sh`, `node_agent.sh`, `box_run.sh`, `convert_tasks.sh`, `feeder.sh`:**
  each sources `env.sh` relative to itself (`readlink -f` on `BASH_SOURCE`), and the yuhw literals are
  replaced by `$QLFM_WORK`, `$QLFM_DATA` and `$BOX`.
  - `box_run.sh` writes the absolute path of `env.sh` into the inbox command at the time it writes it.
  - `feeder.sh` runs `cd $QLFM_WORK/jobs` and `qsub -A "$QLFM_PBS_ACCOUNT" ...`, and its comment about
    the account was updated.
- **All 15 `pbs/*.pbs` files** now have `#PBS -A LArTPC_FM`.
  - 13 files set `QLFM_ROOT=${QLFM_ROOT:-<fork>/multi-modal-qlmatch}` and source `$QLFM_ROOT/scripts/env.sh`.
  - `queue.pbs` sets only `QLFM_ROOT`, because `run_queue.sh` sources `env.sh` itself.
  - `multi_cmd.pbs` changed only its `-A` line.
  - `convert.pbs`, `hold.pbs` and `pimm_pretrain.pbs` lost their `R=`, `QLFM_WORK=`, `PIMM_ROOT=` and `BOX=` literals.
  - **Behaviour change:** `pimm_pretrain.pbs` now sources `env.sh`, which loads the conda module and adds
    `PYTHONPATH`, before running the pimm venv's `torchrun`. Upstream's `infer_hits.pbs` and
    `smoke_r2/r3.pbs` already combine the two this way, but this file was not run to check.
- The `feeder.sh:8` comment "(shared with neutrinoGPU)" about queue limits was left as is.

## Steps 4-5: Python
- **New `qlfm/paths.py`:** defines `WORK`, `REF_WORK` and `PIMM_ROOT`, read from the environment with
  yuhw's work dir as the fallback.
- **Use `WORK` through the helper:** `scripts/{make_tasks_r2,r1_io_figures,r1_mae_demo,r2_event_displays,r4_t5_handscan,pandora_refs,r4_light_stats}.py`.
  - `make_tasks_r2.py` also gets the `sys.path` idiom based on `__file__`, because it had no earlier qlfm import.
- **Read yuhw's run results from `REF_WORK`:**
  - `r2_select_events.py`: `r2p/...` and `round2/displays/...`;
  - `r4_event_displays.py`: `preds_test.npz`;
  - `r4_posthoc.py`: run dirs under `r3/` and `r2p/`, plus `r4/light/*`;
  - `qlfm/paper_metrics.py`: `r4/light/light_stats.npz`.
  - Corpus, cache and `truth.npz` stay on `WORK`.
- **`pimm_train.py` and `pimm_backbone.py`:** take `PIMM_ROOT` from `qlfm.paths`. `pimm_train.py` now adds
  the repo to `sys.path` before importing it.
- **4 pimm configs:** `_base_` and `W` come from `PIMM_ROOT` and `QLFM_WORK`, with yuhw's paths as the fallback.

## Step 6: verification results
- **Passed:**
  - `bash -n` on every `.sh`/`.pbs` file.
  - `grep -L '^#PBS -A LArTPC_FM$' scripts/pbs/*.pbs` prints nothing.
  - Sourcing `env.sh` from `/tmp` gives the expected `QLFM_*`, `PIMM_ROOT` and `BOX` values.
  - `qlfm.paths` gives our `WORK` and `PIMM_ROOT` with yuhw's `REF_WORK`.
  - `py_compile` succeeds on all edited Python files.
  - All 4 pimm configs load with `Config.fromfile` in the pimm venv. Their corpus and splits point at the
    new work dir, no module objects remain, and `pretty_text` renders.
- **Remaining yuhw/neutrinoGPU strings in code:** only intended defaults. They are in `env.sh`
  (`QLFM_DATA`, `QLFM_REF_WORK`), `qlfm/paths.py` and the 4 pimm configs (fallback literals).
- **Only partly tested:**
  - Seven scripts were import-checked with the conda python.
  - `paper_metrics`, `pandora_refs`, `r4_posthoc` and `r4_light_stats` were import-checked only with
    scipy, sklearn and torch replaced by stubs. On the login node, scipy/sklearn imports hang on Lustre and
    torch can't load `libcublas`.
- **`smoke.pbs` debug run, job 7835098 (submitted from `$QLFM_WORK/logs`): PASSED.**
  - `Exit_status = 0` after 44 s of walltime. `qstat` showed `Account_Name = LArTPC_FM`, and
    `sbank list jobs -p LArTPC_FM` shows 1 job charged 0.01 node-hours.
  - The 4-GPU DDP pretraining ran 300 steps (val_loss 2.09 -> 2.08) and wrote `pre/fm.pt`. All 4 downstream
    jobs (hit frozen, energy finetune, energy linear, flavor scratch) reached `TEST`.
  - All outputs are under `fm4npp-lartpc-work/test/smoke_1794327078/`. No files owned by jpeterso were
    created in yuhw's work dir.
  - Log: `fm4npp-lartpc-work/logs/qlfm_smoke.o7835098`. Its only "error" lines are the NCCL/TCPStore
    teardown warning at exit (`destroy_process_group() was not called`), which is harmless.
  - The flavor-scratch `macro_f1` of 0.33 comes from the tiny test sample containing only one class
    (`nue18`). It is a property of the test data, not a fault.
- **Not done:**
  - the short `pimm_pretrain.pbs` run (pimm configs, `_base_` and the pimm venv after sourcing `env.sh`);
  - running the other analysis scripts to completion.

## Known issues / follow-ups
- **Some PBS jobs still read yuhw run results from `$QLFM_WORK`.** These inputs are not in the new work dir:
  - `infer_hits.pbs` reads `$QLFM_WORK/r2p/...`;
  - `smoke_r2.pbs` reads `pre/c_f0.90` and `pre/cl_f0.90` (these exist through the `pre/` links);
  - `pandora_refs.pbs` writes `round2/pandora` (an output, so it is fine);
  - `r4_ledger.sh` reads `$QLFM_WORK/tasks`.
  To read yuhw's results, change those paths to `$QLFM_REF_WORK`, or regenerate the results under your
  own work dir.
- **Checkpoint names:** reusing a checkpoint name that already exists in `pre/` (e.g. `OUT=$QLFM_WORK/pre/panda`)
  would write into yuhw's read-only directory and fail. Use a new name, or remove that symlink first.
- **Light stats:** `r4_posthoc.py` and `paper_metrics.py` read light stats from `REF_WORK`. If you re-run
  `r4_light_stats.py` into your own work dir, set `QLFM_REF_WORK` (or edit those reads) to use the new copy.
- **Not committed:** commit when you are satisfied, e.g.
  `git add -A multi-modal-qlmatch prepare_repo.md && git commit`.
