---
title: a1_check_round_3_plan.md
---

# A1 check, round 3: plan

Plan for the requests in `a1_check_round_3_preplan.md`:

- **Part 1.** Add the hit-width variant `w_meas3d_noVnorm`.
- **Part 2.** Run cross-domain pretrain → downstream experiments.

The design goal for execution: one command from a login node submits a chain of PBS jobs that moves forward on its own. Claude does not need to stay running, and the SSH session can be closed. Section 6 describes how.

Paths used below:

```
R      = /lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc/multi-modal-qlmatch   (repo)
W      = /lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work                  ($QLFM_WORK)
Y      = /lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work                    (read-only reference)
A1R3   = $W/a1r3                                                                    (all round-3 output)
```

---

## 0. Facts this plan relies on, and corrections to the preplan

Research found the following. File references are to the current working tree, including uncommitted changes.

1. **Light variant name.** The code calls it `l_comb80`, not `l_combo80` (`qlfm/eval_perturb.py:70`). It is defined as `("scale80", "slow", "tail")`. This plan uses `l_comb80`.
2. **"C+FX" in the preplan table means C + XF.** This is the qlfm-M model whose backbone was pretrained on **charge only** (`Y/pre/m_c_f0.90`, `light=0`). Light enters only in the downstream head, through XFusion (`train_multitask --late-light 2`).
   - Consequence: for C + XF, the *pretraining* domain differs only in hit widths.
   - The light correction (`l_comb80`) only affects the downstream and evaluation stages.
   - qlfm-M CL (`Y/pre/m_cl_f0.90`, `light=1`, `--late-light 0`) sees both corrections in pretraining.
3. **Round 2 was inference only.** It perturbed the inputs of trained checkpoints at evaluation time, using `qlfm.eval_perturb`. Part 1 follows the same pattern.
   - Part 2 needs the "real" domain inside *training*. That does not exist yet:
     - `train_multitask --width` accepts only `native|meas|const`.
     - `train_pretrain.py` has no perturbation hook at all.
4. **Double-perturbation bug risk (confirmed by reading the code).**
   - `eval_perturb.Perturbed.forward` (`eval_perturb.py:234`) applies the evaluation variant.
   - `MultiTaskModel.forward` (`train_multitask.py:118-121`) then applies the *training* width again (`load_run` sets `model.width`).
   - Result: a model trained in a perturbed domain and evaluated with a variant gets **both** factors. Its "nominal" is then not the simulation domain.
   - Part 2 must fix this (see 2.4).
5. **Splits.** All splits come from one file, `Y/cache/splits_all.npz`: keys `train/val/test`, `pre_f*` and `lab_f*`.
   - The existing backbones were pretrained on `pre_f0.90`, which is n = 747,068, essentially the whole train partition.
   - Downstream uses `lab_f0.01`, which is n_train = 7,471, also drawn from the train partition.
   - So **pretraining and downstream events overlap** in the existing models. Only `val` and `test` are held out.
   - The preplan asks for separate event lists, so new splits are needed (see 2.2).
6. **Existing models (all in `Y/s3/mt/`).** Used as baselines.
   - CL: `qlfm_M_CL/finetune/f0.01_s{0,1,2}` and `qlfm_M_CL/frozen/f0.01_s0`.
   - C + XF: `qlfm_M_C_xf/finetune/f0.01_s{0,1,2}` and `qlfm_M_C_xf/frozen/f0.01_s0`.
   - Recipe:
     - Five tasks: hit, evt_nu, flavor, energy, light.
     - Batch 32, 6,000 steps, `label_frac` 0.01.
     - Finetune runs take ~60 min (CL) or ~43 min (C + XF) on one A100.
7. **Pretraining cost.** Recipe for `m_cl_f0.90` / `m_c_f0.90`:
   - d=384, depth 12, 6 heads.
   - 34,000 steps, batch 64/rank, `world=4` (1 Polaris node).
   - mask 0.6, lr 1.5e-4, ~1,200 ev/s.
   - That is about **2.0 node-hours per backbone**, done in 53-min segments.
   - `ckpt.pt` is written every `val_every` = 4,000 steps, so preemption loses at most ~15 min.
8. **Evaluation timing.** An `eval_perturb` run takes 12–14 min for width variants and 16–21 min with light variants (rounds 2 and light-1).
   - The round-2 grid (6 runs) used ~1.1 node-hours.
   - Downstream T1 = `hit` and T2 = `evt_nu`.

---

## Part 1: `w_meas3d_noVnorm` (inference only, same as round 2 §2)

### 1.1 Code (`qlfm/eval_perturb.py`)
Add the following next to `F_MEAS_3D`:

```python
F_MEAS_3D_NOVNORM = {2: [1.000, 0.967, 0.941, 0.926, 0.915, 0.903],   # Y (= F_MEAS_3D)
                     1: [0.900, 0.882, 0.859, 0.844, 0.848, 0.847],   # V, not renormalised at the anode
                     0: [1.000, 0.960, 0.922, 0.907, 0.892, 0.880]}   # U (= F_MEAS_3D)
TABLES = {"meas": F_MEAS_2D, "meas3d": F_MEAS_3D, "meas3d_noVnorm": F_MEAS_3D_NOVNORM}
```

New `VARIANTS` entries (`width_factor` already parses `base@planes`):

| variant | kind | purpose |
|---|---|---|
| `w_meas3d_noVnorm` | `meas3d_noVnorm` | the requested variant (all planes) |
| `w_meas3d_V` | `meas3d@1` | V-only, normalised (isolates the V term) |
| `w_meas3d_noVnorm_V` | `meas3d_noVnorm@1` | V-only, not normalised |

The two V-only variants are cheap and show whether the change in the result comes from the V plane alone. They are optional, but recommended.

Check the `"_all"` suffix handling (`width.removesuffix("_all")`) and the `@` partition in `width_factor` so that `meas3d_noVnorm` resolves to the new table and not to `meas3d`.

Unit tests (`tests/test_a1r3.py`):
- the factor for a V hit at `D_C[0]` is 0.900 under noVnorm and 1.000 under meas3d;
- U and Y factors are identical between the two tables;
- non-nu hits are untouched.

### 1.2 Runs
Use the same 6 models as round 2: CL and C + XF, `finetune/f0.01_s{0,1,2}`. The command per model is:

```
python -m qlfm.eval_perturb --run $Y/s3/mt/<model>/finetune/f0.01_s<seed> --tasks hit,evt_nu \
  --variants w_meas3d,w_meas3d_noVnorm,w_meas3d_V,w_meas3d_noVnorm_V \
  --save-preds w_meas3d_noVnorm --n-boot 1000 --out $A1R3/novnorm/s3_mt_<model>_finetune_f0.01_s<seed>
```

- `w_meas3d` is re-run so the deltas are paired within one job. It also checks reproducibility against round 2 (acceptance: within 0.002 of the `a1r2/full` value).
- `nominal` is always included.
- Per-event-median T1 for `w_meas3d_noVnorm` comes from the saved npz, using `paper_metrics.t1` / `t1_by_plane`, the same way as in round 2.
- Cost: about 6 × 15 GPU-min, which fits inside the Stage 0 debug job (section 6).

### 1.3 Output
Write `$A1R3/novnorm/summary.md`, with a table that has the same columns as technote round 2, Table 2.2. It should include:
- Δ vs nominal for `w_meas3d` and `w_meas3d_noVnorm`, plus their difference;
- the V-only pair;
- per-plane T1 (cw and evt-median);
- T2 AUC (all events, and the native bright-flash subset);
- mean ± seed SD over 3 seeds.

---

## Part 2: cross-domain checks

### 2.1 Domain definitions (implemented as composite `VARIANTS`)

| domain | definition | `VARIANTS` entry |
|---|---|---|
| sim | unaltered open data | `nominal` |
| real | `w_meas3d` on nu (sim) hit widths **and** `l_comb80` on light | `d_real = ("meas3d", 0.0, ("scale80","slow","tail"))` |
| real, noVnorm (eval only, optional) | `w_meas3d_noVnorm` + `l_comb80` | `d_real_noVnorm` |

`perturb()` already handles a width kind and a light tuple together: light is applied first, then width.

As in rounds 1–2:
- width applies to hits with `hit_nu_edep > 0` (simulated neutrino hits in the overlay);
- light applies to all ophits of flash events, because sim ophits are not labelled.

This known limitation is carried over and stated in the write-up.

Add a single mapping, used by training, pretraining and evaluation:

```python
DOMAINS = {"sim": "nominal", "real": "d_real", "real_noVnorm": "d_real_noVnorm"}
```

### 2.2 Event lists (disjoint pretrain / downstream / eval)
New script `scripts/a1r3_make_splits.py` writes `$A1R3/splits_r3.npz`:
- It copies all keys of `Y/cache/splits_all.npz` unchanged, so `train/val/test/lab_*` stay identical. Downstream and test sets therefore match the existing models exactly.
- It adds:
  - `pre_r3 = setdiff(pre_f0.90, lab_f0.01 ∪ val ∪ test)` (~739.6k events), used for **pretraining**;
  - `lab_f0.01` (7,471 events), used for **downstream training**;
  - `val`, used for downstream model selection;
  - `test`, used for **evaluation** (10,000 per task, `default_rng(12)` pick, the same as in rounds 1–2).
- It asserts that `pre_r3`, `lab_f0.01`, `val` and `test` are pairwise disjoint and prints the sizes. Because splits are hashed by (run, subrun), it also asserts that `val`/`test` share no (run, subrun) with `pre_r3 ∪ lab_f0.01`.

The same three lists are used for **every** variant. Each variant then has disjoint pretrain/downstream/eval sets, and variants stay directly comparable. This is how this plan reads "separate event lists for each variant training".

So that each run is self-documenting, `train_pretrain.py` and `train_multitask.py` write `event_lists.npz` into their output dir with the global indices actually used:
- pretrain subset;
- downstream train pool after task-eligibility filtering;
- val per task;
- test per task.

The summary script asserts disjointness again from these files.

Note: the existing yuhw baselines were pretrained on `pre_f0.90`, which overlaps `lab_f0.01`. That is why the sim-domain backbones are **re-pretrained** on `pre_r3` (2.3), and why the existing models appear only as a reference row.

### 2.3 Runs

**Pretraining (4 new backbones).** These use the qlfm-M recipe from `Y/pre/m_*_f0.90/args.json`, with only `--subset`/`--splits`/`--domain`/`--out` changed:

| id | backbone | `--light` | `--domain` |
|---|---|---|---|
| `pre/sim_cl` | CL | 1 | sim |
| `pre/real_cl` | CL | 1 | real |
| `pre/sim_c` | C (for C + XF) | 0 | sim |
| `pre/real_c` | C (for C + XF) | 0 | real (width only; no light in the encoder) |

Settings: 34,000 steps, bs 64, `world=4` (exactly 1 node per run, to keep the global batch at 256), seed 0.

Re-pretraining `sim_*` costs ~4 nh. It gives a clean, disjoint sim backbone and a like-for-like partner for `real_*`, trained identically except for the domain. The "existing models" row then tests whether this re-pretraining changed anything.

**Downstream (`train_multitask`).** Same recipe as the existing models: five tasks, `--label-frac 0.01`, 6,000 steps, bs 32, `--splits $A1R3/splits_r3.npz`. Mode is `--mode frozen` or `--mode finetune`. The flags are `--late-light 0` for CL and `--late-light 2` for C + XF. The downstream domain is set with `--domain`.

Required matrix (from the preplan), with run-dir names:

| # | model | pretrain | downstream | mode | seeds | dir under `$A1R3/down/` |
|---|---|---|---|---|---|---|
| 1 | CL | sim | real | frozen | 0, 1 | `cl_sim2real_frozen_s{0,1}` |
| 2 | CL | sim | real | finetune | 0, 1 | `cl_sim2real_finetune_s{0,1}` |
| 3 | C + XF | sim | real | frozen | 0 | `cxf_sim2real_frozen_s0` |
| 4 | C + XF | sim | real | finetune | 0 | `cxf_sim2real_finetune_s0` |
| 5 | CL | real | sim | frozen | 0, 1 | `cl_real2sim_frozen_s{0,1}` |
| 6 | CL | real | sim | finetune | 0, 1 | `cl_real2sim_finetune_s{0,1}` |
| 7 | C + XF | real | sim | frozen | 0 | `cxf_real2sim_frozen_s0` |
| 8 | C + XF | real | sim | finetune | 0 | `cxf_real2sim_finetune_s0` |

That is 12 runs: 8 at seed 0, plus seed 1 for the 4 CL runs.

- The preplan asks for a second downstream seed for CL "if it isn't resource intensive". It costs about 4 GPU-h, roughly 1 nh, so it is included. The seed-1 runs use the same backbone.
- Setting `EXTRA_SEEDS_CXF=1` in `a1r3_common.sh` adds the 4 C + XF seed-1 runs for about 0.8 nh more.

**Recommended controls (seed 0, on by default; drop with `CONTROLS=0`).** These are needed to read the results: without them, a sim→real number has no matched in-domain reference.

| model | pretrain → downstream | modes | dir |
|---|---|---|---|
| CL, C + XF | sim → sim | frozen, finetune | `{cl,cxf}_sim2sim_{mode}_s0` |
| CL, C + XF | real → real | frozen, finetune | `{cl,cxf}_real2real_{mode}_s0` |

That adds 8 runs, about 7 GPU-h (~2 nh).

- `sim2sim` reproduces the existing models on the disjoint split.
- `real2real` is the "trained entirely in the target domain" reference.

**Baselines (no training).** The existing models in `Y/s3/mt/`:
- CL `finetune/f0.01_s{0,1,2}` and `frozen/f0.01_s0`;
- C + XF, the same set.

That is 8 runs.

### 2.4 Evaluation
**Code fix first (item 0.4).** Add an `--absolute-domain` flag to `eval_perturb`.
- When it is set, `load_run` sets the model's internal `width`/`domain` to native, and the `--variants` name the absolute input domain.
- `nominal` then means the sim domain and `d_real` the real domain, whatever the model was trained on.
- The default stays off, so rounds 1–2 behaviour is unchanged.
- All round-3 evaluation uses `--absolute-domain`.

For every new downstream run (20) and every baseline (8):

```
python -m qlfm.eval_perturb --run <run> --absolute-domain --tasks hit,evt_nu \
  --variants d_real,w_meas3d,l_comb80,d_real_noVnorm --save-preds d_real --n-boot 1000 \
  --out $A1R3/eval/<name>
```

- **Primary result:** T1 and T2 on `d_real`, which is the real domain requested by the preplan.
- `nominal` (sim domain) is reported as context, and to compute the domain gap.
- `w_meas3d` and `l_comb80` alone split the real-domain loss into its width and light parts.
- `d_real_noVnorm` links Part 2 to Part 1.

Metrics, the same definitions as round 2:
- T1: charge-weighted F1 (bootstrap, paired vs nominal) and per-event median F1 (`paper_metrics.t1` on `preds_d_real.npz`), plus per-plane values.
- T2: AUC on all events and on the bright-flash subset. The subset is defined from **native** light (`pe_win_native`), so the event set is the same in both domains.

The paper's `truth.npz` 200-PE cut and `BRIGHT_PE` must not be mixed without saying so; label which one each table uses.

### 2.5 Code changes for Part 2

| file | change |
|---|---|
| `qlfm/eval_perturb.py` | `d_real`, `d_real_noVnorm`, `DOMAINS`, `--absolute-domain` |
| `qlfm/train_multitask.py` | `--domain sim\|real\|real_noVnorm` (default sim). `MultiTaskModel.forward` applies `perturb(b, DOMAINS[domain], 0.0)`. Raise an error if both `--width` and `--domain` are non-default. Record `domain` in `metrics.json`. Write `event_lists.npz`. |
| `qlfm/train_pretrain.py` | `--domain`. Apply `perturb` to the device batch in both the training loop and `validate()`, so val loss is in the training domain. Write `event_lists.npz`. Check that the pretraining batch carries `hit_nu_edep`, `hit_plane`, `hit_time`, `oph_t`, `oph_pe` and `oph_evt`. If the collate drops any of them, add them back. |
| `scripts/a1r3_make_splits.py` | 2.2 |
| `scripts/a1r3_common.sh` | paths, run matrix, task-file generators (pretrain RUNS_FILE, downstream tasks, eval tasks, novnorm tasks) |
| `scripts/a1r3_submit.sh`, `scripts/a1r3_status.sh` | section 6 |
| `scripts/pbs/a1r3_stage{0,1,2,3}.pbs` | section 6 |
| `scripts/a1r3_summary.py` | tables for Part 1 and Part 2 as markdown + JSON; disjointness re-check |
| `tests/test_a1r3.py` | unit tests (below) |

What the domain does to T5 (light) targets in training:
- `perturb` returns a modified batch. Wherever the light target is read from that batch, real-domain downstream training will regress the real-domain light.
- Only T1/T2 are reported, so this is acceptable. Check where the target is read and state it in the write-up.
- The five-task recipe is kept so the runs match the existing models.

Unit tests:
- the noVnorm factor checks (1.1);
- `d_real` changes `hit_rms` only on nu hits, and `oph_pe` only on flash events;
- `perturb(b, "nominal")` is the identity;
- `--absolute-domain` gives a model internal width `native`;
- `--width` together with `--domain` raises an error.

---

## 3. Acceptance gates (Stage 0 smoke, automated)

Stage 0 writes `$A1R3/GATE_OK` only if **all** of these pass. Otherwise it writes `$A1R3/GATE_FAIL` with the reason and does not submit Stage 1.

1. `pytest tests/test_a1r2.py tests/test_a1_light.py tests/test_a1r3.py` passes, so round-1/2 behaviour is unchanged.
2. `splits_r3.npz` passes its disjointness assertions.
3. **Pretrain smoke**: `train_pretrain --domain real --steps 300 --light 1` on 1 node.
   - Loss is finite and `event_lists.npz` is written.
   - Throughput `ev/s` is ≥ 0.8 × a `--domain sim` 300-step run. The ratio is recorded in `GATE_OK` and used to set the Stage 1 walltime.
4. **Downstream smoke**: `train_multitask --domain real --max-steps 300 --n-test 500` from the smoke backbone.
   - `eval_perturb --absolute-domain` on it gives a `d_real` T1 score equal to the run's own in-training test score (to 1e-4, same test set). This proves no double perturbation.
   - `nominal` differs from it.
5. **Part 1** (runs on the second node at the same time):
   - all 6 novnorm runs finish;
   - `w_meas3d` reproduces `a1r2/full` within 0.002 (T1 cw) and 0.001 (T2).
6. A test `qsub` from the compute node works: `qsub -q debug -l walltime=00:01:00 -h` followed by `qdel`. Stage chaining (section 6) depends on it.

---

## 4. Outputs

```
$A1R3/
  splits_r3.npz
  STATUS                      # one line per stage transition (time, job id, stage, state)
  GATE_OK | GATE_FAIL
  smoke/
  novnorm/<run>/metrics.json, preds_w_meas3d_noVnorm.npz ; novnorm/summary.md
  pre/{sim,real}_{cl,c}/      # fm.pt, ckpt.pt, args.json, log.jsonl, done, event_lists.npz
  down/<name>/                # best.pt, metrics.json, run.log, event_lists.npz
  eval/<name>/                # metrics.json, preds_d_real.npz
  tasks/ logs/
  summary.md, summary.json    # final Part 2 tables
```

Disk use: preds are ~0.5 GB per saved variant, so 34 × 0.5 ≈ 17 GB. Backbones and heads add about 4 GB.

Final Part 2 table, one row per (model, pretrain→downstream, mode):
- T1 cw, T1 evt-median and T2 AUC on the **real** domain, as mean ± SD where there are 2+ seeds;
- the same values on sim;
- the domain gap (real − sim);
- Δ vs the matched in-domain control, `real2real`.

Existing-model rows are included for both modes.

---

## 5. Compute budget (Polaris, 4 × A100 per node, account `LArTPC_FM`)

| stage | queue | nodes × walltime | expected use | max charged |
|---|---|---|---|---|
| 0: tests, splits, smoke, Part 1 | debug | 2 × 1:00 | ~1.5 nh | 2 nh |
| 1: 4 pretrains | preemptable (`-r y`) | 4 × 3:30 | ~9 nh (2.0–2.5 h each) | 14 nh |
| 2: 20 downstream runs | preemptable | 5 × 2:00 | ~6 nh | 10 nh |
| 3: 28 evaluations + summary | preemptable | 4 × 1:30 | ~4 nh | 6 nh |
| **total** | | | **~20 nh** | **~32 nh** |

Without controls the total drops by about 2 nh. With `EXTRA_SEEDS_CXF` it rises by about 1 nh.

- Check the allocation with `sbank` before submitting. Paper-campaign mentions `neutrinoGPU::WireCellDev`; the scripts default to `LArTPC_FM` through `QLFM_PBS_ACCOUNT` in `env.sh`.
- Add the job IDs and node-hours to the ledger in `paper-campaign.md` afterwards. Round 2 skipped this step.

Preemptable queue waits can be hours. To use debug-scaling instead (≤1 h jobs), set `QUEUE_STAGE{1,2,3}=debug-scaling`; the retry loop (section 6) then runs each stage as several 1-hour segments.

---

## 6. Unattended execution (no Claude, no SSH session needed)

**Principle.** All state lives on the filesystem. Each stage is idempotent and **submits its successor itself** from inside its PBS job ("chain-forward"). Nothing runs on the login node after submission: no `feeder.sh`, no `nohup`, no `screen`, no Claude.

### Stage scripts

- **`a1r3_stage0.pbs`** (debug, 2 nodes, 1 h). Node 1 runs gates 1–4 and 6. Node 2 runs the 6 Part 1 evals through `run_queue.sh` (4 GPUs). Then it checks gate 5 and runs `a1r3_summary.py --part 1`.
  - If everything passes: write `GATE_OK`, then `qsub a1r3_stage1.pbs`.
  - Otherwise: write `GATE_FAIL` and stop.
- **`a1r3_stage1.pbs`** (4 nodes). Uses the `pretrain_multi.pbs` pattern with `NODES_PER_RUN=1`, one pretrain per node, `--time-limit = walltime − 10 min`. It resumes from `ckpt.pt` automatically.
  - At the end, if all 4 `done` files exist: `qsub a1r3_stage2.pbs`.
  - Otherwise, if `RETRY < 4`: `qsub -v RETRY=$((RETRY+1)) a1r3_stage1.pbs` (resume).
  - Otherwise: write FAILED to `STATUS`.
  - A preemption requeues the job (`-r y`), and it resumes.
- **`a1r3_stage2.pbs`** (5 nodes). Runs `run_queue.sh` on the downstream task file. The task file is generated at stage start from the `done` backbones. Lock dirs and skip-if-`metrics.json` make reruns safe.
  - The end-of-job logic is the same: if all present, submit stage 3; else retry ≤3; else fail.
- **`a1r3_stage3.pbs`** (4 nodes). Runs the eval task file the same way, then `a1r3_summary.py --part 2`, then writes `STATUS: COMPLETE`.

Every transition appends `time jobid stage state` to `$A1R3/STATUS`. Logs go to `$A1R3/logs/`.

### Commands the user runs

```bash
# once, after the implementation is committed (from a login node, any shell):
bash $R/scripts/a1r3_submit.sh            # submits stage 0 only; prints the job id
exit                                      # SSH can be closed now

# any later login:
bash $R/scripts/a1r3_status.sh            # STATUS tail, qstat for a1r3 jobs, per-stage done counts
bash $R/scripts/a1r3_submit.sh --resume   # only if STATUS shows FAILED/stalled: re-submits the first
                                          # incomplete stage (all stages skip finished work)
```

### Fallbacks

- **qsub not allowed from compute nodes** (gate 6 fails). `a1r3_submit.sh --chain` submits all four stages from the login node at once, linked by `-W depend=afterok`. Each stage then **exits non-zero** if its work is incomplete, so later stages are not released early. Recovery is `--resume`.
- **Stage 0 Part 1 overruns 1 h.** The Part 1 tasks are also appended to the Stage 3 eval task file (skipped if already done), so they are never lost.

---

## 7. Implementation order (interactive session, before submission)

1. Make a branch, `a1r3`. The working tree has many uncommitted round-2 changes, including `eval_perturb.py` and `train_multitask.py`. Commit those first, as their own commit, so the round-3 diff stays reviewable. **Ask the user before committing.**
2. Part 1 code and its tests (1.1).
3. Domain plumbing: `DOMAINS`, `d_real*`, `--domain` in both trainers, `--absolute-domain`, `event_lists.npz` (2.5).
4. `a1r3_make_splits.py`, `a1r3_common.sh` (task generators), stage PBS scripts, submit/status scripts, `a1r3_summary.py`.
5. Run the unit tests on the login node (CPU) where possible.
6. A code-review pass (subagent) on the diff.
7. Commit, then the user runs `a1r3_submit.sh` (section 6).

After `STATUS: COMPLETE`, a short interactive session writes `docs/technote_a1_round3.md`. It covers Part 1 table, Part 2 table, domain-gap figure, caveats and the node-hour ledger.

Cost-saving note for the implementation session: delegate items 2–4 to an implementer subagent with this plan as the spec, and item 6 to a reviewer subagent.

---

## 8. Open choices (defaults chosen; change before implementation if needed)

| choice | default in this plan | alternative |
|---|---|---|
| Sim-domain backbones | re-pretrain on `pre_r3` (disjoint, +4 nh) | reuse `Y/pre/m_*_f0.90` (overlaps `lab_f0.01`, so it violates the separate-lists request) |
| In-domain controls (sim2sim, real2real) | included (+2 nh) | `CONTROLS=0` |
| Second seed | CL only (as requested) | `EXTRA_SEEDS_CXF=1` (+1 nh) |
| Event lists | one disjoint set of lists shared by all variants | a different random split per variant (makes comparison noisier; not recommended) |
| Queue for stages 1–3 | preemptable | debug-scaling segments |
| Downstream tasks | all five (matches existing models) | `hit,evt_nu` only (slightly faster; heads not comparable to the existing models) |

## 9. Caveats to carry into the write-up

- "Real" is a modelled data-like domain: the round-2 width factors and the light-round `l_comb80`. It is not actual detector data. Widths are corrected on simulated neutrino hits only; light is corrected on all ophits of flash events.
- C + XF pretraining sees only the width part of the domain shift.
- One pretraining seed per backbone, so seed variance is downstream only.
- There is one sample per arm for C + XF (unless `EXTRA_SEEDS_CXF`). Quote the bootstrap CIs and do not over-read differences smaller than the round-2 seed SD.
