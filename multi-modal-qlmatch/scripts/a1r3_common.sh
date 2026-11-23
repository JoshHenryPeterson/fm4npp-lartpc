# Source me: shared helpers for the A1 round-3 pipeline (submit/status scripts and the 4 stage PBS jobs).
# Design (docs/a1_check_round_3_plan.md sec. 6): all state is on the filesystem; every stage is idempotent and
# chain-forwards (submits its successor from inside its PBS job). CHAIN_MODE=1: no resubmission, stages exit nonzero
# when incomplete (used with a1r3_submit.sh --chain, linked by -W depend=afterok).
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
export A1R3=${A1R3:-$QLFM_WORK/a1r3}
export Y=$QLFM_REF_WORK
export REF_RUNS=$Y/s3/mt                      # read-only baselines (yuhw)
export SPLITS=$A1R3/splits_r3.npz
export CONTROLS=${CONTROLS:-1}
export CHAIN_MODE=${CHAIN_MODE:-0}
export RETRY=${RETRY:-0}
SEEDS_DOWN="0 1"
NOVNORM_VARIANTS=w_meas3d,w_meas3d_noVnorm,w_meas3d_V,w_meas3d_noVnorm_V
EVAL_VARIANTS=d_real,w_meas3d,l_comb80,d_real_noVnorm
PRE_IDS="sim_cl real_cl sim_c real_c"

# ---- per-stage PBS resources (overridable: QUEUE_STAGE{0..3}, NODES_STAGE{n}, WALL_STAGE{n}, MAX_RETRY_S{n}) --------
stage_queue() { local v=QUEUE_STAGE$1 d=preemptable; [ $1 -eq 0 ] && d=debug; echo ${!v:-$d}; }
stage_nodes() {
  local v=NODES_STAGE$1; local -a d=(2 4 6 4)
  if [ -n "${!v}" ]; then echo ${!v}; return; fi
  if [ $1 -eq 1 ]; then   # one node per unfinished pretrain (min 1): retries/resumes do not hold idle nodes
    local r=$(( $(echo $PRE_IDS | wc -w) - $(n_pre_done) )); [ $r -lt 1 ] && r=1; echo $r; return
  fi
  echo ${d[$1]}
}
stage_wall()  {   # debug* queues: 1 h segments
  local v=WALL_STAGE$1; local -a d=(01:00:00 03:30:00 02:00:00 01:30:00)
  if [ -n "${!v}" ]; then echo ${!v}; elif [[ "$(stage_queue $1)" == debug* ]]; then echo 01:00:00; else echo ${d[$1]}; fi
}
stage_maxretry() {
  local v=MAX_RETRY_S$1; local -a d=(0 4 3 3)
  if [ -n "${!v}" ]; then echo ${!v}; elif [[ "$(stage_queue $1)" == debug* ]]; then echo $(( ${d[$1]} * 3 )); else echo ${d[$1]}; fi
}

status() {   # status STAGE_NUMBER MESSAGE...   -> "date jobid stageN message" line in $A1R3/STATUS
  local st=$1 jid=login; shift
  [ -n "$PBS_JOBID" ] && jid=${PBS_JOBID%%.*}
  mkdir -p $A1R3; echo "$(/bin/date +%F_%T) $jid stage${st} $*" >> $A1R3/STATUS
}

# ---- run matrix ---------------------------------------------------------------------------------------------------
# down_matrix: lines "NAME ARM PRETRAIN DOMAIN MODE LATE SEED"  (ARM cl|cxf; backbone sim_cl/real_cl or sim_c/real_c)
down_matrix() {
  local arm late bb pd dd mode s
  for arm in cl cxf; do
    if [ $arm = cl ]; then late=0; else late=2; fi
    for mode in finetune frozen; do
      for pair in sim:real real:sim; do      # pretrain:downstream domain
        pd=${pair%%:*}; dd=${pair##*:}
        for s in $SEEDS_DOWN; do echo "${arm}_${pd}2${dd}_${mode}_s$s $arm $pd $dd $mode $late $s"; done
      done
      if [ "$CONTROLS" = 1 ]; then
        for pd in sim real; do echo "${arm}_${pd}2${pd}_${mode}_s0 $arm $pd $pd $mode $late 0"; done
      fi
    done
  done
}
n_down_expected() { down_matrix | wc -l; }
bb_id() { [ $1 = cl ] && echo ${2}_cl || echo ${2}_c; }   # bb_id ARM PRETRAIN_DOMAIN

# baselines: lines "NAME RUNDIR"
baseline_runs() {
  local arm m
  for arm in cl cxf; do
    if [ $arm = cl ]; then m=qlfm_M_CL; else m=qlfm_M_C_xf; fi
    for s in 0 1 2; do echo "base_${arm}_finetune_s$s $REF_RUNS/$m/finetune/f0.01_s$s"; done
    echo "base_${arm}_frozen_s0 $REF_RUNS/$m/frozen/f0.01_s0"
  done
}
novnorm_runs() {   # lines "NAME RUNDIR" (Part 1, plan 1.2)
  local m s
  for m in qlfm_M_CL qlfm_M_C_xf; do for s in 0 1 2; do
    echo "s3_mt_${m}_finetune_f0.01_s$s $REF_RUNS/$m/finetune/f0.01_s$s"
  done; done
}
n_eval_expected() { echo $(( $(n_down_expected) + $(baseline_runs | wc -l) )); }

# ---- task-file generators (deterministic, idempotent) -------------------------------------------------------------
PRE_ARGS_COMMON="--splits $SPLITS --subset pre_r3 --bs 64 --lr 1.5e-4 --wd 0.05 --warmup 0.05 --d 384 --depth 12 --heads 6 --max-tokens 4096 --mask-ratio 0.6 --log-every 100 --val-every 4000 --n-val 2048 --seed 0"
pre_args() {   # pre_args ID : recipe of Y/pre/m_{cl,c}_f0.90 with only splits/subset/domain/out changed
  local id=$1 light=1 dom=${1%%_*}
  [ ${id#*_} = c ] && light=0
  echo "$PRE_ARGS_COMMON --steps 34000 --light $light --domain $dom"
}
make_pretrain_runs() {   # $A1R3/tasks/pretrain_runs.txt: "OUT ARGS" for runs without a done file
  mkdir -p $A1R3/tasks $A1R3/logs $A1R3/pre
  : > $A1R3/tasks/pretrain_runs.txt
  local id
  for id in $PRE_IDS; do
    [ -f $A1R3/pre/$id/done ] || echo "$A1R3/pre/$id $(pre_args $id)" >> $A1R3/tasks/pretrain_runs.txt
  done
}
make_down_tasks() {      # only for backbones that are done
  mkdir -p $A1R3/tasks $A1R3/logs $A1R3/down
  : > $A1R3/tasks/down.txt
  local name arm pd dd mode late s bb
  while read -r name arm pd dd mode late s; do
    bb=$(bb_id $arm $pd)
    [ -f $A1R3/pre/$bb/done ] || continue
    # recipe of Y/s3/mt/qlfm_M_*: 5 tasks (default), bs 32, 6000 steps, label-frac 0.01, n-test 10000 per task
    echo "qlfm.train_multitask --mode $mode --fm $A1R3/pre/$bb/fm.pt --backbone qlfm --late-light $late --label-frac 0.01 --seed $s --max-steps 6000 --n-test 10000 --splits $SPLITS --domain $dd --out $A1R3/down/$name" >> $A1R3/tasks/down.txt
  done < <(down_matrix)
}
eval_line() {  # eval_line RUNDIR VARIANTS SAVE NBOOT OUT [extra]
  echo "qlfm.eval_perturb --run $1 $6 --tasks hit,evt_nu --variants $2 --save-preds $3 --n-boot $4 --out $5"
}
make_novnorm_tasks() {   # Part 1; $1 = file
  local name run
  : > $1
  while read -r name run; do
    eval_line $run $NOVNORM_VARIANTS w_meas3d_noVnorm 1000 $A1R3/novnorm/$name >> $1
  done < <(novnorm_runs)
}
make_eval_tasks() {      # new runs whose training is finished + baselines, then Part-1 novnorm (skipped if done)
  mkdir -p $A1R3/tasks $A1R3/logs $A1R3/eval $A1R3/novnorm
  : > $A1R3/tasks/eval.txt
  local name arm rest run
  while read -r name arm rest; do
    [ -f $A1R3/down/$name/metrics.json ] || continue
    eval_line $A1R3/down/$name $EVAL_VARIANTS d_real 1000 $A1R3/eval/$name --absolute-domain >> $A1R3/tasks/eval.txt
  done < <(down_matrix)
  while read -r name run; do
    eval_line $run $EVAL_VARIANTS d_real 1000 $A1R3/eval/$name --absolute-domain >> $A1R3/tasks/eval.txt
  done < <(baseline_runs)
  make_novnorm_tasks $A1R3/tasks/novnorm_tmp.txt
  cat $A1R3/tasks/novnorm_tmp.txt >> $A1R3/tasks/eval.txt
}
make_novnorm_tasks_file() { mkdir -p $A1R3/tasks $A1R3/logs $A1R3/novnorm; make_novnorm_tasks $A1R3/tasks/novnorm.txt; }

# ---- completion state (from files) -------------------------------------------------------------------------------
count_files() { local n=0 f; for f in "$@"; do [ -f "$f" ] && n=$((n+1)); done; echo $n; }
n_pre_done()   { local id l=(); for id in $PRE_IDS; do l+=($A1R3/pre/$id/done); done; count_files "${l[@]}"; }
n_down_done()  { local n a l=(); while read -r n a; do l+=($A1R3/down/$n/metrics.json); done < <(down_matrix); count_files "${l[@]}"; }
n_eval_done()  { local n a l=(); while read -r n a; do l+=($A1R3/eval/$n/metrics.json); done < <(down_matrix; baseline_runs); count_files "${l[@]}"; }
n_novnorm_done() { local n a l=(); while read -r n a; do l+=($A1R3/novnorm/$n/metrics.json); done < <(novnorm_runs); count_files "${l[@]}"; }
stage_complete() {   # stage_complete N -> 0 if finished
  case $1 in
    0) [ -f $A1R3/GATE_OK ] ;;
    1) [ $(n_pre_done) -eq 4 ] ;;
    2) [ $(n_down_done) -eq $(n_down_expected) ] ;;
    3) [ -f $A1R3/COMPLETE ] ;;
  esac
}
first_incomplete_stage() { local k; for k in 0 1 2 3; do stage_complete $k || { echo $k; return; }; done; echo 4; }

# ---- submission (login node or chain-forward from inside a job) ---------------------------------------------------
# submit_stage N [RETRY] [DEPEND_JOBID]: prints the job id (qsub output); the exact qsub line is logged to STATUS.
submit_stage() {
  local n=$1 retry=${2:-0} dep=$3 q nn w rr=y
  q=$(stage_queue $n); nn=$(stage_nodes $n); w=$(stage_wall $n)
  [ $n -eq 0 ] && rr=n
  mkdir -p $A1R3/logs
  local log=$A1R3/logs/a1r3_s${n}_$(/bin/date +%m%d_%H%M%S)_r$retry.log
  local v="QLFM_ROOT=$QLFM_ROOT,QLFM_WORK=$QLFM_WORK,QLFM_REF_WORK=$QLFM_REF_WORK,QLFM_PBS_ACCOUNT=$QLFM_PBS_ACCOUNT,RETRY=$retry,CHAIN_MODE=$CHAIN_MODE,CONTROLS=$CONTROLS"
  local k; for k in 0 1 2 3; do
    for p in QUEUE_STAGE NODES_STAGE WALL_STAGE MAX_RETRY_S; do local vn=$p$k; [ -n "${!vn}" ] && v="$v,$vn=${!vn}"; done
  done
  local cmd=(qsub -A $QLFM_PBS_ACCOUNT -N a1r3_s$n -q $q -l select=$nn:system=polaris -l walltime=$w
             -l filesystems=home:eagle -j oe -r $rr -o $log)
  [ -n "$dep" ] && cmd+=(-W depend=afterok:$dep)
  cmd+=(-v "$v" $QLFM_ROOT/scripts/pbs/a1r3_stage$n.pbs)
  local job
  job=$("${cmd[@]}") || { status $n "SUBMIT FAILED: ${cmd[*]}"; return 1; }
  status $n "submitted $job retry=$retry${dep:+ after $dep}: ${cmd[*]}"
  echo "$job"
}

# chain_forward N: called at the end of stage N (N=1..3 work stages). Never returns.
chain_forward() {
  local n=$1 mr
  if stage_complete $n; then
    status $n "COMPLETE-STAGE"
    if [ $CHAIN_MODE = 1 ] || [ $n -ge 3 ]; then exit 0; fi
    submit_stage $((n+1)) 0 >/dev/null || { status $n "FAILED: cannot submit stage $((n+1))"; exit 4; }
    exit 0
  fi
  if [ $CHAIN_MODE = 1 ]; then status $n "INCOMPLETE (chain mode: exiting nonzero; recover with a1r3_submit.sh --resume)"; exit 1; fi
  mr=$(stage_maxretry $n)
  if [ $RETRY -lt $mr ]; then
    status $n "incomplete; resubmitting (retry $((RETRY+1))/$mr)"
    submit_stage $n $((RETRY+1)) >/dev/null || { status $n "FAILED: cannot resubmit"; exit 4; }
    exit 0
  fi
  status $n "FAILED: incomplete after $RETRY retries (recover with a1r3_submit.sh --resume)"
  exit 2
}

# ---- run_queue helpers (as a1r2_common.sh) ------------------------------------------------------------------------
wall_secs() { qstat -f $PBS_JOBID | awk -F'= ' '/Resource_List.walltime/{split($2,a,":"); print a[1]*3600+a[2]*60+a[3]}'; }
watch_tasks() {  # watch_tasks STAGE TASKFILE: log "task done"/"task FAILED" once per task (run in background)
  local st=$1 tf=$2; declare -A seen
  while true; do
    while read -r line; do
      [ -z "$line" ] && continue
      local out tag
      out=$(echo "$line" | sed -E 's/.*--out ([^ ]+).*/\1/'); tag=$(basename $out)
      [ -n "${seen[$tag]}" ] && continue
      if [ -f $out/metrics.json ]; then status $st "task done: $tag"; seen[$tag]=1
      elif [ $(cat $out.fails 2>/dev/null | wc -l) -ge ${MAX_FAIL:-2} ]; then status $st "task FAILED: $tag (see $out/run.log)"; seen[$tag]=1
      fi
    done < $tf
    sleep 60
  done
}
run_tasks() {  # run_tasks TASKFILE RESERVE_MIN : all nodes of the job, deadline = JOB_T0 + walltime - RESERVE
  local t0=${JOB_T0:-$(/bin/date +%s)} nn=$(wc -l < $PBS_NODEFILE)
  local deadline=$(( t0 + $(wall_secs) - $2 * 60 ))
  mpiexec -n $nn --ppn 1 --cpu-bind none env RUN_ID="${PBS_JOBID%%.*}@$t0" QLFM_DETECTOR=${QLFM_DETECTOR:-uboone} \
    bash $QLFM_ROOT/scripts/run_queue.sh $1 $deadline
}

# ---- stage 1 helpers ----------------------------------------------------------------------------------------------
fix_done() {   # a run killed between its last save() and writing `done` would early-return "already done" forever
  local id d s
  for id in $PRE_IDS; do
    d=$A1R3/pre/$id
    [ -f $d/done ] && continue; [ -f $d/ckpt.pt ] || continue
    s=$($PY - $d/ckpt.pt <<'PYEOF'
import sys, torch
s = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(s["step"] if s["step"] >= (s.get("steps_final") or s["args"]["steps"]) else "")
PYEOF
)
    [ -n "$s" ] && { echo "$s" > $d/done; status 1 "wrote missing done for $id (step $s)"; }
  done
}

# ---- stage 0 helpers ----------------------------------------------------------------------------------------------
s0_fail() { echo "$*" > $A1R3/smoke/A_FAIL; status 0 "node A FAIL: $*"; return 1; }
pre_rate() {   # pre_rate LOG_JSONL -> "RATE FINITE" (median ev_per_s of steps>=200; FINITE=1 if all losses finite and >=1)
  $PY - "$1" <<'PYEOF'
import json, math, statistics, sys
r = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
v = [x["ev_per_s"] for x in r if "ev_per_s" in x and x.get("step", 0) >= 200]
L = [x[k] for x in r for k in ("loss", "val_loss") if k in x]
print(statistics.median(v) if v else 0.0, int(bool(L) and all(math.isfinite(x) for x in L)))
PYEOF
}
s0_pretrain_smoke() {   # s0_pretrain_smoke sim|real  (4 ranks on this node)
  local dom=$1 out=$A1R3/smoke/pre_$1 port=$((29590 + $( [ $1 = sim ] && echo 1 || echo 0 )))
  [ -f $out/SMOKE_OK ] && return 0
  rm -rf $out; mkdir -p $out
  status 0 "smoke pretrain --domain $dom started"
  ( export MASTER_ADDR=$(head -1 $PBS_NODEFILE) MASTER_PORT=$port
    mpiexec -n 4 --ppn 4 --hostfile $A1R3/tasks/hosts_A --cpu-bind depth -d 8 env QLFM_DETECTOR=${QLFM_DETECTOR:-uboone} \
      $PY -u -m qlfm.train_pretrain --out $out $PRE_ARGS_COMMON --steps 600 --val-every 10000 --log-every 50 \
      --time-limit 25 --light 1 --domain $dom 2>&1 | grep -v -E "Warn|bind: warning" > $out/stdout.log )
  [ -f $out/event_lists.npz ] && [ -f $out/fm.pt ] && [ -f $out/log.jsonl ] || { s0_fail "smoke pretrain $dom: missing outputs (see $out/stdout.log)"; return 1; }
  local r; r=($(pre_rate $out/log.jsonl))
  [ "${r[1]}" = 1 ] || { s0_fail "smoke pretrain $dom: non-finite or missing loss"; return 1; }
  echo ${r[0]} > $out/ev_per_s; touch $out/SMOKE_OK
}
s0_node_a() {   # gates 1-4 and 6; run on the head node of the job
  mkdir -p $A1R3/smoke $A1R3/logs $A1R3/tasks; rm -f $A1R3/smoke/A_FAIL
  cd $QLFM_ROOT
  # gate 1: unit tests (round-1/2 behaviour unchanged + round 3)
  if [ ! -f $A1R3/smoke/tests_ok ]; then
    # round-3 tests via pytest if available (else as a script; test_a1r3.py has no pytest import), and the round-1/2
    # script-style tests directly; all must exit 0
    local rc=0 t; : > $A1R3/logs/s0_tests.log
    if $PY -m pytest --version >/dev/null 2>&1; then $PY -m pytest -q tests/test_a1r3.py >> $A1R3/logs/s0_tests.log 2>&1 || rc=1
    else $PY tests/test_a1r3.py >> $A1R3/logs/s0_tests.log 2>&1 || rc=1; fi
    for t in tests/test_a1r2.py tests/test_a1_light.py; do $PY $t >> $A1R3/logs/s0_tests.log 2>&1 || rc=1; done
    [ $rc -eq 0 ] || { s0_fail "gate1: unit tests failed (see logs/s0_tests.log)"; return 1; }
    touch $A1R3/smoke/tests_ok; status 0 "gate1 tests passed"
  fi
  # gate 2: splits
  if [ ! -f $A1R3/smoke/splits_ok ] || [ ! -f $SPLITS ]; then
    $PY scripts/a1r3_make_splits.py --out $SPLITS > $A1R3/logs/s0_splits.log 2>&1 || { s0_fail "gate2: a1r3_make_splits.py failed (see logs/s0_splits.log)"; return 1; }
    touch $A1R3/smoke/splits_ok; status 0 "gate2 splits ok"
  fi
  # gate 3: pretrain smoke (real, then sim for the ev/s ratio)
  s0_pretrain_smoke sim || return 1
  s0_pretrain_smoke real || return 1
  local rr rs ratio
  rr=$(cat $A1R3/smoke/pre_real/ev_per_s); rs=$(cat $A1R3/smoke/pre_sim/ev_per_s)
  ratio=$($PY -c "print(f'{$rr/$rs:.3f}' if $rs>0 else '0')")
  echo "ratio=$ratio real_ev_per_s=$rr sim_ev_per_s=$rs" > $A1R3/smoke/ratio.txt
  $PY -c "import sys; sys.exit(0 if float('$ratio')>=0.8 else 1)" || { s0_fail "gate3: real/sim ev/s ratio $ratio < 0.8 (real $rr, sim $rs)"; return 1; }
  status 0 "gate3 pretrain smoke ok ($(cat $A1R3/smoke/ratio.txt))"
  # gate 4: downstream smoke from the smoke backbone + absolute-domain eval + checks
  if [ ! -f $A1R3/smoke/down_ok ]; then
    rm -rf $A1R3/smoke/down_real $A1R3/smoke/eval_real; mkdir -p $A1R3/smoke/down_real
    CUDA_VISIBLE_DEVICES=0 $PY -m qlfm.train_multitask --mode finetune --fm $A1R3/smoke/pre_real/fm.pt --backbone qlfm \
      --late-light 0 --label-frac 0.01 --seed 0 --max-steps 300 --eval-every 150 --n-test 500 --n-test-light 500 --n-val 200 \
      --splits $SPLITS --domain real --time-limit 20 --out $A1R3/smoke/down_real > $A1R3/smoke/down_real/run.log 2>&1 \
      || { s0_fail "gate4: smoke train_multitask failed (see smoke/down_real/run.log)"; return 1; }
    CUDA_VISIBLE_DEVICES=0 $PY -m qlfm.eval_perturb --run $A1R3/smoke/down_real --absolute-domain --tasks hit,evt_nu \
      --variants d_real --save-preds d_real --n-boot 50 --bs 32 --out $A1R3/smoke/eval_real > $A1R3/logs/s0_smoke_eval.log 2>&1 \
      || { s0_fail "gate4: smoke eval_perturb failed (see logs/s0_smoke_eval.log)"; return 1; }
    $PY scripts/a1r3_check_smoke.py $A1R3/smoke/down_real $A1R3/smoke/eval_real > $A1R3/logs/s0_smoke_check.log 2>&1 \
      || { s0_fail "gate4: a1r3_check_smoke.py failed (see logs/s0_smoke_check.log)"; return 1; }
    touch $A1R3/smoke/down_ok; status 0 "gate4 downstream smoke / no double perturbation ok"
  fi
  # gate 6: qsub from a compute node (held test job, then qdel). Not needed in --chain mode.
  if [ $CHAIN_MODE = 1 ]; then status 0 "gate6 skipped (chain mode)"; return 0; fi
  if [ ! -f $A1R3/smoke/qsub_ok ]; then
    local ts=$A1R3/tasks/qsub_test.sh j="" q
    printf '#!/bin/bash\ntrue\n' > $ts
    for q in "debug 00:01:00" "preemptable 00:05:00"; do
      j=$(qsub -A $QLFM_PBS_ACCOUNT -N a1r3_qsubtest -q ${q% *} -l select=1:system=polaris -l walltime=${q#* } \
          -l filesystems=home:eagle -h $ts 2>$A1R3/logs/s0_qsub.err) && [ -n "$j" ] && break; j=""
    done
    if [ -z "$j" ]; then   # not a gate failure: the user chains from a login node instead
      echo "qsub_compute=no: $(head -c 200 $A1R3/logs/s0_qsub.err)" > $A1R3/smoke/qsub_no; return 0
    fi
    qdel $j >/dev/null 2>&1 || { s0_fail "gate6: qdel of $j failed"; return 1; }
    rm -f $A1R3/smoke/qsub_no; touch $A1R3/smoke/qsub_ok; status 0 "gate6 qsub/qdel from compute node ok ($j)"
  fi
  return 0
}
