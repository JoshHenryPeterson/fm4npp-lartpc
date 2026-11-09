#!/bin/bash
# Per-node worker launcher: 4 GPU workers scan TASKFILE (one `module args... --out DIR` per line).
# A task is skipped if DIR/metrics.json exists or DIR.lock is held (claimed by another worker/job),
# so several jobs can share one task file. Tasks that cannot finish before DEADLINE are left for later jobs.
# A lock records its owner (RUN_ID = jobid@start) and a heartbeat. A lock is stale, and is reclaimed, when its
# heartbeat is older than STALE_MIN (job killed, e.g. preempted) or it belongs to an earlier run of this same job (rerun).
# A task that exits with an error (not the deadline timeout) is logged to DIR.fails and skipped after MAX_FAIL errors
# (delete DIR.fails to retry it).
# Usage (inside PBS): mpiexec -n NNODES --ppn 1 env RUN_ID=... bash run_queue.sh TASKFILE DEADLINE_EPOCH
source /lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc/multi-modal-qlmatch/scripts/env.sh
TASKS=$1; DEADLINE=$2
RUN_ID=${RUN_ID:-${PBS_JOBID%%.*}@0}
STALE_MIN=${STALE_MIN:-15}
MAX_FAIL=${MAX_FAIL:-2}
claim() {   # claim() LOCKDIR: 0 if we now own it
  local lk=$1
  mkdir $lk 2>/dev/null && { echo "$RUN_ID $(hostname)" > $lk/owner; return 0; }
  local own=$(cat $lk/owner 2>/dev/null | cut -d' ' -f1)
  local hb=$lk/owner; [ -f $hb ] || hb=$lk
  if { [ -n "$own" ] && [ "${own%@*}" = "${RUN_ID%@*}" ] && [ "$own" != "$RUN_ID" ]; } || \
     [ -n "$(find $hb -maxdepth 0 -mmin +$STALE_MIN 2>/dev/null)" ]; then
    local tmp=$lk.stale.$$.$RANDOM
    mv $lk $tmp 2>/dev/null || return 1      # another worker got there first
    rm -rf $tmp
    echo "[$(hostname) $(/bin/date +%T)] reclaimed stale lock $lk (owner ${own:-unknown})"
    mkdir $lk 2>/dev/null && { echo "$RUN_ID $(hostname)" > $lk/owner; return 0; }
  fi
  return 1
}
worker() {
  local gpu=$1
  while true; do
    local ran=0
    while read -r line; do
      [ -z "$line" ] && continue
      local out=$(echo "$line" | sed -E 's/.*--out ([^ ]+).*/\1/')
      [ -f $out/metrics.json ] && continue
      [ $(cat $out.fails 2>/dev/null | wc -l) -ge $MAX_FAIL ] && continue
      local left=$(( (DEADLINE - $(/bin/date +%s)) / 60 - 3 ))
      [ $left -lt 10 ] && return
      # long runs (large label budgets, encoder trained) need a fresh slot
      if { { [[ "$line" =~ --label-frac\ (1|0\.1|0\.01)\  ]] && [[ ! "$line" =~ --mode\ (frozen|linear) ]]; } || { [[ "$line" =~ --accum\ [2-9] ]] && [[ "$line" =~ --label-frac\ (1|0\.1|0\.01)\  ]]; }; } && [ $left -lt 45 ]; then continue; fi
      mkdir -p $(dirname $out)
      claim $out.lock || continue
      mkdir -p $out
      echo "[$(hostname) gpu$gpu $(/bin/date +%T)] -> $out (${left} min left)"
      # optional interpreter prefix: "@pimm " (pimm venv) or "@mamba " (mamba_ssm venv)
      local py=$PY cmd=$line
      case "$line" in
        "@pimm "*)  py=$QLFM_WORK/tools/pimm/.venv/bin/python; cmd=${line#@pimm } ;;
        "@mamba "*) py=$QLFM_WORK/tools/venv-mamba/bin/python; cmd=${line#@mamba } ;;
      esac
      ( while sleep 120; do touch $out.lock/owner; done ) & local hb=$!; disown $hb
      CUDA_VISIBLE_DEVICES=$gpu PIMM_ROOT=$QLFM_WORK/tools/pimm timeout $((left*60)) $py -m $cmd --time-limit $((left-5)) > $out/run.log 2>&1
      local rc=$?
      kill $hb 2>/dev/null
      if [ $rc -ne 0 ] && [ $rc -ne 124 ] && [ ! -f $out/metrics.json ]; then
        echo "$RUN_ID gpu$gpu $(/bin/date +%F_%T) rc=$rc" >> $out.fails
        echo "[$(hostname) gpu$gpu $(/bin/date +%T)] FAILED rc=$rc $out ($(wc -l < $out.fails)/$MAX_FAIL)"
      fi
      rm -rf $out.lock
      ran=1
    done < $TASKS
    [ $ran -eq 0 ] && return
  done
}
for g in 0 1 2 3; do worker $g & done
wait
