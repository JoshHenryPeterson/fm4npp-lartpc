# Source me: shared helpers for the A1 round-2 pipeline (submit script and PBS jobs).
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
export A1R2=${A1R2:-$QLFM_WORK/a1r2}
export REF_RUNS=$QLFM_REF_WORK/s3/mt            # read-only inputs (yuhw)
export R1_METRICS=$QLFM_REF_WORK/a1/full/s3_mt_qlfm_M_CL_finetune_f0.01_s0/metrics.json   # round-1 reference
MODELS="qlfm_M_CL qlfm_M_C_xf"
SEEDS="0 1 2"
status() { mkdir -p $A1R2; echo "$(/bin/date +%F_%T) $*" >> $A1R2/STATUS; }
make_tasks() {   # idempotent: rewrites the (deterministic) task files
  mkdir -p $A1R2/tasks $A1R2/logs
  local newv=w_meas,w_meas3d,w_meas3d_Y,w_meas3d_UV,w_diff,w_diff_vd
  # round 1 invoked qlfm.eval_perturb with no interpreter prefix (default conda python) for qlfm_* runs
  echo "qlfm.eval_perturb --run $REF_RUNS/qlfm_M_CL/finetune/f0.01_s0 --tasks hit,evt_nu --variants $newv,w_inv3d --save-preds nominal,w_meas3d --n-boot 200 --out $A1R2/smoke/s3_mt_qlfm_M_CL_finetune_f0.01_s0" > $A1R2/tasks/smoke.txt
  : > $A1R2/tasks/full.txt
  for m in $MODELS; do for s in $SEEDS; do
    echo "qlfm.eval_perturb --run $REF_RUNS/$m/finetune/f0.01_s$s --tasks hit,evt_nu --variants $newv --save-preds nominal,w_meas3d --n-boot 1000 --out $A1R2/full/s3_mt_${m}_finetune_f0.01_s$s" >> $A1R2/tasks/full.txt
  done; done
}
watch_tasks() {  # watch_tasks TASKFILE: log "task done" / "task FAILED" to STATUS once per task (run in background)
  local tf=$1; declare -A seen
  while true; do
    while read -r line; do
      [ -z "$line" ] && continue
      local out=$(echo "$line" | sed -E 's/.*--out ([^ ]+).*/\1/') tag=$(basename $out)
      [ -n "${seen[$tag]}" ] && continue
      if [ -f $out/metrics.json ]; then status "task done: $tag"; seen[$tag]=1
      elif [ $(cat $out.fails 2>/dev/null | wc -l) -ge ${MAX_FAIL:-2} ]; then status "task FAILED: $tag (see $out/run.log)"; seen[$tag]=1
      fi
    done < $tf
    sleep 60
  done
}
# run_queue on this node with a deadline reserving $2 minutes at the end of the job's walltime
run_tasks() {  # run_tasks TASKFILE RESERVE_MIN   (deadline measured from JOB_T0 = job start if set, else now)
  local t0=${JOB_T0:-$(/bin/date +%s)}
  local wall=$(qstat -f $PBS_JOBID | awk -F'= ' '/Resource_List.walltime/{split($2,a,":"); print a[1]*3600+a[2]*60+a[3]}')
  local deadline=$(( t0 + wall - $2 * 60 ))
  local nn=$(wc -l < $PBS_NODEFILE)
  mpiexec -n $nn --ppn 1 --cpu-bind none env RUN_ID="${PBS_JOBID%%.*}@$t0" \
    bash $QLFM_ROOT/scripts/run_queue.sh $1 $deadline
}
