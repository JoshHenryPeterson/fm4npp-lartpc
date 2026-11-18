#!/bin/bash
# Run on each node: convert the subset of tasks assigned to this node (round-robin), NPROC at a time.
# Tasks: "IN.h5 OUTDIR PART NPARTS" lines from scripts/convert_tasks.sh, or from the file $CONVERT_TASKS (SBND: also set QLFM_DETECTOR=sbnd).
source /lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc/multi-modal-qlmatch/scripts/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
NODE=${PMI_RANK:-0}; NNODES=${PMI_SIZE:-1}; NPROC=${NPROC:-28}
{ if [ -n "$CONVERT_TASKS" ]; then cat $CONVERT_TASKS; else bash $QLFM_ROOT/scripts/convert_tasks.sh; fi; } | awk -v n=$NNODES -v r=$NODE 'NR%n==r' | \
  xargs -P $NPROC -L 1 bash -c '[ -f $1/meta.json ] || { mkdir -p $1; $PY -m qlfm.data.convert $0 $1 --part $2 --nparts $3 > $1.log 2>&1; echo "done $1 $?"; }'
