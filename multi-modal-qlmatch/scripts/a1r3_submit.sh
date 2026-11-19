#!/bin/bash
# Submit the A1 round-3 pipeline (docs/a1_check_round_3_plan.md sec. 6). Nothing runs on the login node afterwards.
#   a1r3_submit.sh                 submit stage 0 only (debug: tests, splits, smoke, Part 1); it chain-forwards stages 1-3
#   a1r3_submit.sh --resume        find the first incomplete stage from files on disk and submit that stage
#                                  (chain-forward continues from there; all stages skip finished work)
#   a1r3_submit.sh --chain         fallback if qsub from compute nodes is not allowed: submit stages 0..3 at once,
#                                  linked by -W depend=afterok; stages exit nonzero when incomplete (CHAIN_MODE=1)
#   a1r3_submit.sh --resume --chain   same, starting at the first incomplete stage
# Env: CONTROLS=0 (drop the 8 control runs), QUEUE_STAGE{0..3}, NODES_STAGE{n}, WALL_STAGE{n}, MAX_RETRY_S{n}, QLFM_PBS_ACCOUNT.
RESUME=0; CHAIN=0
for a in "$@"; do case $a in --resume) RESUME=1;; --chain) CHAIN=1;; -h|--help) sed -n 2,11p "$0"; exit 0;; *) echo "unknown arg $a"; exit 2;; esac; done
[ $CHAIN = 1 ] && export CHAIN_MODE=1
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/a1r3_common.sh"
mkdir -p $A1R3/logs $A1R3/tasks
if qstat -u $USER -w 2>/dev/null | grep -Eq ' a1r3_s[0-3] '; then
  echo "an a1r3 job is already queued/running:"; qstat -u $USER -w | grep -E ' a1r3_s[0-3] '; exit 1
fi
first=$(first_incomplete_stage)
if [ $RESUME = 1 ]; then
  [ $first -ge 4 ] && { echo "all stages complete ($A1R3/COMPLETE); nothing to do"; exit 0; }
  start=$first
else
  [ $first -ge 1 ] && { echo "stage 0 already passed (GATE_OK exists); use --resume to continue from stage $first"; exit 1; }
  start=0
fi
rm -f $A1R3/GATE_FAIL
[ $start -eq 0 ] && make_novnorm_tasks_file
[ $start -le 1 ] && make_pretrain_runs
echo "starting at stage $start (chain=$CHAIN controls=$CONTROLS A1R3=$A1R3)"
if [ $CHAIN = 1 ]; then
  prev=""
  for k in $(seq $start 3); do
    prev=$(submit_stage $k 0 $prev) || exit 1
    echo "stage $k: $prev"
  done
else
  job=$(submit_stage $start 0) || exit 1
  echo "stage $start: $job"
fi
echo "progress: bash $QLFM_ROOT/scripts/a1r3_status.sh   (STATUS: $A1R3/STATUS, logs: $A1R3/logs/)"
