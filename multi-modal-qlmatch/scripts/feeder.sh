#!/bin/bash
# Job feeder: submit PBS jobs from a plan, keeping each queue within its per-user limit.
# PLAN lines:  QUEUE|REQUIRE|qsub arguments...
#   QUEUE   : debug, debug-scaling, preemptable or capacity
#   REQUIRE : file that must exist before submitting (e.g. a pretrained fm.pt), or "-"
# Limits (our own jobs, states counted):
#   debug          1 queued (one more may run)        debug-scaling  1 in any state
#   preemptable    $PREEMPT_MAX in any state (default 6; the queue allows 20 per project, shared with neutrinoGPU)
#   capacity       1 in any state (1 running per user)
# Account: the #PBS -A line of the .pbs script (neutrinoGPU::WireCellDev). Preemptable jobs need `-r y` in the
# script (queue.pbs, multi_cmd.pbs, pretrain_multi.pbs, pimm_pretrain.pbs have it) and resumable commands.
# Submitted lines are recorded in PLAN.submitted (with job ids); rerunning the feeder skips them.
# Usage: setsid nohup bash feeder.sh PLAN > PLAN.feeder.log 2>&1 &
PLAN=$1
DONE=$PLAN.submitted
touch $DONE
cd /lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/jobs
declare -A LIMIT=([debug]=1 [debug-scaling]=1 [preemptable]=${PREEMPT_MAX:-6} [capacity]=1)
declare -A STATES=([debug]=Q [debug-scaling]=QRHE [preemptable]=QRHE [capacity]=QRHE)
count() {   # our jobs in queue $1 whose state is in $2 (qstat truncates queue names, so compare 7 chars)
  qstat -u $USER 2>/dev/null | awk -v q="$1" -v s="$2" 'NR>5 && substr($3,1,7)==substr(q,1,7) && index(s,$10)>0' | wc -l
}
while true; do
  pending=0
  for Q in "${!LIMIT[@]}"; do
    nq=$(count $Q ${STATES[$Q]})
    while IFS='|' read -r q req args; do
      [ -z "$q" ] && continue
      grep -qxF "$q|$req|$args" <(cut -d'#' -f1 $DONE | sed 's/ *$//') && continue
      pending=$((pending+1))
      [ "$q" != "$Q" ] && continue
      [ "$nq" -ge "${LIMIT[$Q]}" ] && break
      [ "$req" != "-" ] && [ ! -f "$req" ] && continue
      id=$(eval qsub -q $q $args 2>&1)
      if [[ "$id" =~ ^[0-9]+\.polaris ]]; then
        echo "$(/bin/date '+%F %T') submitted [$q] $id :: $args"
        echo "$q|$req|$args #$id" >> $DONE
        nq=$((nq+1))
      else
        echo "$(/bin/date '+%F %T') deferred [$q] ($id)"
        break
      fi
    done < $PLAN
  done
  [ $pending -eq 0 ] && { echo "$(/bin/date '+%F %T') all submitted"; exit 0; }
  sleep 60
done
