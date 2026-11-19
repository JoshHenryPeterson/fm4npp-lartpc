#!/bin/bash
# File-based command runner for interactive development inside a batch job.
# Drop executable scripts into $BOX/inbox/<name>.sh; output goes to $BOX/out/<name>.log,
# and $BOX/out/<name>.done holds the exit code. Scripts run concurrently in the background.
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
mkdir -p $BOX/inbox $BOX/out $BOX/taken
hostname > $BOX/host
END=$(( $(date +%s) + ${HOLD_SECONDS:-3500} ))
while [ $(date +%s) -lt $END ]; do
  for f in $BOX/inbox/*.sh; do
    [ -e "$f" ] || continue
    n=$(basename $f .sh); mv $f $BOX/taken/$n.sh
    ( bash $BOX/taken/$n.sh > $BOX/out/$n.log 2>&1; echo $? > $BOX/out/$n.done ) &
  done
  sleep 2
done
