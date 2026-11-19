#!/bin/bash
# Submit the whole A1 light check round 1 pipeline as ONE PBS job (scripts/pbs/a1l1_all.pbs: smoke gate -> full -> summary).
# The debug queue allows one queued job per user, so there is no dependency chain. Nothing needs to stay running on
# the login node. Progress: cat $QLFM_WORK/a1l1/STATUS. Idempotent: refuses if an a1l1 job is queued/running;
# smoke/PASSED skips the smoke stage; finished tasks are never rerun.
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/a1l1_common.sh"
if qstat -u $USER -w 2>/dev/null | grep -Eq ' a1l1_'; then
  echo "a1l1 job already queued/running:"; qstat -u $USER -w | grep -E ' a1l1_'; exit 1
fi
make_tasks
nfull=$(ls $A1L1/full/*/metrics.json 2>/dev/null | wc -l)
if [ $nfull -eq 6 ] && [ -f $A1L1/full/summary.md ]; then echo "all 6 runs and summary.md already exist in $A1L1/full; nothing to do"; exit 0; fi
JOB=$(qsub -o $A1L1/logs/a1l1_all.log $QLFM_ROOT/scripts/pbs/a1l1_all.pbs) || exit 1
echo "job: $JOB"
status "submitted $JOB"
echo "progress: cat $A1L1/STATUS   logs: $A1L1/logs/"
