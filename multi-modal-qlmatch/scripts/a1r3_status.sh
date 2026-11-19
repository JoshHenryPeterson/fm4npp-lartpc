#!/bin/bash
# A1 round-3 status: STATUS tail, queued a1r3 jobs, per-stage completion counts (all from files).
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/a1r3_common.sh"
echo "== $A1R3/STATUS (tail) =="; tail -n ${1:-15} $A1R3/STATUS 2>/dev/null || echo "(none)"
echo; echo "== qstat (a1r3 jobs) =="; qstat -u $USER -w 2>/dev/null | grep -E 'a1r3_s[0-3] ' || echo "(none queued/running)"
echo; echo "== completion =="
echo "stage0 gate:      $([ -f $A1R3/GATE_OK ] && { echo OK; sed 's/^/                  /' $A1R3/smoke/ratio.txt 2>/dev/null; } || { [ -f $A1R3/GATE_FAIL ] && { echo FAIL; sed 's/^/                  /' $A1R3/GATE_FAIL; } || echo "not passed"; })"
if grep -q "qsub_compute=no" $A1R3/GATE_OK 2>/dev/null; then
  echo; echo "!!! ACTION NEEDED: stage 0 OK but compute-node qsub is unavailable (no chain-forward)."
  echo "!!! From a login node run:  bash $QLFM_ROOT/scripts/a1r3_submit.sh --resume --chain"; echo
fi
echo "part1 novnorm:    $(n_novnorm_done)/6   summary: $([ -f $A1R3/novnorm/summary.md ] && echo yes || echo no)"
echo "stage1 pretrain:  $(n_pre_done)/4 done"
for id in $PRE_IDS; do
  l=$A1R3/pre/$id/log.jsonl; st=$([ -f $l ] && tail -n 3 $l | grep -o '"step": [0-9]*' | tail -1 | grep -o '[0-9]*')
  echo "    $id: step ${st:-0}/34000 $([ -f $A1R3/pre/$id/done ] && echo done)"
done
echo "stage2 downstream: $(n_down_done)/$(n_down_expected)"
echo "stage3 eval:      $(n_eval_done)/$(n_eval_expected)   summary: $([ -f $A1R3/summary.md ] && echo yes || echo no)   COMPLETE: $([ -f $A1R3/COMPLETE ] && echo yes || echo no)"
echo "next incomplete stage: $(first_incomplete_stage)  (4 = all done)"
echo "failed tasks (.fails):"; ls $A1R3/down/*.fails $A1R3/eval/*.fails $A1R3/novnorm/*.fails 2>/dev/null | sed 's/^/    /' || true
