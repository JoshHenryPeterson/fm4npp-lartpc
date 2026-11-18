#!/bin/bash
# One node of pbs/run-multi.pbs: run its WORK/node<rank>.tasks with NPAR converter processes; log to WORK/node<rank>.log.
set -u
WORK=${1:?}
k=${PALS_RANKID:-${PMI_RANK:-${PMIX_RANK:-}}}
[ -n "$k" ] || { echo "node-many.sh: no rank on $(hostname)" >&2; exit 2; }
Y=/lus/flare/projects/neutrinoGPU/yuhw
export H=$Y/fm4npp-lartpc/multi-modal-qlmatch/sbnd-h5maker
export SL7_SETUP=$H/env.sh WRAP=$Y/sbnd-wirecell-production/aurora/env/aurora/in-aurora-sl7.sh
LOG=$WORK/node$k.log
echo "$(/bin/date -u '+%F %T') node$k $(hostname) tasks $(wc -l < "$WORK/node$k.tasks") NPAR=$NPAR" >> "$LOG"
( while true; do echo "$(/bin/date -u '+%F %T') load $(cut -d' ' -f1 /proc/loadavg) avail_GB $(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo) py $(pgrep -u "$USER" -f sbnd_h5maker.py | wc -l)"; sleep 120; done >> "$WORK/node$k.sampler" ) &
S=$!
one() {
  local part=$1 out=$2
  [ -s "$out" ] && { echo "skip $out"; return 0; }
  mkdir -p "$(dirname "$out")"
  ( cd "$H" && /usr/bin/time -v -o "$out.time" "$WRAP" python sbnd_h5maker.py "$out" --units-file "$part" $ARGS > "$out.log" 2>&1 )
  echo "$(/bin/date -u '+%F %T') $(basename "$out") rc $? wall $(awk -F': ' '/Elapsed/{print $2}' "$out.time") rss_kb $(awk -F': ' '/Maximum resident/{print $2}' "$out.time")"
}
export -f one
export ARGS
tr '\t' '\n' < "$WORK/node$k.tasks" | xargs -P "$NPAR" -n 2 bash -c 'one "$0" "$1"' >> "$LOG" 2>&1
kill $S 2>/dev/null
echo "$(/bin/date -u '+%F %T') node$k done" >> "$LOG"
