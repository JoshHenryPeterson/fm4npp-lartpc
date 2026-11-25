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
# 0.6 (#16): with DUMPS=<dir> (and WCT_SETUP = the WCT 0.38.0 run setup), each task first writes the blob dumps of
# its units into DUMPS/<out name>/ (wct/blobdump.sh, one unit after the other), runs the converter with --blob-dir,
# and deletes the dumps when the converter succeeded.
export DUMPS=${DUMPS:-} WCT_SETUP=${WCT_SETUP:-}
one() {
  local part=$1 out=$2 bargs="" brc=0 t0
  [ -s "$out" ] && { echo "skip $out"; return 0; }
  mkdir -p "$(dirname "$out")"
  if [ -n "$DUMPS" ]; then
    local D=$DUMPS/$(basename "$out" .h5); mkdir -p "$D"; t0=$(/bin/date +%s)
    while read -r u _; do
      [ -n "$u" ] || continue
      ls "$D/$(basename "$u")".blobs*.tar.gz > /dev/null 2>&1 && continue
      SL7_SETUP=$WCT_SETUP "$WRAP" "$H/wct/blobdump.sh" "$u" "$D" sim > "$D/$(basename "$u").dump.log" 2>&1 || brc=1
    done < "$part"
    echo "$(/bin/date -u '+%F %T') $(basename "$out") dumps rc $brc wall $(( $(/bin/date +%s) - t0 ))"
    [ $brc -eq 0 ] || return 0
    bargs="--blob-dir $D"
  fi
  ( cd "$H" && /usr/bin/time -v -o "$out.time" "$WRAP" python sbnd_h5maker.py "$out" --units-file "$part" $bargs $ARGS > "$out.log" 2>&1 )
  local rc=$?
  echo "$(/bin/date -u '+%F %T') $(basename "$out") rc $rc wall $(awk -F': ' '/Elapsed/{print $2}' "$out.time") rss_kb $(awk -F': ' '/Maximum resident/{print $2}' "$out.time")"
  [ -n "$DUMPS" ] && [ $rc -eq 0 ] && [ -s "$out" ] && rm -rf "$DUMPS/$(basename "$out" .h5)"
}
export -f one
export ARGS
tr '\t' '\n' < "$WORK/node$k.tasks" | xargs -P "$NPAR" -n 2 bash -c 'one "$0" "$1"' >> "$LOG" 2>&1
kill $S 2>/dev/null
echo "$(/bin/date -u '+%F %T') node$k done" >> "$LOG"
