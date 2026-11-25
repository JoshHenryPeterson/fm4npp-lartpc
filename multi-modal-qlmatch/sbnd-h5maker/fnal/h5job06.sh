#!/bin/bash
# fm4npp #16 FermiGrid job (porting_sbndbuild03): real-data pairs -> sbnd-h5maker 0.6 HDF5 (Wire-Cell blobs) on dCache.
# The #14 job (h5job.sh) plus, per pair, the kept Wire-Cell step-1 tar(s) and wct/blobdump.sh (prod_aurora's recipe),
# run in a clean environment with the WCLocal2026Nov16 WCT 0.38.0 runtime ($INPUT_TAR_DIR_LOCAL/wc).
#   h5job06.sh <list.tsv name> <pairs per job> <output base>
# list rows: id <tab> sample <tab> reco1 <tab> tracking-pr tar <tab> reco1 events <tab> step-1 tar(s)
#   (space-separated: one qlpctree.tar.gz, or one per segment qlpctree.<seg>.tar.gz of a PARTIAL pair)
# Job $PROCESS takes rows [P*K, P*K+K). Per pair, into <output base>/<sample>/:
#   skip if <id>.done is there; ifdh cp the reco1 and the tar (size-checked); h5pair.sh (all events);
#   put sbnd_<sample>_<id>.h5, .h5.stats.json, .check.json, logs/<id>.logs.tar.gz, LAST <id>.done
#   (status, job, size + sha256 of each file). Nothing is overwritten; a later try's logs get .tryK.
set -u
LIST=${1:?list}; K=${2:?pairs per job}; OUTBASE=${3:?output base}
if [ -n "${DAGMANJOBID:-}" ] && [ -n "${JOBSUBJOBSECTION:-}" ]; then P=$JOBSUBJOBSECTION; else P=${PROCESS:-0}; fi
T0=$(date +%s); echo "h5job06.sh $(date '+%F %T') host $(hostname) index $P list $LIST K $K out $OUTBASE"
H=${INPUT_TAR_DIR_LOCAL:?}/h5; WCT=$INPUT_TAR_DIR_LOCAL/wc; cat "$H/PROVENANCE.txt"
export SBNDH5_SITE=$H/site SBNDH5_MAKER=$H/maker SBNDH5_STEP0=$H/scripts/step0-frameshift.sh
set +u; source "$H/scripts/env-fnal.sh" > setup.log 2>&1; set -u
command -v lar > /dev/null && command -v ifdh > /dev/null || { echo "setup failed"; tail setup.log; exit 3; }
listf=${CONDOR_DIR_INPUT:-.}/$(basename "$LIST"); [ -f "$listf" ] || listf=$LIST
S=${_CONDOR_SCRATCH_DIR:-$PWD}/h5job; mkdir -p "$S"; NAP=${H5_RETRY_SLEEP:-60}
size_of() { ifdh ll "$1" 0 2>/dev/null | awk '$1 ~ /^-/ && NF >= 9 {print $5; exit}'; }
fetch() { local src=$1 dst=$2/$(basename "$1") want got k
  for k in 1 2 3; do rm -f "$dst"; want=$(size_of "$src"); got=
    if [ -n "$want" ]; then ifdh cp "$src" "$dst" > /dev/null 2>&1; got=$(stat -c %s "$dst" 2>/dev/null); [ "$got" = "$want" ] && return 0; fi
    echo "  fetch try $k of $(basename "$src"): source ${want:-?} copy ${got:-none}"; sleep $((NAP*k)); done; return 1; }
put() { local want got k; want=$(stat -c %s "$1")
  for k in 1 2 3; do ifdh cp "$1" "$2" > /dev/null 2>&1; got=$(size_of "$2"); [ "$got" = "$want" ] && return 0
    echo "  put try $k of $(basename "$2"): local $want dest ${got:-none}"; [ -n "$got" ] && ifdh rm "$2" > /dev/null 2>&1; sleep $((NAP*k)); done; return 1; }
n=0; nok=0; nskip=0
grep -v '^#' "$listf" | sed -n "$((P*K+1)),$((P*K+K))p" > "$S/my.tsv"; echo "pairs in this job: $(wc -l < "$S/my.tsv")"
# blobdump <step-1 tar> <out dir> <name>: WCT 0.38.0 env in a clean shell (the h5 env above is a different UPS setup)
blobdump() { env -i HOME="${HOME:-/tmp}" USER="${USER:-}" PATH=/usr/bin:/bin TMPDIR="${TMPDIR:-/tmp}" WC_TOP="$WCT" \
  bash -c 'set +u; source "$WC_TOP/sbnd-wirecell-production/aurora/env/fnal/setup-fnal-run.sh" > "$2/$3.setup.log" 2>&1
    command -v wire-cell > /dev/null || { echo "WCT setup failed"; tail -5 "$2/$3.setup.log"; exit 3; }
    bash "$4" "$1" "$2" data "$3"' _ "$1" "$2" "$3" "$H/maker/wct/blobdump.sh"; }
while IFS=$'\t' read -r id sample r1 tar nev qls; do
  [ -z "$id" ] && continue; n=$((n+1)); t=$(date +%s); D=$OUTBASE/$sample; B=sbnd_${sample}_$id
  echo "=== pair $n $id ($sample, $nev events) $(date '+%F %T')"
  [ -n "$(size_of "$D/$id.done")" ] && { echo "  done at the destination, skipped"; nok=$((nok+1)); nskip=$((nskip+1)); continue; }
  in=$S/in; o=$S/out; rm -rf "$in" "$o"; mkdir -p "$in" "$o"
  read -r -a qa <<< "${qls:-}"; [ ${#qa[@]} -gt 0 ] || { echo "  NO STEP-1 TAR in the list"; continue; }
  ok=1; fetch "$r1" "$in" && fetch "$tar" "$in" || ok=0
  for q in "${qa[@]}"; do [ $ok = 1 ] && { fetch "$q" "$in" || ok=0; }; done
  [ $ok = 1 ] || { echo "  FETCH FAILED"; continue; }
  # blob dumps: one per step-1 tar (a PARTIAL pair has one per segment: <id>.<seg>.blobs.tar.gz)
  mkdir -p "$o/logs" "$o/dump"; brc=0; tb=$(date +%s)
  for q in "${qa[@]}"; do qb=$(basename "$q"); seg=${qb#$id.qlpctree}; seg=${seg%.tar.gz}
    blobdump "$in/$qb" "$o/dump" "$id$seg" > "$o/logs/blobdump$seg.out" 2>&1 || { brc=$?; echo "  blobdump rc $brc on $qb"; }
    mv "$o/dump/$id$seg.setup.log" "$o/logs/" 2>/dev/null; done
  for wd in "$o"/dump/.wct.*; do [ -d "$wd" ] && cp "$wd/wct.log" "$o/logs/blobdump-$(basename "$wd").wct.log"; done
  echo "  blobdump: ${#qa[@]} tar(s), rc $brc, $(( $(date +%s)-tb )) s"
  if [ $brc = 0 ]; then
    SBNDH5_BLOBDUMPS="$(ls "$o"/dump/*.blobs.tar.gz | tr '\n' ' ')" bash "$H/scripts/h5pair.sh" "$in/$(basename "$r1")" "$in/$(basename "$tar")" "$o/$B.h5" 0 > "$o/h5pair.out" 2>&1; rc=$?
  else echo "FAIL blobdump rc $brc" > "$o/h5pair.out"; rc=15; fi
  st=$(grep -E '^(OK|FAIL)' "$o/h5pair.out" | tail -1); echo "  rc $rc: ${st:-no status} ($(( $(date +%s)-t )) s)"
  (cd "$o" && tar -czf "$S/$id.logs.tar.gz" h5pair.out logs) 2>/dev/null
  failed=0; files=()
  if [ $rc = 0 ]; then
    for f in "$B.h5" "$B.h5.stats.json" "$B.check.json"; do
      have=$(size_of "$D/$f"); if [ -n "$have" ] && [ "$have" = "$(stat -c %s "$o/$f")" ]; then echo "  $f already there"; else
        [ -n "$have" ] && ifdh rm "$D/$f" > /dev/null 2>&1; put "$o/$f" "$D/$f" || { echo "  COPY FAILED $f"; failed=1; }; fi
      files+=("$o/$f"); done
  fi
  ln=logs/$id.logs.tar.gz; k=2; while [ -n "$(size_of "$D/$ln")" ]; do ln=logs/$id.logs.try$k.tar.gz; k=$((k+1)); done
  put "$S/$id.logs.tar.gz" "$D/$ln" || { echo "  COPY FAILED $ln"; failed=1; }
  if [ $rc = 0 ] && [ $failed = 0 ]; then
    { echo "status	$st"; echo "job	${CLUSTER:-}.${PROCESS:-} ${JOBSUBJOBID:-} index $P host $(hostname) $(date '+%F %T')"; echo "logs	$ln"
      for f in "${files[@]}"; do echo "file	$(stat -c %s "$f")	$(sha256sum "$f" | cut -d' ' -f1)	$(basename "$f")"; done; } > "$S/$id.done"
    put "$S/$id.done" "$D/$id.done" && nok=$((nok+1)) || echo "  COPY FAILED $id.done"
  fi
  rm -rf "$in" "$o" "$S/$id.logs.tar.gz" "$S/$id.done"
done < "$S/my.tsv"
echo "job done $(date '+%F %T'): $nok/$n pairs ($nskip already there), $(( $(date +%s)-T0 )) s"
[ "$n" -gt 0 ] && [ "$nok" -eq "$n" ]
