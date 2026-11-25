#!/bin/bash
# sbnd-h5maker 0.6 (fm4npp #16): the post-split Wire-Cell point-cloud tree of one unit, for --blob-dir.
#   blobdump.sh <step-1 tar | unit dir> <out dir> <sim|data> [name]
# Runs WCT (0.38.0, the release of the step-2 pass the wc_hit_table labels come from) with wct-blobdump.jsonnet:
# step 2's cluster-splitting visitors (switch_scope ... protect_bundle), no neutrino tagger / DL vertex / tracking,
# and writes <out>/<name>.blobs.tar.gz (unit dir with one ql/<seg>/qlpctree.tar.gz) or
# <out>/<name>.blobs.<seg>.tar.gz (several segments, a PARTIAL unit).  <name> defaults to the unit dir / tar basename.
# Needs the WCT run environment (wire-cell on PATH, WIRECELL_PATH with the 0.38.0 cfg); this directory is
# prepended to WIRECELL_PATH.
set -u
IN=${1:?step-1 tar or unit dir}; OUT=${2:?out dir}; REAL=${3:?sim|data}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WIRECELL_PATH=$HERE${WIRECELL_PATH:+:$WIRECELL_PATH}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
if [ -d "$IN" ]; then
  NAME=${4:-$(basename "$IN")}; mapfile -t TARS < <(ls "$IN"/ql/*/qlpctree.tar.gz 2>/dev/null)
else
  NAME=${4:-$(basename "$IN" .qlpctree.tar.gz)}; TARS=("$IN")
fi
[ ${#TARS[@]} -gt 0 ] || { echo "blobdump.sh: no step-1 tar under $IN" >&2; exit 2; }
rc=0
for T in "${TARS[@]}"; do
  if [ ${#TARS[@]} -gt 1 ]; then seg=$(basename "$(dirname "$T")"); O=$OUT/$NAME.blobs.$seg.tar.gz; else O=$OUT/$NAME.blobs.tar.gz; fi
  W=$(mktemp -d "$OUT/.wct.XXXXXX")
  ( cd "$W" && wire-cell -l stdout -L info -c wct-blobdump.jsonnet --tla-str input="$T" --tla-str reality="$REAL" \
      --tla-str out="$O.tmp" > wct.log 2>&1 )
  r=$?
  if [ $r -eq 0 ] && [ -s "$O.tmp" ]; then mv "$O.tmp" "$O"; rm -rf "$W"
  else echo "blobdump.sh: wire-cell rc $r on $T (log kept in $W)" >&2; rm -f "$O.tmp"; rc=1; fi
done
exit $rc
