#!/bin/bash
# Round-1 production: HDF5Maker on each input artroot, then WC labels from its nusel files.
#   bash run_round1.sh SAMPLE convert|label
# SAMPLE: bnb_overlay | nue_overlay | beam_on | ext   (lists in ../datasets/uboone_round1/)
# Files still on tape (NEARLINE only) are skipped. Rerun to pick them up. Existing outputs are kept.
# Output: $OUT/SAMPLE/h5/<input>.h5 (HDF5Maker) and $OUT/SAMPLE/wc/<input>_wc.h5 (wc_hit_labels.py)
set -u
SAMPLE=$1; STAGE=$2
WORK=${WORK:-/tmp/$USER/fm4npp_round1}
OUT=${OUT:-/pnfs/uboone/persistent/users/yuhw/fm4npp_round1}
PYLIB=${PYLIB:-/exp/uboone/app/users/yuhw/pylib-uproot}
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=$H/../datasets/uboone_round1
case $SAMPLE in
  *overlay) FCL=hdf5maker_uB_public-nowire_job.fcl ;;
  *)        FCL=hdf5maker_uB_data-nowire_job.fcl ;;
esac
CK=/pnfs/uboone/persistent/users/jjo/processed_chekcout_rootfiles/bnb/run1
case $SAMPLE in
  bnb_overlay) CHECKOUT=$CK/cv/checkout_prodgenie_bnb_nu_overlay_run1_PF.root ;;
  nue_overlay) CHECKOUT=$CK/cv/checkout_prodgenie_bnb_intrinsic_nue_overlay_run1_PF.root ;;
  beam_on)     CHECKOUT=$CK/data/checkout_data_bnb_run1_PF.root ;;
  ext)         CHECKOUT=$CK/data/checkout_data_extbnb_run1_PF.root ;;
esac
online() { [[ $(cat "$(dirname "$1")/.(get)($(basename "$1"))(locality)") == *ONLINE* ]]; }
mkdir -p "$WORK/$SAMPLE" "$OUT/$SAMPLE/h5" "$OUT/$SAMPLE/wc"

if [ "$STAGE" = convert ]; then
  todo=()
  while read -r p; do
    n=$(basename "$p" .root)
    [ -e "$OUT/$SAMPLE/h5/$n.h5" ] && continue
    online "$p" || { echo "tape: $n"; continue; }
    todo+=("$p")
  done < "$D/${SAMPLE}_inputs_paths.txt"
  echo "$SAMPLE: converting ${#todo[@]} files"
  [ ${#todo[@]} -eq 0 ] && exit 0
  printf '%s\n' "${todo[@]}" > "$WORK/$SAMPLE/todo.txt"
  bash "$H/sl7.sh" "source $H/setup_env.sh >/dev/null 2>&1
    while read -r p; do
      n=\$(basename \$p .root); d=$WORK/$SAMPLE/\$n; mkdir -p \$d && cd \$d
      if lar -c $FCL -s \$p > lar.log 2>&1; then
        mv NeutrinoML_*.h5 \$n.h5 && cp \$n.h5 $OUT/$SAMPLE/h5/ && cd .. && rm -rf \$d && echo ok \$n
      else echo FAIL \$n; tail -3 lar.log; fi
    done < $WORK/$SAMPLE/todo.txt"
elif [ "$STAGE" = label ]; then
  PYTHONPATH=$PYLIB python3 - "$SAMPLE" "$D" "$OUT" "$CHECKOUT" "$H/../scripts/wc_hit_labels.py" <<'EOF'
import json, os, subprocess, sys
sample, D, OUT, checkout, script = sys.argv[1:]
sel = json.load(open(f"{D}/selection.json"))[sample]
npath = {os.path.basename(p): p for p in open(f"{D}/{sample}_nusel_paths.txt").read().split()}
def online(p):
    return "ONLINE" in open(os.path.join(os.path.dirname(p), f".(get)({os.path.basename(p)})(locality)")).read()
for r in sel:
    n = (r.get("public") or r.get("reco2"))[:-5]
    h5, out = f"{OUT}/{sample}/h5/{n}.h5", f"{OUT}/{sample}/wc/{n}_wc.h5"
    if os.path.exists(out) or not os.path.exists(h5):
        continue
    nus = [npath[x] for x, _ in r["nusel"]]
    if not nus or not all(online(p) for p in nus):
        print("nusel on tape:", n); continue
    tmp = f"/tmp/{os.environ['USER']}_{n}_wc.h5"
    rc = subprocess.call([sys.executable, script, tmp, "--h5", h5, "--checkout", checkout, "--nusel", *nus])
    if rc == 0:
        subprocess.check_call(["cp", tmp, out]); os.remove(tmp); print("ok", n)
    else:
        print("FAIL", n)
EOF
fi
