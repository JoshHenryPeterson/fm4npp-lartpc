#!/bin/bash
# fm4npp #14 option B: copy one WCLocal2026Nov16 tracking-pr tar from scratch to persistent and verify ADLER32.
# usage: copy-trpr.sh <scratch tar path>   -> prints: status <tab> dest
src=$1; P=/pnfs/sbnd/persistent/users/yuhw/prod-fermigrid/WCLocal2026Nov16
rel=${src#/pnfs/sbnd/scratch/users/yuhw/prod-fermigrid/WCLocal2026Nov16/}; dst=$P/$rel
ck() { cat "$(dirname "$1")/.(get)($(basename "$1"))(checksum)" 2>/dev/null | tr -d '\n'; }
mkdir -p "$(dirname "$dst")"
ss=$(stat -c %s "$src")
if [ -e "$dst" ] && [ "$(stat -c %s "$dst")" != "$ss" ]; then echo -e "SIZEMISMATCH_EXISTING\t$dst"; exit 1; fi
[ -e "$dst" ] || cp "$src" "$dst" || { echo -e "CPFAIL\t$dst"; exit 1; }
for i in 1 2 3 4 5 6 7 8 9 10; do a=$(ck "$src"); b=$(ck "$dst"); [ -n "$b" ] && break; sleep 3; done
if [ -n "$a" ] && [ "$a" = "$b" ]; then echo -e "OK\t$dst\t$b"; else echo -e "CKFAIL\t$dst\t$a\t$b"; fi
