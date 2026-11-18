#!/usr/bin/env python3
"""Split an aurora-r2 sample into converter part lists of ~TARGET events (fm4npp-lartpc #11).

    python make_parts.py UNITS_TSV SAMPLE_DIR OUTDIR [--target 500]

UNITS_TSV: production-prep/dlspace/lists/<s>-units.tsv (node, unit, status, verdict, ...). Units with status OK or
PARTIAL are kept (FAIL units have no Wire-Cell output). Event counts come from each unit's reco1_files.json.
Writes OUTDIR/units.txt (unit dir, events) and OUTDIR/parts/part-NNNN.txt.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("tsv")
ap.add_argument("sample_dir")
ap.add_argument("outdir")
ap.add_argument("--target", type=int, default=500)
a = ap.parse_args()
units = []
for l in open(a.tsv):
    if l.startswith("#"):
        continue
    node, unit, status = l.rstrip("\n").split("\t")[:3]
    if status not in ("OK", "PARTIAL"):
        continue
    d = os.path.join(a.sample_dir, node, unit)
    n = sum(int(x.get("events", 0)) for x in json.load(open(os.path.join(d, "reco1_files.json"))))
    units.append((d, n))
os.makedirs(os.path.join(a.outdir, "parts"), exist_ok=True)
with open(os.path.join(a.outdir, "units.txt"), "w") as f:
    for d, n in units:
        f.write("%s\t%d\n" % (d, n))
parts, cur, ne = [], [], 0
for d, n in units:
    cur.append(d)
    ne += n
    if ne >= a.target:
        parts.append(cur)
        cur, ne = [], 0
if cur:
    parts.append(cur)
for i, p in enumerate(parts):
    with open(os.path.join(a.outdir, "parts", "part-%04d.txt" % i), "w") as f:
        f.write("\n".join(p) + "\n")
print("%s: %d units, %d events, %d parts" % (a.outdir, len(units), sum(n for _, n in units), len(parts)))
