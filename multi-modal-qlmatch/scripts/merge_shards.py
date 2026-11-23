#!/usr/bin/env python
"""Merge consecutive converted shards of one sample into larger shards (same column files, same event order).

Why: every memory-mapped column holds a file descriptor, and each data-loader worker maps the columns of every
shard it touches. The full SBND sample converts to 1,781 shards (~14 mapped columns each), which exceeds the
per-process limit (16,384). Merging groups of N shards keeps the global event order, so existing split files
(indices into the corpus) stay valid.

Usage: merge_shards.py SRC_SAMPLE_DIR DST_SAMPLE_DIR [--group 30] [--jobs 8]
Output shards: DST/<name>_m000_p0, _m001_p0, ... (sorted order = source order).
"""
import argparse
import glob
import json
import os
import shutil
from multiprocessing import Pool

import numpy as np

OFFSETS = {"ev_hit_off.npy": "n_hits", "ev_oph_off.npy": "n_ophits"}


def merge(job):
    srcs, dst = job
    if os.path.exists(os.path.join(dst, "meta.json")):
        return dst, "exists"
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    metas = [json.load(open(os.path.join(s, "meta.json"))) for s in srcs]
    cols = sorted(f for f in os.listdir(srcs[0]) if f.endswith(".npy"))
    for s in srcs[1:]:
        assert sorted(f for f in os.listdir(s) if f.endswith(".npy")) == cols, f"column set differs in {s}"
    for c in cols:
        parts = [np.load(os.path.join(s, c)) for s in srcs]
        if c in OFFSETS:
            base = np.cumsum([0] + [m[OFFSETS[c]] for m in metas[:-1]])
            parts = [p.astype(np.int64) + b for p, b in zip(parts, base)]
        np.save(os.path.join(tmp, c), np.concatenate(parts))
    meta = dict(n_events=sum(m["n_events"] for m in metas), n_hits=sum(m["n_hits"] for m in metas),
                n_ophits=sum(m["n_ophits"] for m in metas), edep_rows=sum(m.get("edep_rows", 0) for m in metas),
                hit_id_mismatch=sum(m.get("hit_id_mismatch", 0) for m in metas),
                neg_hit_time=sum(m.get("neg_hit_time", 0) for m in metas),
                min_ophit_t=min(m.get("min_ophit_t", 0) for m in metas),
                max_ophit_t=max(m.get("max_ophit_t", 0) for m in metas),
                merged_from=[os.path.abspath(s) for s in srcs], sources=[m.get("source") for m in metas])
    json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"), indent=1)
    os.rename(tmp, dst)
    return dst, meta["n_events"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--group", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    shards = sorted(d for d in glob.glob(os.path.join(a.src, "*")) if os.path.exists(os.path.join(d, "meta.json")))
    name = os.path.basename(os.path.normpath(a.dst))
    os.makedirs(a.dst, exist_ok=True)
    jobs = [(shards[i:i + a.group], os.path.join(a.dst, f"{name}_m{k:03d}_p0"))
            for k, i in enumerate(range(0, len(shards), a.group))]
    print(f"{len(shards)} shards -> {len(jobs)} merged shards in {a.dst}", flush=True)
    with Pool(a.jobs) as p:
        for dst, n in p.imap_unordered(merge, jobs):
            print(dst, n, flush=True)


if __name__ == "__main__":
    main()
