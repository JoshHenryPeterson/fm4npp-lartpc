#!/usr/bin/env python
"""Add the sbnd-h5maker 0.6 columns (#16: Wire-Cell blobs, hit-blob pairs, G4 tracks, per-hit truth position) to an
existing merged SBND cache, in place, without re-converting the hits (the 0.5 datasets in the v06 files are equal by
value to the files the cache was built from; checked here event by event: run/subrun/event and hits per event).

Per merged shard: its meta.json "sources" (full/<name>.h5) -> v06/<name>.h5 (--src-sub), qlfm.data.convert.
convert_blobs on each source in memory, concatenated with global per-event offsets, saved into the shard; meta.json
gains n_blobs / n_hit_blob / n_tracks; marker blobs_v06.done. Reruns skip done shards.

Usage (QLFM_DETECTOR=sbnd, on a compute node): add_blobs_v06.py SAMPLE_CACHE_DIR [--jobs 8] [--src-sub v06] [--only N]
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

from qlfm.data.convert import convert_blobs
from qlfm.data.h5reader import OpenSampleFile


def offsets(cnt):
    return np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)


def add(job):
    d, sub = job
    done = os.path.join(d, "blobs_v06.done")
    if os.path.exists(done):
        return d, "skip"
    meta = json.load(open(os.path.join(d, "meta.json")))
    run, subrun, event = (np.load(os.path.join(d, f"ev_{k}.npy")) for k in ("run", "subrun", "event"))
    hcnt = np.load(os.path.join(d, "ev_hit_cnt.npy")).astype(np.int64)
    parts, e0 = [], 0
    for src in meta["sources"]:
        path = src.replace("/full/", f"/{sub}/")
        f = OpenSampleFile(path)
        n = f.n_events
        eid = f.f["event_table/event_id"][:]
        assert (eid[:, 0] == run[e0:e0 + n]).all() and (eid[:, 1] == subrun[e0:e0 + n]).all() \
            and (eid[:, 2] == event[e0:e0 + n]).all(), f"{path}: events differ from the cache"
        hs, hc = f.offsets("hit_table")
        assert (hc == hcnt[e0:e0 + n]).all(), f"{path}: hits per event differ from the cache"
        parts.append(convert_blobs(f, None, 0, n, int(hc.sum()), hs - hs[0]))
        f.close()
        e0 += n
    assert e0 == meta["n_events"], (d, e0, meta["n_events"])
    out = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    ev = {}
    for name in ("blob", "hb", "trk"):
        if f"{name}_cnt" in parts[0][1]:
            cnt = np.concatenate([p[1][f"{name}_cnt"] for p in parts])
            ev[f"{name}_cnt"], ev[f"{name}_off"] = cnt.astype(np.int32), offsets(cnt)
    if "hit_txyz" in out:
        assert len(out["hit_txyz"]) == meta["n_hits"]
    for k, v in list(out.items()) + [(f"ev_{k}", v) for k, v in ev.items()]:
        np.save(os.path.join(d, f"{k}.tmp.npy"), v)
        os.replace(os.path.join(d, f"{k}.tmp.npy"), os.path.join(d, f"{k}.npy"))
    stats = dict(n_blobs=int(len(out["blob_x"])), n_hit_blob=int(len(out["hb_hit"])),
                 n_tracks=int(len(out.get("trk_trackid", []))), sources=len(meta["sources"]))
    meta.update({k: v for k, v in stats.items() if k != "sources"})
    json.dump(meta, open(os.path.join(d, "meta.json.tmp"), "w"), indent=1)
    os.replace(os.path.join(d, "meta.json.tmp"), os.path.join(d, "meta.json"))
    json.dump(stats, open(done, "w"))
    return d, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--src-sub", default="v06")
    ap.add_argument("--only", type=int, default=None, help="first N shards (tests)")
    a = ap.parse_args()
    shards = sorted(d for d in glob.glob(os.path.join(a.sample, "*_p0")) if os.path.exists(os.path.join(d, "meta.json")))
    if a.only:
        shards = shards[:a.only]
    with Pool(a.jobs) as p:
        for d, st in p.imap_unordered(add, [(d, a.src_sub) for d in shards]):
            print(os.path.basename(d), st, flush=True)


if __name__ == "__main__":
    main()
