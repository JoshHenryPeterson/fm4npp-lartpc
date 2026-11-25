#!/usr/bin/env python
"""Refresh the Wire-Cell label columns of a merged SBND qlfm cache from rebuilt HDF5 files (#14, option B: the
WCT 0.38.0 files in <sample>/wcl1116/ have the same events, hits and order as full/; only the WC tables differ).

Per merged shard (meta.json "sources" = the full/ files it was built from, in order):
  hit_pnd_nu.npy      <- WC nu-candidate footprint, wc_hit_table.wc_nu_index >= 0 (as convert.py)
  hit_pnd_nu_ct0.npy  <- the same, restricted to candidates that pass the PR cosmic tagger (wc_event_table.cosmict_flag == 0)
  hit_pnd_nu_r2.npy   <- the previous (aurora-r2) hit_pnd_nu, kept once for reference
A marker file wc_labels_<tag>.done makes reruns skip finished shards.
--tag full (real data, whose full/ files are already WCT 0.38.0): reads the sources as they are, checks that the
footprint equals the converter's hit_pnd_nu and only adds hit_pnd_nu_ct0 (no _r2 copy).

Usage: refresh_wc_labels.py SAMPLE_CACHE_DIR [--tag wcl1116|full] [--jobs 3]
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import h5py
import numpy as np


def offsets(f, table, n):
    sc = f[f"{table}/event_id.seq_cnt"][:]
    cnt = np.zeros(n, np.int64)
    cnt[sc[:, 0]] = sc[:, 1]
    return cnt


def labels(path):
    with h5py.File(path, "r") as f:
        n = f["event_table/event_id"].shape[0]
        hc, wc = offsets(f, "hit_table", n), offsets(f, "wc_hit_table", n)
        assert (hc == wc).all(), f"{path}: wc_hit_table rows != hit_table rows per event"
        nu = f["wc_hit_table/wc_nu_index"][:].ravel().astype(np.int64)
        hid = f["wc_hit_table/hit_id"][:].ravel()
        evt = np.repeat(np.arange(n), wc)
        start = np.concatenate([[0], np.cumsum(wc)[:-1]])
        assert (hid == np.arange(len(hid)) - start[evt]).all(), f"{path}: wc_hit_table not in hit order"
        pnd = nu >= 0
        ec = offsets(f, "wc_event_table", n)
        cevt = np.repeat(np.arange(n), ec)
        cnu = f["wc_event_table/nu_index"][:].ravel().astype(np.int64)
        good = f["wc_event_table/cosmict_flag"][:].ravel() == 0
        keys = np.unique(cevt[good] * 100000 + cnu[good])
        ct0 = pnd & np.isin(evt * 100000 + nu, keys)
    return pnd.astype(np.uint8), ct0.astype(np.uint8)


def refresh(job):
    d, tag = job
    done = os.path.join(d, f"wc_labels_{tag}.done")
    if os.path.exists(done):
        return d, "skip"
    meta = json.load(open(os.path.join(d, "meta.json")))
    parts = [labels(src.replace("/full/", f"/{tag}/")) for src in meta["sources"]]
    pnd = np.concatenate([p for p, _ in parts]); ct0 = np.concatenate([c for _, c in parts])
    assert len(pnd) == meta["n_hits"], f"{d}: {len(pnd)} hits vs {meta['n_hits']}"
    if tag == "full":
        cur = np.load(os.path.join(d, "hit_pnd_nu.npy"))
        assert (cur == pnd).all(), f"{d}: footprint differs from the converted hit_pnd_nu"
        np.save(os.path.join(d, "hit_pnd_nu_ct0.tmp.npy"), ct0)
        os.replace(os.path.join(d, "hit_pnd_nu_ct0.tmp.npy"), os.path.join(d, "hit_pnd_nu_ct0.npy"))
        stats = dict(n_hits=int(len(pnd)), frac_new=float(pnd.mean()), frac_ct0=float(ct0.mean()))
        json.dump(stats, open(done, "w"))
        return d, stats
    old = os.path.join(d, "hit_pnd_nu.npy"); keep = os.path.join(d, "hit_pnd_nu_r2.npy")
    if not os.path.exists(keep):
        os.replace(old, keep)
    prev = np.load(keep, mmap_mode="r")
    for name, arr in (("hit_pnd_nu", pnd), ("hit_pnd_nu_ct0", ct0)):
        np.save(os.path.join(d, name + ".tmp.npy"), arr)
        os.replace(os.path.join(d, name + ".tmp.npy"), os.path.join(d, name + ".npy"))
    stats = dict(n_hits=int(len(pnd)), frac_r2=float(np.mean(prev)), frac_new=float(pnd.mean()),
                 frac_ct0=float(ct0.mean()), agree_r2=float(np.mean(prev == pnd)))
    json.dump(stats, open(done, "w"))
    return d, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample")
    ap.add_argument("--tag", default="wcl1116")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    shards = sorted(d for d in glob.glob(os.path.join(a.sample, "*_p0")) if os.path.exists(os.path.join(d, "meta.json")))
    with Pool(a.jobs) as p:
        for d, st in p.imap_unordered(refresh, [(d, a.tag) for d in shards]):
            print(os.path.basename(d), st, flush=True)


if __name__ == "__main__":
    main()
