#!/usr/bin/env python3
"""Merge per-file HDF5Maker outputs (+ WC labels) of one round-1 sample into one file that
`qlfm/data/h5reader.py` reads (ph5concat layout: per table `event_id` (N,3) and
`event_id.seq_cnt` (n_nonempty, 2) = [event sequence number, row count]).

  python merge_round1.py SAMPLE_DIR OUT.h5
SAMPLE_DIR has h5/<name>.h5 (HDF5Maker) and optionally wc/<name>_wc.h5 (wc_hit_labels.py).

WC labels become extra columns (defaults where a file has no WC labels, e.g. WC output on tape):
  hit_table:   wc_cluster_id (-1 = on no WC cluster), wc_main (in the WC in-time main cluster)
  event_table: wc_in_nusel, wc_has_intime_main, wc_in_checkout, wc_match_found, wc_match_isFC,
               wc_generic_nu, wc_numu_score, wc_nue_score, wc_stm_*
WC neutrino hit label = hit_table/wc_main & event_table/wc_generic_nu (of the hit's event).
"""
import glob
import os
import sys

import h5py
import numpy as np

K = np.array([1 << 40, 1 << 20, 1], dtype=np.int64)
HIT_WC = {"wc_cluster_id": (np.int32, -1), "wc_main": (bool, False)}
DTYPES = {}
EVENT_WC = {"in_nusel": bool, "has_intime_main": bool, "in_checkout": bool, "match_found": bool,
            "match_isFC": bool, "generic_nu": bool, "numu_score": np.float32, "nue_score": np.float32,
            "stm_eventtype": np.int32, "stm_lowenergy": np.int32, "stm_LM": np.int32, "stm_TGM": np.int32,
            "stm_STM": np.int32, "stm_FullDead": np.int32, "stm_clusterlength": np.float32}


def per_file(fn):
    """{table: {column: array}} for one HDF5Maker file, with the WC columns added."""
    f = h5py.File(fn, "r")
    t = {g: {c: f[g][c][:] for c in f[g] if not c.startswith("event_id.")} for g in f}
    for g in f:                                   # remember on-disk dtypes (vlen strings come back as object)
        for c in f[g]:
            DTYPES.setdefault((g, c), f[g][c].dtype)
    nhit, nev = len(t["hit_table"]["event_id"]), len(t["event_table"]["event_id"])
    wfn = fn.replace("/h5/", "/wc/")[:-3] + "_wc.h5"
    W = h5py.File(wfn, "r") if os.path.exists(wfn) else None
    for c, (dt, dv) in HIT_WC.items():
        t["hit_table"][c] = np.full(nhit, dv, dt)
    for c, dt in EVENT_WC.items():
        t["event_table"]["wc_" + c] = np.zeros(nev, dt)
    if W is not None:
        assert np.array_equal(W["hit_wc/event_id"][:], t["hit_table"]["event_id"]), fn
        assert np.array_equal(W["event_wc/event_id"][:], t["event_table"]["event_id"]), fn
        for c in HIT_WC:
            t["hit_table"][c] = W["hit_wc/" + c][:].astype(HIT_WC[c][0])
        for c, dt in EVENT_WC.items():
            if c in W["event_wc"]:
                t["event_table"]["wc_" + c] = W["event_wc/" + c][:].astype(dt)
    return t, W is not None


def main():
    sdir, out = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(f"{sdir}/h5/*.h5"))
    parts, nwc = [], 0
    for fn in files:
        t, has = per_file(fn)
        parts.append(t)
        nwc += has
    ev = np.concatenate([p["event_table"]["event_id"] for p in parts]).astype(np.int64)
    evkey = ev @ K
    seq_of = {int(k): i for i, k in enumerate(evkey)}
    assert len(seq_of) == len(ev), "duplicate events across files"
    with h5py.File(out, "w") as o:
        for tab in parts[0]:
            g = o.create_group(tab)
            for c in parts[0][tab]:
                a = np.concatenate([p[tab][c] for p in parts])
                if a.ndim == 1:
                    a = a[:, None]
                dt = DTYPES.get((tab, c), a.dtype)
                if len(a):
                    g.create_dataset(c, data=a, dtype=dt, compression="gzip", compression_opts=4)
                else:
                    g.create_dataset(c, shape=a.shape, dtype=dt)
            eid = np.concatenate([p[tab]["event_id"] for p in parts]).astype(np.int64)
            sc = np.zeros((0, 2), np.int64)
            if len(eid):
                seq = np.array([seq_of[int(k)] for k in eid @ K], np.int64)
                assert np.all(np.diff(seq) >= 0), f"{tab}: rows not ordered by event"
                u, cnt = np.unique(seq, return_counts=True)
                sc = np.stack([u, cnt], 1)
            g.create_dataset("event_id.seq_cnt", data=sc)
    nhit = sum(len(p["hit_table"]["event_id"]) for p in parts)
    print(f"{out}: {len(files)} files ({nwc} with WC labels), {len(ev)} events, {nhit} hits")


if __name__ == "__main__":
    main()
