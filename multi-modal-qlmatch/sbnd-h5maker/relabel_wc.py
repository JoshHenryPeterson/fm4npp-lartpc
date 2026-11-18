#!/usr/bin/env python3
"""Rewrite wc_hit_table of an existing sbnd_h5maker file from the Wire-Cell tracking-pr.root footprints.

    python relabel_wc.py FILE.h5 [FILE.h5 ...]

Needs h5py, numpy, uproot and awkward, but not ROOT/gallery. Uses the hit_table (channel, tpc, local_plane,
local_time) and the wc_dirs / detector / wc_tick_offset / wc_ticks_per_slice attributes of the file, so the
result is the same as a fresh conversion with this version of sbnd_h5maker.py. wc_event_table is unchanged.
Made for the 0.3 -> 0.4 fix of the WC channel numbering; sets the creator attribute to "<old> + relabel_wc <VERSION>".
"""
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbnd_h5maker import VERSION, WC  # noqa: E402


def col(g, c):
    a = g[c][:]
    return a[:, 0] if a.ndim == 2 and a.shape[1] == 1 else a


def relabel(path):
    with h5py.File(path, "r+") as f:
        det = json.loads(f.attrs["detector"])
        wc = WC(json.loads(f.attrs["wc_dirs"]), float(f.attrs["wc_tick_offset"]),
                int(f.attrs.get("wc_ticks_per_slice", 4)), det)
        ev = f["event_table/event_id"][:]
        h = f["hit_table"]
        sc = h["event_id.seq_cnt"][:]
        cnt = np.zeros(len(ev), np.int64)
        cnt[sc[:, 0]] = sc[:, 1]
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        wcch = wc.wc_channel(col(h, "channel"), col(h, "tpc"), col(h, "local_plane"))
        tick = col(h, "local_time")
        n = len(tick)
        out = {k: np.full(n, -1, np.int32) for k in ("wc_cluster_id", "wc_nu_index")}
        out.update({k: np.zeros(n, np.int32) for k in ("wc_main", "wc_in_enu")})
        for i, rse in enumerate(ev):
            if not cnt[i]:
                continue
            s = slice(start[i], start[i] + cnt[i])
            status, _, fp = wc.event(tuple(int(x) for x in rse))
            r = wc.hit_labels(fp, wcch[s], tick[s])
            for k, v in zip(("wc_cluster_id", "wc_nu_index", "wc_main", "wc_in_enu"), r):
                out[k][s] = v
        g = f["wc_hit_table"]
        for k, v in out.items():
            assert g[k].shape[0] == n
            g[k][:, 0] = v
        f.attrs["wc_channel_offsets"] = json.dumps(wc.ch_off.tolist())
        f.attrs["creator"] = "%s + relabel_wc %s" % (f.attrs["creator"], VERSION)
        print("%s: %d events, %d hits, %d on WC nu clusters" % (path, len(ev), n, int((out["wc_cluster_id"] >= 0).sum())),
              flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        relabel(p)
