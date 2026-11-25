#!/usr/bin/env python
"""Test utility (#16): copy the first N events of one converted SBND shard and add SYNTHETIC blob columns, so the
blob / 3D code paths can be smoke-tested before the sbnd-h5maker 0.6 files exist. Not physics.

Blobs: collection-plane (Y) hits grouped per (TPC, 8 wires, 16 ticks); centre x = drift coordinate, z = wire
coordinate, y = 0; charge = 100 x summed integral. Hit-blob pairs: the Y hits of each blob. Per-hit truth position:
(x, 0, z) for Y hits, NaN otherwise. Writes the cache layout that qlfm.data.convert writes for 0.6 files.

Usage: QLFM_DETECTOR=sbnd make_fake_blob_cache.py SRC_SHARD DST_SAMPLE_DIR [--n 64]
"""
import argparse
import json
import os

import numpy as np

from qlfm import geometry as G

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--n", type=int, default=64)
a = ap.parse_args()
dst = os.path.join(a.dst, "fake_p0")
os.makedirs(dst, exist_ok=True)
L = lambda k: np.load(os.path.join(a.src, k + ".npy"), mmap_mode="r")
n = a.n
hc = np.asarray(L("ev_hit_cnt")[:n]).astype(np.int64)
oc = np.asarray(L("ev_oph_cnt")[:n]).astype(np.int64)
nh, no = int(hc.sum()), int(oc.sum())
for f in sorted(os.listdir(a.src)):
    if not f.endswith(".npy"):
        continue
    k = f[:-4]
    x = L(k)
    if k.startswith("ev_"):
        v = np.asarray(x[:n])
    elif k.startswith("hit_"):
        v = np.asarray(x[:nh])
    elif k.startswith("oph_"):
        v = np.asarray(x[:no])
    else:
        continue
    if k in ("ev_hit_off", "ev_oph_off"):
        c = hc if k == "ev_hit_off" else oc
        v = np.concatenate([[0], np.cumsum(c)[:-1]]).astype(v.dtype)
    np.save(os.path.join(dst, f), v)

plane = np.load(os.path.join(dst, "hit_plane.npy")).astype(np.int64)
wire = np.load(os.path.join(dst, "hit_wire.npy")).astype(np.int64)
tick = np.load(os.path.join(dst, "hit_time.npy")).astype(np.float64) / 8
q = np.load(os.path.join(dst, "hit_integral.npy")).astype(np.float64)
tpc = np.load(os.path.join(dst, "hit_tpc.npy")).astype(np.int64)
evt = np.repeat(np.arange(n), hc)
local = np.arange(nh) - np.repeat(np.concatenate([[0], np.cumsum(hc)[:-1]]), hc)
xdrift = (tick - G.TRIG_TICK) * G.TICK_TO_CM * np.where(tpc == 0, -1, 1)      # TPC 0 at x < 0
zc = wire * G.WIRE_PITCH_CM
y = plane == 2
key = ((evt * 2 + tpc) * 1000 + wire // 8) * 1000 + (tick // 16).astype(np.int64)
uk, inv = np.unique(key[y], return_inverse=True)
nb = len(uk)
bq = np.bincount(inv, q[y], nb) * 100.0
bx = np.bincount(inv, xdrift[y], nb) / np.bincount(inv, minlength=nb)
bz = np.bincount(inv, zc[y], nb) / np.bincount(inv, minlength=nb)
bevt = (uk // 1000 // 1000) // 2
btpc = (uk // 1000 // 1000) % 2
order = np.argsort(bevt, kind="stable")                  # blobs contiguous per event
rank = np.empty(nb, np.int64)
rank[order] = np.arange(nb)
bcnt = np.bincount(bevt, minlength=n)
boff = np.concatenate([[0], np.cumsum(bcnt)[:-1]])
blob_local = rank - boff[bevt]
cols = dict(x=bx, y=np.zeros(nb), z=bz, q=bq)
for k, v in cols.items():
    np.save(os.path.join(dst, f"blob_{k}.npy"), v[order].astype(np.float32))
np.save(os.path.join(dst, "blob_ext.npy"), np.tile([[0.8, 2.4, 2.4, 2.4]], (nb, 1)).astype(np.float32))
np.save(os.path.join(dst, "blob_tpc.npy"), btpc[order].astype(np.uint8))
np.save(os.path.join(dst, "blob_cluster.npy"), np.zeros(nb, np.int32))
np.save(os.path.join(dst, "blob_nu.npy"), np.full(nb, -1, np.int16))
np.save(os.path.join(dst, "blob_trackid.npy"), np.full(nb, -1, np.int32))
# hit-blob pairs, contiguous per event, event-local indices
hy = np.nonzero(y)[0]
pe = evt[hy]
po = np.argsort(pe, kind="stable")
np.save(os.path.join(dst, "hb_hit.npy"), local[hy][po].astype(np.int32))
np.save(os.path.join(dst, "hb_blob.npy"), blob_local[inv][po].astype(np.int32))
pcnt = np.bincount(pe, minlength=n)
np.save(os.path.join(dst, "ev_blob_off.npy"), boff.astype(np.int64))
np.save(os.path.join(dst, "ev_blob_cnt.npy"), bcnt.astype(np.int32))
np.save(os.path.join(dst, "ev_hb_off.npy"), np.concatenate([[0], np.cumsum(pcnt)[:-1]]).astype(np.int64))
np.save(os.path.join(dst, "ev_hb_cnt.npy"), pcnt.astype(np.int32))
txyz = np.full((nh, 3), np.nan, np.float16)
txyz[y] = np.stack([xdrift[y], np.zeros(y.sum()), zc[y]], 1)
np.save(os.path.join(dst, "hit_txyz.npy"), txyz)
meta = json.load(open(os.path.join(a.src, "meta.json")))
meta.update(n_events=n, n_hits=nh, n_ophits=no, n_blobs=nb, fake_blobs=True, source=a.src)
json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)
np.savez(os.path.join(a.dst, "splits.npz"), train=np.arange(0, n - 16), val=np.arange(n - 16, n - 8),
         test=np.arange(n - 8, n), lab_f1=np.arange(0, n - 16))
print(json.dumps(dict(events=n, hits=nh, blobs=nb, pairs=int(len(hy)))))
