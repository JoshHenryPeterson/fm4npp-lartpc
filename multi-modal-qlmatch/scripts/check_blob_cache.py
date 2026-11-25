#!/usr/bin/env python
"""Checks of a converted sbnd-h5maker 0.6 cache (#16), shard by shard in event chunks (low memory).

  * per event: hits, blobs, hit-blob pairs, charge tokens and blob voxel tokens (default token settings)
  * 2D -> 3D lifting quality: per hit, the blob-charge-weighted blob centre (what --pos3d uses) vs the per-hit truth
    position (hit_txyz = SCE-displaced mean deposit position; simulation): coverage and |residual| per axis
  * blob-level nu label from the blob trackid (track_table.nu_idx >= 0) vs the hit truth (hit_nu_edep > 0.05 MeV),
    charge-weighted over the hits of each blob (simulation)

Usage: QLFM_DETECTOR=sbnd check_blob_cache.py SAMPLE_DIR [SAMPLE_DIR ...] [--max-events 2000]
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from qlfm.data.labels import HIT_NU_EDEP_MEV
from qlfm.tokenize import TokCfg, hit_xyz_from_blobs, tokenize_blobs, tokenize_charge

ap = argparse.ArgumentParser()
ap.add_argument("samples", nargs="+")
ap.add_argument("--max-events", type=int, default=2000)
ap.add_argument("--chunk", type=int, default=50)
a = ap.parse_args()
cfg = TokCfg()
for sdir in a.samples:
    acc = dict(ev=0, hits=0, blobs=0, pairs=0, ctok=0, btok=0, q=0.0, q3=0.0, h3=0, res=[], nu_agree=0.0, nu_q=0.0)
    shards = sorted(d for d in glob.glob(os.path.join(sdir, "*_p0")) if os.path.exists(os.path.join(d, "ev_blob_off.npy")))
    for d in shards:
        L = lambda k: np.load(os.path.join(d, k + ".npy"), mmap_mode="r")
        hoff, hcnt = np.asarray(L("ev_hit_off")), np.asarray(L("ev_hit_cnt"))
        boff, bcnt = np.asarray(L("ev_blob_off")), np.asarray(L("ev_blob_cnt"))
        poff, pcnt = np.asarray(L("ev_hb_off")), np.asarray(L("ev_hb_cnt"))
        sim = os.path.exists(os.path.join(d, "hit_txyz.npy"))
        if sim and os.path.exists(os.path.join(d, "ev_trk_off.npy")):
            toff, tcnt = np.asarray(L("ev_trk_off")), np.asarray(L("ev_trk_cnt"))
        n = len(hcnt)
        for e0 in range(0, n, a.chunk):
            if acc["ev"] >= a.max_events:
                break
            e1 = min(n, e0 + a.chunk)
            for e in range(e0, e1):
                h0, hn, b0, bn, p0, pn = int(hoff[e]), int(hcnt[e]), int(boff[e]), int(bcnt[e]), int(poff[e]), int(pcnt[e])
                q = np.asarray(L("hit_integral")[h0:h0 + hn], np.float64).clip(min=0)
                bxyz = torch.from_numpy(np.stack([np.asarray(L("blob_" + c)[b0:b0 + bn]) for c in "xyz"], 1))
                bq = torch.from_numpy(np.asarray(L("blob_q")[b0:b0 + bn], np.float32))
                hh = torch.from_numpy(np.asarray(L("hb_hit")[p0:p0 + pn]).astype(np.int64))
                hb = torch.from_numpy(np.asarray(L("hb_blob")[p0:p0 + pn]).astype(np.int64))
                xyz = hit_xyz_from_blobs(hn, hh, hb, bxyz, bq).numpy()
                ok = ~np.isnan(xyz[:, 0])
                acc["hits"] += hn; acc["blobs"] += bn; acc["pairs"] += pn
                acc["q"] += q.sum(); acc["q3"] += q[ok].sum(); acc["h3"] += int(ok.sum())
                plane = torch.from_numpy(np.asarray(L("hit_plane")[h0:h0 + hn]).astype(np.int64))
                ct = tokenize_charge(plane, torch.from_numpy(np.asarray(L("hit_wire")[h0:h0 + hn]).astype(np.int64)),
                                     torch.from_numpy(np.asarray(L("hit_time")[h0:h0 + hn]).astype(np.float32) / 8),
                                     torch.from_numpy(q.astype(np.float32)), torch.zeros(hn), torch.zeros(hn, dtype=torch.long),
                                     1, cfg, tpc=torch.from_numpy(np.asarray(L("hit_tpc")[h0:h0 + hn]).astype(np.int64)))
                bt = tokenize_blobs(bxyz, bq, torch.from_numpy(np.asarray(L("blob_ext")[b0:b0 + bn])),
                                    torch.from_numpy(np.asarray(L("blob_tpc")[b0:b0 + bn]).astype(np.int64)),
                                    torch.zeros(bn, dtype=torch.long), 1, cfg)
                acc["ctok"] += ct.tok_evt.numel(); acc["btok"] += bt.tok_evt.numel()
                if sim:
                    t = np.asarray(L("hit_txyz")[h0:h0 + hn], np.float32)
                    both = ok & ~np.isnan(t[:, 0])
                    if both.any():
                        acc["res"].append(np.abs(xyz[both] - t[both])[::7])     # thinned sample
                    if os.path.exists(os.path.join(d, "ev_trk_off.npy")):
                        tid = np.asarray(L("trk_trackid")[int(toff[e]):int(toff[e]) + int(tcnt[e])])
                        tnu = np.asarray(L("trk_nu_idx")[int(toff[e]):int(toff[e]) + int(tcnt[e])])
                        btid = np.asarray(L("blob_trackid")[b0:b0 + bn])
                        lut = dict(zip(tid.tolist(), (tnu >= 0).tolist()))
                        bnu = np.array([lut.get(int(x), False) for x in btid], bool)
                        hnu = np.asarray(L("hit_nu_edep")[h0:h0 + hn], np.float32) > HIT_NU_EDEP_MEV
                        hq = q[hh.numpy()]
                        acc["nu_agree"] += (hq * (bnu[hb.numpy()] == hnu[hh.numpy()])).sum()
                        acc["nu_q"] += hq.sum()
                acc["ev"] += 1
        if acc["ev"] >= a.max_events:
            break
    ev = max(acc["ev"], 1)
    r = dict(sample=sdir, events=acc["ev"], hits_per_ev=acc["hits"] / ev, blobs_per_ev=acc["blobs"] / ev,
             pairs_per_hit=acc["pairs"] / max(acc["hits"], 1), charge_tokens_per_ev=acc["ctok"] / ev,
             blob_tokens_per_ev=acc["btok"] / ev, hit_frac_with_3d=acc["h3"] / max(acc["hits"], 1),
             charge_frac_with_3d=acc["q3"] / max(acc["q"], 1e-9))
    if acc["res"]:
        res = np.concatenate(acc["res"])
        r["lift_residual_cm_median_xyz"] = np.median(res, 0).round(2).tolist()
        r["lift_residual_cm_p90_xyz"] = np.quantile(res, 0.9, axis=0).round(2).tolist()
    if acc["nu_q"] > 0:
        r["blob_nu_label_agreement_q"] = acc["nu_agree"] / acc["nu_q"]
    print(json.dumps(r), flush=True)
