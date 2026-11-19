#!/usr/bin/env python
"""Select easy / hard test events per task for the round-2 event displays.

Algorithms: 6 foundation models (frozen @ 1 % labels, seed 0, late light) + Pandora direct + Pandora+BDT(+light)
(for T1: Pandora nu-slice hits). Per-event correctness:
  T2/T3: predicted class == truth;  T4: |E_pred/E_true - 1| < 0.2;  T1: per-event charge-weighted F1 >= 0.8.
T1 hard = 5 worst overall (all have a tiny nu charge fraction) + 5 worst with nu charge fraction >= 5 %.
easy = most algorithms correct, hard = fewest correct, boundary = about half correct (4 of 8; T1: 3-4 of 7, random
order); picks are spread over truth classes (T4: energy bins).

Usage: r2_select_events.py OUT.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qlfm.paths import REF_WORK  # noqa: E402

FMS = ["qlfm_M_CL", "qlfm_S_CL", "panda", "polarmae", "fm4npp", "qlfm_SD_CL"]
N_PICK = 10


def load(bb, task):
    return np.load(f"{REF_WORK}/r2p/{bb}/{task}/frozen/f0.01_s0/preds_test.npz")


def event_level(task, pnd):
    g = pnd[f"{task}_gidx"]
    y = pnd[f"{task}_y"]
    preds, correct = {}, {}
    for bb in FMS:
        r = load(bb, task)
        assert np.array_equal(r["gidx"], g), bb
        p = r["pred"]
        preds[bb] = p.argmax(1) if task != "energy" else p.reshape(-1)
    preds["pandora_direct"] = pnd[f"{task}_direct"]
    preds["pandora_bdt"] = pnd[f"{task}_bdt"]
    for k, p in preds.items():
        correct[k] = (np.abs(np.exp(p - y) - 1) < 0.2) if task == "energy" else (p == y)
    return g, y, preds, correct


def hit_level():
    per = {}
    g0 = None
    for bb in FMS + ["pandora_slice"]:
        r = load("qlfm_M_CL" if bb == "pandora_slice" else bb, "hit")
        evt, q, t = r["hit_evt"], r["hit_q"].clip(min=0), r["hit_tgt"].astype(bool)
        p = (r["hit_pnd"] > 0.5) if bb == "pandora_slice" else (r["pred"] > 0.5)
        g, inv = np.unique(evt, return_inverse=True)
        per["_nu_frac"] = np.bincount(inv, weights=q * t) / np.maximum(np.bincount(inv, weights=q), 1e-9)
        tp = np.bincount(inv, weights=q * (p & t)); pt = np.bincount(inv, weights=q * p); tt = np.bincount(inv, weights=q * t)
        f1 = 2 * tp / np.maximum(pt + tt, 1e-9)
        if g0 is None:
            g0 = g
        assert np.array_equal(g, g0), bb
        per[bb] = f1
    return g0, per


def pick(order, cls, n, rng):
    """Take n events from `order` (best-first), round-robin over classes in `cls`."""
    groups = {c: [i for i in order if cls[i] == c] for c in np.unique(cls)}
    out = []
    while len(out) < n and any(groups.values()):
        for c in list(groups):
            if groups[c] and len(out) < n:
                out.append(groups[c].pop(0))
    return out


def main():
    rng = np.random.default_rng(0)
    rng_b = np.random.default_rng(1)          # boundary picks (separate stream: easy/hard picks unchanged)
    pnd = np.load(f"{REF_WORK}/round2/displays/pandora_event_preds.npz")
    sel = {}
    for task in ("evt_nu", "flavor", "energy"):
        g, y, preds, correct = event_level(task, pnd)
        algs = list(correct)
        n_ok = sum(correct[a].astype(int) for a in algs)
        jitter = rng.random(len(g)) * 1e-3
        cls = y if task != "energy" else np.digitize(np.exp(y), [0.7, 1.5])     # E bins (GeV)
        easy = pick(list(np.argsort(-(n_ok + jitter))), cls, N_PICK, rng)
        hard = pick(list(np.argsort(n_ok + jitter)), cls, N_PICK, rng)
        mid = [int(i) for i in rng_b.permutation(len(g)) if n_ok[i] == len(algs) // 2]
        bound = pick(mid, cls, N_PICK, rng_b)
        sel[task] = [dict(gidx=int(g[i]), kind=kind, n_correct=int(n_ok[i]), n_alg=len(algs), truth=float(y[i]),
                          preds={a: float(preds[a][i]) for a in algs})
                     for kind, idx in (("easy", easy), ("boundary", bound), ("hard", hard)) for i in idx]
        print(task, "n_correct easy", [int(n_ok[i]) for i in easy], "boundary", [int(n_ok[i]) for i in bound],
              "hard", [int(n_ok[i]) for i in hard])
    g, f1 = hit_level()
    nu_frac = f1.pop("_nu_frac")
    algs = list(f1)
    n_ok = sum((f1[a] >= 0.8).astype(int) for a in algs)
    mean_f1 = np.mean([f1[a] for a in algs], 0)
    score = n_ok + 0.01 * mean_f1
    order_e, order_h = list(np.argsort(-score)), list(np.argsort(score))
    # the worst events overall are those with a tiny nu charge fraction; show 5 of them + 5 typical-fraction failures
    hard = order_h[:N_PICK // 2] + [i for i in order_h if nu_frac[i] >= 0.05][:N_PICK - N_PICK // 2]
    easy = order_e[:N_PICK]
    bound = [int(i) for i in rng_b.permutation(len(g)) if n_ok[i] in (len(algs) // 2, len(algs) // 2 + 1)][:N_PICK]
    sel["hit"] = [dict(gidx=int(g[i]), kind=kind, n_correct=int(n_ok[i]), n_alg=len(algs), nu_frac=float(nu_frac[i]),
                       preds={a: float(f1[a][i]) for a in algs})
                  for kind, idx in (("easy", easy), ("boundary", bound), ("hard", hard)) for i in idx]
    print("hit n_correct easy", [int(n_ok[i]) for i in easy], "boundary", [int(n_ok[i]) for i in bound],
          "hard", [int(n_ok[i]) for i in hard],
          "hard nu_frac", [round(float(nu_frac[i]), 3) for i in hard])
    json.dump(sel, open(sys.argv[1], "w"), indent=1)
    allg = sorted({e["gidx"] for v in sel.values() for e in v})
    np.save(sys.argv[1].replace(".json", "_gidx.npy"), np.array(allg))
    print("unique events", len(allg))


if __name__ == "__main__":
    main()
