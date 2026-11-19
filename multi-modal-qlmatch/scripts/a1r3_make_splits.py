#!/usr/bin/env python
"""A1 round 3 event lists: python scripts/a1r3_make_splits.py OUT.npz [--splits IN.npz] [--corpus DIRS]

Copies every key of splits_all.npz unchanged (train/val/test/pre_f*/lab_f*) and adds
  pre_r3 = setdiff(pre_f0.90, lab_f0.01 U val U test)     pretraining events, disjoint from downstream and eval
Asserts pre_r3, lab_f0.01, val, test pairwise disjoint and that val/test share no (run, subrun) with pre_r3 U lab_f0.01.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qlfm.paths import REF_WORK  # noqa: E402


def check_disjoint(lists, run, sub):
    """lists: {name: index array}. Pairwise index disjointness and (run, subrun) separation of val/test from the rest."""
    names = list(lists)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = len(np.intersect1d(lists[a], lists[b]))
            assert n == 0, f"{a} and {b} share {n} events"
    key = lambda idx: set(map(tuple, np.unique(np.stack([run[idx], sub[idx]], 1), axis=0).tolist()))
    train_like = key(np.concatenate([lists[k] for k in lists if k not in ("val", "test")]))
    for k in ("val", "test"):
        if k in lists:
            n = len(key(lists[k]) & train_like)
            assert n == 0, f"{k} shares {n} (run, subrun) with the training-side lists"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=os.path.join(REF_WORK, "cache/splits_all.npz"))
    ap.add_argument("--corpus", default=f"{REF_WORK}/cache/bnb,{REF_WORK}/cache/nue")
    a = ap.parse_args(argv)
    from qlfm.data.dataset import Corpus
    sp = np.load(a.splits)
    out = {k: sp[k] for k in sp.files}
    c = Corpus(a.corpus)
    run, sub = c.ev_array("run"), c.ev_array("subrun")
    assert c.n == len(out["train"]) + len(out["val"]) + len(out["test"]), "corpus does not match the splits file"
    excl = np.union1d(np.union1d(out["lab_f0.01"], out["val"]), out["test"])
    out["pre_r3"] = np.setdiff1d(out["pre_f0.90"], excl)
    lists = {"pre_r3": out["pre_r3"], "lab_f0.01": out["lab_f0.01"], "val": out["val"], "test": out["test"]}
    check_disjoint(lists, run, sub)
    assert np.isin(out["pre_r3"], out["pre_f0.90"]).all()
    np.savez(a.out + ".tmp.npz", **out)
    os.replace(a.out + ".tmp.npz", a.out)
    for k, v in lists.items():
        print(f"{k:10s} {len(v):8d}")
    print(f"pre_f0.90 {len(out['pre_f0.90']):8d}  (overlap with lab_f0.01: {len(np.intersect1d(out['pre_f0.90'], out['lab_f0.01']))})")
    print(f"wrote {a.out} ({len(out)} keys); disjointness OK")


if __name__ == "__main__":
    main()
