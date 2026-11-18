#!/usr/bin/env python
"""Create the (run, subrun)-hashed splits and nested pretraining / label subsets for a corpus.

Keys written:
  train, val, test                 partitions (90/5/5 % by default)
  pre_f{F}                         pretraining subsets, F = fraction of ALL events (0.01, 0.10, 0.50, 0.90)
  lab_f{F}                         labeled-pool subsets, F = fraction of the train partition (0.001 ... 1.0)
"""
import argparse

import numpy as np

from qlfm.data.dataset import Corpus
from qlfm.data.splits import make_splits, nested_subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--frac-val", type=float, default=0.05)
    ap.add_argument("--frac-test", type=float, default=0.05)
    a = ap.parse_args()
    c = Corpus(a.corpus)
    run, sub = c.ev_array("run"), c.ev_array("subrun")
    s = make_splits(run, sub, a.frac_val, a.frac_test)
    n = c.n
    out = dict(s)
    for f in (0.01, 0.10, 0.50, 0.90):
        out[f"pre_f{f:.2f}"] = nested_subset(s["train"], f, n, seed=1) if f < 0.9 else s["train"]
    for f in (0.001, 0.01, 0.10, 1.0):
        out[f"lab_f{f:g}"] = nested_subset(s["train"], f, len(s["train"]), seed=2)
    np.savez(a.out, **out)
    for k, v in out.items():
        print(f"{k:12s} {len(v):8d}  ({len(v)/n:.4f} of all)")
    print("unique (run,subrun):", len(np.unique(np.stack([run, sub], 1), axis=0)))


if __name__ == "__main__":
    main()
