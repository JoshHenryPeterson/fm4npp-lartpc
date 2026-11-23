#!/usr/bin/env python
"""D5 (#10): k-fold splits over the real-data events with Wire-Cell labels (wc_in_nusel == 1, is_data == 1) of a
round-1 corpus, stratified by the WC generic-nu decision. Fold k: test = fold k (1/K of the events), train =
the rest (lab_f1), val = a fixed 10% of train used only for monitoring (train_multitask --select last).

Usage: make_wc_folds.py CORPUS OUT_PREFIX [--k 10] [--seed 0]   -> OUT_PREFIX_f{k}.npz
"""
import argparse

import numpy as np

from qlfm.data.dataset import Corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("out_prefix")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    c = Corpus(a.corpus)
    real = (c.ev_array("is_data") == 1) & (c.ev_array("wc_in_nusel") == 1)
    gen = c.ev_array("wc_generic_nu") == 1
    rng = np.random.default_rng(a.seed)
    fold = np.full(c.n, -1)
    for cls in (True, False):                       # stratified: positives spread evenly over folds
        idx = rng.permutation(np.flatnonzero(real & (gen == cls)))
        fold[idx] = np.arange(len(idx)) % a.k
    for k in range(a.k):
        te = np.flatnonzero(fold == k)
        tr = np.flatnonzero((fold >= 0) & (fold != k))
        va = np.sort(np.random.default_rng(100 + k).permutation(tr)[: max(1, len(tr) // 10)])
        np.savez(f"{a.out_prefix}_f{k}.npz", train=tr, val=va, test=te, lab_f1=tr)
        print(f"fold {k}: train {len(tr)} ({int(gen[tr].sum())} WC nu), test {len(te)} ({int(gen[te].sum())} WC nu)")


if __name__ == "__main__":
    main()
