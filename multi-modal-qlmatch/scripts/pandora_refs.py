#!/usr/bin/env python
"""Round-2 Part A: open-data (Pandora) reference baselines for T2-T4 on the same splits/test subsets.

A1  direct reco outputs (no training):
    T2 max nu_score; T3 slice_pdg (14 -> numuCC, 12 -> nueCC, no nu slice -> NC/other);
    T4 calorimetric log E = a + b * log(sum collection-plane charge in the nu slice), fit on train.
A2  Pandora features (+ light summary) -> HistGradientBoosting at label budgets 0.1/1/10/100 %.
T1 (per-hit) stays Pandora's nu-slice hits (reported by every T1 run as pandora_*_q).

Usage: pandora_refs.py OUTDIR [--corpus ...] [--splits ...]
"""
import argparse
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from qlfm.data.dataset import Corpus
from qlfm.data.labels import EV_KEYS, event_labels
from qlfm.evaluate import metrics_for
from qlfm.paths import WORK as W



def pick(idx, n, seed):     # identical to train_downstream's val/test subset selection
    return np.sort(np.random.default_rng(seed).permutation(idx)[:n])


def load_features(c):
    X = np.concatenate([np.load(os.path.join(s.dir, "pnd_feat.npy")) for s in c.shards])
    names = json.load(open(os.path.join(c.shards[0].dir, "pnd_feat.json")))
    return X, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--corpus", default=f"{W}/cache/bnb,{W}/cache/nue")
    ap.add_argument("--splits", default=f"{W}/cache/splits_all.npz")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = Corpus(a.corpus)
    sp = np.load(a.splits)
    X, names = load_features(c)
    ci = list(range(len(names["charge"])))
    allf = list(range(len(names["charge"]) + len(names["light"])))
    col = {k: i for i, k in enumerate(names["charge"] + names["light"])}
    lab = event_labels({k: c.ev_array(k) for k in EV_KEYS})
    results = []

    for task in ("evt_nu", "flavor", "energy"):
        elig, y = lab[task]
        te = pick(sp["test"][elig[sp["test"]]], 20000, 12)
        tr_all = sp["train"][elig[sp["train"]]]
        # ---------------- A1: direct reco outputs ----------------
        if task == "evt_nu":
            s = X[te, col["best_nu_score"]]
            logits = np.stack([np.zeros_like(s), np.log(np.clip(s, 1e-6, 1) / np.clip(1 - s, 1e-6, 1)), np.zeros_like(s)], 1)
        elif task == "flavor":
            cls = np.where(X[te, col["best_is_numu_like"]] > 0, 0, np.where(X[te, col["best_is_nue_like"]] > 0, 1, 2))
            logits = np.eye(3)[cls] * 10.0
        else:
            xq = X[tr_all, col["slice_logq_y"]]
            okq = xq > 0
            b, a0 = np.polyfit(xq[okq], y[tr_all][okq], 1)
            pred = a0 + b * X[te, col["slice_logq_y"]]
            logits = pred
        m = metrics_for(task, {"y": y[te], "pred": logits})
        results.append(dict(task=task, method="pandora_direct", input="C", label_frac=None, seed=0, n_train=0, test=m))
        print(task, "pandora_direct", json.dumps({k: v for k, v in m.items() if not isinstance(v, list)}), flush=True)

        # ---------------- A2: Pandora features + BDT ----------------
        for inp, feats in (("C", ci), ("CL", allf)):
            for lf in (0.001, 0.01, 0.1, 1.0):
                pool = sp[f"lab_f{lf:g}"]
                tr = pool[elig[pool]]
                for seed in ((0, 1, 2) if lf <= 0.01 else (0,)):
                    kw = dict(max_iter=500, learning_rate=0.1, early_stopping=True, validation_fraction=0.15,
                              n_iter_no_change=20, random_state=seed)
                    if task == "energy":
                        mdl = HistGradientBoostingRegressor(loss="squared_error", **kw).fit(X[tr][:, feats], y[tr])
                        pred = mdl.predict(X[te][:, feats])
                    else:
                        mdl = HistGradientBoostingClassifier(class_weight="balanced", **kw).fit(X[tr][:, feats], y[tr])
                        pr = np.zeros((len(te), 3))
                        pr[:, mdl.classes_] = mdl.predict_proba(X[te][:, feats])
                        pred = np.log(np.clip(pr, 1e-9, 1))
                    m = metrics_for(task, {"y": y[te], "pred": pred})
                    results.append(dict(task=task, method="pandora_bdt", input=inp, label_frac=lf, seed=seed,
                                        n_train=int(len(tr)), test=m))
                    print(task, "bdt", inp, lf, seed, len(tr), "%.4f" % m["score"], flush=True)
    json.dump(results, open(os.path.join(a.out, "pandora_refs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
