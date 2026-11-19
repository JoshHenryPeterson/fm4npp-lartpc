#!/usr/bin/env python
"""Round 4 (issue #6) parts 1-3: post hoc analysis of the saved test predictions (no GPU, no retraining).

1. T1-nu: per-event neutrino completeness / purity / F1 (charge- and hit-count weighted), median and mean over
   events, next to the pooled T1 (all hits of all test events in one sum). Pandora's nu-slice hits on identical
   events and with identical truth.
2. Pandora-consistency checks (pooled vs per-event, events without a nu slice, wrong-slice events) and the event
   selection for the displays (scripts/r4_event_displays.py).
3. T2 and T1 on the sub-samples passing / failing the large-light cut (scripts/r4_light_stats.py).

Usage: r4_posthoc.py OUT_DIR TEX_DIR FIG_DIR
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from qlfm.data.dataset import Corpus
from qlfm.data.labels import EV_KEYS, event_labels
from qlfm.paths import REF_WORK
from qlfm.paths import WORK as W

MODELS = [("qlfm-M CL", "qlfm_M_CL"), ("qlfm-M C", "qlfm_M_C"), ("Panda", "panda")]
# (mode, label fraction, short name); fine-tune and no-light controls only at 1 % labels
CONFIGS = [("frozen_xf", "0.01", "XF 1\\%"), ("frozen", "0.01", "LL 1\\%"), ("finetune", "0.01", "FT 1\\%"),
           ("frozen_nolight", "0.01", "no light 1\\%"), ("frozen_xf", "0.1", "XF 10\\%"), ("frozen", "0.1", "LL 10\\%")]
PRIMARY = ("frozen_xf", "0.01")
HIT_DISPLAY_N = 6                 # events per display category


def run_dir(key, task, mode, lf):
    for root in ("r3", "r2p"):
        d = f"{REF_WORK}/{root}/{key}/{task}/{mode}/f{lf}_s0"
        if os.path.exists(f"{d}/preds_test.npz"):
            return d
    return None


# ------------------------------------------------------------------ T1-nu
def event_sums(sel, tgt, q, inv):
    """Per-event sums of true-positive / selected / true weight, charge (q) and hit-count (n) weighted.
    inv: per-hit event index (0..n_events-1)."""
    S = {}
    for wn, w in (("q", q.astype(np.float64)), ("n", np.ones(len(q)))):
        S[f"tp_{wn}"] = np.bincount(inv, w * (sel & tgt))
        S[f"ns_{wn}"] = np.bincount(inv, w * sel)
        S[f"nt_{wn}"] = np.bincount(inv, w * tgt)
    return S


def per_event(S, wn="q"):
    tp, ns, nt = S[f"tp_{wn}"], S[f"ns_{wn}"], S[f"nt_{wn}"]
    comp = tp / np.maximum(nt, 1e-9)
    pur = np.where(ns > 0, tp / np.maximum(ns, 1e-9), np.nan)
    f1 = 2 * tp / np.maximum(ns + nt, 1e-9)
    return comp, pur, f1, ns


def summarize_t1(S, m=None):
    """Pooled and per-event T1 numbers from per-event sums; m: optional boolean event mask."""
    if m is not None:
        S = {k: v[m] for k, v in S.items()}
    out = {"n_events": int(len(S["tp_q"]))}
    for wn in ("q", "n"):
        tp, ns, nt = S[f"tp_{wn}"].sum(), S[f"ns_{wn}"].sum(), S[f"nt_{wn}"].sum()
        e, p = tp / max(nt, 1e-9), tp / max(ns, 1e-9)
        out[f"pooled_eff_{wn}"], out[f"pooled_pur_{wn}"] = float(e), float(p)
        out[f"pooled_f1_{wn}"] = float(2 * e * p / max(e + p, 1e-9))
        comp, pur, f1, nsel = per_event(S, wn)
        has = nsel > 0
        out.update({f"ev_f1_med_{wn}": float(np.median(f1)), f"ev_f1_mean_{wn}": float(f1.mean()),
                    f"ev_comp_med_{wn}": float(np.median(comp)), f"ev_comp_mean_{wn}": float(comp.mean()),
                    f"ev_pur_med_{wn}": float(np.nanmedian(pur)) if has.any() else float("nan"),
                    f"ev_pur_mean_{wn}": float(np.nanmean(pur)) if has.any() else float("nan"),
                    f"ev_nosel_{wn}": float((~has).mean()),
                    f"ev_good_{wn}": float(((comp > 0.5) & (np.nan_to_num(pur) > 0.5)).mean()),
                    f"ev_f1_gt08_{wn}": float((f1 > 0.8).mean())})
    return out


def load_hit(d):
    r = np.load(f"{d}/preds_test.npz")
    return dict(evt=r["hit_evt"], p=r["pred"], tgt=r["hit_tgt"].astype(bool), q=r["hit_q"].clip(min=0),
                pnd=r["hit_pnd"] > 0.5)


# ------------------------------------------------------------------ T2
def t2_auc(y, s, sub=None):
    m = (y != 2) if sub is None else ((y != 2) & sub)
    yy = y[m] == 1
    if yy.sum() == 0 or (~yy).sum() == 0:
        return float("nan"), int(m.sum()), int((y[m] == 0).sum())
    return float(roc_auc_score(yy, s[m])), int(m.sum()), int((y[m] == 0).sum())


def eff_at_rej(y, s, sub, rej=0.95):
    m = (y != 2) & sub
    yy, ss = y[m] == 1, s[m]
    if (~yy).sum() < 20:
        return float("nan")
    thr = np.quantile(ss[~yy], rej)
    return float((ss[yy] > thr).mean())


def pandora_t2_scores(c, sp, gidx, lab):
    """Pandora direct (max nu_score) and Pandora-feature BDTs (C, C+L) at 1 % and 10 % labels -> p(in-AV nu)."""
    X = np.concatenate([np.load(os.path.join(s.dir, "pnd_feat.npy")) for s in c.shards])
    names = json.load(open(os.path.join(c.shards[0].dir, "pnd_feat.json")))
    col = {k: i for i, k in enumerate(names["charge"] + names["light"])}
    ci = list(range(len(names["charge"])))
    allf = list(range(len(names["charge"]) + len(names["light"])))
    elig, y = lab["evt_nu"]
    out = {"Pandora direct": X[gidx, col["best_nu_score"]]}
    for lf in ("0.01", "0.1"):
        pool = sp[f"lab_f{lf}"]
        tr = pool[elig[pool]]
        for inp, feats in (("C", ci), ("C+L", allf)):
            mdl = HistGradientBoostingClassifier(class_weight="balanced", max_iter=500, learning_rate=0.1,
                                                 early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                                                 random_state=0).fit(X[tr][:, feats], y[tr])
            pr = np.zeros((len(gidx), 3))
            pr[:, mdl.classes_] = mdl.predict_proba(X[gidx][:, feats])
            out[f"Pandora BDT {inp} {float(lf) * 100:g}\\%"] = pr[:, 1]
    return out


# ------------------------------------------------------------------ tex helpers
def f3(v):
    return "--" if v is None or not np.isfinite(v) else f"{v:.3f}"


def write(path, s):
    with open(path, "w") as f:
        f.write(s)


def main():
    out, tex, figd = sys.argv[1:4]
    for d in (out, tex, figd):
        os.makedirs(d, exist_ok=True)
    c = Corpus(f"{W}/cache/bnb,{W}/cache/nue")
    sp = np.load(f"{W}/cache/splits_all.npz")
    lab = event_labels({k: c.ev_array(k) for k in EV_KEYS})
    ls = np.load(f"{REF_WORK}/r4/light/light_stats.npz")
    cut = json.load(open(f"{REF_WORK}/r4/light/light_cut.json"))["cut"]
    pe_win = ls["pe_win"]
    res = {"cut_pe": cut}

    # ================= T1 / T1-nu =================
    t1 = {}
    ev_tab = {}                  # per-event quantities for the primary config, keyed by method
    U = INV = HEVT = None
    prim_S = {}
    for mname, key in MODELS:
        for mode, lf, short in CONFIGS:
            d = run_dir(key, "hit", mode, lf)
            if d is None:
                continue
            h = load_hit(d)
            if U is None:
                HEVT = h["evt"]
                U, INV = np.unique(HEVT, return_inverse=True)
                pas = pe_win[U] >= cut
                SP = event_sums(h["pnd"], h["tgt"], h["q"], INV)
                t1["Pandora|nu slice"] = {"all": summarize_t1(SP), "pass": summarize_t1(SP, pas),
                                          "fail": summarize_t1(SP, ~pas)}
                comp, pur, f1, ns = per_event(SP)
                ev_tab["Pandora"] = dict(comp=comp, pur=pur, f1=f1, has=ns > 0)
                ev_tab["nu_frac"] = SP["nt_q"] / np.maximum(np.bincount(INV, h["q"].astype(np.float64)), 1e-9)
                res["t1_n_pass_fail"] = [int(pas.sum()), int((~pas).sum())]
            assert len(h["evt"]) == len(HEVT) and (h["evt"][::9973] == HEVT[::9973]).all(), d   # same hit order
            S = event_sums(h["p"] > 0.5, h["tgt"], h["q"], INV)
            r = {"all": summarize_t1(S), "pass": summarize_t1(S, pas), "fail": summarize_t1(S, ~pas)}
            t1[f"{mname}|{short}"] = r
            if (mode, lf) == PRIMARY:
                comp, pur, f1, ns = per_event(S)
                ev_tab[mname] = dict(comp=comp, pur=pur, f1=f1)
                prim_S[mname] = S
            print("T1", mname, short, json.dumps({k: round(v, 3) for k, v in r["all"].items()}), flush=True)
    res["t1"] = t1

    # Pandora consistency: events without a nu slice; wrong-slice events (the slice holds < 10 % of the nu charge)
    P = ev_tab["Pandora"]
    has = P["has"]
    wrong = has & (P["comp"] < 0.1)
    good = has & ~wrong
    res["pandora_consistency"] = dict(
        n_events=int(len(has)), frac_no_slice=float((~has).mean()), frac_wrong_slice=float(wrong.mean()),
        ev_f1_med_good_slice=float(np.median(P["f1"][good])),
        ev_pur_med_good_slice=float(np.nanmedian(P["pur"][good])),
        ev_comp_med_good_slice=float(np.median(P["comp"][good])),
        pooled_Pandora_good_slice=summarize_t1(SP, good),
        **{f"pooled_{m}_good_slice": summarize_t1(S, good) for m, S in prim_S.items()})
    print("Pandora consistency", json.dumps({k: v for k, v in res["pandora_consistency"].items()
                                             if not isinstance(v, dict)}), flush=True)

    # ---- display selection (per-event charge-weighted F1, primary config, qlfm-M CL vs Pandora) ----
    Q = ev_tab["qlfm-M CL"]
    Q["gidx"] = U
    fq, fp = Q["f1"], P["f1"]
    rng = np.random.default_rng(4)
    cats = {
        "pnd_wrong_slice": np.where(wrong & (fq > 0.8))[0],                               # Pandora picks a cosmic slice
        "pnd_low_purity": np.where(has & ~wrong & (P["pur"] < 0.6) & (fq > 0.8))[0],      # nu slice absorbs cosmics
        "model_worse": np.where((fp - fq > 0.25) & (fp > 0.7))[0],                         # model clearly worse
        "typical": np.where((np.abs(fq - np.median(fq)) < 0.03) & (np.abs(fp - np.median(fp[has])) < 0.1))[0],
    }
    sel = []
    for k, idx in cats.items():
        pick = rng.choice(idx, min(HIT_DISPLAY_N, len(idx)), replace=False) if len(idx) else []
        for i in sorted(pick, key=lambda i: fq[i] - fp[i]):
            sel.append(dict(kind=k, gidx=int(Q["gidx"][i]), f1={m: float(ev_tab[m]["f1"][i]) for m in
                                                                 ("Pandora", "qlfm-M CL", "qlfm-M C", "Panda") if m in ev_tab},
                            pnd_comp=float(P["comp"][i]), pnd_pur=float(np.nan_to_num(P["pur"][i])),
                            nu_frac=float(ev_tab["nu_frac"][i]), pe_win=float(pe_win[Q["gidx"][i]])))
        res.setdefault("display_cat_sizes", {})[k] = int(len(idx))
    json.dump(sel, open(f"{out}/display_selection.json", "w"), indent=1)

    # ---- figure: per-event F1 distributions and model-vs-Pandora scatter ----
    f, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    b = np.linspace(0, 1, 41)
    for m, col in (("Pandora", "k"), ("qlfm-M CL", "tab:red"), ("qlfm-M C", "tab:orange"), ("Panda", "tab:blue")):
        if m in ev_tab:
            v = ev_tab[m]["f1"]
            ax[0].hist(v, b, histtype="step", lw=1.6, color=col,
                       label=f"{m}: median {np.median(v):.2f}, mean {v.mean():.2f}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"per-event $\nu$ F1 (charge weighted)")
    ax[0].set_ylabel("events")
    ax[0].legend(fontsize=8, loc="upper left")
    for m, col in (("Pandora", "k"), ("qlfm-M CL", "tab:red"), ("Panda", "tab:blue")):
        pp = ev_tab[m]["pur"] if m != "Pandora" else np.where(has, P["pur"], np.nan)
        ax[1].scatter(ev_tab[m]["comp"], np.nan_to_num(pp, nan=-0.05), s=2, alpha=0.35, color=col, label=m)
    ax[1].set_xlabel(r"per-event $\nu$ completeness (charge)")
    ax[1].set_ylabel(r"per-event $\nu$ purity (charge); $-0.05$ = nothing selected")
    ax[1].legend(fontsize=8, markerscale=4)
    ax[2].scatter(fp, fq, s=3, alpha=0.4, c=np.log10(np.clip(ev_tab["nu_frac"], 1e-3, 1)), cmap="viridis")
    ax[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[2].set_xlabel("Pandora per-event F1")
    ax[2].set_ylabel("qlfm-M CL (XF, 1%) per-event F1")
    ax[2].set_title(r"colour: log$_{10}$ $\nu$ charge fraction of the event", fontsize=9)
    f.tight_layout()
    f.savefig(f"{figd}/round4_t1nu.png", dpi=130)

    # ================= T2 on light-cut sub-samples =================
    t2 = {}
    base = None
    for mname, key in MODELS:
        for mode, lf, short in CONFIGS:
            d = run_dir(key, "evt_nu", mode, lf)
            if d is None:
                continue
            r = np.load(f"{d}/preds_test.npz")
            g, y = r["gidx"], r["y"]
            if base is None:
                base = (g, y)
            assert (g == base[0]).all()
            s = softmax(r["pred"], 1)[:, 1]
            p = pe_win[g] >= cut
            row = {}
            for nm, sub in (("all", np.ones_like(p)), ("pass", p), ("fail", ~p)):
                a, n, n0 = t2_auc(y, s, sub)
                row[nm] = dict(auc=a, n=n, n_nonu=n0, eff_rej95=eff_at_rej(y, s, sub))
            t2[f"{mname}|{short}"] = row
            print("T2", mname, short, {k: round(v["auc"], 3) for k, v in row.items()}, flush=True)
    g, y = base
    p = pe_win[g] >= cut
    refs = pandora_t2_scores(c, sp, g, lab)
    refs["light only (beam-window PE)"] = pe_win[g]
    for nm, s in refs.items():
        row = {}
        for sn, sub in (("all", np.ones_like(p)), ("pass", p), ("fail", ~p)):
            a, n, n0 = t2_auc(y, s, sub)
            row[sn] = dict(auc=a, n=n, n_nonu=n0, eff_rej95=eff_at_rej(y, s, sub))
        t2[f"{nm}|"] = row
        print("T2", nm, {k: round(v["auc"], 3) for k, v in row.items()}, flush=True)
    res["t2"] = t2
    res["t2_counts"] = {sn: np.bincount(y[sub], minlength=3).tolist()
                        for sn, sub in (("all", np.ones_like(p)), ("pass", p), ("fail", ~p))}
    json.dump(res, open(f"{out}/posthoc.json", "w"), indent=1)

    # ================= tex fragments =================
    # T1-nu table (charge weighted): pooled vs per-event, all test events
    L = []
    for k, r in t1.items():
        m, s = k.split("|")
        a = r["all"]
        L.append(f"{m} & {s} & {f3(a['pooled_eff_q'])} & {f3(a['pooled_pur_q'])} & {f3(a['pooled_f1_q'])} & "
                 f"{f3(a['ev_comp_med_q'])} & {f3(a['ev_pur_med_q'])} & {f3(a['ev_f1_med_q'])} & {f3(a['ev_f1_mean_q'])} & "
                 f"{f3(a['ev_good_q'])} & {f3(a['ev_nosel_q'])} \\\\")
    write(f"{tex}/t1nu_q.tex", "\n".join(L) + "\n")
    L = []
    for k, r in t1.items():
        m, s = k.split("|")
        a = r["all"]
        L.append(f"{m} & {s} & {f3(a['pooled_eff_n'])} & {f3(a['pooled_pur_n'])} & {f3(a['pooled_f1_n'])} & "
                 f"{f3(a['ev_comp_med_n'])} & {f3(a['ev_pur_med_n'])} & {f3(a['ev_f1_med_n'])} & {f3(a['ev_f1_mean_n'])} & "
                 f"{f3(a['ev_good_n'])} \\\\")
    write(f"{tex}/t1nu_n.tex", "\n".join(L) + "\n")
    # T1 light-cut table
    L = []
    for k, r in t1.items():
        m, s = k.split("|")
        L.append(f"{m} & {s} & " + " & ".join(f"{f3(r[sn]['pooled_f1_q'])} & {f3(r[sn]['ev_f1_med_q'])}"
                                             for sn in ("all", "pass", "fail")) + " \\\\")
    write(f"{tex}/t1cut.tex", "\n".join(L) + "\n")
    # T2 light-cut table
    L = []
    for k, r in t2.items():
        m, s = k.split("|")
        L.append(f"{m} & {s} & " + " & ".join(f"{f3(r[sn]['auc'])}" for sn in ("all", "pass", "fail")) + " & " +
                 " & ".join(f"{f3(r[sn]['eff_rej95'])}" for sn in ("all", "pass")) + " \\\\")
    write(f"{tex}/t2cut.tex", "\n".join(L) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
