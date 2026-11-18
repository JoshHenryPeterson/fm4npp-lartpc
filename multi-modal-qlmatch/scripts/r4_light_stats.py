#!/usr/bin/env python
"""Round 4 (issue #6) part 3.1: beam-window light statistics and the large-light cut.

Per event (all events of the corpus), from the cached optical hits that the light tokens see:
  pe_all : sum PE of ophits in the 0-24 us light-token window
  pe_win : sum PE of ophits in the beam-flash window [WIN_LO, WIN_HI] us
plus the PE-weighted ophit time spectrum per T2 class, and the largest reconstructed flash (fl_pe, fl_t). Writes OUT/light_stats.npz (global event index order),
OUT/light_cut.json (chosen cut, light-only T2 AUCs) and the figure FIG.

Usage: r4_light_stats.py OUT_DIR FIG_PNG
Other corpora (e.g. SBND, QLFM_DETECTOR=sbnd): set QLFM_CORPUS and QLFM_SPLITS; the windows come from qlfm.geometry.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from qlfm import geometry as G
from qlfm.data.dataset import Corpus
from qlfm.data.labels import EV_KEYS, event_labels

W = os.environ.get("QLFM_WORK", "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work")
WIN_LO, WIN_HI = G.BEAM_WINDOW_US   # MicroBooNE 3-6 us: truth nu time 3.1-4.7 us + 0.48 us flash lag, with margin
T_LO, T_HI = G.LIGHT_EDGES[0], G.LIGHT_EDGES[-1]   # light-token window (MicroBooNE 0-24 us)
CUTS = G.T2_PE_CUTS


TB = np.arange(T_LO, T_HI + 0.25, 0.25)


def per_event(c, cls):
    pe_all, pe_win, spec = [], [], np.zeros((3, len(TB) - 1))
    g0 = 0
    for sh in c.shards:
        off, cnt = np.asarray(sh.ev("oph_off")).astype(np.int64), np.asarray(sh.ev("oph_cnt")).astype(np.int64)
        t, pe = np.asarray(sh.oph("t")).astype(np.float32), np.asarray(sh.oph("pe")).astype(np.float32).clip(min=0)
        ev = np.repeat(np.arange(sh.n), cnt)
        o = off[0]
        t, pe = t[o:o + cnt.sum()], pe[o:o + cnt.sum()]
        inw = (t >= T_LO) & (t < T_HI)
        bw = (t >= WIN_LO) & (t < WIN_HI)
        a = np.bincount(ev, pe * inw, minlength=sh.n)
        pe_all.append(a)
        pe_win.append(np.bincount(ev, pe * bw, minlength=sh.n))
        k = cls[g0 + ev]
        for j in range(3):
            spec[j] += np.histogram(t[k == j], TB, weights=pe[k == j])[0]
        g0 += sh.n
    return np.concatenate(pe_all), np.concatenate(pe_win), spec


def main():
    out, fig = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    c = Corpus(os.environ.get("QLFM_CORPUS", f"{W}/cache/bnb,{W}/cache/nue"))
    sp = np.load(os.environ.get("QLFM_SPLITS", f"{W}/cache/splits_all.npz"))
    lab = event_labels({k: c.ev_array(k) for k in EV_KEYS})
    elig, y = lab["evt_nu"]
    pe_all, pe_win, spec = per_event(c, np.where(elig, y, -1))
    fl_pe, fl_t = c.ev_array("fl_pe"), c.ev_array("fl_t")
    np.savez_compressed(os.path.join(out, "light_stats.npz"), pe_all=pe_all, pe_win=pe_win, spec=spec, tb=TB,
                        fl_pe=fl_pe, fl_t=fl_t, t2_elig=elig, t2_y=y)

    te = sp["test"][elig[sp["test"]]]
    yt = y[te]
    m01 = yt != 2
    res = dict(window_us=[WIN_LO, WIN_HI], n_test=int(len(te)), n_class=np.bincount(yt, minlength=3).tolist())
    # light-only T2 discriminants (in-AV nu vs no-nu)
    for name, s in (("pe_win", pe_win), ("pe_all", pe_all), ("fl_pe", fl_pe)):
        res[f"auc_{name}"] = float(roc_auc_score(yt[m01] == 1, s[te][m01]))
    rows = []
    for cut in CUTS:
        p = pe_win[te] >= cut
        rows.append(dict(cut=cut, eff_nu=float(p[yt == 1].mean()), pass_nonu=float(p[yt == 0].mean()),
                         pass_outav=float(p[yt == 2].mean()), n_pass=int(p.sum()),
                         n_pass_class=np.bincount(yt[p], minlength=3).tolist(),
                         n_fail_class=np.bincount(yt[~p], minlength=3).tolist()))
    res["cuts"] = rows
    # chosen cut: best balanced accuracy (in-AV nu vs no-nu) over a fine scan, rounded to the CUTS grid
    grid = np.unique(np.round(np.logspace(0, 3.5 + np.log10(CUTS[-1] / 500), 200)))   # MicroBooNE: 10**3.5
    ba = [0.5 * ((pe_win[te][yt == 1] >= g).mean() + (pe_win[te][yt == 0] < g).mean()) for g in grid]
    g_best = float(grid[int(np.argmax(ba))])
    res["best_balanced_cut"] = g_best
    res["best_balanced_acc"] = float(max(ba))
    res["cut"] = float(min(CUTS, key=lambda x: abs(np.log(x) - np.log(g_best))))
    json.dump(res, open(os.path.join(out, "light_cut.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "cuts"}, indent=1))
    for r in rows:
        print(r)

    # ---- figure: PE spectra by class, ophit timing, ROC of light-only discriminants ----
    names = {0: r"no $\nu$", 1: r"in-AV visible $\nu$", 2: r"out-of-AV $\nu$"}
    cols = {0: "tab:gray", 1: "tab:red", 2: "tab:blue"}
    f, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = np.logspace(-1, 4, 61)
    for k in (0, 1, 2):
        v = pe_win[te][yt == k]
        ax[0].hist(np.clip(v, 0.11, None), bins, histtype="step", lw=1.6, color=cols[k],
                   weights=np.full(len(v), 1 / len(v)), label=f"{names[k]} (n={len(v)}, {100 * (v < 0.1).mean():.1f}% $\\leq$0.1 PE)")
    ax[0].axvline(res["cut"], color="k", ls="--", lw=1)
    ax[0].text(res["cut"] * 1.1, 0.6 * ax[0].get_ylim()[1], f"cut {res['cut']:g} PE", fontsize=9)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(f"beam-window PE (ophits, {WIN_LO:g}-{WIN_HI:g} $\\mu$s)")
    ax[0].set_ylabel("fraction of events")
    ax[0].legend(fontsize=8)
    for k in (0, 1, 2):
        ax[1].stairs(spec[k] / spec[k].sum(), TB, lw=1.6, color=cols[k], label=names[k])
    ax[1].axvspan(WIN_LO, WIN_HI, color="y", alpha=0.25, label="beam window")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"ophit time ($\mu$s), PE-weighted, all events")
    ax[1].set_ylabel("fraction of PE per 0.25 $\\mu$s")
    ax[1].legend(fontsize=8)
    for name, s, ls in (("pe_win", pe_win, "-"), ("pe_all", pe_all, "--"), ("fl_pe", fl_pe, ":")):
        fpr, tpr, _ = roc_curve(yt[m01] == 1, s[te][m01])
        ax[2].plot(fpr, tpr, ls, label=f"{name}: AUC {res['auc_' + name]:.3f}")
    ax[2].set_xlabel(r"no-$\nu$ pass rate")
    ax[2].set_ylabel(r"in-AV $\nu$ efficiency")
    ax[2].set_title(r"light-only T2 (in-AV $\nu$ vs no $\nu$)", fontsize=10)
    ax[2].legend(fontsize=8)
    f.tight_layout()
    f.savefig(fig, dpi=130)


if __name__ == "__main__":
    main()
