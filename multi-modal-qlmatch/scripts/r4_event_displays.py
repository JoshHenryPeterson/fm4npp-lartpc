#!/usr/bin/env python
"""Round-4 T1 event displays (issue #6 part 2): events where the per-event nu F1 of the foundation models and of
Pandora's nu slice differ (selection: scripts/r4_posthoc.py -> display_selection.json).

Panels: TPC charge, PMT view of the flash matched to the true nu time (as in round 2), Truth, Pandora nu-slice hits,
qlfm-M CL, qlfm-M C, Panda (frozen, strong late light XF, 1 % labels; hit colour = p(nu)), and a box with each
method's per-event completeness / purity / F1 (charge and hit-count weighted).

Usage: r4_event_displays.py SELECTION.json OUT_DIR
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from r2_event_displays import CH0, HIT_NU_EDEP_MEV, frame, light_panel, load_event, nu_boxes  # noqa: E402

from qlfm.data.dataset import Corpus
from qlfm.paths import REF_WORK
from qlfm.paths import WORK as W

FMS = [("qlfm-M CL", "r3/qlfm_M_CL/hit/frozen_xf/f0.01_s0"), ("qlfm-M C", "r3/qlfm_M_C/hit/frozen_xf/f0.01_s0"),
       ("Panda", "r3/panda/hit/frozen_xf/f0.01_s0")]
KIND = {"pnd_wrong_slice": "Pandora selects a cosmic slice", "pnd_low_purity": "Pandora ν slice absorbs cosmic hits",
        "model_worse": "foundation model worse than Pandora", "typical": "typical event (both near their median)"}
_P = {}


def preds(g, q):
    out = {}
    for name, d in FMS:
        if name not in _P:
            r = np.load(f"{REF_WORK}/{d}/preds_test.npz")
            o = np.argsort(r["hit_evt"], kind="stable")
            _P[name] = (r["hit_evt"][o], r["pred"][o], r["hit_q"][o])
        evt, p, hq = _P[name]
        a, b = np.searchsorted(evt, g), np.searchsorted(evt, g, side="right")
        assert b - a == len(q) and np.allclose(hq[a:b], q, rtol=1e-2, atol=1), (name, g)   # same hit order as the cache
        out[name] = p[a:b]
    return out


def ev_metrics(sel, tgt, q):
    r = []
    for w in (q, np.ones_like(q)):
        tp, ns, nt = (w * (sel & tgt)).sum(), (w * sel).sum(), (w * tgt).sum()
        c, p = tp / max(nt, 1e-9), (tp / ns if ns > 0 else float("nan"))
        r += [c, p, 2 * tp / max(ns + nt, 1e-9)]
    return r


def render(item, k, e, pr, out):
    tgt = e["nu_edep"] > HIT_NU_EDEP_MEV
    q = np.clip(e["integral"].astype(np.float32), 0, None)
    lo, hi = np.percentile(e["tick"], [0, 100])
    ylim = (hi + 30, lo - 30)
    fig = plt.figure(figsize=(8.27, 11.0))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.05, 1, 1, 1], hspace=0.3, wspace=0.2,
                          left=0.06, right=0.985, top=0.935, bottom=0.04)
    nu = (f"{'ν' if e['nu_pdg'] > 0 else 'ν̄'}{ {12: 'e', 14: 'μ'}.get(abs(int(e['nu_pdg'])), '?') } "
          f"{'CC' if e['is_cc'] else 'NC'}, E = {float(e['nu_energy']):.2f} GeV, "
          f"vtx = ({e['nu_vtx'][0]:.0f}, {e['nu_vtx'][1]:.0f}, {e['nu_vtx'][2]:.0f}) cm")
    fig.text(0.06, 0.988, f"T1 per-event — {KIND[item['kind']]} #{k + 1}    run {int(e['run'])} / subrun "
                          f"{int(e['subrun'])} / event {int(e['event'])}  [gidx {item['gidx']}]",
             fontsize=8.5, weight="bold", va="top")
    fig.text(0.06, 0.968, f"Truth: {nu};  ν charge fraction {item['nu_frac']:.1%};  beam-window light "
                          f"{item['pe_win']:.0f} PE", fontsize=7.5, va="top")
    ax = fig.add_subplot(gs[0, 0])
    o = np.argsort(q)
    sc = ax.scatter(e["ch"][o], e["tick"][o], c=np.log10(1 + q[o]), s=0.4, lw=0, cmap="viridis", vmin=1, vmax=3.5,
                    rasterized=True)
    frame(ax, e, "Charge: hit integral (log10)", ylim)
    ax.set_ylabel("tick", fontsize=6, labelpad=1)
    for i, n in enumerate("UVY"):
        ax.text(CH0[i] + 60, ylim[1] + 40, n, fontsize=7, va="top", color="0.3")
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01).ax.tick_params(labelsize=5)
    light_panel(fig.add_subplot(gs[0, 1]), e)
    panels = [("Truth (ν: E_dep > 0.05 MeV)", tgt.astype(np.float32)), ("Pandora (ν-slice hits)", (e["pnd_nu"] > 0.5) * 1.0)]
    panels += [(n, pr[n]) for n, _ in FMS]
    rows = []
    for i, (name, p) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        p = np.asarray(p, np.float32)
        o = np.argsort(p, kind="stable")
        ax.scatter(e["ch"][o], e["tick"][o], c=p[o], s=0.4, lw=0, cmap="coolwarm", vmin=0, vmax=1, rasterized=True)
        if i > 0:
            m = ev_metrics(p > 0.5, tgt, q)
            rows.append((name.split(" (")[0], m))
            name += f"   F1_q = {m[2]:.2f}"
        frame(ax, e, name, ylim)
        nu_boxes(ax, e, tgt)
        if i % 2 == 0:
            ax.set_ylabel("tick", fontsize=6, labelpad=1)
        if i >= 4:
            ax.set_xlabel("global channel (U | V | Y)", fontsize=6, labelpad=1)
    # per-event metrics in the free panel
    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    f = lambda v: " -- " if not np.isfinite(v) else f"{v:.2f}"
    txt = "Per-event ν metrics (completeness / purity / F1)\n\n" + f"{'':<10s} {'charge-weighted':<19s} hit-count\n"
    for n, m in rows:
        txt += f"{n:<10s} {f(m[0])} {f(m[1])} {f(m[2])}     {f(m[3])} {f(m[4])} {f(m[5])}\n"
    txt += ("\nPandora: its ν slice = all hits of the slice with\nthe highest ν score (no slice -> nothing selected);"
            "\nmodels: hits with p(ν) > 0.5.")
    ax.text(0.0, 0.95, txt, fontsize=7, va="top", family="DejaVu Sans Mono", transform=ax.transAxes)
    fig.text(0.985, 0.006, "hit colour: p(ν), blue = cosmic, red = ν; dashed boxes: 1–99 % extent of true-ν hits per plane;"
                           " models: frozen + strong late light (XF), 1 % labels", fontsize=5.5, ha="right", color="0.3")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    sel = json.load(open(sys.argv[1]))
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    c = Corpus(f"{W}/cache/bnb,{W}/cache/nue")
    cnt = {}
    for item in sel:
        k = cnt.get(item["kind"], 0)
        cnt[item["kind"]] = k + 1
        e = load_event(c, item["gidx"])
        pr = preds(item["gidx"], np.clip(e["integral"].astype(np.float32), 0, None))
        out = f"{out_dir}/{item['kind']}{k + 1:02d}.png"
        render(item, k, e, pr, out)
        print(out, flush=True)


if __name__ == "__main__":
    main()
