#!/usr/bin/env python
"""Round-4 T5 hand scan: 20 random test events, measured light vs the charge -> light prediction of four encoders
(90 % pretraining, fine-tuned with 10 % labels): qlfm-M CL, qlfm-S CL, qlfm-M C, Panda.

One page per event:
  TPC charge | measured PMT light (total PE per PMT, 0-24 us)
  four predicted PMT views (filled = predicted, black ring = measured, common area scale per event)
  PMT-summed time profile (measured + 4 predictions) | per-PMT totals
Per model and event: MSE over the 32 x 17 log1p features, total-PE error Sum_pred/Sum_meas - 1, PMT-pattern TV.

Usage: r4_t5_handscan.py T5_ROOT OUT_DIR TEX_DIR [N_EVENTS] [SEED]
  -> OUT_DIR/hs{NN}.png, OUT_DIR/handscan.json, TEX_DIR/t5_handscan.tex (per-event table)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(__file__))
from r2_event_displays import CH0, HIT_NU_EDEP_MEV, frame, load_event  # noqa: E402

from qlfm.data.dataset import Corpus
from qlfm.geometry import AV, PMT_YZ, in_volume
from qlfm.tokenize import TokCfg
from qlfm.paths import WORK as W

MODELS = [("qlfm-M CL", "qlfm_M_CL_pre0.90", "tab:red"), ("qlfm-S CL", "qlfm_S_CL_pre0.90", "tab:orange"),
          ("qlfm-M C", "qlfm_M_C_pre0.90", "tab:blue"), ("Panda", "panda_pre0.90", "tab:green")]
EDGES = np.array(TokCfg().light_edges)
WIDTH = np.diff(EDGES)
BEAM = (3.0, 6.0)


def ev_metrics(p, t):
    """p, t: [32, 17] log1p features of one event."""
    beam = (EDGES[:-1] >= BEAM[0]) & (EDGES[1:] <= BEAM[1])
    pe_p, pe_t = np.expm1(p[:, -1]).clip(min=0), np.expm1(t[:, -1]).clip(min=0)
    sp, st = pe_p.sum(), pe_t.sum()
    tv = 0.5 * np.abs(pe_p / max(sp, 1e-9) - pe_t / max(st, 1e-9)).sum() if st > 0 else float("nan")
    return dict(mse=float(((p - t) ** 2).mean()), dtot=float(sp / st - 1) if st > 0 else float("nan"),
                tv=float(tv), pe_pred=float(sp), pe_meas=float(st),
                beam_pred=float(np.expm1(p[:, :-1]).clip(min=0)[:, beam].sum()))


def pmt_view(ax, pe, ref, scale, title, color):
    ax.add_patch(Rectangle((AV["z"][0], AV["y"][0]), AV["z"][1] - AV["z"][0], AV["y"][1] - AV["y"][0],
                           fill=False, lw=0.6, color="0.4"))
    z, y = PMT_YZ[:, 1], PMT_YZ[:, 0]
    ax.scatter(z, y, s=8, facecolors="none", edgecolors="0.8", lw=0.3)
    ax.scatter(z, y, s=scale * pe, c=color, alpha=0.5, edgecolors=color, lw=0.4)
    if ref is not None:
        ax.scatter(z, y, s=scale * ref, facecolors="none", edgecolors="k", lw=0.6)
    ax.set_xlim(-30, 1070)
    ax.set_ylim(-140, 140)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=7, pad=2)
    ax.tick_params(labelsize=5, length=2, pad=1)
    ax.set_xlabel("z [cm]", fontsize=5, labelpad=0)
    ax.set_ylabel("y [cm]", fontsize=5, labelpad=0)


def describe(e):
    if e["nu_pdg"] == 0:
        return "no ν interaction (cosmics only)"
    fl = {12: "e", 14: "μ"}.get(abs(int(e["nu_pdg"])), "?")
    where = "in AV" if in_volume(np.asarray(e["nu_vtx"])[None])[0] else "outside AV"
    return (f"{'ν' if e['nu_pdg'] > 0 else 'ν̄'}{fl} {'CC' if e['is_cc'] else 'NC'} {where}, "
            f"E = {float(e['nu_energy']):.2f} GeV")


def render(k, g, e, t, preds, mets, out):
    q = np.clip(e["integral"].astype(np.float32), 0, None)
    nu_frac = q[e["nu_edep"] > HIT_NU_EDEP_MEV].sum() / max(q.sum(), 1e-9)
    pe_t = np.expm1(t[:, -1]).clip(min=0)
    pe_p = {n: np.expm1(p[:, -1]).clip(min=0) for n, p in preds.items()}
    scale = 260.0 / max(pe_t.max(), max(v.max() for v in pe_p.values()), 1e-9)
    fig = plt.figure(figsize=(8.27, 11.0))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.5, 0.62, 0.62, 0.62, 1.05], hspace=0.38, wspace=0.16,
                          left=0.07, right=0.985, top=0.94, bottom=0.05)
    fig.text(0.07, 0.988, f"T5 hand scan #{k + 1}    run {int(e['run'])} / subrun {int(e['subrun'])} / event "
                          f"{int(e['event'])}  [gidx {g}]", fontsize=8.5, weight="bold", va="top")
    fig.text(0.07, 0.968, f"Truth: {describe(e)};  ν charge fraction {nu_frac:.1%};  measured light "
                          f"{pe_t.sum():.0f} PE (0–24 µs), {mets['beam_pe']:.0f} PE in the 3–6 µs beam window",
             fontsize=7.5, va="top")
    ax = fig.add_subplot(gs[0, :])
    lo, hi = np.percentile(e["tick"], [0, 100])
    ylim = (hi + 30, lo - 30)
    o = np.argsort(q)
    sc = ax.scatter(e["ch"][o], e["tick"][o], c=np.log10(1 + q[o]), s=0.4, lw=0, cmap="viridis", vmin=1, vmax=3.5,
                    rasterized=True)
    frame(ax, e, "Charge (model input): hit integral (log10)", ylim)
    ax.set_ylabel("tick", fontsize=6, labelpad=1)
    ax.set_xlabel("global channel (U | V | Y)", fontsize=6, labelpad=1)
    for i, n in enumerate("UVY"):
        ax.text(CH0[i] + 60, ylim[1] + 40, n, fontsize=7, va="top", color="0.3")
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01).ax.tick_params(labelsize=5)
    pmt_view(fig.add_subplot(gs[1, 0]), pe_t, None, scale, f"Measured light: {pe_t.sum():.0f} PE", "0.35")
    for i, (n, _, c) in enumerate(MODELS):
        m = mets["models"][n]
        ax = fig.add_subplot(gs[1 + (i + 1) // 2, (i + 1) % 2])
        pmt_view(ax, pe_p[n], pe_t, scale, f"{n}: {m['pe_pred']:.0f} PE ({m['dtot']:+.0%}),  TV = {m['tv']:.2f},  "
                                         f"MSE = {m['mse']:.2f}", c)
    # metrics box
    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    txt = f"{'':<10s} {'MSE':>5s} {'total PE':>9s} {'ΔΣ':>6s} {'TV':>5s} {'3–6 µs PE':>9s}\n"
    txt += f"{'measured':<10s} {'':>5s} {pe_t.sum():9.0f} {'':>6s} {'':>5s} {mets['beam_pe']:9.0f}\n"
    for n, _, _ in MODELS:
        m = mets["models"][n]
        txt += f"{n:<10s} {m['mse']:5.2f} {m['pe_pred']:9.0f} {m['dtot']:+6.0%} {m['tv']:5.2f} {m['beam_pred']:9.0f}\n"
    txt += "\nMSE: mean over 32 PMTs x 17 log(1+PE) features\nΔΣ = Σpred/Σmeas − 1;  TV = ½Σ|f_pred − f_meas|\n(PE fractions per PMT)"
    ax.text(0.0, 1.0, txt, fontsize=6.3, va="top", family="DejaVu Sans Mono", transform=ax.transAxes)
    # PMT-summed time profile
    ax = fig.add_subplot(gs[4, 0])
    x = np.repeat(EDGES, 2)[1:-1]
    prof = lambda a: np.repeat(np.expm1(a[:, :-1]).clip(min=0).sum(0) / WIDTH, 2)
    ax.plot(x, np.maximum(prof(t), 0.5), color="k", lw=1.6, label="measured")
    for n, _, c in MODELS:
        ax.plot(x, np.maximum(prof(preds[n]), 0.5), color=c, lw=0.9, label=n)
    ax.axvspan(*BEAM, color="0.9", zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.5, None)
    ax.set_xlim(0, 24)
    ax.set_xlabel("time [µs] (beam window shaded)", fontsize=6, labelpad=1)
    ax.set_ylabel("PE / µs, summed over PMTs", fontsize=6, labelpad=1)
    ax.set_title("Time profile", fontsize=7, pad=2)
    ax.tick_params(labelsize=5, length=2, pad=1)
    ax.legend(fontsize=5.5, frameon=False)
    # per-PMT totals
    ax = fig.add_subplot(gs[4, 1])
    ch = np.arange(32)
    ax.bar(ch, np.maximum(pe_t, 0.5), color="0.75", width=0.8, label="measured")
    for j, (n, _, c) in enumerate(MODELS):
        ax.scatter(ch + (j - 1.5) * 0.15, np.maximum(pe_p[n], 0.5), s=6, color=c, label=n, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.5, None)
    ax.set_xlim(-0.7, 31.7)
    ax.set_xlabel("PMT channel", fontsize=6, labelpad=1)
    ax.set_ylabel("total PE (0–24 µs)", fontsize=6, labelpad=1)
    ax.set_title("Per-PMT totals", fontsize=7, pad=2)
    ax.tick_params(labelsize=5, length=2, pad=1)
    fig.text(0.985, 0.008, "Encoders pretrained on 90 % of the pretraining split, fine-tuned on 10 % labels; input = "
                           "charge only. PMT views: area ∝ PE (same scale on all five), black ring = measured.",
             fontsize=5.5, ha="right", color="0.3")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    root, out_dir, tex_dir = sys.argv[1:4]
    n_ev = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    os.makedirs(out_dir, exist_ok=True)
    P = {}
    for n, key, _ in MODELS:
        r = np.load(f"{root}/{key}/light/finetune/f0.1_s0/preds_test.npz")
        P[n] = dict(zip(r["gidx"].tolist(), r["pred"].astype(np.float32)))
        T = dict(zip(r["gidx"].tolist(), r["light_tgt"].astype(np.float32)))
    common = sorted(set.intersection(*(set(v) for v in P.values())))
    sel = np.random.default_rng(seed).choice(common, n_ev, replace=False)
    c = Corpus(f"{W}/cache/bnb,{W}/cache/nue")
    beam = (EDGES[:-1] >= BEAM[0]) & (EDGES[1:] <= BEAM[1])
    res = []
    for k, g in enumerate(sel.tolist()):
        e = load_event(c, g)
        t = T[g]
        preds = {n: P[n][g] for n, _, _ in MODELS}
        mets = dict(gidx=g, truth=describe(e), beam_pe=float(np.expm1(t[:, :-1]).clip(min=0)[:, beam].sum()),
                    models={n: ev_metrics(preds[n], t) for n in preds})
        mets["meas_pe"] = mets["models"][MODELS[0][0]]["pe_meas"]
        out = f"{out_dir}/hs{k + 1:02d}.png"
        render(k, g, e, t, preds, mets, out)
        res.append(mets)
        print(out, flush=True)
    json.dump(res, open(f"{out_dir}/handscan.json", "w"), indent=1)
    # LaTeX: one row per event, total-PE error and TV per model; bold = lowest event MSE
    rows = []
    for k, m in enumerate(res):
        best = min(m["models"], key=lambda n: m["models"][n]["mse"])
        cells = []
        for n, _, _ in MODELS:
            v = m["models"][n]
            s = f"{v['dtot']:+.0%} / {v['tv']:.2f}".replace("%", r"\%") if np.isfinite(v["tv"]) else "--"
            cells.append(rf"\textbf{{{s}}}" if n == best else s)
        truth = m["truth"].replace(" (cosmics only)", "").replace("outside AV", "out of AV").replace(", E = ", ", ")
        for a, b in (("ν̄μ", r"$\bar\nu_\mu$"), ("ν̄e", r"$\bar\nu_e$"), ("νμ", r"$\nu_\mu$"), ("νe", r"$\nu_e$")):
            truth = truth.replace(a, b)
        rows.append(" & ".join([str(k + 1), truth, f"{m['meas_pe']:.0f}"] + cells) + r" \\")
    best = {n: sum(min(m["models"], key=lambda x: m["models"][x]["mse"]) == n for m in res) for n, _, _ in MODELS}
    med = {n: (np.median([m["models"][n]["mse"] for m in res]), np.nanmedian([abs(m["models"][n]["dtot"]) for m in res]),
               np.nanmedian([m["models"][n]["tv"] for m in res])) for n, _, _ in MODELS}
    rows.append(r"\midrule")
    rows.append(" & ".join([r"\multicolumn{3}{l}{median $|\Delta\Sigma|$ / TV}"]
                           + [f"{med[n][1]:.0%} / {med[n][2]:.2f}".replace("%", r"\%") for n, _, _ in MODELS]) + r" \\")
    rows.append(" & ".join([r"\multicolumn{3}{l}{median event MSE}"] + [f"{med[n][0]:.2f}" for n, _, _ in MODELS]) + r" \\")
    rows.append(" & ".join([r"\multicolumn{3}{l}{events with the lowest MSE}"] + [str(best[n]) for n, _, _ in MODELS])
                + r" \\")
    open(f"{tex_dir}/t5_handscan.tex", "w").write("\n".join(rows) + "\n")
    print("best:", best, "median:", {n: [round(float(x), 3) for x in v] for n, v in med.items()})


if __name__ == "__main__":
    main()
