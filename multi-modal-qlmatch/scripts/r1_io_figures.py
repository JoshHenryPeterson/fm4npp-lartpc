#!/usr/bin/env python
"""Round-1 tech-note figures: what the qlfm model takes as input and what the MAE predicts, on one event.

  r1_charge_input.png  charge hits -> occupied 16-wire x 64-tick cell tokens (3 planes)
  r1_charge_token.png  one charge token: its hits, per-hit features, centroid/position, 4x8 MAE target
  r1_light_input.png   optical hits -> 32 PMT tokens (16-bin log PE time profile + total); one PMT token
  r1_mae_io.png        MAE: masked/visible tokens, charge reconstructions, charge -> light prediction
                       (needs the GPU output of scripts/r1_mae_demo.py)

Tokenization uses the model's own code (qlfm.tokenize) on CPU.
Usage: r1_io_figures.py GIDX MAE_NPZ OUT_DIR
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from qlfm import geometry as G
from qlfm.data.dataset import Corpus, EventDataset, collate
from qlfm.tokenize import TokCfg, tokenize_charge, tokenize_light
from qlfm.paths import WORK as W

CFG = TokCfg()
PL = "UVY"


def load(gidx):
    c = Corpus(f"{W}/cache/bnb,{W}/cache/nue")
    it = EventDataset(c, [gidx], truth=True)[0]
    b = collate([it])
    ct = tokenize_charge(b["hit_plane"], b["hit_wire"], b["hit_time"], b["hit_q"], b["hit_rms"], b["hit_evt"], 1, CFG)
    lt = tokenize_light(b["oph_ch"], b["oph_t"], b["oph_pe"], b["oph_evt"], 1, CFG)
    s, j = c.locate(gidx)
    sh = c.shards[s]
    ev = {k: np.asarray(sh.ev(k)[j]) for k in ("run", "subrun", "event", "nu_pdg", "is_cc", "nu_energy", "nu_vtx")}
    return b, ct, lt, ev


def charge_input(b, ct, ev, out):
    p, w, t, q = (b[k].numpy() for k in ("hit_plane", "hit_wire", "hit_time", "hit_q"))
    nu = b["hit_nu_edep"].numpy() > 0.05
    cell = ct.tok_cell.numpy()
    tp = ct.tok_plane.numpy()
    tq = ct.tok_stat[:, 1].numpy()
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), gridspec_kw=dict(hspace=0.32, wspace=0.18))
    for k in range(3):
        m = p == k
        o = np.argsort(q[m])
        ax = axs[0, k]
        sc = ax.scatter(w[m][o], t[m][o], c=np.log10(1 + q[m][o]), s=0.4, lw=0, cmap="viridis", vmin=1, vmax=3.5,
                        rasterized=True)
        ax.set_xlim(0, G.N_WIRES[k])
        ax.set_ylim(6400, 0)
        ax.set_title(f"plane {PL[k]}: {m.sum():,} hits (input)", fontsize=10)
        ax.set_xlabel("wire", fontsize=8)
        ax.tick_params(labelsize=7)
        nw, ntc = G.N_WIRES[k] // CFG.cell_wires + 1, 6400 // CFG.cell_ticks + 1
        img = np.full((ntc, nw), np.nan)
        mm = tp == k
        img[np.clip(cell[mm, 1], 0, ntc - 1), np.clip(cell[mm, 0], 0, nw - 1)] = tq[mm]
        ax = axs[1, k]
        im = ax.imshow(img, origin="upper", aspect="auto", cmap="magma_r", interpolation="nearest",
                       extent=(0, nw * CFG.cell_wires, ntc * CFG.cell_ticks, 0), vmin=0, vmax=np.nanpercentile(tq, 99))
        ax.set_xlim(0, G.N_WIRES[k])
        ax.set_ylim(6400, 0)
        ax.set_title(f"plane {PL[k]}: {mm.sum():,} cell tokens", fontsize=10)
        ax.set_xlabel("wire", fontsize=8)
        ax.tick_params(labelsize=7)
        if k == 0:
            axs[0, 0].set_ylabel("tick (0.5 µs)", fontsize=8)
            axs[1, 0].set_ylabel("tick (0.5 µs)", fontsize=8)
        if nu[m].any():
            for a_ in axs[:, k]:
                x0, x1 = np.percentile(w[m & nu], [1, 99])
                y0, y1 = np.percentile(t[m & nu], [1, 99])
                a_.add_patch(Rectangle((x0 - 20, y0 - 60), x1 - x0 + 40, y1 - y0 + 120, fill=False, ls="--", lw=0.7,
                                       color="tab:red"))
    fig.colorbar(sc, ax=axs[0, :], fraction=0.015, pad=0.01).set_label("log10(1 + hit integral)", fontsize=8)
    fig.colorbar(im, ax=axs[1, :], fraction=0.015, pad=0.01).set_label("cell log(1 + Σq/200)", fontsize=8)
    fig.suptitle(f"Charge input of one event (run {int(ev['run'])}/{int(ev['subrun'])}/{int(ev['event'])}; true "
                 f"{'νe' if abs(int(ev['nu_pdg'])) == 12 else 'νμ'} {'CC' if ev['is_cc'] else 'NC'}, "
                 f"{float(ev['nu_energy']):.2f} GeV, red box = neutrino).  Top: reconstructed wire hits.  "
                 f"Bottom: one token per occupied {CFG.cell_wires}-wire × {CFG.cell_ticks}-tick cell "
                 f"({len(tq):,} tokens in total).", fontsize=9.5, y=0.995)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pick_token(b, ct):
    """A Y-plane token on the neutrino with many hits (the 'example token')."""
    nu = b["hit_nu_edep"].numpy() > 0.05
    h2t = ct.hit2tok.numpy()
    nt = len(ct.tok_plane)
    nh = np.bincount(h2t[h2t >= 0], minlength=nt)
    nnu = np.bincount(h2t[(h2t >= 0) & nu], minlength=nt)
    ok = (ct.tok_plane.numpy() == 2) & (nnu >= 0.8 * nh) & (nh >= 6) & (nh <= 30)
    cand = np.nonzero(ok)[0]
    return cand[np.argmax(nh[cand])] if len(cand) else int(np.argmax(nh))


def charge_token(b, ct, out):
    k = pick_token(b, ct)
    p, w, t, q, rms = (b[x].numpy() for x in ("hit_plane", "hit_wire", "hit_time", "hit_q", "hit_rms"))
    h2t = ct.hit2tok.numpy()
    wc, tc = ct.tok_cell[k].numpy()
    pl = int(ct.tok_plane[k])
    cw, ctk = CFG.cell_wires, CFG.cell_ticks
    sel = h2t == k
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1, 1, 1.25], height_ratios=[1, 1], hspace=0.35, wspace=0.32)
    # (a) neighbourhood with the cell grid
    ax = fig.add_subplot(gs[:, 0])
    W0, W1 = wc * cw - 4 * cw, wc * cw + 5 * cw
    T0, T1 = tc * ctk - 4 * ctk, tc * ctk + 5 * ctk
    m = (p == pl) & (w >= W0) & (w < W1) & (t >= T0) & (t < T1)
    ax.scatter(w[m], t[m], s=4 + 40 * np.log1p(q[m] / 200), c=np.log10(1 + q[m]), cmap="viridis", vmin=1, vmax=3.5,
               lw=0, zorder=3)
    for x in range(W0, W1 + 1, cw):
        ax.axvline(x, color="0.85", lw=0.6, zorder=1)
    for y in range(T0, T1 + 1, ctk):
        ax.axhline(y, color="0.85", lw=0.6, zorder=1)
    tm = (ct.tok_plane.numpy() == pl)
    cpos = ct.tok_cpos.numpy()
    cw_wire = cpos[:, 1] / G.WIRE_PITCH_CM
    cw_tick = cpos[:, 0] / G.TICK_TO_CM + G.TRIG_TICK
    inwin = tm & (cw_wire >= W0) & (cw_wire < W1) & (cw_tick >= T0) & (cw_tick < T1)
    ax.scatter(cw_wire[inwin], cw_tick[inwin], marker="x", s=25, c="tab:red", lw=1, zorder=4,
               label="token position (charge centroid)")
    ax.add_patch(Rectangle((wc * cw, tc * ctk), cw, ctk, fill=False, lw=2, color="tab:orange", zorder=5))
    ax.set_xlim(W0, W1)
    ax.set_ylim(T1, T0)
    ax.set_xlabel(f"wire (plane {PL[pl]})", fontsize=9)
    ax.set_ylabel("tick", fontsize=9)
    ax.set_title("(a) neighbourhood: hits, cell grid,\ntoken positions; orange = example token", fontsize=9.5)
    ax.legend(fontsize=7, loc="lower left")
    # (b) the token's hits in in-cell coordinates, with the 4x8 target sub-grid
    ax = fig.add_subplot(gs[0, 1])
    rw = (w[sel] - wc * cw) / cw
    rt = (t[sel] - tc * ctk) / ctk
    ax.scatter(rw, rt, s=10 + 120 * np.log1p(q[sel] / 200), c=np.log10(1 + q[sel]), cmap="viridis", vmin=1, vmax=3.5,
               edgecolor="k", lw=0.3, zorder=3)
    for x in np.linspace(0, 1, CFG.sub_w + 1):
        ax.axvline(x, color="0.7", lw=0.6, ls="--")
    for y in np.linspace(0, 1, CFG.sub_t + 1):
        ax.axhline(y, color="0.7", lw=0.6, ls="--")
    cr = ct.tok_crel[k].numpy() + 0.5
    ax.scatter([cr[0]], [cr[1]], marker="x", s=80, c="tab:red", lw=2, zorder=4, label="centroid")
    ax.scatter([0.5], [0.5], marker="+", s=80, c="0.4", lw=1.5, zorder=4, label="cell centre")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel(f"relative wire (×{cw} wires)", fontsize=8)
    ax.set_ylabel(f"relative tick (×{ctk} ticks)", fontsize=8)
    ax.set_title(f"(b) the token's {sel.sum()} hits (dashed:\n4×8 target sub-grid)", fontsize=9.5)
    ax.legend(fontsize=7, loc="lower right")
    # (c) target sub-grid
    ax = fig.add_subplot(gs[0, 2])
    tg = ct.tok_target[k].numpy().reshape(CFG.sub_w, CFG.sub_t).T
    im = ax.imshow(tg, cmap="magma_r", aspect="auto", extent=(0, 1, 1, 0), vmin=0)
    fig.colorbar(im, ax=ax, fraction=0.05).set_label("log(1 + Σq/200)", fontsize=8)
    st = ct.tok_stat[k].numpy()
    ax.set_title(f"(c) MAE charge target (34 values):\n4×8 grid + log n_hits = {st[0]:.2f}, "
                 f"log(1+Σq/200) = {st[1]:.2f}", fontsize=9.5)
    ax.set_xlabel("relative wire", fontsize=8)
    ax.set_ylabel("relative tick", fontsize=8)
    # (d) per-hit features (input of the hit MLP)
    ax = fig.add_subplot(gs[1, 1:3])
    ax.axis("off")
    f = ct.feat_hit.numpy()[sel]
    o = np.argsort(-q[sel])[:8]
    rows = [[f"{i}", f"{w[sel][j]}", f"{t[sel][j]:.0f}", f"{q[sel][j]:.0f}", f"{rms[sel][j]:.1f}",
             f"{f[j, 0]:+.2f}", f"{f[j, 1]:+.2f}", f"{f[j, 2]:.2f}", f"{f[j, 3]:.2f}"] for i, j in enumerate(o)]
    tb = ax.table(cellText=rows, colLabels=["hit", "wire", "tick", "q", "rms", "Δw", "Δt", "ln(1+q/200)", "rms/5"],
                  loc="center", cellLoc="center", colWidths=[0.06, 0.09, 0.09, 0.08, 0.08, 0.09, 0.09, 0.15, 0.09])
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.5)
    tb.scale(1, 1.25)
    ax.set_title(f"(d) per-hit input features (top {len(o)} hits by charge): raw → 4-vector → shared MLP 4→64→128\n"
                 f"Δw, Δt = position inside the cell (relative wire/tick − 0.5)", fontsize=9.5)
    # (e) token summary / embedding recipe
    ax = fig.add_subplot(gs[:, 3])
    ax.axis("off")
    cp = ct.tok_cpos[k].numpy()
    txt = (f"(e) token = cell (plane {PL[pl]}, wire-cell {wc}, time-cell {tc})\n\n"
           f"content embedding\n"
           f"  {sel.sum()} hit vectors → MLP → 128-d each\n"
           f"  pool: max ‖ mean → 256-d\n"
           f"  ‖ (log n_hits, log Σq) = ({st[0]:.2f}, {st[1]:.2f})\n"
           f"  → MLP → d (256 for S, 384 for M)\n\n"
           f"position embedding\n"
           f"  charge-weighted centroid:\n"
           f"    drift x = {cp[0]:.1f} cm,  wire coord = {cp[1]:.1f} cm\n"
           f"    → Fourier features (10 octaves)\n"
           f"  in-cell offset = ({ct.tok_crel[k, 0]:+.2f}, {ct.tok_crel[k, 1]:+.2f})\n"
           f"    → MLP\n"
           f"  + plane embedding ({PL[pl]})\n\n"
           f"+ modality embedding (charge)\n\n"
           f"token vector = content + position + modality\n"
           f"MAE: if masked, the decoder must predict\n"
           f"panel (c) from the other tokens")
    ax.text(0, 1, txt, va="top", ha="left", fontsize=8.6, family="DejaVu Sans Mono")
    fig.suptitle("One charge token: hits in a 16-wire × 64-tick cell (≈ 4.8 cm × 3.5 cm) become one vector",
                 fontsize=11, y=0.99)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return k


def light_input(b, lt, out):
    ch, tt, pe = (b[k].numpy() for k in ("oph_ch", "oph_t", "oph_pe"))
    pe = np.clip(pe, 0, None)
    edges = np.array(CFG.light_edges)
    feat = lt.feat.numpy()                                                   # [32, 17]
    tot = np.expm1(feat[:, -1])
    kp = int(np.argmax(tot))
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.1, 1], hspace=0.4, wspace=0.45)
    # (a) all optical hits
    ax = fig.add_subplot(gs[0, 0])
    m = (tt >= edges[0]) & (tt < edges[-1])
    ax.scatter(tt[m], ch[m], s=3 + 30 * np.sqrt(pe[m] / pe[m].max()), c=np.log10(pe[m] + 1), cmap="plasma", lw=0)
    for e in edges:
        ax.axvline(e, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel("optical-hit time (µs, readout clock)", fontsize=8)
    ax.set_ylabel("PMT channel", fontsize=8)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(-1, 32)
    ax.set_title(f"(a) input: {m.sum()} optical hits in the 0–24 µs window\n(grey lines: the 16 time bins)",
                 fontsize=9.5)
    # (b) token matrix
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(feat, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(17))
    ax.set_xticklabels([f"{edges[i]:g}–{edges[i + 1]:g}" for i in range(16)] + ["total"], rotation=90, fontsize=6)
    ax.set_ylabel("PMT channel = token", fontsize=8)
    ax.axhline(kp, color="tab:red", lw=0.8)
    fig.colorbar(im, ax=ax, fraction=0.04).set_label("log(1 + PE)", fontsize=8)
    ax.set_title("(b) 32 light tokens × 17 features\n(16 time bins in µs + total PE)", fontsize=9.5)
    # (c) PMT map
    ax = fig.add_subplot(gs[0, 2])
    ax.add_patch(Rectangle((0, -116.5), 1036.8, 233, fill=False, color="0.5", lw=0.6))
    ax.scatter(G.PMT_YZ[:, 1], G.PMT_YZ[:, 0], s=50, facecolors="none", edgecolors="0.7", lw=0.5)
    ax.scatter(G.PMT_YZ[:, 1], G.PMT_YZ[:, 0], s=300 * tot / tot.max(), c="tab:orange", alpha=0.6,
               edgecolors="tab:red", lw=0.4)
    ax.scatter([G.PMT_YZ[kp, 1]], [G.PMT_YZ[kp, 0]], s=160, facecolors="none", edgecolors="k", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-30, 1070)
    ax.set_ylim(-140, 140)
    ax.set_xlabel("z [cm]", fontsize=8)
    ax.set_ylabel("y [cm]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("(c) token positions: PMT (y, z); circle area ∝ total PE\n(black ring: example PMT)", fontsize=9.5)
    # (d) one PMT: raw hits and its feature vector
    ax = fig.add_subplot(gs[1, 0:2])
    mk = m & (ch == kp)
    ax.vlines(tt[mk], 0, pe[mk], color="tab:blue", lw=1.2, label="optical hits (PE)")
    ax.set_xlabel("time (µs)", fontsize=8)
    ax.set_ylabel("PE per optical hit", fontsize=8, color="tab:blue")
    ax.set_xlim(edges[0], edges[-1])
    for e in edges:
        ax.axvline(e, color="0.85", lw=0.6, zorder=0)
    ax2 = ax.twinx()
    ax2.stairs(feat[kp, :16], edges, color="tab:red", lw=1.8, label="token feature: log(1 + PE per bin)")
    ax2.set_ylabel("log(1 + PE in bin)", fontsize=8, color="tab:red")
    ax.set_title(f"(d) one light token, PMT {kp} at (y, z) = ({G.PMT_YZ[kp, 0]:.0f}, {G.PMT_YZ[kp, 1]:.0f}) cm: "
                 f"{mk.sum()} optical hits → 16 binned values + log(1 + total PE) = {feat[kp, 16]:.2f}", fontsize=9.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper center")
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(0, 1, ("light token embedding\n\n"
                   "  17 features → MLP 17→d→d\n"
                   "  + Fourier(PMT y, z)\n"
                   "  + PMT-ID embedding (32)\n"
                   "  + modality embedding (light)\n\n"
                   "always 32 tokens per event,\n"
                   "next to ~2,000 charge tokens;\n"
                   "both attend to each other in\n"
                   "every encoder layer\n\n"
                   "MAE: per event either 50 % of\n"
                   "PMTs (p=0.4), all 32 (p=0.4)\n"
                   "or none (p=0.2) are masked;\n"
                   "the decoder predicts the 17\n"
                   "values of each masked PMT"), va="top", ha="left", fontsize=8.6, family="DejaVu Sans Mono")
    fig.suptitle("Light input: optical hits of each PMT become one token (its PE time profile)", fontsize=11, y=0.99)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def mae_io(npz, out):
    d = np.load(npz)
    cell, tp, mc = d["tok_cell"], d["tok_plane"], d["mask_c"]
    tg, st, pc = d["tok_target"], d["tok_stat"], d["pred_c"]
    sw, stt = d["sub"]
    edges = d["light_edges"]
    fig = plt.figure(figsize=(15, 11.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 0.9, 1.1], hspace=0.5, wspace=0.3)
    # (a) Y plane: visible vs masked tokens, and the reconstruction
    k = 2
    nw, ntc = G.N_WIRES[k] // CFG.cell_wires + 1, 6400 // CFG.cell_ticks + 1
    mm = tp == k
    vmax = np.percentile(st[:, 1], 99)
    for i, (title, val) in enumerate([("encoder input: grey = visible,\nred = masked", st[:, 1]),
                                      ("target: all tokens\n(total charge per cell)", st[:, 1]),
                                      ("output: visible +\nreconstructed masked", np.where(mc, pc[:, -1], st[:, 1]))]):
        ax = fig.add_subplot(gs[0, i])
        img = np.full((ntc, nw), np.nan)
        msk = np.zeros((ntc, nw))
        sel = mm & (~mc if i == 0 else np.ones_like(mc))
        img[np.clip(cell[sel, 1], 0, ntc - 1), np.clip(cell[sel, 0], 0, nw - 1)] = val[sel]
        ax.imshow(img, origin="upper", aspect="auto", cmap="magma_r", interpolation="nearest", vmin=0, vmax=vmax,
                  extent=(0, nw * CFG.cell_wires, ntc * CFG.cell_ticks, 0))
        if i == 0:                                            # replace by a categorical visible / masked map
            ax.images[-1].remove()
            cat = np.full((ntc, nw), np.nan)
            cat[np.clip(cell[mm & ~mc, 1], 0, ntc - 1), np.clip(cell[mm & ~mc, 0], 0, nw - 1)] = 0
            cat[np.clip(cell[mm & mc, 1], 0, ntc - 1), np.clip(cell[mm & mc, 0], 0, nw - 1)] = 1
            from matplotlib.colors import ListedColormap
            ax.imshow(cat, origin="upper", aspect="auto", cmap=ListedColormap(["0.35", "tab:red"]), vmin=0, vmax=1,
                      interpolation="nearest", extent=(0, nw * CFG.cell_wires, ntc * CFG.cell_ticks, 0))
        ax.set_ylim(6400, 0)
        ax.set_title(f"({'abc'[i]}) plane Y {title}", fontsize=9)
        ax.set_xlabel("wire", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("tick", fontsize=8)
    ax = fig.add_subplot(gs[0, 3])
    m2 = mc
    ax.scatter(st[m2, 1], pc[m2, -1], s=3, alpha=0.4, c="tab:blue", lw=0, label="log(1+Σq/200)")
    ax.scatter(st[m2, 0], pc[m2, -2], s=3, alpha=0.4, c="tab:green", lw=0, label="log n_hits")
    lim = [0, max(st[m2].max(), pc[m2, -2:].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.7)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("target", fontsize=8)
    ax.set_ylabel("prediction", fontsize=8)
    ax.legend(fontsize=7, markerscale=3)
    ax.set_title(f"(d) all {m2.sum()} masked tokens\n(of {len(mc)}): cell summaries", fontsize=9)
    # (e) example masked cells: target vs prediction sub-grids
    cand = np.nonzero(mc & (st[:, 0] >= np.log(8)))[0]
    ex = cand[np.argsort(-st[cand, 1])][[0, len(cand) // 3, 2 * len(cand) // 3]] if len(cand) >= 3 else cand[:3]
    for i, j in enumerate(ex):
        sub = gs[1, i].subgridspec(1, 2, wspace=0.1)
        vm = max(tg[j].max(), pc[j, :sw * stt].max(), 0.1)
        for r, (lab, v) in enumerate([("target", tg[j]), ("pred.", pc[j, :sw * stt])]):
            ax = fig.add_subplot(sub[0, r])
            ax.imshow(np.clip(v, 0, None).reshape(sw, stt).T, cmap="magma_r", vmin=0, vmax=vm, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{lab}", fontsize=8)
            if r == 0:
                ax.set_ylabel(f"masked cell {i + 1}\n({PL[tp[j]]}, n_hits={np.exp(st[j, 0]):.0f})", fontsize=7.5)
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ax.text(0, 1, ("(e) masked cells: the 4×8 in-cell\n"
                   "charge pattern (log(1+Σq/200))\n"
                   "predicted by the decoder only from\n"
                   "the visible charge + light tokens\n"
                   "and the masked cell's position.\n\n"
                   "Masking: 60 % of charge tokens,\n"
                   "half in 256-tick drift strips cut\n"
                   "across all three planes.\n\n"
                   "Predictions are smoother than the\n"
                   "truth: an MSE decoder predicts the\n"
                   "average of the plausible patterns."), va="top", fontsize=8.6, family="DejaVu Sans Mono")
    # (f) light: all 32 PMTs masked -> predicted from charge
    lf, pl_ = d["light_feat"], d["pred_l"]
    tot_t, tot_p = np.expm1(lf[:, -1]), np.expm1(np.clip(pl_[:, -1], 0, None))
    lsub = gs[2, 0:2].subgridspec(2, 1, hspace=0.6)
    for i, (lab, tot) in enumerate([("true", tot_t), ("predicted from charge only", tot_p)]):
        ax = fig.add_subplot(lsub[i, 0])
        ax.add_patch(Rectangle((0, -116.5), 1036.8, 233, fill=False, color="0.5", lw=0.6))
        ax.scatter(G.PMT_YZ[:, 1], G.PMT_YZ[:, 0], s=50, facecolors="none", edgecolors="0.7", lw=0.5)
        ax.scatter(G.PMT_YZ[:, 1], G.PMT_YZ[:, 0], s=300 * tot / max(tot_t.max(), tot_p.max()), c="tab:orange",
                   alpha=0.6, edgecolors="tab:red", lw=0.4)
        ax.set_aspect("equal")
        ax.set_xlim(-30, 1070)
        ax.set_ylim(-140, 140)
        ax.tick_params(labelsize=7)
        if i == 1:
            ax.set_xlabel("z [cm]", fontsize=8)
        ax.set_title(f"({'fg'[i]}) light with all 32 PMTs masked — {lab}: total PE per PMT "
                     f"(sum {tot.sum():,.0f} PE)", fontsize=9)
    ax = fig.add_subplot(gs[2, 2:])
    top = np.argsort(-tot_t)[:4]
    cols = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    for c_, kp in zip(cols, top):
        ax.stairs(lf[kp, :16], edges, color=c_, lw=1.6, label=f"PMT {kp} true")
        ax.stairs(np.clip(pl_[kp, :16], 0, None), edges, color=c_, lw=1.2, ls="--", label=f"PMT {kp} predicted")
    ax.set_xlabel("time (µs)", fontsize=8)
    ax.set_ylabel("log(1 + PE in bin)", fontsize=8)
    ax.set_xlim(0, 12)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("(h) 4 brightest PMTs: true (solid) vs predicted from charge (dashed)\n"
                 "charge cannot tell when in the 1.6 µs beam spill the ν interacted → flash smeared over 3.5–5 µs",
                 fontsize=9)
    fig.suptitle("Masked autoencoder on the same event (qlfm-S CL, round 1): input → output", fontsize=11, y=0.995)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    gidx, npz, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    b, ct, lt, ev = load(gidx)
    charge_input(b, ct, ev, f"{out}/r1_charge_input.png")
    k = charge_token(b, ct, f"{out}/r1_charge_token.png")
    light_input(b, lt, f"{out}/r1_light_input.png")
    print("example token", k, "tokens", len(ct.tok_plane))
    if os.path.exists(npz):
        mae_io(npz, f"{out}/r1_mae_io.png")
        print("MAE figure done")


if __name__ == "__main__":
    main()
