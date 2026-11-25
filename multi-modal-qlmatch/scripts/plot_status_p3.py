#!/usr/bin/env python
"""Figures for the 2026-11-25 status deck (#16): P3 vs earlier SBND models on simulation (T1-T5) and real data.

Usage: python scripts/plot_status_p3.py OUT_DIR
Reads fine-tune metrics.json (sbnd_full/*, p3/ft/*) and sbnd_data/eval/report_all.jsonl + report_p3.json.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

W = "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work"
NW = "/lus/eagle/projects/LArTPC_FM/yuhw/fm4npp-lartpc-work"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3dd"
C = {"P3": "#d23f3f", "CL": "#2a78d6", "CXF": "#1baf7a", "UB": "#eda100", "SCR": "#8a8980", "WID": "#7b5cd6",
     "WCFT": "#e87ba4"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 220})
MODELS = [("P3", "P3 (3D blobs + new tokenizers)", f"{NW}/p3/ft/f0.01_s*"),
          ("CL", "SBND-pre M CL", f"{W}/sbnd_full/sbndpre_M_CL/finetune/f0.01_s*"),
          ("CXF", "SBND-pre M C+XF", f"{W}/sbnd_full/sbndpre_M_C_xf/finetune/f0.01_s*"),
          ("UB", "MicroBooNE-pre M CL", f"{W}/sbnd_full/ubpre_M_CL/finetune/f0.01_s*"),
          ("SCR", "no pretraining", f"{W}/sbnd_full/scratch_M_CL/finetune/f0.01_s*")]

# 1. simulation T1-T5
tests = {k: [json.load(open(f + "/metrics.json"))["test"] for f in sorted(glob.glob(p))] for k, _, p in MODELS}
panels = [("T1 hit F1", lambda t: t["hit"]["score"], (0.75, 1.0), 0.931),
          ("T3 flavour macro-F1", lambda t: t["flavor"]["score"], (0.6, 0.85), None),
          ("T4 energy resolution (lower = better)", lambda t: t["energy"]["resolution"], (0.15, 0.27), None),
          ("T5 light MSE (lower = better)", lambda t: t["light"]["mse"], (0.5, 1.0), None)]
fig, axs = plt.subplots(1, 4, figsize=(11, 2.9))
for ax, (title, f, lim, wc) in zip(axs, panels):
    for i, (k, lab, _) in enumerate(MODELS):
        v = [f(t) for t in tests[k]]
        ax.bar(i, np.mean(v), 0.7, color=C[k], label=lab)
        ax.plot([i, i], [min(v), max(v)], color=INK, lw=1)
    if wc:
        ax.axhline(wc, color=INK, ls="--", lw=1)
        ax.text(len(MODELS) - 0.5, wc + 0.004, "Wire-Cell", ha="right", fontsize=7)
    ax.set_ylim(*lim)
    ax.set_xticks([])
    ax.set_title(title, fontsize=8.5)
    ax.grid(axis="x", visible=False)
fig.legend(*axs[0].get_legend_handles_labels(), ncol=5, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(os.path.join(OUT, "p3_sim.png"))
plt.close(fig)

# 2. real data (held-out files) + width sensitivity
rows = [json.loads(l) for l in open(f"{W}/sbnd_data/eval/report_all.jsonl")] + json.load(open(f"{W}/sbnd_data/eval/report_p3.json"))
def key(r):
    s = r["run"]
    if "/p3/ft/" in s: return "P3"
    if "/wcft/sbndpre_M_CL/" in s: return "WCFT"
    if "/width/sbndpre_M_CL/" in s: return "WID"
    for k, sub in (("CL", "sbndpre_M_CL/finetune"), ("CXF", "sbndpre_M_C_xf/finetune"), ("UB", "ubpre_M_CL/finetune"),
                   ("SCR", "scratch_M_CL/finetune")):
        if "/width/" not in s and f"/sbnd_full/{sub}/" in s:
            return k
    return None
DM = [("P3", "P3"), ("CL", "SBND-pre M CL"), ("CXF", "SBND-pre M C+XF"), ("WID", "M CL, data-like-width FT (E6)"),
      ("WCFT", "M CL, real-data FT with WC labels (E7)"), ("SCR", "no pretraining")]
by = {k: [r for r in rows if key(r) == k] for k, _ in DM}
panels = [("sim T1 F1: nominal (dark) vs data-like widths (light)", None, (0.75, 1.0)),
          ("data: share of WC ν charge tagged", lambda r: r["data_test"]["hit/wc/beam_on"]["wc_in_model"], (0.3, 0.85)),
          ("data: beam-on selected at WC's off-beam rate", lambda r: r["data_test"]["evt/wc/all"]["model_on"], (0.1, 0.6))]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.0))
for ax, (title, f, lim) in zip(axs, panels):
    for i, (k, lab) in enumerate(DM):
        rs = by[k]
        if f is None:
            for j, (g, a) in enumerate(((lambda r: r["sim_nom"]["t1_f1"], 1.0), (lambda r: r["sim_w"]["t1_f1"], 0.45))):
                v = [g(r) for r in rs]
                ax.bar(i + (j - 0.5) * 0.38, np.mean(v), 0.36, color=C[k], alpha=a, label=lab if j == 0 else None)
                ax.plot([i + (j - 0.5) * 0.38] * 2, [min(v), max(v)], color=INK, lw=1)
        else:
            v = [f(r) for r in rs]
            ax.bar(i, np.mean(v), 0.7, color=C[k])
            ax.plot([i, i], [min(v), max(v)], color=INK, lw=1)
    if title.startswith("data: beam-on"):
        wc = by["P3"][0]["data_test"]["evt/wc/all"]["wc_on"]
        ax.axhline(wc, color=INK, ls="--", lw=1)
        ax.text(-0.45, wc + 0.01, f"Wire-Cell {100 * wc:.1f}%", ha="left", fontsize=7)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(*lim)
    ax.set_xticks([])
    ax.set_title(title, fontsize=8.5)
    ax.grid(axis="x", visible=False)
fig.legend(*axs[0].get_legend_handles_labels(), ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(os.path.join(OUT, "p3_data.png"))
plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
