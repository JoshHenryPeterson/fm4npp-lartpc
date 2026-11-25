#!/usr/bin/env python
"""Figures for the SBND real-data campaign slides (#14), from the first-look and model-vs-Wire-Cell outputs.

Usage: python scripts/plot_sbnd_realdata_campaign.py OUT_DIR
Reads $QLFM_WORK/sbnd_data/firstlook.json and $QLFM_WORK/sbnd_data/eval/report_all.jsonl.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

W = os.environ.get("QLFM_WORK", "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # categorical slots 1-5, fixed order
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3dd"
TICK_TO_CM = 0.07815
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 220})

fl = json.load(open(os.path.join(W, "sbnd_data", "firstlook.json")))

# 1. hit-width ratio data / sim vs drift, per plane
e = np.array(fl["drift_edges_ticks"])
x = 0.5 * (e[1:] + e[:-1]) * TICK_TO_CM
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for p, (name, c) in enumerate(zip(("U", "V", "Y"), C)):
    ax.plot(x, fl["on"][f"rms_ratio_to_sim_p{p}_drift"], "-o", color=c, lw=2, ms=5, label=f"{name}, beam-on")
    ax.plot(x, fl["off"][f"rms_ratio_to_sim_p{p}_drift"], "--s", color=c, lw=1.5, ms=4, mfc="white",
            label=f"{name}, off-beam")
ax.axhline(1.0, color=MUTED, lw=1)
ax.set_xlabel("drift distance from the anode (cm)")
ax.set_ylabel("median hit RMS, data / sim")
ax.set_ylim(0.94, 1.02)
ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "width_ratio.png"))
plt.close(fig)

# 2. light time profile on the beam clock (shape: fraction of PE per bin)
t = np.array(fl["t_edges_us"])
tc = 0.5 * (t[1:] + t[:-1])
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.axvspan(*fl["beam_window_us"], color="#f1f0ea", zorder=0, label="beam window 0-3 µs")
for (k, lab), c, ls in zip((("on", "beam-on"), ("off", "off-beam"), ("sim", "BNB CV sim")), C, ("-", "--", ":")):
    h = np.array(fl[k]["light_profile_pe_per_event"])
    ax.plot(tc, h / h.sum(), ls, color=c, lw=2, label=lab, drawstyle="steps-mid")
ax.set_xlim(-4, 8)
ax.set_xlabel("optical-hit time, beam clock (µs)")
ax.set_ylabel("fraction of PE per 0.25 µs")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "light_profile.png"))
plt.close(fig)

# 3 + 4: model results (E4: the 13 simulation-trained models on all data events)
rows_all = [json.loads(l) for l in open(os.path.join(W, "sbnd_data", "eval", "report_all.jsonl"))]


def kind(r):
    """'base' (E2/E4 models), 'width' (E6: fine-tuned with data-like widths) or 'wcft' (E7: real data + WC labels)."""
    return "wcft" if "/sbnd_data/wcft/" in r["run"] else "width" if "/sbnd_full/width/" in r["run"] else "base"


FAM = [("sbndpre_M_CL/finetune", "SBND-pre CL, fine-tuned"), ("sbndpre_M_C_xf/finetune", "SBND-pre C+XF"),
       ("ubpre_M_CL/finetune", "MicroBooNE-pre CL"), ("sbndpre_M_CL/frozen", "SBND-pre CL, frozen"),
       ("scratch_M_CL/finetune", "no pretraining")]
MK = ["o", "s", "D", "^", "v"]
rows = [r for r in rows_all if kind(r) == "base"]
fam = lambda r: next(i for i, (k, _) in enumerate(FAM) if f"/sbnd_full/{k}/" in r["run"])

fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot([0.3, 1.0], [0.3, 1.0], color=MUTED, lw=1, ls="--")
ax.text(0.95, 0.92, "data = sim", color=MUTED, fontsize=7, ha="right", rotation=38)
for i, (k, lab) in enumerate(FAM):
    rr = [r for r in rows if fam(r) == i]
    ax.plot([r["sim_nom"]["wc_in_model"] for r in rr], [r["data"]["hit/wc/beam_on"]["wc_in_model"] for r in rr],
            MK[i], color=C[i], ms=7, mec="white", mew=1, label=lab, linestyle="none")
ax.set_xlim(0.72, 0.95)
ax.set_ylim(0.3, 0.95)
ax.set_xlabel("simulation test: share of WC ν charge also tagged")
ax.set_ylabel("beam-on data: same share")
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "agreement.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 2.9))
wc = rows[0]["data"]["evt/wc/all"]["wc_on"]
labels, vals = ["Wire-Cell"], [[wc]]
for i, (k, lab) in enumerate(FAM):
    labels.append(lab)
    vals.append([r["data"]["evt/wc/all"]["model_on"] for r in rows if fam(r) == i])
y = np.arange(len(labels))[::-1]
for j, (yy, v) in enumerate(zip(y, vals)):
    col = INK if j == 0 else C[j - 1]
    m = float(np.mean(v))
    ax.barh(yy, m, height=0.55, color=col, alpha=0.9 if j else 0.75)
    if len(v) > 1:
        ax.plot([min(v), max(v)], [yy, yy], color=INK, lw=1.2)
    ax.text(max(v) + 0.01, yy, f"{100 * m:.1f}%" + (f" ({100 * min(v):.1f}-{100 * max(v):.1f})" if len(v) > 1 else ""),
            va="center", fontsize=7, color=INK)
off = rows[0]["data"]["evt/wc/all"]["wc_off"]
ax.axvline(off, color=MUTED, lw=1, ls=":")
ax.text(off + 0.005, y[0] + 0.45, f"off-beam {100 * off:.1f}%", fontsize=7, color=MUTED)
ax.set_yticks(y, labels, fontsize=7.5)
ax.set_xlim(0, 0.62)
ax.set_ylim(y[-1] - 0.6, y[0] + 0.8)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("beam-on pass rate, off-beam fixed at 5.8%")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "event_level.png"))
plt.close(fig)

# 5. E5 width test: share of WC nu charge tagged, simulation nominal vs data-like widths vs beam-on data (held-out)
fig, ax = plt.subplots(figsize=(6.2, 2.9))
xs = np.arange(len(FAM))
bw = 0.26
for j, (lab, get, col) in enumerate((
        ("simulation, nominal widths", lambda r: r["sim_nom"]["wc_in_model"], C[0]),
        ("simulation, data-like widths (V×0.97, Y×0.955)", lambda r: r["sim_w"]["wc_in_model"], C[1]),
        ("beam-on data (held-out files)", lambda r: r["data_test"]["hit/wc/beam_on"]["wc_in_model"], INK))):
    for i in range(len(FAM)):
        v = [get(r) for r in rows if fam(r) == i]
        ax.bar(xs[i] + (j - 1) * bw, np.mean(v), bw * 0.92, color=col, alpha=0.85 if j < 2 else 0.7,
               label=lab if i == 0 else None)
        if len(v) > 1:
            ax.plot([xs[i] + (j - 1) * bw] * 2, [min(v), max(v)], color=INK, lw=1)
ax.set_xticks(xs, [l.replace(", fine-tuned", "").replace(", ", "\n").replace("-pre ", "-pre\n") for _, l in FAM], fontsize=7.5)
ax.set_ylim(0, 1.0)
ax.set_ylabel("share of WC ν charge also tagged")
ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "width_test.png"))
plt.close(fig)

# 6. E6 / E7: the fixes, on the held-out data files (CL and C+XF, SBND-pretrained)
VAR = [("base", "simulation FT", C[0]), ("width", "data-like-width FT (E6)", C[1]), ("wcft", "real-data FT, WC labels (E7)", C[2])]
ARCH = [("sbndpre_M_CL", "SBND-pre CL"), ("sbndpre_M_C_xf", "SBND-pre C+XF")]
pick = lambda v, a: [r for r in rows_all if kind(r) == v and f"/{a}/" in r["run"] and "/frozen/" not in r["run"]]
fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
for ax, (title, get, ref) in zip(axs, (
        ("hit level: share of WC ν charge tagged", lambda r: r["data_test"]["hit/wc/beam_on"]["wc_in_model"], None),
        ("event level: beam-on pass rate", lambda r: r["data_test"]["evt/wc/all"]["model_on"],
         rows[0]["data_test"]["evt/wc/all"]["wc_on"]))):
    for i, (a_, alab) in enumerate(ARCH):
        for j, (v_, vlab, col) in enumerate(VAR):
            vals = [get(r) for r in pick(v_, a_)]
            x = i + (j - 1) * 0.27
            ax.bar(x, np.mean(vals), 0.25, color=col, alpha=0.85, label=vlab if i == 0 else None)
            ax.plot([x, x], [min(vals), max(vals)], color=INK, lw=1)
    if ref is not None:
        ax.axhline(ref, color=INK, lw=1, ls="--")
        ax.text(-0.45, ref + 0.012, f"Wire-Cell {100 * ref:.1f}%", fontsize=7, color=INK, ha="left")
    ax.set_xticks(range(len(ARCH)), [l for _, l in ARCH], fontsize=8)
    ax.set_ylim(0, 1.0 if ref is None else 0.65)
    ax.set_title(title, fontsize=8.5)
    ax.grid(axis="x", visible=False)
axs[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
fig.legend(*axs[0].get_legend_handles_labels(), fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(os.path.join(OUT, "fixes.png"))
plt.close(fig)

# 7. light tokens: PMT layout per TPC and the 16 token time bins over the measured light profiles
det = json.load(open(os.path.join(os.path.dirname(__file__), "..", "sbnd-h5maker", "sbnd_detector.json")))
pm = sorted((o for o in det["opdets"] if o["type"].startswith("pmt")), key=lambda o: o["channel"])
EDGES = (-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 16.0, 21.0)   # geometry.LIGHT_EDGES
fig = plt.figure(figsize=(9.6, 3.2))
gsl = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.5), wspace=0.32, left=0.06, right=0.99, top=0.86, bottom=0.17)
for k, (side, lab) in enumerate(((False, "TPC 0 (x = −213.75 cm)"), (True, "TPC 1 (x = +213.75 cm)"))):
    ax = fig.add_subplot(gsl[0, k])
    for typ, mk, col, tl in (("pmt_coated", "o", C[0], "coated (96)"), ("pmt_uncoated", "s", C[1], "uncoated (24)")):
        q = [o for o in pm if (o["x"] > 0) == side and o["type"] == typ]
        ax.scatter([o["z"] for o in q], [o["y"] for o in q], s=26, marker=mk, color=col, edgecolors="white", lw=0.5,
                   label=tl if k == 0 else None, zorder=3)
    ax.add_patch(plt.Rectangle((0, -200), 500, 400, fill=False, ec=MUTED, lw=1))
    ax.set_xlim(-20, 520); ax.set_ylim(-230, 230); ax.set_aspect("equal")
    ax.set_xlabel("z (cm)", fontsize=8); ax.set_ylabel("y (cm)", fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title(lab + ": 60 PMTs", fontsize=8.5)
fig.axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(1.1, 1.32), ncol=2)
ax = fig.add_subplot(gsl[0, 2])
e = np.asarray(EDGES)
ax.axvspan(0, 3, color="#e9e3c8", zorder=0, label="beam window 0-3 µs")
ax.axvspan(12, 21, color="#f1f0ea", zorder=0)
ax.text(16.5, 0.5, "not in the\nfirst-look\nprofile", transform=ax.get_xaxis_transform(), ha="center", va="center",
        fontsize=6.5, color=MUTED)
t = np.array(fl["t_edges_us"])
for (key, lab), c, ls in zip((("on", "beam-on"), ("off", "off-beam"), ("sim", "BNB CV sim")), C, ("-", "--", ":")):
    h = np.array(fl[key]["light_profile_pe_per_event"])          # PE per event per 0.25 us
    tok = np.array([h[(t[:-1] >= e[i]) & (t[1:] <= e[i + 1])].sum() / (e[i + 1] - e[i]) if e[i + 1] <= t[-1] else np.nan
                    for i in range(len(e) - 1)])
    ax.stairs(np.nan_to_num(tok, nan=0), e, color=c, lw=1.8, ls=ls, label=lab)
for x in e:
    ax.axvline(x, color=MUTED, lw=0.5, zorder=1)
ax.set_xlim(-3, 21)
ax.set_yscale("log")
ax.set_xlabel("optical-hit time, beam clock (µs): 16 token bins", fontsize=8)
ax.set_ylabel("PE / µs per event, all PMTs", fontsize=8)
ax.tick_params(labelsize=7)
ax.set_title("16 token time bins and the mean light", fontsize=8.5)
ax.legend(fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.92))
fig.savefig(os.path.join(OUT, "light_tokens.png"))
plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
