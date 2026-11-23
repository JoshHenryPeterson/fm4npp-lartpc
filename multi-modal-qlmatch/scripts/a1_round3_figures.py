#!/usr/bin/env python
"""Bar-chart PDFs for docs/technote_a1_round3.md (sections 2.2, 2.3, 3.2-3.5).

Same style as scripts/a1_round2_figures.py. Plotted numbers come from the source data
(a1r3/novnorm/*/metrics.json, a1r3/novnorm/*/t1_npz_*.json, a1r3/summary.json); every
plotted bar is cross-checked against the Markdown tables of the tech note (tol 2e-4).
Mean over seeds, error bars = seed SD (ddof = 1); single-seed bars have no error bar.
Writes vector PDFs, a1_round3_figures_data.json and README.md to OUT.
Optional: set A1R3_PNG_DIR to also write PNG previews (default /tmp/a1r3_fig_preview).
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE = os.path.join(HERE, "..", "docs", "technote_a1_round3.md")
WORK = "/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work"
OUT = WORK + "/a1_round3/figures"
PNG = os.environ.get("A1R3_PNG_DIR", "/tmp/a1r3_fig_preview")
NOTE_NAME = "technote_a1_round3.md"
C_CL, C_XF = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"
MODELS = [("CL", C_CL), ("C+XF", C_XF)]
RUNDIR = {"CL": "CL", "C+XF": "C_xf"}
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "pdf.fonttype": 42})
os.makedirs(OUT, exist_ok=True)

LINES = open(NOTE, encoding="utf-8").read().split("\n")
TOL = 2e-4
CHECK = []
DATA = {}
README = []


def die(msg):
    raise SystemExit("PARSE ERROR: " + msg)


def section(prefix):
    s = [i for i, l in enumerate(LINES) if l.startswith("### " + prefix)]
    if len(s) != 1:
        die("section " + prefix)
    e = next(i for i in range(s[0] + 1, len(LINES)) if LINES[i].startswith("#"))
    return LINES[s[0] + 1:e]


def cell(t):
    t = t.replace("**", "").replace("−", "-").replace("`", "").strip()
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)(?:\s*±\s*(\d+(?:\.\d+)?))?", t)
    if not m:
        die("cell %r" % t)
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else None)


def parse_table(lines, start):
    """Parse the first pipe table at/after index start; return (cols, {rowkey: {col: text}}, end)."""
    j = start
    while j < len(lines) and not lines[j].startswith("|"):
        j += 1
    cols = [c.strip().replace("`", "") for c in lines[j].strip().strip("|").split("|")]
    j += 2
    rows = {}
    while j < len(lines) and lines[j].startswith("|"):
        c = [x.strip() for x in lines[j].strip().strip("|").split("|")]
        if len(c) != len(cols):
            die("column count")
        key = " ".join(x.replace("`", "").replace("**", "") for x in c[:1])
        rows[key] = {cols[k]: c[k] for k in range(1, len(cols))}
        j += 1
    if not rows:
        die("empty table")
    return cols, rows, j


def model_tables(sec):
    """{(model, kind): rows} for tables headed by '**qlfm-M CL**' / '**qlfm-M C + XF**' lines."""
    res = {}
    for i, l in enumerate(sec):
        if l.startswith("**qlfm-M"):
            mod = "CL" if l.startswith("**qlfm-M CL") else "C+XF"
            _, rows, _ = parse_table(sec, i + 1)
            res[mod] = rows
    return res


def msd(vals):
    a = np.array(vals, float)
    return (float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def check(tag, model, var, got, ref, tol=TOL):
    dev = abs(got - ref)
    CHECK.append((tag, model, var, got, ref, dev))
    if dev >= tol:
        raise SystemExit("CROSS-CHECK FAILED: %s %s %s plotted %.5f vs table %.5f (dev %.2e)"
                         % (tag, model, var, got, ref, dev))


def check_sd(tag, model, var, got, ref):
    check(tag + " (SD)", model, var, got, ref, 2.5e-4)


def readme(name, src, desc):
    README.append("- `%s`: %s (source: %s)" % (name, desc, src))


# ------------------------------------------------------------------ style (as round 2)
def style(ax, ylabel=None, zero=True):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    if zero:
        ax.axhline(0, color=INK, lw=0.8, zorder=3)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(length=2.5)


def bars(ax, x, vals, color, w=0.38):
    xs, ms, es = [], [], []
    for xi, v in zip(x, vals):
        if v is None:
            continue
        xs.append(xi); ms.append(v[0]); es.append(v[1] or 0)
    ax.bar(xs, ms, width=w, color=color, zorder=2, linewidth=0)
    pts = [(a, b, c) for a, b, c in zip(xs, ms, es) if c > 0]
    if pts:
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], fmt="none",
                    ecolor=INK, elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=4)


def pair_axes(ax, names, cl, xf, mono=True, rot=0, fs=8):
    n = len(names)
    off = 0.21
    x = list(range(n))
    bars(ax, [i - off for i in x], cl, C_CL)
    bars(ax, [i + off for i in x], xf, C_XF)
    ax.set_xticks(x)
    ax.set_xticklabels(names, family="monospace" if mono else None, fontsize=fs,
                       rotation=rot, ha="right" if rot else "center")
    ax.set_xlim(-0.6, n - 0.4)


def legend(fig, y=0.99, nominal=False, top=0.76, fs=None, nomlabel=" nominal"):
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MODELS]
    l = [m for m, _ in MODELS]
    if nominal:
        for m, c in MODELS:
            h.append(plt.Line2D([0], [0], color=c, lw=0.9, ls="--")); l.append(m + nomlabel)
    fig.subplots_adjust(top=top)
    kw = {"fontsize": fs} if fs else {}
    fig.legend(h, l, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, y),
               handlelength=1.6, **kw)


def save(fig, fn):
    fig.savefig(os.path.join(OUT, fn), bbox_inches="tight")
    if PNG:
        os.makedirs(PNG, exist_ok=True)
        fig.savefig(os.path.join(PNG, fn.replace(".pdf", ".png")), bbox_inches="tight", dpi=100)
    plt.close(fig)


def rec(vals):
    return [None if v is None else {"mean": v[0], "sd": v[1]} for v in vals]


FOOT = "Mean over 3 seeds; error bars = seed SD. Source: " + NOTE_NAME + " "

# ------------------------------------------------------------------ load part-1 runs
RUNS = {}
T1J = {}
for m, d in RUNDIR.items():
    RUNS[m] = []
    T1J[m] = []
    for s in range(3):
        p = "%s/a1r3/novnorm/s3_mt_qlfm_M_%s_finetune_f0.01_s%d" % (WORK, d, s)
        RUNS[m].append(json.load(open(p + "/metrics.json"))["results"])
        T1J[m].append(json.load(open(p + "/t1_npz_w_meas3d_noVnorm.json")))
R2 = {m: [json.load(open("%s/a1r2/full/s3_mt_qlfm_M_%s_finetune_f0.01_s%d/metrics.json" % (WORK, d, s)))["results"]
          for s in range(3)] for m, d in RUNDIR.items()}

# Gate-5-style reference check: w_meas3d in a1r3/novnorm reproduces the a1r2/full run
g5 = 0.0
for m, _ in MODELS:
    for r3, r2 in zip(RUNS[m], R2[m]):
        g5 = max(g5, abs(r3["hit"]["w_meas3d"]["score"] - r2["hit"]["w_meas3d"]["score"]),
                 abs(r3["evt_nu"]["w_meas3d"]["score"] - r2["evt_nu"]["w_meas3d"]["score"]))
if g5 > 2e-3:
    raise SystemExit("w_meas3d in novnorm does not reproduce a1r2/full (max dev %.4f)" % g5)

s22 = model_tables(section("2.2"))
s23 = model_tables(section("2.3"))

VAR = ["nominal", "w_meas3d", "w_meas3d_noVnorm"]
COLS = [("T1 charge-wtd", "T1 (charge-weighted F1)", lambda r, v: r["hit"][v]["score"]),
        ("T1 evt-median", "T1 (per-event median F1)", lambda r, v: r["hit"][v]["evt_med"]["score"]),
        ("T2 all", "T2 (AUC), all events", lambda r, v: r["evt_nu"][v]["score"])]

# ================================================================ Fig A: absolute, 2.2
fig, axs = plt.subplots(1, 3, figsize=(8.4, 3.7), sharey=True, gridspec_kw={"wspace": 0.08})
recA = {}
for ax, (col, ttl, get) in zip(axs, COLS):
    res = {}
    for m, _ in MODELS:
        vals = []
        nomd = cell(s22[m]["nominal"][col])
        for v in VAR:
            ms = msd([get(r, v) for r in RUNS[m]])
            ref = nomd[0] if v == "nominal" else nomd[0] + cell(s22[m][v][col])[0]
            check("FigA", m, "%s/%s" % (col, v), ms[0], ref)
            if v == "nominal":
                check_sd("FigA", m, "%s/%s" % (col, v), ms[1], nomd[1])
            vals.append(ms)
        res[m] = vals
    pair_axes(ax, VAR, res["CL"], res["C+XF"], rot=30)
    style(ax, zero=False)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for m, c in MODELS:
        ax.axhline(res[m][0][0], color=c, lw=0.9, ls="--", zorder=3)
    ax.set_title(ttl, fontsize=9, color=INK2)
    recA[col] = {"x": VAR, "CL": rec(res["CL"]), "C+XF": rec(res["C+XF"])}
axs[0].set_ylabel("score (absolute)")
legend(fig, 0.99, nominal=True, top=0.76)
fig.suptitle("Part 1: absolute T1 and T2 under 3D hit-width variants (V table normalised vs not)",
             fontsize=10, y=1.02)
fig.text(0.01, -0.13, FOOT + "§2.2 and per-run metrics.json (a1r3/novnorm). Dashed lines: nominal per model. "
         "T2 bright omitted.", fontsize=7.5, color=INK2, ha="left", va="top")
save(fig, "a1r3_noVnorm_abs.pdf")
DATA["a1r3_noVnorm_abs.pdf"] = {"source": "per-run metrics.json (a1r3/novnorm), checked vs §2.2 nominal+delta",
                                "panels": recA}
readme("a1r3_noVnorm_abs.pdf", "per-run metrics.json, checked vs §2.2",
       "Fig A: absolute T1 cw / T1 evt-median / T2 all, nominal, w_meas3d, w_meas3d_noVnorm")

# ================================================================ Fig B: delta, 2.2
fig, axs = plt.subplots(1, 3, figsize=(8.4, 3.5), gridspec_kw={"wspace": 0.28})
recB = {}
DV = VAR[1:]
for ax, (col, ttl, get) in zip(axs, COLS):
    res = {}
    for m, _ in MODELS:
        vals = []
        for v in DV:
            per = [get(r, v) - get(r, "nominal") for r in RUNS[m]]
            ms = msd(per)
            tb = cell(s22[m][v][col])
            check("FigB", m, "%s/%s" % (col, v), ms[0], tb[0])
            check_sd("FigB", m, "%s/%s" % (col, v), ms[1], tb[1])
            vals.append(ms)
        res[m] = vals
    pair_axes(ax, DV, res["CL"], res["C+XF"], rot=20)
    style(ax)
    ax.set_title(ttl, fontsize=9, color=INK2)
    recB[col] = {"x": DV, "CL": rec(res["CL"]), "C+XF": rec(res["C+XF"])}
axs[0].set_ylabel("Δ vs nominal")
legend(fig, 0.99, top=0.76)
fig.suptitle("Part 1: change vs nominal under 3D hit-width variants (V table normalised vs not)",
             fontsize=10, y=1.02)
nomtxt = "; ".join("%s nominal (absolute) " % m + ", ".join(
    "%s %.4f" % (c.split()[0] + " " + c.split()[1], msd([g(r, "nominal") for r in RUNS[m]])[0])
    for c, _, g in COLS) for m, _ in MODELS)
fig.text(0.01, -0.07, FOOT + "§2.2. Panels have independent y-scales. T2 bright omitted.\n" + nomtxt,
         fontsize=7.5, color=INK2, ha="left", va="top")
save(fig, "a1r3_noVnorm_delta.pdf")
DATA["a1r3_noVnorm_delta.pdf"] = {"source": "per-run metrics.json (a1r3/novnorm) deltas, checked vs §2.2", "panels": recB}
readme("a1r3_noVnorm_delta.pdf", "per-run metrics.json, checked vs §2.2",
       "Fig B: delta T1 cw / T1 evt-median / T2 all, w_meas3d and w_meas3d_noVnorm")

# ================================================================ Fig C / D: per plane, 2.3
PL = ["U", "V", "Y"]
GAP = 1.0
GROUPS = ["nominal", "w_meas3d", "w_meas3d_noVnorm"]
fig, ax = plt.subplots(figsize=(8.4, 3.9))
pos, lab, recC, nom = [], [], {}, {}
for gi, g in enumerate(GROUPS):
    base = gi * (3 + GAP)
    for pi, p in enumerate(PL):
        c = base + pi
        two = {}
        for m, _ in MODELS:
            tb = s23[m][p]
            nm = cell(tb["Nominal"])
            v = msd([r["hit"][g]["by_plane"][p]["score"] for r in RUNS[m]])
            ref = nm[0] if g == "nominal" else nm[0] + cell(tb[g])[0]
            check("FigC", m, "%s/%s" % (g, p), v[0], ref)
            if g == "nominal":
                check_sd("FigC", m, "%s/%s" % (g, p), v[1], nm[1])
                nom.setdefault(m, {})[p] = v[0]
            else:
                dv = msd([r["hit"][g]["by_plane"][p]["score"] - r["hit"]["nominal"]["by_plane"][p]["score"]
                          for r in RUNS[m]])
                check_sd("FigC delta", m, "%s/%s" % (g, p), dv[1], cell(tb[g])[1])
            two[m] = v
        bars(ax, [c - 0.21], [two["CL"]], C_CL)
        bars(ax, [c + 0.21], [two["C+XF"]], C_XF)
        pos.append(c); lab.append(p)
        recC.setdefault(g, {})[p] = {m: {"mean": two[m][0], "sd": two[m][1]} for m, _ in MODELS}
    ax.text(base + 1, -0.17, g, transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=9, color=INK, family="monospace" if g != "nominal" else None)
for gi in range(len(GROUPS)):
    base = gi * (3 + GAP)
    for m, col in MODELS:
        for pi, p in enumerate(PL):
            c = base + pi + (-0.21 if m == "CL" else 0.21)
            ax.plot([c - 0.19, c + 0.19], [nom[m][p]] * 2, color=col, lw=0.9, ls="--", zorder=3)
ax.set_xticks(pos); ax.set_xticklabels(lab, fontsize=9)
ax.set_xlim(-0.7, (len(GROUPS) - 1) * (3 + GAP) + 2.7)
style(ax, "T1 (charge-weighted F1)", zero=False)
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
legend(fig, 0.99, nominal=True, top=0.76, fs=8, nomlabel=" nominal (same plane)")
ax.set_title("Part 1: T1 (charge-weighted F1) per plane, absolute", fontsize=10, pad=6)
fig.text(0.01, -0.09, FOOT + "§2.3 and per-run metrics.json (a1r3/novnorm)", fontsize=7.5, color=INK2,
         ha="left", va="top")
save(fig, "a1r3_perplane_T1_qwtd_abs.pdf")
DATA["a1r3_perplane_T1_qwtd_abs.pdf"] = {"source": "per-run metrics.json (a1r3/novnorm), checked vs §2.3 first two tables",
                                         "groups": recC}
readme("a1r3_perplane_T1_qwtd_abs.pdf", "per-run metrics.json, checked vs §2.3 charge-weighted tables",
       "Fig C: per-plane absolute T1 cw, nominal / w_meas3d / w_meas3d_noVnorm")

# ---- Fig D: third table (per-event median, absolute) -> delta vs nominal on the plane
sec23 = section("2.3")
i3 = next(i for i, l in enumerate(sec23) if l.startswith("| Model | Plane"))
cols3 = [c.strip().replace("`", "") for c in sec23[i3].strip().strip("|").split("|")]
T3 = {}
for l in sec23[i3 + 2:]:
    if not l.startswith("|"):
        break
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    T3[("CL" if c[0] == "CL" else "C+XF", c[1])] = dict(zip(cols3[2:], c[2:]))
NV = "w_meas3d_noVnorm"
fig, ax = plt.subplots(figsize=(8.4, 3.9))
recD, pos, lab, nomD = {}, [], [], {}
DG = [("nominal", "nominal (round 2)"), ("w_meas3d", "w_meas3d (round 2)"), (NV, NV)]
for gi, (gname, colkey) in enumerate(DG):
    base = gi * (3 + GAP)
    for pi, p in enumerate(PL):
        c = base + pi
        two = {}
        for m, _ in MODELS:
            if colkey == NV:
                ab = msd([t["by_plane"][p]["evt_med"] for t in T1J[m]])
                tb = cell(T3[(m, p)][NV])
                check("FigD", m, "%s/%s" % (NV, p), ab[0], tb[0])
                check_sd("FigD", m, "%s/%s" % (NV, p), ab[1], tb[1])
            else:
                ab = cell(T3[(m, p)][colkey])  # absolute value and SD as printed in the table
            if gname == "nominal":
                nomD.setdefault(m, {})[p] = ab[0]
            two[m] = ab
        bars(ax, [c - 0.21], [two["CL"]], C_CL)
        bars(ax, [c + 0.21], [two["C+XF"]], C_XF)
        pos.append(c); lab.append(p)
        recD.setdefault(gname, {})[p] = {m: {"mean": two[m][0], "sd": two[m][1]} for m, _ in MODELS}
    ax.text(base + 1, -0.17, gname, transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=9, color=INK, family="monospace" if gname != "nominal" else None)
for gi in range(len(DG)):
    base = gi * (3 + GAP)
    for m, col in MODELS:
        for pi, p in enumerate(PL):
            c = base + pi + (-0.21 if m == "CL" else 0.21)
            ax.plot([c - 0.19, c + 0.19], [nomD[m][p]] * 2, color=col, lw=0.9, ls="--", zorder=3)
ax.set_xticks(pos); ax.set_xticklabels(lab, fontsize=9)
ax.set_xlim(-0.7, (len(DG) - 1) * (3 + GAP) + 2.7)
style(ax, "T1 (per-event median F1)", zero=False)
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
legend(fig, 0.99, nominal=True, top=0.76, fs=8, nomlabel=" nominal (same plane)")
ax.set_title("Part 1: T1 (per-event median F1) per plane, absolute", fontsize=10, pad=6)
fig.text(0.01, -0.09, FOOT + "§2.3 (third table; nominal and w_meas3d are round-2 values as printed)",
         fontsize=7.5, color=INK2, ha="left", va="top")
save(fig, "a1r3_perplane_T1_evtmed_abs.pdf")
DATA["a1r3_perplane_T1_evtmed_abs.pdf"] = {
    "source": "§2.3 third table (round-2 nominal and w_meas3d as printed); noVnorm from t1_npz_w_meas3d_noVnorm.json, "
              "checked vs table", "groups": recD}
readme("a1r3_perplane_T1_evtmed_abs.pdf", "§2.3 per-event-median table + t1_npz jsons",
       "Fig D: per-plane absolute per-event-median T1, nominal / w_meas3d (round 2) / w_meas3d_noVnorm")

# ================================================================ Fig C2: per-plane qwtd delta (paired per seed)
def delta_fig(fn, ttl, ylab, vals, foot, source, readme_txt):
    """vals[(group, plane)][model] = (mean, sd); groups = DV."""
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    pos, lab, rc = [], [], {}
    for gi, g in enumerate(DV):
        base = gi * (3 + GAP)
        for pi, p in enumerate(PL):
            c = base + pi
            two = vals[(g, p)]
            bars(ax, [c - 0.21], [two["CL"]], C_CL)
            bars(ax, [c + 0.21], [two["C+XF"]], C_XF)
            pos.append(c); lab.append(p)
            rc.setdefault(g, {})[p] = {m: {"mean": two[m][0], "sd": two[m][1]} for m, _ in MODELS}
        ax.text(base + 1, -0.17, g, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9, color=INK, family="monospace")
    ax.set_xticks(pos); ax.set_xticklabels(lab, fontsize=9)
    ax.set_xlim(-0.7, (len(DV) - 1) * (3 + GAP) + 2.7)
    style(ax, ylab)
    legend(fig, 0.99, top=0.76)
    ax.set_title(ttl, fontsize=10, pad=6)
    fig.text(0.01, -0.11, foot, fontsize=7.5, color=INK2, ha="left", va="top")
    save(fig, fn)
    DATA[fn] = {"source": source, "groups": rc}
    readme(fn, source, readme_txt)


vq = {}
for g in DV:
    for p in PL:
        vq[(g, p)] = {}
        for m, _ in MODELS:
            ms = msd([r["hit"][g]["by_plane"][p]["score"] - r["hit"]["nominal"]["by_plane"][p]["score"]
                      for r in RUNS[m]])
            tb = cell(s23[m][p][g])
            check("FigC2", m, "%s/%s" % (g, p), ms[0], tb[0])
            check_sd("FigC2", m, "%s/%s" % (g, p), ms[1], tb[1])
            vq[(g, p)][m] = ms
delta_fig("a1r3_perplane_T1_qwtd_delta.pdf", "T1 (charge-weighted F1) per plane: change vs nominal",
          "Δ T1 vs nominal (same plane)", vq,
          FOOT + "§2.3 and per-run metrics.json (a1r3/novnorm).\nΔ paired per seed; error bar = seed SD of the paired Δ.",
          "per-run metrics.json (a1r3/novnorm) by_plane paired deltas, checked vs §2.3 charge-weighted tables",
          "per-plane charge-weighted T1 delta vs nominal, w_meas3d and w_meas3d_noVnorm")

# ================================================================ Fig D2: per-plane evt-median delta (paired per seed)
R2T = {m: [json.load(open("%s/a1r2/full/s3_mt_qlfm_M_%s_finetune_f0.01_s%d/t1_npz.json" % (WORK, d, s)))
           for s in range(3)] for m, d in RUNDIR.items()}
ve = {}
for g in DV:
    for p in PL:
        ve[(g, p)] = {}
        for m, _ in MODELS:
            nom_ps = [t["nominal"]["by_plane"][p]["evt_med"] for t in R2T[m]]
            if g == "w_meas3d":
                var_ps = [t["w_meas3d"]["by_plane"][p]["evt_med"] for t in R2T[m]]
                colkey = "w_meas3d (round 2)"
            else:
                var_ps = [t["by_plane"][p]["evt_med"] for t in T1J[m]]
                colkey = NV
            ms = msd([v - n for v, n in zip(var_ps, nom_ps)])
            # check absolute means against the table, and the delta against table value - table nominal
            check("FigD2 nominal", m, "%s/%s" % (g, p), float(np.mean(nom_ps)), cell(T3[(m, p)]["nominal (round 2)"])[0])
            check("FigD2 variant", m, "%s/%s" % (g, p), float(np.mean(var_ps)), cell(T3[(m, p)][colkey])[0])
            check("FigD2 delta", m, "%s/%s" % (g, p), ms[0],
                  cell(T3[(m, p)][colkey])[0] - cell(T3[(m, p)]["nominal (round 2)"])[0], 3e-4)
            ve[(g, p)][m] = ms
delta_fig("a1r3_perplane_T1_evtmed_delta.pdf", "T1 (per-event median F1) per plane: change vs nominal",
          "Δ T1 vs nominal (same plane)", ve,
          FOOT + "§2.3 (third table), a1r2/full t1_npz.json and a1r3/novnorm t1_npz json.\n"
          "Δ = variant − a1r2 nominal on the same plane, paired per seed; error bar = seed SD of the paired Δ.",
          "a1r2/full t1_npz.json + a1r3/novnorm t1_npz jsons, paired per seed, checked vs §2.3 third table",
          "per-plane per-event-median T1 delta vs nominal, w_meas3d (round 2) and w_meas3d_noVnorm")

# ================================================================ Fig E / F: Part 2 primary tables
SJ = json.load(open(WORK + "/a1r3/summary.json"))["rows"]
ARMS = [("real → real", "real2real"), ("sim → real", "sim2real"), ("real → sim", "real2sim"),
        ("sim → sim", "sim2sim"), ("existing (sim) → sim", "existing2sim")]
P2COLS = [("T1 charge-wtd", "t1_cw", "T1 (charge-weighted F1)"),
          ("T1 evt-median", "t1_med", "T1 (per-event median F1)"),
          ("T2 all", "t2_all", "T2 (AUC), all events")]


def primary(sec):
    i = next(i for i, l in enumerate(sec) if l.startswith("**Primary"))
    _, rows, _ = parse_table(sec, i + 1)
    return rows


def part2(fn, sec_cl, sec_xf, mode, label):
    tabs = {"CL": primary(section(sec_cl)), "C+XF": primary(section(sec_xf))}
    mname = {"CL": "CL", "C+XF": "C + XF"}
    fig, axs = plt.subplots(1, 3, figsize=(8.0, 4.3), gridspec_kw={"wspace": 0.18})
    rc = {}
    for ax, (col, key, ttl) in zip(axs, P2COLS):
        res = {}
        for m, _ in MODELS:
            vals = []
            for an, ak in ARMS:
                row = SJ["%s|%s|%s" % (mname[m], mode, ak)]
                v = msd(row["real"][key])
                tb = cell(tabs[m][an][col])
                check("Fig" + label, m, "%s/%s" % (col, an), v[0], tb[0], 1e-4 + 5e-5)
                check_sd("Fig" + label, m, "%s/%s" % (col, an), v[1], tb[1])
                nseed = int(float(tabs[m][an]["n seeds"]))
                if nseed != row["n_seeds"]:
                    die("n seeds mismatch %s %s" % (m, an))
                vals.append(v if nseed > 1 else (v[0], None))
            res[m] = vals
        pair_axes(ax, [a[0] for a in ARMS], res["CL"], res["C+XF"], mono=False, rot=30, fs=7)
        style(ax, zero=False)
        if key == "t2_all":
            ax.set_ylim(0.8, 1.0)
            ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
        else:
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(ttl, fontsize=9, color=INK2)
        rc[col] = {"x": [a[0] for a in ARMS], "CL": rec(res["CL"]), "C+XF": rec(res["C+XF"])}
    axs[0].set_ylabel("score (real domain)")
    axs[2].set_ylabel("")
    legend(fig, 0.99, top=0.76)
    fig.suptitle("Part 2, %s: real-domain scores by pretrain → downstream arm" % mode, fontsize=10, y=1.02)
    ns = "; ".join("%s %s" % (an, "/".join(str(SJ["%s|%s|%s" % (mname[m], mode, ak)]["n_seeds"]) for m, _ in MODELS))
                   for an, ak in ARMS)
    fig.text(0.01, -0.2, "Mean over the downstream seeds present; error bars = seed SD, none for single-seed bars.\n"
             "Seeds (CL/C+XF): " + ns + ".\n"
             "Source: " + NOTE_NAME + " §%s and §%s 'Primary' tables, checked against a1r3/summary.json.\n"
             "T2 panel y-axis starts at 0.8." % (sec_cl, sec_xf), fontsize=7.5, color=INK2, ha="left", va="top")
    save(fig, fn)
    DATA[fn] = {"source": "a1r3/summary.json rows (real-domain), checked vs §%s and §%s Primary tables" % (sec_cl, sec_xf),
                "panels": rc,
                "n_seeds": {m: {an: SJ["%s|%s|%s" % (mname[m], mode, ak)]["n_seeds"] for an, ak in ARMS}
                            for m, _ in MODELS}}
    readme(fn, "a1r3/summary.json, checked vs §%s / §%s Primary tables" % (sec_cl, sec_xf),
           "Fig %s: real-domain T1 cw / T1 evt-median / T2 all by arm, %s; bars CL and C+XF" % (label, mode))


part2("a1r3_part2_finetune_primary.pdf", "3.2", "3.4", "finetune", "E")
part2("a1r3_part2_frozen_primary.pdf", "3.3", "3.5", "frozen", "F")

# ------------------------------------------------------------------ report + outputs
print("%-26s %-5s %-34s %9s %9s %9s" % ("figure", "model", "bar", "plotted", "table", "dev"))
for t in CHECK:
    print("%-26s %-5s %-34s %9.4f %9.4f %9.2e" % t)
print("CROSS-CHECK OK: %d values, max deviation %.2e; w_meas3d vs a1r2/full max dev %.1e" %
      (len(CHECK), max(c[-1] for c in CHECK), g5))
json.dump(DATA, open(os.path.join(OUT, "a1_round3_figures_data.json"), "w"), indent=1, ensure_ascii=False)
with open(os.path.join(OUT, "README.md"), "w") as f:
    f.write("# A1 round 3 figures\n\nGenerated by `scripts/a1_round3_figures.py` from `docs/%s` and the source "
            "data in `fm4npp-lartpc-work/a1r3`. Error bars = seed SD (ddof = 1).\n\n" % NOTE_NAME)
    f.write("\n".join(README) + "\n- `a1_round3_figures_data.json`: exactly the numbers plotted (mean, sd).\n")
print("wrote", len(DATA), "figures to", OUT)
