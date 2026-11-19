#!/usr/bin/env python
"""Bar-chart PDFs for docs/technote_a1_round2.md (sections 2.2, 2.3, 3.2, 4.3).

Numbers are parsed from the tech note's Markdown tables (mean +- seed SD).
Writes vector PDFs, a1_round2_figures_data.json and README.md to OUT.
Optional: set A1R2_PNG_DIR to also write PNG previews.
"""
import json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE = os.path.join(HERE, "..", "docs", "technote_a1_round2.md")
OUT = "/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work/a1_round2/figures"
PNG = os.environ.get("A1R2_PNG_DIR", "/tmp/a1r2_fig_preview")
C_CL, C_XF = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"
MODELS = [("CL", C_CL), ("C+XF", C_XF)]
NOTE_NAME = "technote_a1_round2.md"
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "pdf.fonttype": 42})

LINES = open(NOTE, encoding="utf-8").read().split("\n")


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
    if t == "n/a":
        return None
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)(?:\s*±\s*(\d+(?:\.\d+)?))?", t)
    if not m:
        die("cell %r" % t)
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else None)


def tables(lines, hdr_pred):
    """Return {header_text: {rowkey: {col: (mean, sd)}}} for tables after header lines."""
    out = {}
    for i, l in enumerate(lines):
        if not hdr_pred(l):
            continue
        j = i + 1
        while j < len(lines) and not lines[j].startswith("|"):
            j += 1
        cols = [c.strip().replace("`", "") for c in lines[j].strip().strip("|").split("|")]
        j += 2
        rows = {}
        while j < len(lines) and lines[j].startswith("|"):
            c = [x.strip() for x in lines[j].strip().strip("|").split("|")]
            if len(c) != len(cols):
                die("column count in " + l)
            key = re.sub(r"\s*\(2D\)", "", c[0].replace("`", "").replace("**", ""))
            rows[key] = {cols[k]: c[k] for k in range(1, len(cols))}
            j += 1
        if not rows:
            die("empty table " + l)
        out[l] = rows
    return out


def model_tables(sec, extra=None):
    t = tables(sec, lambda l: l.startswith("**qlfm-M"))
    res = {}
    for h, rows in t.items():
        mod = "CL" if h.startswith("**qlfm-M CL") else "C+XF"
        kind = ""
        if "per-event median" in h:
            kind = "evtmed"
        elif "charge-weighted" in h:
            kind = "qwtd"
        res[(mod, kind)] = rows
    return res


def get(rows, key, col):
    try:
        return cell(rows[key][col])
    except KeyError:
        die("missing %s / %s" % (key, col))


DATA = {}


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


def bars(ax, x, vals, color, label=None, w=0.38):
    """vals: list of (mean, sd or None) or None."""
    xs, ms, es = [], [], []
    for xi, v in zip(x, vals):
        if v is None:
            continue
        xs.append(xi); ms.append(v[0]); es.append(v[1] or 0)
    ax.bar(xs, ms, width=w, color=color, label=label, zorder=2, linewidth=0)
    ax.errorbar(xs, ms, yerr=es, fmt="none", ecolor=INK, elinewidth=0.8,
                capsize=1.8, capthick=0.8, zorder=4)


def pair_axes(ax, names, cl, xf, mono=True, rot=0):
    n = len(names)
    off = 0.21
    x = list(range(n))
    bars(ax, [i - off for i in x], cl, C_CL, "CL")
    bars(ax, [i + off for i in x], xf, C_XF, "C+XF")
    ax.set_xticks(x)
    ax.set_xticklabels(names, family="monospace" if mono else None, fontsize=8,
                       rotation=rot, ha="right" if rot else "center")
    ax.set_xlim(-0.6, n - 0.4)


def legend(fig, y=1.0):
    fig.subplots_adjust(top=0.76)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MODELS]
    fig.legend(h, [m for m, _ in MODELS], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, y), handlelength=1.2)


def finish(fig, name, title, foot, top=0.86, titley=None):
    fig.suptitle(title, fontsize=10, color=INK, y=top + 0.1 if titley is None else titley)
    fig.text(0.01, -0.02, foot, fontsize=7.5, color=INK2, ha="left", va="top", wrap=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    if PNG:
        os.makedirs(PNG, exist_ok=True)
        fig.savefig(os.path.join(PNG, name.replace(".pdf", ".png")), bbox_inches="tight", dpi=100)
    plt.close(fig)


def record(name, source, labels, cl, xf, extra=None):
    d = {"source": source, "x": labels,
         "CL": [None if v is None else {"mean": v[0], "sd": v[1]} for v in cl],
         "C+XF": [None if v is None else {"mean": v[0], "sd": v[1]} for v in xf]}
    if extra:
        d.update(extra)
    DATA[name] = d


FOOT = "Mean over 3 seeds; error bars = seed SD. Source: " + NOTE_NAME + " "
README = []


def readme(name, src, desc):
    README.append("- `%s`: %s (source: %s)" % (name, desc, src))


# ---------------------------------------------------------------- 2.2
s22 = model_tables(section("2.2"))
W = ["w_meas", "w_meas3d", "w_meas3d_Y", "w_meas3d_UV", "w_diff", "w_diff_vd"]
for fn, col, short, ttl in [("a1r2_width_T1_qwtd.pdf", "T1 charge-wtd", "T1",
                             "T1 (charge-weighted F1)"),
                            ("a1r2_width_T2.pdf", "T2", "T2", "T2 (AUC)")]:
    cl = [get(s22[("CL", "")], v, col) for v in W]
    xf = [get(s22[("C+XF", "")], v, col) for v in W]
    nom = {m: get(s22[(m, "")], "nominal", col) for m in ("CL", "C+XF")}
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    pair_axes(ax, W, cl, xf, rot=30)
    style(ax, "Δ %s vs nominal" % short)
    legend(fig, 0.99)
    ax.set_title("%s: change vs nominal under hit-width variants" % ttl, fontsize=10, pad=6)
    foot = (FOOT + "§2.2\nNominal (absolute): CL %.4f ± %.4f, C+XF %.4f ± %.4f."
            % (nom["CL"][0], nom["CL"][1], nom["C+XF"][0], nom["C+XF"][1]))
    fig.text(0.01, -0.17, foot, fontsize=7.5, color=INK2, ha="left", va="top")
    fig.savefig(os.path.join(OUT, fn), bbox_inches="tight")
    if PNG:
        os.makedirs(PNG, exist_ok=True)
        fig.savefig(os.path.join(PNG, fn.replace(".pdf", ".png")), bbox_inches="tight", dpi=100)
    plt.close(fig)
    record(fn, "§2.2 column '%s'" % col, W, cl, xf,
           {"nominal_absolute": {m: {"mean": v[0], "sd": v[1]} for m, v in nom.items()}})
    readme(fn, "§2.2 '%s' column" % col, "%s delta vs nominal, hit-width variants" % short)

# ---------------------------------------------------------------- 2.3
s23 = model_tables(section("2.3"))
PL = ["U", "V", "Y"]


def perplane(fn, kind, groups, note):
    fig, ax = plt.subplots(figsize=(7.2 if len(groups) > 1 else 4.2, 3.7))
    gap = 1.0
    rec = {}
    pos, lab = [], []
    for gi, (gname, col) in enumerate(groups):
        base = gi * (3 + gap)
        for pi, p in enumerate(PL):
            c = base + pi
            cl = get(s23[("CL", kind)], p, col)
            xf = get(s23[("C+XF", kind)], p, col)
            bars(ax, [c - 0.21], [cl], C_CL)
            bars(ax, [c + 0.21], [xf], C_XF)
            pos.append(c); lab.append(p)
            rec.setdefault(gname, {})[p] = {
                "CL": {"mean": cl[0], "sd": cl[1]}, "C+XF": {"mean": xf[0], "sd": xf[1]}}
        ax.text(base + 1, -0.17, gname, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=9, color=INK)
    ax.set_xticks(pos)
    ax.set_xticklabels(lab, fontsize=9)
    ax.set_xlim(-0.7, (len(groups) - 1) * (3 + gap) + 2.7)
    style(ax, "Δ T1 vs nominal (same plane)")
    legend(fig, 0.99)
    return fig, ax, rec


for fn, kind, groups, ttl, sub in [
    ("a1r2_perplane_T1_qwtd.pdf", "qwtd",
     [("Δ w_meas3d (all planes)", "Δ"), ("Y-only Δ", "Y-only Δ"),
      ("UV-only Δ", "UV-only Δ")],
     "T1 (charge-weighted F1) per plane: change vs nominal", ""),
    ("a1r2_perplane_T1_evtmed.pdf", "evtmed", [("Δ w_meas3d (all planes)", "Δ")],
     "T1 (per-event median) per plane: change vs nominal",
     "Y-only / UV-only per-event-median values do not exist (predictions saved only for nominal and w_meas3d).")]:
    fig, ax, rec = perplane(fn, kind, groups, sub)
    ax.set_title(ttl, fontsize=10, pad=6)
    foot = FOOT + "§2.3" + ("\n" + sub if sub else "")
    fig.text(0.01, -0.07, foot, fontsize=7.5, color=INK2, ha="left", va="top")
    fig.savefig(os.path.join(OUT, fn), bbox_inches="tight")
    if PNG:
        fig.savefig(os.path.join(PNG, fn.replace(".pdf", ".png")), bbox_inches="tight", dpi=100)
    plt.close(fig)
    DATA[fn] = {"source": "§2.3 %s tables" % kind, "groups": rec}
    readme(fn, "§2.3 per-plane %s tables" % kind,
           "per-plane T1 delta, " + ("w_meas3d / Y-only / UV-only" if len(groups) > 1
                                     else "w_meas3d only (Y-only/UV-only not available)"))

# ---------------------------------------------------------------- 3.2
s32 = model_tables(section("3.2"))
L6 = ["l_scale80", "l_scale82", "l_slow", "l_tail", "l_comb80", "l_comb82"]
for fn, col, ttl, ylab in [
    ("a1l1_light_T1_qwtd.pdf", "T1 charge-wtd", "T1 (charge-weighted F1)", "Δ T1 vs nominal"),
    ("a1l1_light_T1_evtmed.pdf", "T1 evt-median", "T1 (per-event median)", "Δ T1 vs nominal"),
    ("a1l1_light_T2_all.pdf", "T2 all", "T2 (AUC), all events", "Δ T2 vs nominal"),
    ("a1l1_light_T2_bright.pdf", "T2 bright", "T2 (AUC), bright subset", "Δ T2 vs nominal")]:
    cl = [get(s32[("CL", "")], v, col) for v in L6 + ["l_prompt"]]
    xf = [get(s32[("C+XF", "")], v, col) for v in L6 + ["l_prompt"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"width_ratios": [6, 1.3], "wspace": 0.3})
    pair_axes(a1, L6, cl[:6], xf[:6], rot=30)
    pair_axes(a2, ["l_prompt"], cl[6:], xf[6:], rot=30)
    style(a1, ylab); style(a2)
    a1.set_title("Correction variants", fontsize=9, color=INK2)
    a2.set_title("Prompt", fontsize=9, color=INK2)
    legend(fig, 1.0)
    fig.suptitle("%s: change vs nominal under input-light variants" % ttl, fontsize=10, y=1.0 + 0.1)
    fig.text(0.01, -0.12, FOOT + "§3.2. Panels have independent y-scales; cells without ± have no error bar.",
             fontsize=7.5, color=INK2, ha="left", va="top")
    fig.savefig(os.path.join(OUT, fn), bbox_inches="tight")
    if PNG:
        fig.savefig(os.path.join(PNG, fn.replace(".pdf", ".png")), bbox_inches="tight", dpi=100)
    plt.close(fig)
    record(fn, "§3.2 column '%s'" % col, L6 + ["l_prompt"], cl, xf)
    readme(fn, "§3.2 '%s' column" % col, "light-variant delta, 2 panels (l_prompt separate)")

# ---------------------------------------------------------------- 4.3
s43 = model_tables(section("4.3"))
TV = ["t_scale80", "t_scale82", "t_slow", "t_tail", "t_comb80", "t_comb82"]
fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
dg = [("Δgap", "Δgap nu−dark"), ("Δgap out_av", "Δgap out_av"),
      ("Δgap nonu_flash", "Δgap nonu_flash")]
rec = {}
for ax, (col, ttl) in zip(axs, dg):
    cl = [get(s43[("CL", "")], v, col) for v in TV]
    xf = [get(s43[("C+XF", "")], v, col) for v in TV]
    pair_axes(ax, TV, cl, xf, rot=40)
    style(ax)
    ax.set_title(ttl, fontsize=9, color=INK2)
    rec[col] = {"x": TV, "CL": [{"mean": v[0], "sd": v[1]} for v in cl],
                "C+XF": [{"mean": v[0], "sd": v[1]} for v in xf]}
axs[0].set_ylabel("Δ gap vs nominal (MSE group − MSE nonu_dark)")
legend(fig, 1.0)
fig.suptitle("T5: change in MSE gap to nonu_dark under target variants", fontsize=10, y=1.1)
fig.text(0.01, -0.14, FOOT + "§4.3. Nominal Δgap = 0 (omitted). Shared y-axis.",
         fontsize=7.5, color=INK2, ha="left", va="top")
fn = "a1l1_T5_dgap.pdf"
fig.savefig(os.path.join(OUT, fn), bbox_inches="tight")
if PNG:
    fig.savefig(os.path.join(PNG, fn.replace(".pdf", ".png")), bbox_inches="tight", dpi=100)
plt.close(fig)
DATA[fn] = {"source": "§4.3 delta-gap columns", "panels": rec}
readme(fn, "§4.3 'Δgap', 'Δgap out_av', 'Δgap nonu_flash' columns", "T5 gap change, 3 panels")

REF = {"CL": 0.2158, "C+XF": 0.2593}
# cross-check against 4.2
s42 = "\n".join(section("4.2"))
for m, v in REF.items():
    if ("%.4f" % v) not in s42:
        die("reference MSE %s not found in 4.2" % m)
XV = ["nominal"] + TV
fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.7), sharey=True, gridspec_kw={"wspace": 0.08})
rec = {}
for ax, col in zip(axs, ["MSE nu", "MSE out_av", "MSE nonu_flash"]):
    cl = [get(s43[("CL", "")], v, col) for v in XV]
    xf = [get(s43[("C+XF", "")], v, col) for v in XV]
    pair_axes(ax, XV, cl, xf, rot=40)
    style(ax)
    for m, c in MODELS:
        ax.axhline(REF[m], color=c, lw=1.0, ls="--", zorder=5)
    ax.set_title(col, fontsize=9, color=INK2)
    rec[col] = {"x": XV, "CL": [{"mean": v[0], "sd": v[1]} for v in cl],
                "C+XF": [{"mean": v[0], "sd": v[1]} for v in xf]}
axs[0].set_ylim(0, 0.8)
axs[0].set_ylabel("MSE")
axs[2].text(1.01, REF["CL"], "nonu_dark (CL)", transform=axs[2].get_yaxis_transform(), ha="left", va="top", fontsize=7, color=INK2)
axs[2].text(1.01, REF["C+XF"], "nonu_dark (C+XF)", transform=axs[2].get_yaxis_transform(), ha="left", va="bottom", fontsize=7, color=INK2)
legend(fig, 1.0)
fig.suptitle("T5: MSE by group under target variants (dashed = MSE of nonu_dark)", fontsize=10, y=1.1)
fig.text(0.01, -0.14, FOOT + "§4.3; dashed lines from §4.2 (CL 0.2158, C+XF 0.2593, no band drawn).",
         fontsize=7.5, color=INK2, ha="left", va="top")
fn = "a1l1_T5_mse.pdf"
fig.savefig(os.path.join(OUT, fn), bbox_inches="tight")
if PNG:
    fig.savefig(os.path.join(PNG, fn.replace(".pdf", ".png")), bbox_inches="tight", dpi=100)
plt.close(fig)
DATA[fn] = {"source": "§4.3 MSE columns; reference §4.2", "panels": rec, "nonu_dark_ref": REF}
readme(fn, "§4.3 MSE nu / out_av / nonu_flash columns + §4.2 reference", "T5 absolute MSE, 3 panels")


# ================================================================ ABSOLUTE versions (suffix _abs)
import numpy as np
WORK = "/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work"
RUNDIR = {"CL": "CL", "C+XF": "C_xf"}


def load_runs(sub):
    out = {}
    for m, d in RUNDIR.items():
        out[m] = [json.load(open("%s/%s/s3_mt_qlfm_M_%s_finetune_f0.01_s%d/metrics.json"
                                 % (WORK, sub, d, s)))["results"] for s in range(3)]
    return out


def msd(vals):
    a = np.array(vals, float)
    return (float(a.mean()), float(a.std(ddof=1)))


CHECK = []
TOL = 2e-4


def check(tag, model, var, got, ref):
    dev = abs(got - ref)
    CHECK.append((tag, model, var, got, ref, dev))
    if dev >= TOL:
        raise SystemExit("CROSS-CHECK FAILED: %s %s %s mean %.5f vs nominal+delta %.5f (dev %.2e)"
                         % (tag, model, var, got, ref, dev))


def abs_style(ax, ylab):
    style(ax, ylab, zero=False)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])


def abs_legend(fig, ax, nom):
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MODELS]
    l = [m for m, _ in MODELS]
    for m, c in MODELS:
        h.append(plt.Line2D([0], [0], color=c, lw=0.9, ls="--")); l.append(m + " nominal")
    fig.subplots_adjust(top=0.76)
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.99),
               handlelength=1.6)


def nom_lines(ax, nom):
    for m, c in MODELS:
        ax.axhline(nom[m], color=c, lw=0.9, ls="--", zorder=3)


def save(fig, fn):
    fig.savefig(os.path.join(OUT, fn), bbox_inches="tight")
    if PNG:
        os.makedirs(PNG, exist_ok=True)
        fig.savefig(os.path.join(PNG, fn.replace(".pdf", ".png")), bbox_inches="tight", dpi=100)
    plt.close(fig)


# ---- width
RW = load_runs("a1r2/full")
WX = ["nominal"] + W
for fn, col, key, ttl, ylab in [
        ("a1r2_width_T1_qwtd_abs.pdf", "T1 charge-wtd", "hit", "T1 (charge-weighted F1) under hit-width variants", "T1 (charge-weighted F1)"),
        ("a1r2_width_T2_abs.pdf", "T2", "evt_nu", "T2 (AUC) under hit-width variants", "T2 (AUC)")]:
    res = {}
    for m, _ in MODELS:
        nomd = get(s22[(m, "")], "nominal", col)[0]
        vals = []
        for v in WX:
            ms = msd([r[key][v]["score"] for r in RW[m]])
            ref = nomd if v == "nominal" else nomd + get(s22[(m, "")], v, col)[0]
            check(fn, m, v, ms[0], ref)
            vals.append(ms)
        res[m] = vals
    nom = {m: res[m][0][0] for m, _ in MODELS}
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    pair_axes(ax, WX, res["CL"], res["C+XF"], rot=30)
    abs_style(ax, ylab); nom_lines(ax, nom); abs_legend(fig, ax, nom)
    ax.set_title(ttl, fontsize=10, pad=6)
    fig.text(0.01, -0.17, "Mean over 3 seeds; error bars = seed SD of the absolute score. Source: "
             + NOTE_NAME + " §2.2 and per-run metrics.json", fontsize=7.5, color=INK2, ha="left", va="top")
    save(fig, fn)
    record(fn, "per-run metrics.json (a1r2/full), checked vs §2.2 nominal+delta", WX, res["CL"], res["C+XF"])
    readme(fn, "per-run metrics.json, checked vs §2.2", "%s absolute score, nominal + hit-width variants" % ylab)

# ---- per-plane
PLX = {"nominal": "nominal", "w_meas3d": "w_meas3d", "Y-only": "w_meas3d_Y", "UV-only": "w_meas3d_UV"}
DCOL = {"w_meas3d": "Δ", "Y-only": "Y-only Δ", "UV-only": "UV-only Δ"}


def perplane_abs(fn, kind, groups, ttl, sub):
    fig, ax = plt.subplots(figsize=(7.6 if len(groups) > 2 else 4.8, 3.9))
    gap, pos, lab, rec, nom = 1.0, [], [], {}, {}
    for gi, g in enumerate(groups):
        base = gi * (3 + gap)
        for pi, p in enumerate(PL):
            c = base + pi
            two = {}
            for m, col in MODELS:
                tb = s23[(m, kind)]
                nm = get(tb, p, "Nominal")
                if kind == "qwtd":
                    v = msd([r["hit"][PLX[g]]["by_plane"][p]["score"] for r in RW[m]])
                    ref = nm[0] if g == "nominal" else nm[0] + get(tb, p, DCOL[g])[0]
                    check(fn, m, "%s/%s" % (g, p), v[0], ref)
                else:  # per-event median: table absolute values
                    v = nm if g == "nominal" else get(tb, p, "w_meas3d")
                    if g != "nominal":
                        check(fn, m, "%s/%s (table)" % (g, p), v[0], nm[0] + get(tb, p, "Δ")[0])
                two[m] = v
                if g == "nominal":
                    nom.setdefault(m, {})[p] = v[0]
            bars(ax, [c - 0.21], [two["CL"]], C_CL)
            bars(ax, [c + 0.21], [two["C+XF"]], C_XF)
            pos.append(c); lab.append(p)
            rec.setdefault(g, {})[p] = {m: {"mean": two[m][0], "sd": two[m][1]} for m, _ in MODELS}
        ax.text(base + 1, -0.17, g, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9, color=INK)
        # per-plane nominal dashed segment within each group (same plane's nominal)
    for gi in range(len(groups)):
        base = gi * (3 + gap)
        for m, col in MODELS:
            for pi, p in enumerate(PL):
                c = base + pi + (-0.21 if m == "CL" else 0.21)
                ax.plot([c - 0.19, c + 0.19], [nom[m][p]] * 2, color=col, lw=0.9, ls="--", zorder=3)
    ax.set_xticks(pos); ax.set_xticklabels(lab, fontsize=9)
    ax.set_xlim(-0.7, (len(groups) - 1) * (3 + gap) + 2.7)
    abs_style(ax, "T1 (%s)" % ("charge-weighted F1" if kind == "qwtd" else "per-event median F1"))
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MODELS]
    l = [m for m, _ in MODELS]
    for m, c in MODELS:
        h.append(plt.Line2D([0], [0], color=c, lw=0.9, ls="--")); l.append(m + " nominal (same plane)")
    fig.subplots_adjust(top=0.76)
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.99),
               handlelength=1.6, fontsize=8)
    ax.set_title(ttl, fontsize=10, pad=6)
    foot = ("Mean over 3 seeds; error bars = seed SD of the absolute score. Source: " + NOTE_NAME
            + " §2.3 and per-run metrics.json" + ("\n" + sub if sub else ""))
    fig.text(0.01, -0.09, foot, fontsize=7.5, color=INK2, ha="left", va="top")
    save(fig, fn)
    DATA[fn] = {"source": "§2.3 %s (per-run metrics.json for qwtd; tech-note table for evtmed)" % kind,
                "groups": rec}


perplane_abs("a1r2_perplane_T1_qwtd_abs.pdf", "qwtd", ["nominal", "w_meas3d", "Y-only", "UV-only"],
             "T1 (charge-weighted F1) per plane, absolute", "")
readme("a1r2_perplane_T1_qwtd_abs.pdf", "per-run metrics.json, checked vs §2.3",
       "per-plane absolute T1 (charge-wtd): nominal, w_meas3d, Y-only, UV-only")
perplane_abs("a1r2_perplane_T1_evtmed_abs.pdf", "evtmed", ["nominal", "w_meas3d"],
             "T1 (per-event median F1) per plane, absolute",
             "Y-only / UV-only per-event-median values do not exist (predictions saved only for nominal and w_meas3d).")
readme("a1r2_perplane_T1_evtmed_abs.pdf", "§2.3 per-event-median tables (absolute values and SD as printed)",
       "per-plane absolute T1 (per-event median): nominal, w_meas3d only")

# ---- light
RL = load_runs("a1l1/full")
LX = ["nominal"] + L6 + ["l_prompt"]
HDR = {}
for m, h in (("CL", "**qlfm-M CL**"), ("C+XF", "**qlfm-M C + XF**")):
    ln = next(l for l in section("3.2") if l.startswith(h) and "nominal:" in l)
    n = re.search(r"T1 ([\d.]+) ± [\d.]+ / ([\d.]+) ± [\d.]+ evt-median, T2 ([\d.]+) ± [\d.]+ all, ([\d.]+) ± [\d.]+ bright", ln)
    if not n:
        die("light nominal header " + m)
    HDR[m] = dict(zip(["T1 charge-wtd", "T1 evt-median", "T2 all", "T2 bright"], map(float, n.groups())))
GET = {"T1 charge-wtd": lambda r, v: r["hit"][v]["score"],
       "T1 evt-median": lambda r, v: r["hit"][v]["evt_med"]["score"],
       "T2 all": lambda r, v: r["evt_nu"][v]["score"],
       "T2 bright": lambda r, v: r["evt_nu"][v]["bright"]["score"]}
for fn, col, ttl, ylab in [
        ("a1l1_light_T1_qwtd_abs.pdf", "T1 charge-wtd", "T1 (charge-weighted F1) under input-light variants", "T1 (charge-weighted F1)"),
        ("a1l1_light_T1_evtmed_abs.pdf", "T1 evt-median", "T1 (per-event median F1) under input-light variants", "T1 (per-event median F1)"),
        ("a1l1_light_T2_all_abs.pdf", "T2 all", "T2 (AUC, all events) under input-light variants", "T2 (AUC), all events"),
        ("a1l1_light_T2_bright_abs.pdf", "T2 bright", "T2 (AUC, bright subset) under input-light variants", "T2 (AUC), bright subset")]:
    res = {}
    for m, _ in MODELS:
        vals = []
        for v in LX:
            ms = msd([GET[col](r, v) for r in RL[m]])
            ref = HDR[m][col] if v == "nominal" else HDR[m][col] + get(s32[(m, "")], v, col)[0]
            check(fn, m, v, ms[0], ref)
            vals.append(ms)
        res[m] = vals
    nom = {m: res[m][0][0] for m, _ in MODELS}
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    pair_axes(ax, LX, res["CL"], res["C+XF"], rot=30)
    abs_style(ax, ylab); nom_lines(ax, nom); abs_legend(fig, ax, nom)
    ax.set_title(ttl, fontsize=10, pad=6)
    fig.text(0.01, -0.17, "Mean over 3 seeds; error bars = seed SD of the absolute score. Source: "
             + NOTE_NAME + " §3.2 and per-run metrics.json", fontsize=7.5, color=INK2, ha="left", va="top")
    save(fig, fn)
    record(fn, "per-run metrics.json (a1l1/full), checked vs §3.2 nominal+delta", LX, res["CL"], res["C+XF"])
    readme(fn, "per-run metrics.json, checked vs §3.2", "%s absolute score, nominal + 7 light variants, one panel" % ylab)

print("%-34s %-5s %-22s %8s %8s %9s" % ("figure", "model", "bar", "mean", "nom+d", "dev"))
for t in CHECK:
    print("%-34s %-5s %-22s %8.4f %8.4f %9.2e" % t)
print("CROSS-CHECK OK: %d bars, max deviation %.2e (tol %.0e)" % (len(CHECK), max(c[-1] for c in CHECK), TOL))

json.dump(DATA, open(os.path.join(OUT, "a1_round2_figures_data.json"), "w"), indent=1, ensure_ascii=False)
with open(os.path.join(OUT, "README.md"), "w") as f:
    f.write("# A1 round 2 figures\n\nGenerated by `scripts/a1_round2_figures.py` from "
            "`docs/%s`. Mean over 3 seeds, error bars = seed SD.\n\n" % NOTE_NAME)
    f.write("\n".join(README) + "\n- `a1_round2_figures_data.json`: exactly the numbers plotted (mean, sd).\n")
print("wrote", len(DATA), "figures to", OUT)
