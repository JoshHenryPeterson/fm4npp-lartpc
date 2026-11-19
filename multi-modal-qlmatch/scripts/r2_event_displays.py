#!/usr/bin/env python
"""Round-2 event displays: one page per selected event (scripts/r2_select_events.py).

Panels: TPC charge (global channel U|V|Y vs tick, colour = log10 hit integral), BEE-style PMT view (circle area ~ PE at
each PMT's (z, y) of the optical flash matched to the true neutrino time, read from the open-data opflash tables), then the per-hit nu/cosmic separation of Truth, Pandora (nu-slice hits) and the
6 frozen@1% foundation-model hit heads (scripts/infer_hits.py), colour = p(nu).

Usage: r2_event_displays.py SELECTION.json HITS_DIR OUT_DIR
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from qlfm.data.dataset import Corpus
from qlfm.data.h5reader import OpenSampleFile
from qlfm.geometry import AV, N_WIRES, PMT_YZ
from qlfm.paths import WORK as W

FMS = ["qlfm_M_CL", "qlfm_S_CL", "panda", "polarmae", "fm4npp", "qlfm_SD_CL"]
LABEL = {"qlfm_M_CL": "qlfm-M CL", "qlfm_S_CL": "qlfm-S CL", "panda": "Panda", "polarmae": "PoLAr-MAE",
         "fm4npp": "FM4NPP", "qlfm_SD_CL": "qlfm-SD CL", "pandora_direct": "Pandora", "pandora_bdt": "Pandora+BDT",
         "pandora_slice": "Pandora"}
CLASSES = {"evt_nu": ["no-ν", "in-AV ν", "out-AV ν"], "flavor": ["νμCC", "νeCC", "NC"]}
TASK_TITLE = {"hit": "T1 hit ν/cosmic", "evt_nu": "T2 event ν-presence", "flavor": "T3 flavor", "energy": "T4 energy"}
CH0 = np.array([0, N_WIRES[0], N_WIRES[0] + N_WIRES[1]])
NCH = sum(N_WIRES)
HIT_NU_EDEP_MEV = 0.05
# Reconstructed flash time minus true interaction time (event_table nu_vtx_wire_time, ns): median 0.478 us,
# 5-95 % range 0.457-0.506 us, over all visible in-AV nu events with a beam flash (largest flash). Used to pick the
# flash belonging to the nu interaction; without a truth time, the flash closest to the beam-flash median time.
# If no flash lies within MATCH_TOL_US, the optical hits within OPH_WIN_US of the reference time are summed per PMT.
FLASH_DT_US = 0.478
BEAM_FLASH_T_US = 4.56
MATCH_TOL_US = 1.0
OPH_WIN_US = (-0.5, 3.5)        # fallback without a matched flash: sum optical hits in t_ref + this window
_H5 = {}


def flashes(sh, j):
    """All optical flashes of event j of shard sh: list of (time us, total PE, per-PMT PE[32])."""
    src = sh.meta["source"]
    if src not in _H5:
        _H5[src] = OpenSampleFile(src)
    f, e = _H5[src], sh.meta["ev_lo"] + j
    if f.offsets("opflash_table")[1][e] == 0:
        return []
    fl, _ = f.read_range("opflash_table", e, e + 1, ["flash_id", "time", "totalpe"])
    sp, _ = f.read_range("opflashsumpe_table", e, e + 1, ["flash_id", "pmt_channel", "sumpe"])
    out = []
    for fid, t, pe in zip(fl["flash_id"], fl["time"], fl["totalpe"]):
        m = sp["flash_id"] == fid
        v = np.zeros(32)
        v[sp["pmt_channel"][m].astype(int)] = sp["sumpe"][m]
        out.append((float(t), float(pe), v))
    return out


def load_event(c, g):
    s, j = c.locate(g)
    sh = c.shards[s]
    h0, hn = int(sh.ev("hit_off")[j]), int(sh.ev("hit_cnt")[j])
    o0, on = int(sh.ev("oph_off")[j]), int(sh.ev("oph_cnt")[j])
    e = {k: np.asarray(sh.hit(k)[h0:h0 + hn]) for k in ("plane", "wire", "time", "integral", "nu_edep", "pnd_nu")}
    e["ch"] = CH0[e["plane"]] + e["wire"].astype(np.int64)
    e["tick"] = e["time"].astype(np.float32) / 8
    e.update({"oph_" + k: np.asarray(sh.oph(k)[o0:o0 + on]) for k in ("ch", "t", "pe")})
    e.update({k: np.asarray(sh.ev(k)[j]) for k in ("fl_pmt", "fl_t", "fl_pe", "fl_n", "nu_vtx", "nu_pdg", "is_cc",
                                                    "nu_energy", "nu_time", "run", "subrun", "event")})
    e["flashes"] = flashes(sh, j)
    return e


def hit_preds(hits_dir, g, n):
    out = {}
    for bb in FMS:
        f = f"{hits_dir}/{bb}.npz"
        if not os.path.exists(f):
            continue
        if bb not in hit_preds.cache:
            r = np.load(f)
            hit_preds.cache[bb] = (r["hit_evt"], r["pred"])
        evt, p = hit_preds.cache[bb]
        sel = evt == g
        assert sel.sum() == n, (bb, g, sel.sum(), n)
        out[bb] = p[sel]
    return out


hit_preds.cache = {}


def f1_charge(p, t, q):
    tp, pt, tt = (q * (p & t)).sum(), (q * p).sum(), (q * t).sum()
    return 2 * tp / max(pt + tt, 1e-9)


def frame(ax, e, title, ylim):
    ax.set_xlim(0, NCH)
    ax.set_ylim(*ylim)
    for x in CH0[1:]:
        ax.axvline(x, color="0.6", lw=0.5)
    ax.set_title(title, fontsize=7, pad=2)
    ax.tick_params(labelsize=5, length=2, pad=1)


def nu_boxes(ax, e, tgt):
    for pl in range(3):
        m = tgt & (e["plane"] == pl)
        if m.sum() < 2:
            continue
        x0, x1 = np.percentile(e["ch"][m], [1, 99])
        y0, y1 = np.percentile(e["tick"][m], [1, 99])
        ax.add_patch(Rectangle((x0 - 15, y0 - 40), x1 - x0 + 30, y1 - y0 + 80, fill=False, ls="--", lw=0.5, color="k"))


def matched_flash(e):
    """(flash or None, reference time, label): the flash closest to t_nu + FLASH_DT_US (truth), else to the beam time."""
    has_t = np.isfinite(e["nu_time"]) and e["nu_time"] > 0
    t_ref = float(e["nu_time"]) / 1000 + FLASH_DT_US if has_t else BEAM_FLASH_T_US
    if not e["flashes"]:
        return None, t_ref, has_t
    best = min(e["flashes"], key=lambda fl: abs(fl[0] - t_ref))
    return (best if abs(best[0] - t_ref) < MATCH_TOL_US else None), t_ref, has_t


def light_panel(ax, e):
    fl, t_ref, has_t = matched_flash(e)
    if fl is not None:
        pe = fl[2]
    else:
        w = (e["oph_t"] >= t_ref + OPH_WIN_US[0]) & (e["oph_t"] < t_ref + OPH_WIN_US[1])
        pe = np.bincount(e["oph_ch"][w].astype(int), weights=e["oph_pe"][w], minlength=32)[:32].astype(np.float64)
    ax.add_patch(Rectangle((AV["z"][0], AV["y"][0]), AV["z"][1] - AV["z"][0], AV["y"][1] - AV["y"][0],
                           fill=False, lw=0.6, color="0.4"))
    z, y = PMT_YZ[:, 1], PMT_YZ[:, 0]
    ax.scatter(z, y, s=60, facecolors="none", edgecolors="0.75", lw=0.4)
    smax = 350.0
    s = smax * pe / max(pe.max(), 1e-9)
    ax.scatter(z, y, s=s, c="tab:orange", alpha=0.55, edgecolors="tab:red", lw=0.4)
    if e["nu_pdg"] != 0:
        ax.scatter([e["nu_vtx"][2]], [e["nu_vtx"][1]], marker="*", s=70, c="tab:blue", edgecolors="k", lw=0.3,
                   zorder=5, label="true ν vertex")
        ax.legend(fontsize=5, loc="lower right", frameon=False, handletextpad=0.2)
    ax.set_xlim(-30, 1070)
    ax.set_ylim(-140, 140)
    ax.set_aspect("equal")
    ax.set_xlabel("z [cm]", fontsize=6, labelpad=1)
    ax.set_ylabel("y [cm]", fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5, length=2, pad=1)
    ref = f"t_ν + {FLASH_DT_US:.2f} µs = {t_ref:.2f} µs" if has_t else f"beam time {t_ref:.2f} µs"
    nfl = len(e["flashes"])
    if fl is None:
        t = (f"PMT view: no reco flash within {MATCH_TOL_US:.0f} µs of {ref} ({nfl} flash{'es' if nfl != 1 else ''});\n"
             f"optical hits in [{t_ref + OPH_WIN_US[0]:.2f}, {t_ref + OPH_WIN_US[1]:.2f}] µs: {pe.sum():.0f} PE")
    else:
        largest = max(e["flashes"], key=lambda f: f[1])[0] == fl[0]
        t = (f"PMT view: flash at {fl[0]:.2f} µs, {fl[1]:.0f} PE (matched to {ref});\n"
             f"{nfl} flash{'es' if nfl != 1 else ''} in event, matched = {'largest' if largest else 'NOT the largest'}")
    ax.set_title(t, fontsize=6.5, pad=2)


def fmt_pred(task, v, truth):
    if task == "energy":
        r = np.exp(v - truth) - 1
        return f"{np.exp(v):.2f} GeV ({r:+.0%})", abs(r) < 0.2
    return CLASSES[task][int(v)], int(v) == int(truth)


def header(task, item, e, k):
    nu = "no ν" if e["nu_pdg"] == 0 else (f"{'ν' if e['nu_pdg'] > 0 else 'ν̄'}"
                                           f"{ {12: 'e', 14: 'μ'}.get(abs(int(e['nu_pdg'])), '?') } "
                                           f"{'CC' if e['is_cc'] else 'NC'}, E = {float(e['nu_energy']):.2f} GeV, "
                                           f"vtx = ({e['nu_vtx'][0]:.0f}, {e['nu_vtx'][1]:.0f}, {e['nu_vtx'][2]:.0f}) cm, "
                                           f"t_ν = {float(e['nu_time']) / 1000:.2f} µs")
    l1 = (f"{TASK_TITLE[task]} — {item['kind']} #{k + 1}   (correct: {item['n_correct']}/{item['n_alg']})    "
          f"run {int(e['run'])} / subrun {int(e['subrun'])} / event {int(e['event'])}  [gidx {item['gidx']}]")
    l2 = f"Truth: {nu}"
    if task == "hit":
        l2 += f";  ν charge fraction {item.get('nu_frac', float('nan')):.1%}"
        return l1, l2, ""
    tv = item["truth"]
    l2 += f";  task label: " + (f"{np.exp(tv):.2f} GeV" if task == "energy" else CLASSES[task][int(tv)])
    parts = []
    for a, v in item["preds"].items():
        s, ok = fmt_pred(task, v, tv)
        parts.append(f"{LABEL[a]}: {s} {'✓' if ok else '✗'}")
    return l1, l2, "   ".join(parts[:4]) + "\n" + "   ".join(parts[4:])


def render(task, item, k, e, preds, out):
    tgt = e["nu_edep"] > HIT_NU_EDEP_MEV
    q = np.clip(e["integral"].astype(np.float32), 0, None)
    lo, hi = np.percentile(e["tick"], [0, 100])
    ylim = (hi + 30, lo - 30)                       # tick increases downwards (drift away from the anode)
    fig = plt.figure(figsize=(8.27, 11.0))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.05, 1, 1, 1, 1], hspace=0.32, wspace=0.2,
                          left=0.06, right=0.985, top=0.905, bottom=0.05)
    l1, l2, l3 = header(task, item, e, k)
    fig.text(0.06, 0.985, l1, fontsize=8.5, weight="bold", va="top")
    fig.text(0.06, 0.965, l2, fontsize=7.5, va="top")
    if l3:
        fig.text(0.06, 0.947, l3, fontsize=6.5, va="top", family="DejaVu Sans")
    ax = fig.add_subplot(gs[0, 0])
    o = np.argsort(q)
    sc = ax.scatter(e["ch"][o], e["tick"][o], c=np.log10(1 + q[o]), s=0.4, lw=0, cmap="viridis", vmin=1, vmax=3.5,
                    rasterized=True)
    frame(ax, e, "Charge: hit integral (log10)", ylim)
    ax.set_ylabel("tick", fontsize=6, labelpad=1)
    for i, n in enumerate("UVY"):
        ax.text(CH0[i] + 60, ylim[1] + 40, n, fontsize=7, va="top", color="0.3")
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cb.ax.tick_params(labelsize=5)
    light_panel(fig.add_subplot(gs[0, 1]), e)
    panels = [("Truth (ν: E_dep > 0.05 MeV)", tgt.astype(np.float32)), ("Pandora (ν-slice hits)", e["pnd_nu"] > 0.5)]
    panels += [(LABEL[bb], preds[bb]) for bb in FMS if bb in preds]
    for i, (name, p) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        p = np.asarray(p, np.float32)
        o = np.argsort(p, kind="stable")
        ax.scatter(e["ch"][o], e["tick"][o], c=p[o], s=0.4, lw=0, cmap="coolwarm", vmin=0, vmax=1, rasterized=True)
        if i > 0:
            name += f"   F1_q = {f1_charge(p > 0.5, tgt, q):.2f}" if tgt.any() else \
                f"   ν-tagged charge = {(q * (p > 0.5)).sum() / max(q.sum(), 1e-9):.1%}"
        frame(ax, e, name, ylim)
        if tgt.any():
            nu_boxes(ax, e, tgt)
        if i % 2 == 0:
            ax.set_ylabel("tick", fontsize=6, labelpad=1)
        if i >= len(panels) - 2:
            ax.set_xlabel("global channel (U | V | Y)", fontsize=6, labelpad=1)
    fig.text(0.985, 0.006, "hit colour: p(ν), blue = cosmic, red = ν; dashed boxes: 1–99 % extent of true-ν hits per plane",
             fontsize=5.5, ha="right", color="0.3")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    sel = json.load(open(sys.argv[1]))
    hits_dir, out_dir = sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    c = Corpus(f"{W}/cache/bnb,{W}/cache/nue")
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    for task, items in sel.items():
        if only and task not in only:
            continue
        cnt = {"easy": 0, "boundary": 0, "hard": 0}
        for item in items:
            k = cnt[item["kind"]]
            cnt[item["kind"]] += 1
            e = load_event(c, item["gidx"])
            preds = hit_preds(hits_dir, item["gidx"], len(e["tick"]))
            out = f"{out_dir}/{task}_{item['kind']}{k + 1:02d}.png"
            render(task, item, k, e, preds, out)
            print(out, flush=True)


if __name__ == "__main__":
    main()
