"""Data/sim robustness test (issue #9; recipe by Sunny Seo, slides of 6 Nov 2026): re-evaluate trained T1/T2 models
with the simulated neutrino hits made more data-like, no retraining.

python -m qlfm.eval_perturb --run RUN_DIR [--tasks hit,evt_nu] [--variants all] [--out OUT_DIR]   (writes OUT_DIR/metrics.json)

RUN_DIR is a finished qlfm.train_multitask or qlfm.train_downstream run (metrics.json + best.pt); the model and the
test subsets are rebuilt from its saved arguments, so the "nominal" variant reproduces the stored test score.
Perturbations act on the hits of simulated neutrinos only (hit_nu_edep > --sim-edep MeV); real (cosmic) hits and the
light are untouched:
  * width: hit_rms *= f(plane, drift), drift = tick - 830 (0 at the anode)
      meas  measured data/sim width ratio (Sunny, p.13), normalised to 1 at the anode, end values held
            (U and V provisional)
      diff  diffusion only, published D_L (data 3.74, sim 6.40 cm2/s), sigma0^2 = 1.9 us^2:
            f = sqrt((s0 + 2 D_data t) / (s0 + 2 D_sim t)), t = drift time (1.00 -> 0.88 at the cathode)
  * charge: hit_q *= 1 + eps, eps = +-0.02, +-0.05 (MicroBooNE WireMod)
  Controls (is a drop specific to the data/sim mismatch, or any change of width?):
      w_inv       simulated widths divided by the measured factor (sim made broader, away from data)
      w_c98/c95   simulated widths times a constant 0.98 / 0.95 (no drift dependence)
      w_meas_all  measured factor applied to every hit, real cosmics included (relative sim/data width unchanged)
      w_const     every hit's width set to WIDTH_CONST (width carries no information)
Only qlfm reads hit width (its hit encoder; also the "raw" hit adapter); PTv3, PolarMAE and FM4NPP see charge and
position only, so a width-only variant must leave them unchanged.
Light variants (A1 light check, round 1; docs/a1_light_check_round_1_plan.md): l_scale80/l_scale82/l_slow/l_tail/
l_comb80/l_comb82/l_prompt rescale the ophit PE of the beam flash (dt = t - t_b) before tokenization; --t5-targets
applies the same corrections to the T5 (light) target of has_flash events and compares the nu / no-nu error gap.
Metrics are computed with the unperturbed hit charge (T1 is charge-weighted). Uncertainties: bootstrap over test
events of each metric (68/95% intervals) and a paired bootstrap of the change w.r.t. nominal (same resampled events).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from qlfm.data.dataset import Corpus, EventDataset, collate
from qlfm.data.labels import EV_KEYS, HIT_NU_EDEP_MEV, event_labels
from qlfm.evaluate import _softmax, metrics_for
from qlfm.ref import build_backbone

ANODE_TICK = 830.0
D_C = [388.0, 1165.0, 1941.0, 2718.0, 3494.0, 4271.0]          # drift bin centres (ticks)
F_MEAS_2D = {2: [1.000, 0.972, 0.951, 0.936, 0.927, 0.919],      # Y
             1: [1.000, 0.982, 0.965, 0.952, 0.954, 0.955],      # V (provisional)
             0: [1.000, 0.969, 0.936, 0.916, 0.901, 0.895]}      # U (provisional)
F_MEAS = F_MEAS_2D                                               # round-1 name
F_MEAS_3D = {2: [1.000, 0.967, 0.941, 0.926, 0.915, 0.903],      # Y   (Sunny, p.13, 3D; round 2)
             1: [1.000, 0.980, 0.954, 0.938, 0.942, 0.941],      # V
             0: [1.000, 0.960, 0.922, 0.907, 0.892, 0.880]}      # U
F_MEAS_3D_NOVNORM = {2: [1.000, 0.967, 0.941, 0.926, 0.915, 0.903],   # Y (= F_MEAS_3D)
                     1: [0.900, 0.882, 0.859, 0.844, 0.848, 0.847],   # V, not renormalised at the anode (round 3)
                     0: [1.000, 0.960, 0.922, 0.907, 0.892, 0.880]}   # U (= F_MEAS_3D)
TABLES = {"meas": F_MEAS_2D, "meas3d": F_MEAS_3D, "meas3d_noVnorm": F_MEAS_3D_NOVNORM}
US_PER_TICK, V_DRIFT_CM_US, MAX_DRIFT_CM = 0.5, 0.1098, 256.35
V_DRIFT_DATA_CM_US = 0.1076                                      # drift velocity measured in data (round 2, diff_vd)
D_DATA, D_SIM, S0_US2 = 3.74e-6, 6.40e-6, 1.9                    # cm2/us, cm2/us, us2
WIDTH_CONST = 4.0                                                # ticks, about the median hit rms

VARIANTS = {"nominal": (None, 0.0, ()), "w_meas": ("meas", 0.0, ()), "w_diff": ("diff", 0.0, ()),
            "q-5": (None, -0.05, ()), "q-2": (None, -0.02, ()), "q+2": (None, 0.02, ()), "q+5": (None, 0.05, ()),
            "w_meas_q-5": ("meas", -0.05, ()), "w_meas_q+5": ("meas", 0.05, ()),
            "w_inv": ("inv", 0.0, ()), "w_c98": ("c0.98", 0.0, ()), "w_c95": ("c0.95", 0.0, ()), "w_meas_all": ("meas_all", 0.0, ()),
            "w_const": ("const", 0.0, ()),
            # round 2 (3D factors, plane subsets, data drift velocity in the diffusion bracket)
            "w_meas3d": ("meas3d", 0.0, ()), "w_meas3d_Y": ("meas3d@2", 0.0, ()), "w_meas3d_UV": ("meas3d@0,1", 0.0, ()),
            "w_diff_vd": ("diff_vd", 0.0, ()), "w_inv3d": ("inv3d", 0.0, ()),
            # A1 light check round 1
            "l_scale80": (None, 0.0, ("scale80",)), "l_scale82": (None, 0.0, ("scale82",)),
            "l_slow": (None, 0.0, ("slow",)), "l_tail": (None, 0.0, ("tail",)),
            "l_comb80": (None, 0.0, ("scale80", "slow", "tail")), "l_comb82": (None, 0.0, ("scale82", "slow", "tail")),
            "l_prompt": (None, 0.0, ("prompt",)),
            # A1 check round 3: noVnorm table, V-only pair, composite data-like domains (width + light)
            "w_meas3d_noVnorm": ("meas3d_noVnorm", 0.0, ()), "w_meas3d_V": ("meas3d@1", 0.0, ()),
            "w_meas3d_noVnorm_V": ("meas3d_noVnorm@1", 0.0, ()),
            "d_real": ("meas3d", 0.0, ("scale80", "slow", "tail")),
            "d_real_noVnorm": ("meas3d_noVnorm", 0.0, ("scale80", "slow", "tail"))}
DOMAINS = {"sim": "nominal", "real": "d_real", "real_noVnorm": "d_real_noVnorm"}   # train/pretrain/eval --domain
ROUND3_VARIANTS = ["w_meas3d_noVnorm", "w_meas3d_V", "w_meas3d_noVnorm_V", "d_real", "d_real_noVnorm"]
LIGHT_VARIANTS = [v for v, x in VARIANTS.items() if x[2] and x[0] is None and not x[1]]   # pure-light (round 1); d_* are composite
WIDTH_VARIANTS = [v for v, x in VARIANTS.items() if not x[2] and v not in ROUND3_VARIANTS]   # --variants all (round-1/2)
# T5 target variants: the same components applied to the light target of has_flash events (no l_prompt counterpart)
T5_TARGETS = {"t_scale80": ("scale80",), "t_scale82": ("scale82",), "t_slow": ("slow",), "t_tail": ("tail",),
              "t_comb80": ("scale80", "slow", "tail"), "t_comb82": ("scale82", "slow", "tail")}
# light constants (us, PE); dt = t - t_b
TB_WIN = (3.4, 5.6); TB_BIN = 0.1
PB_WIN = (-0.25, 0.5); PB_BRIGHT = 600.0
SCALE_WIN = (-0.25, 3.0)
SLOW_EDGES = [0.5, 1.0, 1.5, 2.0, 3.0]
SLOW_DIM = [0.83, 0.65, 0.61, 0.64]; SLOW_BRIGHT = [0.82, 0.63, 0.48, 0.45]
TAIL_DT, TAIL_F = 8.0, 0.37
PROMPT_DT = 0.5
SCALES = {"scale80": 0.80, "scale82": 0.82}
BRIGHT_WIN, BRIGHT_PE = (3.0, 6.0), 200.0          # paper T2 bright cut (r4_light_stats pe_win >= paper_metrics t2_cut)
DT_DIAG = [(-0.25, 0.5), (0.5, 3.0), (3.0, 8.0), (8.0, float("inf"))]
NB_TB = int(round((TB_WIN[1] - TB_WIN[0]) / TB_BIN))   # 22

PLANE_NAMES = {0: "U", 1: "V", 2: "Y"}


def _interp(x, xp, fp):
    """np.interp on torch tensors (end values held)."""
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1, y0, y1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    return y0 + ((x - x0) / (x1 - x0)).clamp(0, 1) * (y1 - y0)


def width_factor(plane, tick, kind):
    """kind: inv | inv3d | c<value> | meas | meas3d[@p,q] | diff | diff_vd. "@p,q" restricts a table kind to the
    listed planes (f = 1 elsewhere)."""
    d = tick.float() - ANODE_TICK
    if kind.startswith("inv"):                      # simulated widths divided by the measured factor
        return 1 / width_factor(plane, tick, "meas" + kind[3:])
    if kind.startswith("c"):
        return torch.full_like(d, float(kind[1:]))
    base, _, sub = kind.partition("@")
    if base in TABLES:
        keep = {int(x) for x in sub.split(",")} if sub else None
        xp = torch.tensor(D_C, device=tick.device)
        f = torch.ones_like(d)
        for p, fp in TABLES[base].items():
            if keep is not None and p not in keep:
                continue
            m = plane == p
            f[m] = _interp(d[m], xp, torch.tensor(fp, device=tick.device))
        return f
    if kind in ("diff", "diff_vd"):
        t = (d * US_PER_TICK).clamp(0, MAX_DRIFT_CM / V_DRIFT_CM_US)
        v2 = V_DRIFT_CM_US ** 2
        v2d = V_DRIFT_DATA_CM_US ** 2 if kind == "diff_vd" else v2   # only the data term's cm2 -> us2 conversion changes
        return torch.sqrt((S0_US2 + 2 * D_DATA * t / v2d) / (S0_US2 + 2 * D_SIM * t / v2))
    raise ValueError(kind)


def beam_flash(b):
    """Per event from the NATIVE ophits: t_b (lower edge of the max-PE 0.1 us bin in 3.4-5.6 us), P_b (PE summed in
    dt in [-0.25, 0.5)) and has_flash (any PE in 3.4-5.6 us). Returns (t_b, P_b, has_flash), each [n_events]."""
    n, t, pe, ev = b["n_events"], b["oph_t"].float(), b["oph_pe"].float(), b["oph_evt"]
    m = (t >= TB_WIN[0]) & (t < TB_WIN[1])
    k = torch.floor((t[m] - TB_WIN[0]) / TB_BIN).long().clamp(0, NB_TB - 1)
    h = torch.zeros(n * NB_TB, device=t.device).index_add_(0, ev[m] * NB_TB + k, pe[m]).view(n, NB_TB)
    tb = TB_WIN[0] + h.argmax(1).float() * TB_BIN
    has = h.sum(1) > 0
    dt = t - tb[ev]
    w = (dt >= PB_WIN[0]) & (dt < PB_WIN[1]) & has[ev]
    pb = torch.zeros(n, device=t.device).index_add_(0, ev[w], pe[w])
    return tb, pb, has


def light_factor_dt(dt, pb, kinds):
    """Per-ophit multiplicative factor from dt (us) and the event's native P_b (PE), product of the components."""
    f = torch.ones_like(dt)
    bright = pb >= PB_BRIGHT
    for k in kinds:
        if k in SCALES:
            f = f * torch.where((dt >= SCALE_WIN[0]) & (dt < SCALE_WIN[1]), SCALES[k], 1.0)
        elif k == "slow":
            for lo, hi, fd, fb in zip(SLOW_EDGES[:-1], SLOW_EDGES[1:], SLOW_DIM, SLOW_BRIGHT):
                f = f * torch.where((dt >= lo) & (dt < hi), torch.where(bright, fb, fd), 1.0)
        elif k == "tail":
            f = f * torch.where((dt >= TAIL_DT) & bright, TAIL_F, 1.0)
        elif k == "prompt":
            f = f * torch.where(dt >= PROMPT_DT, 0.0, 1.0)
        else:
            raise ValueError(k)
    return f


def light_factor(b, kinds, flash=None):
    """Per-ophit factor [n_ophits] (1 for events without a beam flash). flash = beam_flash(b) if already computed."""
    tb, pb, has = flash if flash is not None else beam_flash(b)
    ev = b["oph_evt"]
    f = light_factor_dt(b["oph_t"].float() - tb[ev], pb[ev], kinds)
    return torch.where(has[ev], f, torch.ones_like(f))


def pe_window_sums(b, flash, f):
    """Summed PE before / after the factor f, per P_b class (dim, bright) and dt window DT_DIAG: [2, 2, 4]
    (flash events only; ophits before dt = -0.25 us are not in any window)."""
    tb, pb, has = flash
    ev = b["oph_evt"]
    dt, pe = b["oph_t"].float() - tb[ev], b["oph_pe"].float()
    out = np.zeros((2, 2, len(DT_DIAG)))
    br = pb[ev] >= PB_BRIGHT
    for ci, cm in enumerate((~br, br)):
        for wi, (lo, hi) in enumerate(DT_DIAG):
            m = cm & has[ev] & (dt >= lo) & (dt < hi)
            out[ci, 0, wi] = float(pe[m].double().sum())
            out[ci, 1, wi] = float((pe * f)[m].double().sum())
    return out


def pe_win_native(b):
    """Per-event summed PE in [3, 6) us (paper T2 bright-flash cut variable)."""
    t, pe, ev = b["oph_t"].float(), b["oph_pe"].float(), b["oph_evt"]
    m = (t >= BRIGHT_WIN[0]) & (t < BRIGHT_WIN[1])
    return torch.zeros(b["n_events"], device=t.device).index_add_(0, ev[m], pe[m])


def perturb(b, variant, sim_edep):
    width, eps, light = VARIANTS[variant]
    if light:                                   # before the early return; all ophits (sim ophits cannot be identified)
        b = dict(b)
        b["oph_pe"] = b["oph_pe"] * light_factor(b, light)
    if width is None and eps == 0:
        return b
    b = dict(b)
    sim = b["hit_nu_edep"] > sim_edep
    if width == "const":
        b["hit_rms"] = torch.full_like(b["hit_rms"], WIDTH_CONST)
    elif width is not None:
        f = width_factor(b["hit_plane"], b["hit_time"], width.removesuffix("_all"))
        b["hit_rms"] = b["hit_rms"] * f if width.endswith("_all") else torch.where(sim, b["hit_rms"] * f, b["hit_rms"])
    if eps:
        b["hit_q"] = torch.where(sim, b["hit_q"] * (1 + eps), b["hit_q"])
    return b


class Perturbed(nn.Module):
    """Applies the current variant to a copy of the batch; prediction code still reads the original batch.
    collect (set before the nominal pass): per-event native flash info and, if t5_targets, the T5 target variants."""

    def __init__(self, model, sim_edep, t5_targets=()):
        super().__init__()
        self.model, self.sim_edep, self.variant = model, sim_edep, "nominal"
        self.count = np.zeros(3)                 # hits, simulated hits, simulated hits below the T1 label threshold
        self.t5_targets = list(t5_targets)
        self.info = {k: [] for k in ("gidx", "tb", "pb", "has", "pe_win")}
        self.tgt = {k: [] for k in ["nominal"] + self.t5_targets}
        self.pe_diag = {}                        # light variant -> [2, 2, 4] summed PE (dim/bright, before/after, dt window)

    def forward(self, b):
        light = VARIANTS[self.variant][2]
        flash = beam_flash(b) if (light or self.variant == "nominal") else None
        if self.variant == "nominal":
            e = b["hit_nu_edep"]
            self.count += [e.numel(), int((e > self.sim_edep).sum()), int(((e > self.sim_edep) & (e <= HIT_NU_EDEP_MEV)).sum())]
            self._collect(b, flash)
        elif light:
            f = light_factor(b, light, flash)
            self.pe_diag[self.variant] = self.pe_diag.get(self.variant, 0) + pe_window_sums(b, flash, f)
        return self.model(perturb(b, self.variant, self.sim_edep))

    @torch.no_grad()
    def _collect(self, b, flash):
        tb, pb, has = flash
        for k, v in (("gidx", b["gidx"]), ("tb", tb), ("pb", pb), ("has", has), ("pe_win", pe_win_native(b))):
            self.info[k].append(v.cpu())
        if self.t5_targets:
            from qlfm.train_downstream import light_target
            self.tgt["nominal"].append(light_target(b).float().cpu())
            for name in self.t5_targets:
                b2 = dict(b)
                b2["oph_pe"] = b["oph_pe"] * light_factor(b, T5_TARGETS[name], flash)
                self.tgt[name].append(light_target(b2).float().cpu())

    def event_info(self):
        """Native per-event info over all loaded events, sorted by gidx: dict of numpy arrays."""
        d = {k: torch.cat(v).numpy() for k, v in self.info.items()}
        o = np.argsort(d["gidx"], kind="stable")
        return {k: v[o] for k, v in d.items()}, o

    def t5_target_arrays(self, order):
        return {k: torch.cat(v).numpy()[order] for k, v in self.tgt.items() if v}


# ---------------- statistics ----------------
def hit_event_sums(r):
    """Per-event charge-weighted (tp, true, selected) sums; T1 score = F1 of the summed counts (= hit_metrics)."""
    q, t, s = r["hit_q"].clip(min=0), r["hit_tgt"].astype(bool), r["pred"] > 0.5
    ev, inv = np.unique(r["hit_evt"], return_inverse=True)
    sums = np.stack([np.bincount(inv, q * (s & t), len(ev)), np.bincount(inv, q * t, len(ev)),
                     np.bincount(inv, q * s, len(ev))], 1)
    return ev, sums


def hit_event_sums_plane(r, ev, plane):
    """(n_events, 3 planes, 3) per-event, per-plane charge-weighted (tp, true, selected) sums; ev = sorted event ids."""
    q, t, s = r["hit_q"].clip(min=0), r["hit_tgt"].astype(bool), r["pred"] > 0.5
    key = np.searchsorted(ev, r["hit_evt"]) * 3 + plane.astype(np.int64)
    n = len(ev) * 3
    return np.stack([np.bincount(key, q * (s & t), n), np.bincount(key, q * t, n),
                     np.bincount(key, q * s, n)], 1).reshape(len(ev), 3, 3)


def bootstrap_by_plane(res, plane, n_boot, seed=0):
    """Per-plane analogue of bootstrap() for the hit task (same resampled events as bootstrap(), same seed)."""
    rng = np.random.default_rng(seed)
    ev = np.unique(res["nominal"]["hit_evt"])
    n = len(ev)
    idx = rng.integers(0, n, size=(n_boot, n))
    sums = {v: hit_event_sums_plane(r, ev, plane) for v, r in res.items()}
    q = lambda a, lo, hi: [float(np.quantile(a, lo)), float(np.quantile(a, hi))]
    stat, point = {}, {}
    for v, s in sums.items():
        for k, name in PLANE_NAMES.items():
            stat[v, k] = f1_from_sums(s[:, k][idx].sum(1))
            point[v, k] = float(f1_from_sums(s[:, k].sum(0)))
    out = {}
    for v in res:
        out[v] = {}
        for k, name in PLANE_NAMES.items():
            d = stat[v, k] - stat["nominal", k]
            out[v][name] = dict(score=point[v, k], ci68=q(stat[v, k], 0.16, 0.84), ci95=q(stat[v, k], 0.025, 0.975),
                                delta=point[v, k] - point["nominal", k], delta_ci68=q(d, 0.16, 0.84),
                                delta_ci95=q(d, 0.025, 0.975), n_events=int(n))
    return out


def save_preds(path, r):
    """Hit predictions in the preds_test.npz column layout read by qlfm.paper_metrics.t1 (+ hit_plane, hit_sim);
    pred = sigmoid probability of the true-neutrino class (threshold 0.5), gidx = per-event ids (sorted)."""
    cols = dict(pred=r["pred"].astype(np.float16), hit_tgt=r["hit_tgt"].astype(np.uint8),
                hit_q=r["hit_q"].astype(np.float32), hit_evt=r["hit_evt"].astype(np.int32), gidx=r["gidx"],
                hit_plane=r["hit_plane"].astype(np.uint8), hit_sim=r["hit_sim"].astype(bool))
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **cols)
    os.replace(tmp, path)
    print(f"saved {path} ({os.path.getsize(path) / 1e9:.2f} GB, {len(cols['pred'])} hits)", flush=True)


def f1_from_sums(s):
    tp, tt, ss = s[..., 0], s[..., 1], s[..., 2]
    eff, pur = tp / np.maximum(tt, 1e-9), tp / np.maximum(ss, 1e-9)
    return 2 * eff * pur / np.maximum(eff + pur, 1e-9)


def evt_scores(r):
    y = r["y"].astype(int)
    m = y != 2
    return r["gidx"][m], (y[m] == 1).astype(int), _softmax(r["pred"])[m, 1]


def bootstrap(task, res, n_boot, seed=0):
    """res: {variant: predictions}. Returns {variant: dict(score, ci68, ci95, delta, delta_ci68, delta_ci95)}."""
    rng = np.random.default_rng(seed)
    if task == "hit":
        per = {v: hit_event_sums(r) for v, r in res.items()}
        ev = per["nominal"][0]
        x = {v: s for v, (e, s) in per.items()}
        assert all(np.array_equal(per[v][0], ev) for v in per)
        n = len(ev)
        idx = rng.integers(0, n, size=(n_boot, n))
        stat = {v: f1_from_sums(s[idx].sum(1)) for v, s in x.items()}
        point = {v: float(f1_from_sums(s.sum(0))) for v, s in x.items()}
    else:
        per = {v: evt_scores(r) for v, r in res.items()}
        g, y, _ = per["nominal"]
        assert all(np.array_equal(per[v][0], g) for v in per)
        n = len(g)
        idx = rng.integers(0, n, size=(n_boot, n))
        stat = {v: np.array([roc_auc_score(y[i], s[i]) for i in idx]) for v, (_, _, s) in per.items()}
        point = {v: float(roc_auc_score(y, s)) for v, (_, _, s) in per.items()}
    q = lambda a, lo, hi: [float(np.quantile(a, lo)), float(np.quantile(a, hi))]
    out = {}
    for v in res:
        d = stat[v] - stat["nominal"]
        out[v] = dict(score=point[v], ci68=q(stat[v], 0.16, 0.84), ci95=q(stat[v], 0.025, 0.975),
                      delta=point[v] - point["nominal"], delta_ci68=q(d, 0.16, 0.84), delta_ci95=q(d, 0.025, 0.975),
                      n_events=int(n))
    return out


def hit_event_counts(r):
    """Per-event UNWEIGHTED (tp, true, selected) hit counts (paper_metrics.t1 definition), same event order as
    hit_event_sums."""
    t, s = r["hit_tgt"].astype(bool), r["pred"] > 0.5
    ev, inv = np.unique(r["hit_evt"], return_inverse=True)
    return ev, np.stack([np.bincount(inv, s & t, len(ev)), np.bincount(inv, t, len(ev)),
                         np.bincount(inv, s, len(ev))], 1).astype(np.float64)


def _ci(stat, point, nom_stat, nom_point, n):
    q = lambda a, lo, hi: [float(np.nanquantile(a, lo)), float(np.nanquantile(a, hi))]
    d = stat - nom_stat
    return dict(score=point, ci68=q(stat, 0.16, 0.84), ci95=q(stat, 0.025, 0.975), delta=point - nom_point,
                delta_ci68=q(d, 0.16, 0.84), delta_ci95=q(d, 0.025, 0.975), n_events=int(n))


def bootstrap_t1_median(res, n_boot, seed=0):
    """Median over events of the unweighted per-event F1 (paper T1; events with no true and no selected hit
    skipped), bootstrapped over events with the resampling indices of bootstrap('hit')."""
    rng = np.random.default_rng(seed)
    per = {v: hit_event_counts(r) for v, r in res.items()}
    ev = per["nominal"][0]
    assert all(np.array_equal(per[v][0], ev) for v in per)
    n = len(ev)
    idx = rng.integers(0, n, size=(n_boot, n))
    f1 = {}
    for v, (_, s) in per.items():
        den = s[:, 1] + s[:, 2]
        f1[v] = np.where(den > 0, 2 * s[:, 0] / np.maximum(den, 1e-9), np.nan)
    stat = {v: np.nanmedian(f[idx], axis=1) for v, f in f1.items()}
    point = {v: float(np.nanmedian(f)) for v, f in f1.items()}
    return {v: _ci(stat[v], point[v], stat["nominal"], point["nominal"], n) for v in res}


def _auc(y, s):
    return float(roc_auc_score(y, s)) if y.min() != y.max() else float("nan")


def bootstrap_auc_subsets(res, n_boot, masks, seed=0):
    """T2 AUC (in-AV nu vs no-nu) on event subsets. masks: {name: bool array aligned with the class != 2 events of
    res['nominal'] (sorted gidx)}. Same resampling indices as bootstrap('evt_nu')."""
    rng = np.random.default_rng(seed)
    per = {v: evt_scores(r) for v, r in res.items()}
    g, y, _ = per["nominal"]
    n = len(g)
    idx = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for name, mk in masks.items():
        stat, point = {}, {}
        for v, (_, _, sc) in per.items():
            point[v] = _auc(y[mk], sc[mk])
            stat[v] = np.array([_auc(y[i][mk[i]], sc[i][mk[i]]) for i in idx])
        out[name] = {v: _ci(stat[v], point[v], stat["nominal"], point["nominal"], int(mk.sum())) for v in res}
    return out


# ---------------- T5 target test ----------------
T5_GROUPS = ("nu", "out_av", "nonu_flash", "nonu_dark")
T5_MIN_DARK = 100


def t5_event_stats(pred, tgt):
    """Per event: MSE over all 32 x 17 features, MSE of the per-PMT total (last feature), signed total bias
    mean_pmt(pred - target). pred, tgt: [n, 32, 17]."""
    d = pred.astype(np.float32) - tgt.astype(np.float32)
    return dict(mse=(d ** 2).mean((1, 2)), mse_tot=(d[..., -1] ** 2).mean(1), bias_tot=d[..., -1].mean(1))


def t5_groups(elig, y, has_flash, pb, nonu_dark_pb=None, min_dark=T5_MIN_DARK):
    """Group label per event: nu (evt_nu class 1), out_av (2), nonu_flash / nonu_dark (class 0), other (not eligible for
    evt_nu). nonu_dark = class 0 without a flash (default), or P_b < nonu_dark_pb if given; if the default has fewer than
    min_dark events it falls back to P_b < 20 PE. Returns (labels, definition dict)."""
    lab = np.full(len(y), "other", dtype=object)
    lab[elig & (y == 1)] = "nu"
    lab[elig & (y == 2)] = "out_av"
    c0 = elig & (y == 0)
    n_noflash = int((c0 & ~has_flash).sum())
    if nonu_dark_pb is not None:
        dark, how, cut = c0 & (pb < nonu_dark_pb), "pb_cut", float(nonu_dark_pb)
    elif n_noflash >= min_dark:
        dark, how, cut = c0 & ~has_flash, "no_flash", None
    else:
        dark, how, cut = c0 & (pb < 20.0), "pb_cut_fallback", 20.0
    lab[c0] = "nonu_flash"
    lab[dark] = "nonu_dark"
    return lab, dict(definition=how, pb_cut=cut, n_class0_noflash=n_noflash, n_class0=int(c0.sum()))


def t5_bootstrap(stats, groups, n_boot, seed=0):
    """stats: {variant: t5_event_stats dict} (key 'nominal' required); groups: label per event. Paired bootstrap over
    events (same resampled events for every variant). For every variant: per group mean of mse / mse_tot / bias_tot
    with 95% CI; gap = group - nonu_dark (groups nu, out_av, nonu_flash), its change from nominal and the fraction
    of the nominal gap closed, (gap_nom - gap)/gap_nom, with CIs. 'other' is reported but excluded from the gaps."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    idx = rng.integers(0, n, size=(n_boot, n))
    W = np.zeros((n_boot, n), np.float32)
    for i in range(n_boot):
        W[i] = np.bincount(idx[i], minlength=n)
    names = list(T5_GROUPS) + ["other"]
    mk = np.stack([groups == g for g in names], 1).astype(np.float64)          # [n, G]
    cnt = mk.sum(0)
    bcnt = W @ mk

    def group_means(x):
        pt = (x[:, None] * mk).sum(0) / np.maximum(cnt, 1)
        bs = (W @ (x[:, None] * mk)) / np.maximum(bcnt, 1)
        return pt, bs
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
    pt, bs = {}, {}
    for v, st in stats.items():
        for m in ("mse", "mse_tot", "bias_tot"):
            pt[v, m], bs[v, m] = group_means(st[m].astype(np.float64))
    gi = {g: i for i, g in enumerate(names)}
    out = {}
    for v in stats:
        o = dict(groups={}, gap={})
        for g in names:
            o["groups"][g] = dict(n=int(cnt[gi[g]]))
            for m in ("mse", "mse_tot", "bias_tot"):
                o["groups"][g][m] = dict(score=float(pt[v, m][gi[g]]), ci95=q(bs[v, m][:, gi[g]]))
        d_i = gi["nonu_dark"]
        for g in ("nu", "out_av", "nonu_flash"):
            gp = lambda vv: (pt[vv, "mse"][gi[g]] - pt[vv, "mse"][d_i], bs[vv, "mse"][:, gi[g]] - bs[vv, "mse"][:, d_i])
            gap, gapb = gp(v)
            g0, g0b = gp("nominal")
            with np.errstate(divide="ignore", invalid="ignore"):
                closed, closedb = (g0 - gap) / g0, (g0b - gapb) / g0b
            o["gap"][g] = dict(gap=float(gap), gap_ci95=q(gapb), nominal_gap=float(g0), change=float(gap - g0),
                               change_ci95=q(gapb - g0b), frac_closed=float(closed),
                               frac_closed_ci95=[float(np.nanquantile(closedb, 0.025)), float(np.nanquantile(closedb, 0.975))])
        out[v] = o
    return out


def t5_analysis(pred, tgts, info, lab_evt, n_boot, nonu_dark_pb=None, bs_seed=0):
    """pred: [n, 32, 17] nominal-input predictions of the T5 test events; tgts: {'nominal'|variant: [n, 32, 17]};
    info: dict(gidx, has, pb) per event (same order); lab_evt = (elig, y) arrays indexed by gidx. Corrections are
    applied only to has_flash events outside nonu_dark (so nonu_dark targets are the nominal ones in every variant)."""
    g = info["gidx"]
    elig, y = lab_evt[0][g], lab_evt[1][g]
    groups, dfn = t5_groups(elig, y, info["has"], info["pb"], nonu_dark_pb)
    corr = info["has"] & (groups != "nonu_dark")
    stats = {}
    for v, t in tgts.items():
        tt = tgts["nominal"] if v == "nominal" else np.where(corr[:, None, None], t, tgts["nominal"])
        stats[v] = t5_event_stats(pred, tt)
    out = t5_bootstrap(stats, groups, n_boot, bs_seed)
    dark = groups == "nonu_dark"
    dm = {v: float(stats[v]["mse"][dark].mean()) if dark.any() else float("nan") for v in stats}
    c0 = elig & (y == 0)
    q = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    return dict(dark_definition=dfn, n_events=int(len(g)), group_counts={k: int((groups == k).sum()) for k in T5_GROUPS + ("other",)},
                class0_pb=dict(n=int(c0.sum()), frac_flash=float(info["has"][c0].mean()) if c0.any() else None,
                               quantiles=dict(zip(map(str, q), np.quantile(info["pb"][c0], q).tolist())) if c0.any() else None,
                               frac_below={str(c): float((info["pb"][c0] < c).mean()) for c in (1, 5, 20, 50, 100, 200)} if c0.any() else None),
                nonu_dark=dict(mse_by_variant=dm, max_abs_diff=float(np.nanmax(np.abs(np.array(list(dm.values())) - dm["nominal"]))),
                               identical=bool(all(dm[v] == dm["nominal"] for v in dm)), n=int(dark.sum())),
                nominal_mse=float(stats["nominal"]["mse"].mean()), variants=out)


# ---------------- model / test-set reconstruction ----------------
def load_run(run, dev, absolute_domain=False):
    a = json.load(open(os.path.join(run, "metrics.json")))
    multitask = "tasks" in a
    a.setdefault("pimm_prep", "legacy")          # runs before audit #6 used the round 2-4 PTv3 preprocessing
    if multitask:
        from qlfm.train_multitask import MultiTaskModel, get_args
        ns = get_args(["--mode", a["mode"], "--fm", a["fm"], "--out", run])
    else:
        from qlfm.train_downstream import DownstreamModel, get_args
        ns = get_args(["--task", a["task"], "--mode", a["mode"], "--fm", a["fm"] or "", "--out", run])
    for k, v in a.items():
        if hasattr(ns, k) and k not in ("n_test", "n_val"):
            setattr(ns, k, v)
    backbone = build_backbone(ns.backbone, ns, ns.fm if ns.mode != "scratch" else None)
    if multitask:
        model = MultiTaskModel(backbone, list(a["tasks"]), ns.mode, ns.late_light, ns.hit_feat)
        model.width = a.get("width", "native")       # widths the model was trained with (applied in its forward)
        model.domain = a.get("domain", "sim")        # input domain the model was trained in (applied in its forward)
        if absolute_domain:                          # variants are then absolute input domains (nominal = sim)
            model.width, model.domain = "native", "sim"
    else:
        model = DownstreamModel(backbone, ns.task, ns.mode, ns.late_light, ns.hit_feat)
    model = model.to(dev)
    sd = torch.load(os.path.join(run, "best.pt"), map_location=dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("backbone.") or "adapter_" in k] + list(unexpected)
    if bad:
        raise RuntimeError(f"checkpoint/model mismatch: {bad[:8]}")
    model.eval()
    return a, ns, model, multitask


def test_sets(a, ns, multitask, tasks):
    corpus = Corpus(ns.corpus)
    sp = np.load(ns.splits)
    lab = event_labels({k: corpus.ev_array(k) for k in EV_KEYS})
    pick = lambda idx, n, seed: np.sort(np.random.default_rng(seed).permutation(idx)[:n])
    n_test = a["n_test"] if multitask else {a["task"]: a["n_test"]}
    te = {t: pick(sp["test"][lab[t][0][sp["test"]]], n_test[t], 12) for t in tasks}
    return corpus, lab, te


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tasks", default="hit,evt_nu")
    ap.add_argument("--variants", default="all", help="comma list of " + ",".join(VARIANTS))
    ap.add_argument("--sim-edep", type=float, default=0.0, help="hit counts as simulated if nu edep > this (MeV)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None, help="output directory (default RUN_DIR/perturb)")
    ap.add_argument("--save-preds", default="", help="comma list of variants whose hit predictions are written to "
                                                     "OUT/preds_<variant>.npz (with hit_plane, hit_sim)")
    ap.add_argument("--t5-targets", default="", help="comma list of " + ",".join(T5_TARGETS) + " (needs 'light' in --tasks)")
    ap.add_argument("--nonu-dark-pb", type=float, default=None,
                    help="nonu_dark = class 0 with P_b < this (PE); default: class 0 without a flash, falling back "
                         "to P_b < 20 PE if that has fewer than %d events in the T5 test subset" % T5_MIN_DARK)
    ap.add_argument("--absolute-domain", action="store_true",
                    help="ignore the model's training width/domain: variants are absolute input domains (nominal = sim)")
    ap.add_argument("--time-limit", type=float, default=None, help="accepted for run_queue.sh; unused")
    args = ap.parse_args(argv)
    t0 = time.time()
    dev = torch.device("cuda")
    out_dir = args.out or os.path.join(args.run, "perturb")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    variants = WIDTH_VARIANTS if args.variants == "all" else ["nominal"] + [v for v in args.variants.split(",")
                                                                             if v != "nominal"]
    save_set = [v for v in args.save_preds.split(",") if v]
    assert all(v in variants for v in save_set), f"--save-preds variants must be run variants: {save_set}"
    a, ns, model, multitask = load_run(args.run, dev, args.absolute_domain)
    run_tasks = list(a["tasks"]) if multitask else [a["task"]]
    tasks = [t for t in args.tasks.split(",") if t in run_tasks]
    t5_targets = [v for v in args.t5_targets.split(",") if v]
    assert all(v in T5_TARGETS for v in t5_targets), t5_targets
    if t5_targets:
        assert multitask and "light" in tasks, "--t5-targets needs a multi-task run with 'light' in --tasks"
    if multitask:                      # only the requested heads; T5 keeps the trained light-off pass (CL), else not needed
        model.tasks = tasks
        if "light" not in tasks:
            model.light_off_pass = False
    corpus, lab, te = test_sets(a, ns, multitask, tasks)
    lab_np = {t: (lab[t][0], lab[t][1]) for t in tasks}
    union = np.unique(np.concatenate([te[t] for t in tasks]))
    loader = torch.utils.data.DataLoader(EventDataset(corpus, union, truth=True), batch_size=args.bs, shuffle=False,
                                         collate_fn=collate, num_workers=args.workers, pin_memory=True)
    wrapped = Perturbed(model, args.sim_edep, t5_targets)

    stored = {t: (a["test"][t] if multitask else a["test"]) for t in tasks}
    res = {t: {} for t in tasks}
    sim_share = None
    for v in variants:
        wrapped.variant = v
        torch.manual_seed(0)           # identical random token caps in every variant -> paired comparison
        if multitask:
            from qlfm.train_multitask import predict
            kw = {}
            if v == "nominal" and "hit" in tasks:     # per-hit plane / simulated flag (same hit order in every variant)
                kw["hit_extra"] = {"hit_plane": lambda b: b["hit_plane"].to(torch.uint8),
                                   "hit_sim": lambda b: b["hit_nu_edep"] > args.sim_edep}
            r = predict(wrapped, loader, dev, {t: te[t] for t in tasks}, lab_np, **kw)
            if v == "nominal" and "hit" in tasks:
                hit_extra = {k: r["hit"].pop(k) for k in ("hit_plane", "hit_sim")}
        else:
            from qlfm.train_downstream import predict
            t = tasks[0]
            r1 = predict(wrapped, loader, dev, t, (lambda g: lab[t][1][g]) if lab[t][1] is not None else (lambda g: None))
            keep = np.isin(r1["gidx"], te[t])
            assert keep.all()
            r = {t: r1}
        for t in tasks:
            res[t][v] = r[t]
        if v in save_set and multitask and "hit" in tasks:
            save_preds(os.path.join(out_dir, f"preds_{v}.npz"), {**r["hit"], **hit_extra})
        m = {t: metrics_for(t, r[t])["score"] for t in tasks}
        print(json.dumps({"variant": v, **m, "min": round((time.time() - t0) / 60, 1)}), flush=True)
        if v == "nominal":
            for t in tasks:
                print(f"nominal {t}: {m[t]:.4f} (stored {stored[t]['score']:.4f})", flush=True)
            c = wrapped.count
            sim_share = dict(n_hits=int(c[0]), frac_sim=float(c[1] / max(c[0], 1)),
                             frac_sim_below_label=float(c[2] / max(c[0], 1)))
            print(json.dumps(sim_share), flush=True)
    stat_tasks = [t for t in tasks if t in ("hit", "evt_nu")]
    summary = {t: bootstrap(t, res[t], args.n_boot) for t in stat_tasks}
    info, order = wrapped.event_info()
    if "hit" in tasks:                 # paper T1: per-event unweighted median F1
        for v, x in bootstrap_t1_median(res["hit"], args.n_boot).items():
            summary["hit"][v]["evt_med"] = x
    if "evt_nu" in tasks:              # T2 subsets from NATIVE light: paper bright cut, native P_b classes
        g = evt_scores(res["evt_nu"]["nominal"])[0]
        ii = np.searchsorted(info["gidx"], g)
        assert (info["gidx"][ii] == g).all()
        masks = {"bright": info["pe_win"][ii] >= BRIGHT_PE, "pb_lt600": info["pb"][ii] < PB_BRIGHT,
                 "pb_ge600": info["pb"][ii] >= PB_BRIGHT}
        for name, x in bootstrap_auc_subsets(res["evt_nu"], args.n_boot, masks).items():
            for v in variants:
                summary["evt_nu"][v][name] = x[v]
    if multitask and "hit" in tasks:
        bp = bootstrap_by_plane(res["hit"], hit_extra["hit_plane"], args.n_boot)
        for v in variants:
            summary["hit"][v]["by_plane"] = bp[v]
    flash = info["has"].astype(bool)
    light_diag = dict(
        n_events=int(len(flash)), frac_flash=float(flash.mean()),
        tb_quantiles=dict(zip(["0.05", "0.25", "0.5", "0.75", "0.95"], np.quantile(info["tb"][flash], [.05, .25, .5, .75, .95]).tolist())),
        tb_hist=np.histogram(info["tb"][flash], bins=NB_TB, range=TB_WIN)[0].tolist(),
        pb_quantiles=dict(zip(["0.05", "0.25", "0.5", "0.75", "0.95"], np.quantile(info["pb"][flash], [.05, .25, .5, .75, .95]).tolist())),
        frac_pb_ge600=float((info["pb"] >= PB_BRIGHT).mean()), frac_pb_ge600_of_flash=float((info["pb"][flash] >= PB_BRIGHT).mean()),
        frac_bright_pe_win=float((info["pe_win"] >= BRIGHT_PE).mean()),
        pe_windows_us=[list(w) if w[1] != float("inf") else [w[0], None] for w in DT_DIAG],
        pe_by_variant={v: dict(dim=dict(before=d[0, 0].tolist(), after=d[0, 1].tolist()),
                               bright=dict(before=d[1, 0].tolist(), after=d[1, 1].tolist()))
                       for v, d in wrapped.pe_diag.items()})
    np.savez_compressed(os.path.join(out_dir, "event_info.npz"), **info)       # native per-event gidx, tb, pb, has, pe_win
    t5 = None
    if t5_targets:
        lr = res["light"]["nominal"]
        sel = np.searchsorted(info["gidx"], lr["gidx"])
        assert (info["gidx"][sel] == lr["gidx"]).all()
        tg = wrapped.t5_target_arrays(order)
        tg = {k: v[sel] for k, v in tg.items()}
        pred = lr["pred"].astype(np.float32)
        t5 = t5_analysis(pred, tg, dict(gidx=lr["gidx"], has=flash[sel], pb=info["pb"][sel]), lab["evt_nu"], args.n_boot,
                         args.nonu_dark_pb)
        t5["stored_mse"] = stored["light"].get("mse")
        t5["input_variants"] = {v: dict(mse=float(((res["light"][v]["pred"].astype(np.float32)
                                                     - res["light"][v]["light_tgt"].astype(np.float32)) ** 2).mean()),
                                        max_abs_pred_diff=float(np.abs(res["light"][v]["pred"].astype(np.float32) - pred).max()))
                                for v in variants}
    out = dict(run=args.run, backbone=ns.backbone, mode=ns.mode, late_light=ns.late_light, multitask=multitask,
               tasks=tasks, variants=variants, sim_edep=args.sim_edep, n_boot=args.n_boot,
               stored={t: stored[t]["score"] for t in tasks}, results=summary, sim_share=sim_share,
               model_domain=getattr(model, "domain", None), model_width=getattr(model, "width", None),
               light_diag=light_diag, t5=t5, minutes=(time.time() - t0) / 60)
    json.dump(out, open(out_path + ".tmp", "w"), indent=1)
    os.replace(out_path + ".tmp", out_path)
    if t5:
        print(json.dumps(dict(t5_dark=t5["dark_definition"], groups=t5["group_counts"], nominal_mse=t5["nominal_mse"],
                              stored=t5["stored_mse"], nonu_dark=t5["nonu_dark"])), flush=True)
    for t in stat_tasks:
        for v in variants:
            s = summary[t][v]
            print(f"{t:7s} {v:12s} {s['score']:.4f}  d={s['delta']:+.4f} "
                  f"[{s['delta_ci95'][0]:+.4f},{s['delta_ci95'][1]:+.4f}]  ci95=[{s['ci95'][0]:.4f},{s['ci95'][1]:.4f}]")


if __name__ == "__main__":
    main()
