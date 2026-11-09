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
F_MEAS = {2: [1.000, 0.972, 0.951, 0.936, 0.927, 0.919],         # Y
          1: [1.000, 0.982, 0.965, 0.952, 0.954, 0.955],         # V (provisional)
          0: [1.000, 0.969, 0.936, 0.916, 0.901, 0.895]}         # U (provisional)
US_PER_TICK, V_DRIFT_CM_US, MAX_DRIFT_CM = 0.5, 0.1098, 256.35
D_DATA, D_SIM, S0_US2 = 3.74e-6, 6.40e-6, 1.9                    # cm2/us, cm2/us, us2
WIDTH_CONST = 4.0                                                # ticks, about the median hit rms

VARIANTS = {"nominal": (None, 0.0), "w_meas": ("meas", 0.0), "w_diff": ("diff", 0.0),
            "q-5": (None, -0.05), "q-2": (None, -0.02), "q+2": (None, 0.02), "q+5": (None, 0.05),
            "w_meas_q-5": ("meas", -0.05), "w_meas_q+5": ("meas", 0.05),
            "w_inv": ("inv", 0.0), "w_c98": ("c0.98", 0.0), "w_c95": ("c0.95", 0.0), "w_meas_all": ("meas_all", 0.0),
            "w_const": ("const", 0.0)}


def _interp(x, xp, fp):
    """np.interp on torch tensors (end values held)."""
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1, y0, y1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    return y0 + ((x - x0) / (x1 - x0)).clamp(0, 1) * (y1 - y0)


def width_factor(plane, tick, kind):
    d = tick.float() - ANODE_TICK
    if kind == "inv":
        return 1 / width_factor(plane, tick, "meas")
    if kind.startswith("c"):
        return torch.full_like(d, float(kind[1:]))
    if kind == "meas":
        xp = torch.tensor(D_C, device=tick.device)
        f = torch.ones_like(d)
        for p, fp in F_MEAS.items():
            m = plane == p
            f[m] = _interp(d[m], xp, torch.tensor(fp, device=tick.device))
        return f
    if kind == "diff":
        t = (d * US_PER_TICK).clamp(0, MAX_DRIFT_CM / V_DRIFT_CM_US)
        v2 = V_DRIFT_CM_US ** 2
        return torch.sqrt((S0_US2 + 2 * D_DATA * t / v2) / (S0_US2 + 2 * D_SIM * t / v2))
    raise ValueError(kind)


def perturb(b, variant, sim_edep):
    width, eps = VARIANTS[variant]
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
    """Applies the current variant to a copy of the batch; prediction code still reads the original batch."""

    def __init__(self, model, sim_edep):
        super().__init__()
        self.model, self.sim_edep, self.variant = model, sim_edep, "nominal"
        self.count = np.zeros(3)                 # hits, simulated hits, simulated hits below the T1 label threshold

    def forward(self, b):
        if self.variant == "nominal":
            e = b["hit_nu_edep"]
            self.count += [e.numel(), int((e > self.sim_edep).sum()), int(((e > self.sim_edep) & (e <= HIT_NU_EDEP_MEV)).sum())]
        return self.model(perturb(b, self.variant, self.sim_edep))


# ---------------- statistics ----------------
def hit_event_sums(r):
    """Per-event charge-weighted (tp, true, selected) sums; T1 score = F1 of the summed counts (= hit_metrics)."""
    q, t, s = r["hit_q"].clip(min=0), r["hit_tgt"].astype(bool), r["pred"] > 0.5
    ev, inv = np.unique(r["hit_evt"], return_inverse=True)
    sums = np.stack([np.bincount(inv, q * (s & t), len(ev)), np.bincount(inv, q * t, len(ev)),
                     np.bincount(inv, q * s, len(ev))], 1)
    return ev, sums


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


# ---------------- model / test-set reconstruction ----------------
def load_run(run, dev):
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
    ap.add_argument("--time-limit", type=float, default=None, help="accepted for run_queue.sh; unused")
    args = ap.parse_args(argv)
    t0 = time.time()
    dev = torch.device("cuda")
    out_dir = args.out or os.path.join(args.run, "perturb")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    variants = list(VARIANTS) if args.variants == "all" else ["nominal"] + [v for v in args.variants.split(",")
                                                                             if v != "nominal"]
    a, ns, model, multitask = load_run(args.run, dev)
    run_tasks = list(a["tasks"]) if multitask else [a["task"]]
    tasks = [t for t in args.tasks.split(",") if t in run_tasks]
    if multitask:                      # only the requested heads; T5's light-off pass is not needed
        model.tasks, model.light_off_pass = tasks, False
    corpus, lab, te = test_sets(a, ns, multitask, tasks)
    lab_np = {t: (lab[t][0], lab[t][1]) for t in tasks}
    union = np.unique(np.concatenate([te[t] for t in tasks]))
    loader = torch.utils.data.DataLoader(EventDataset(corpus, union, truth=True), batch_size=args.bs, shuffle=False,
                                         collate_fn=collate, num_workers=args.workers, pin_memory=True)
    wrapped = Perturbed(model, args.sim_edep)

    stored = {t: (a["test"][t] if multitask else a["test"]) for t in tasks}
    res = {t: {} for t in tasks}
    sim_share = None
    for v in variants:
        wrapped.variant = v
        torch.manual_seed(0)           # identical random token caps in every variant -> paired comparison
        if multitask:
            from qlfm.train_multitask import predict
            r = predict(wrapped, loader, dev, {t: te[t] for t in tasks}, lab_np)
        else:
            from qlfm.train_downstream import predict
            t = tasks[0]
            r1 = predict(wrapped, loader, dev, t, (lambda g: lab[t][1][g]) if lab[t][1] is not None else (lambda g: None))
            keep = np.isin(r1["gidx"], te[t])
            assert keep.all()
            r = {t: r1}
        for t in tasks:
            res[t][v] = r[t]
        m = {t: metrics_for(t, r[t])["score"] for t in tasks}
        print(json.dumps({"variant": v, **m, "min": round((time.time() - t0) / 60, 1)}), flush=True)
        if v == "nominal":
            for t in tasks:
                print(f"nominal {t}: {m[t]:.4f} (stored {stored[t]['score']:.4f})", flush=True)
            c = wrapped.count
            sim_share = dict(n_hits=int(c[0]), frac_sim=float(c[1] / max(c[0], 1)),
                             frac_sim_below_label=float(c[2] / max(c[0], 1)))
            print(json.dumps(sim_share), flush=True)
    summary = {t: bootstrap(t, res[t], args.n_boot) for t in tasks}
    out = dict(run=args.run, backbone=ns.backbone, mode=ns.mode, late_light=ns.late_light, multitask=multitask,
               tasks=tasks, variants=variants, sim_edep=args.sim_edep, n_boot=args.n_boot,
               stored={t: stored[t]["score"] for t in tasks}, results=summary, sim_share=sim_share,
               minutes=(time.time() - t0) / 60)
    json.dump(out, open(out_path + ".tmp", "w"), indent=1)
    os.replace(out_path + ".tmp", out_path)
    for t in tasks:
        for v in variants:
            s = summary[t][v]
            print(f"{t:7s} {v:12s} {s['score']:.4f}  d={s['delta']:+.4f} "
                  f"[{s['delta_ci95'][0]:+.4f},{s['delta_ci95'][1]:+.4f}]  ci95=[{s['ci95'][0]:.4f},{s['ci95'][1]:.4f}]")


if __name__ == "__main__":
    main()
