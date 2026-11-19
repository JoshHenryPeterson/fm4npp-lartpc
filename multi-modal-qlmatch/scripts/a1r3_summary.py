"""A1 round-3 summaries (docs/a1_check_round_3_plan.md).

python scripts/a1r3_summary.py --part 1 [--a1r3 DIR] [--check-gate5]   -> A1R3/novnorm/summary.md
python scripts/a1r3_summary.py --part 2 [--a1r3 DIR]                   -> A1R3/summary.md, summary.json

Part 1 reads A1R3/novnorm/<run>/metrics.json (+ preds_w_meas3d_noVnorm.npz) for the 6 round-2 models and compares w_meas3d to
$QLFM_WORK/a1r2/full/<run>/metrics.json (gate 5: |d T1 charge-wtd| <= 0.002, |d T2| <= 0.001; exit 1 with --check-gate5 if it fails).
Part 2 reads A1R3/eval/<name>/metrics.json (+ preds_d_real.npz), A1R3/down/*/event_lists.npz and A1R3/pre/*/event_lists.npz and
re-checks that pretrain / downstream-train / val / test lists are disjoint. Eval dir names:
  new runs   {cl,cxf}_{sim,real}2{sim,real}_{frozen,finetune}_s<seed>
  baselines  [base_|baseline_]{cl,cxf}_{frozen,finetune}_s<seed>  or  s3_mt_qlfm_M_{CL,C_xf}_{mode}_f0.01_s<seed>
Missing runs/files are reported in the summary and make the exit code 1 (as a1r2_summarize.py)."""
import argparse
import json
import os
import re
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS = {"qlfm_M_CL": "qlfm-M CL", "qlfm_M_C_xf": "qlfm-M C + XF"}
SEEDS = [0, 1, 2]
P1_VARS = ["nominal", "w_meas3d", "w_meas3d_noVnorm", "w_meas3d_V", "w_meas3d_noVnorm_V"]
GATE_T1, GATE_T2 = 0.002, 0.001
PL = "UVY"
errors = []


def ms(x, f="{:+.4f}"):
    x = [a for a in x if a is not None and np.isfinite(a)]
    if not x:
        return "n/a"
    return (f + " ± " + f.replace("+", "")).format(np.mean(x), np.std(x, ddof=1) if len(x) > 1 else 0.0)


def load(p):
    if not os.path.exists(p):
        errors.append(f"missing {p}")
        return None
    return json.load(open(p))


def npz_planes(d, v):
    """Per-event-median / charge-weighted T1 per plane from preds_<v>.npz (paper_metrics.t1_by_plane), cached."""
    cache = os.path.join(d, f"t1_npz_{v}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    p = os.path.join(d, f"preds_{v}.npz")
    if not os.path.exists(p):
        errors.append(f"missing {p}")
        return None
    from qlfm import paper_metrics as PM
    z = np.load(p)
    g, bp = PM.t1(z), PM.t1_by_plane(z)
    out = dict(all=dict(evt_med=g["f1_evt_med"], f1_q=g["f1_q"]),
               by_plane={k: dict(evt_med=x["f1_evt_med"], f1_q=x["f1_q"]) for k, x in bp.items()})
    try:
        json.dump(out, open(cache + ".tmp", "w"), indent=1)
        os.replace(cache + ".tmp", cache)
    except OSError:
        pass
    return out


def write(path, text):
    open(path + ".tmp", "w").write(text)
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------- part 1
def part1(a1r3, work, check_gate5):
    nv = os.path.join(a1r3, "novnorm")
    ref = os.path.join(work, "a1r2/full")
    tag = lambda m, s: f"s3_mt_{m}_finetune_f0.01_s{s}"
    L = ["# A1 round 3, part 1: w_meas3d_noVnorm", "",
         f"Source: `{nv}`. Mean ± seed SD (ddof=1) over the seeds present; deltas vs nominal. T1 charge-wtd and T2 are the "
         "eval_perturb in-run scores; T1 evt-median (paper definition) is from the in-run evt_med bootstrap; T2 bright = native "
         "3-6 us PE >= 200. noVnorm = V table not renormalised at the anode (0.900 at the first drift bin).", ""]
    runs, gate = {}, []
    for m in MODELS:
        for s in SEEDS:
            r = load(os.path.join(nv, tag(m, s), "metrics.json"))
            if r is not None:
                runs[m, s] = r["results"]
    for m, name in MODELS.items():
        rs = [(s, runs[m, s]) for s in SEEDS if (m, s) in runs]
        if not rs:
            continue
        L += [f"## {name} ({len(rs)} seeds)", "",
              "| variant | T1 charge-wtd | T1 evt-median | T2 all | T2 bright |", "|---|---|---|---|---|"]
        g = lambda v, t, k, sub=None: [(r[t][v][sub] if sub else r[t][v])[k] for _, r in rs if v in r[t]]
        for v in P1_VARS:
            if v == "nominal":
                c = [ms(g(v, "hit", "score"), "{:.4f}"), ms([r["hit"][v]["evt_med"]["score"] for _, r in rs], "{:.4f}"),
                     ms(g(v, "evt_nu", "score"), "{:.4f}"), ms(g(v, "evt_nu", "score", "bright"), "{:.4f}")]
            else:
                c = [ms(g(v, "hit", "delta")), ms([r["hit"][v]["evt_med"]["delta"] for _, r in rs if v in r["hit"]]),
                     ms(g(v, "evt_nu", "delta")), ms(g(v, "evt_nu", "delta", "bright"))]
            L.append(f"| {v} | " + " | ".join(c) + " |")
        d = lambda t, k, a, b: [r[t][a][k] - r[t][b][k] for _, r in rs if a in r[t] and b in r[t]]
        L += ["", f"noVnorm - meas3d (all planes): T1 charge-wtd {ms(d('hit', 'score', 'w_meas3d_noVnorm', 'w_meas3d'))}, "
              f"T2 {ms(d('evt_nu', 'score', 'w_meas3d_noVnorm', 'w_meas3d'))}; V-only noVnorm - normalised: "
              f"T1 {ms(d('hit', 'score', 'w_meas3d_noVnorm_V', 'w_meas3d_V'))}, T2 {ms(d('evt_nu', 'score', 'w_meas3d_noVnorm_V', 'w_meas3d_V'))}.",
              "", "### Per-plane T1 charge-wtd (delta vs nominal on the plane)", "",
              "| plane | nominal | " + " | ".join(P1_VARS[1:]) + " |", "|---|---|" + "---|" * (len(P1_VARS) - 1)]
        for p in PL:
            bp = lambda v, k: [r["hit"][v]["by_plane"][p][k] for _, r in rs if v in r["hit"] and "by_plane" in r["hit"][v]]
            L.append(f"| {p} | {ms(bp('nominal', 'score'), '{:.4f}')} | " + " | ".join(ms(bp(v, "delta")) for v in P1_VARS[1:]) + " |")
        L += ["", "### Per-plane T1 evt-median (npz, paper definition; w_meas3d_noVnorm absolute, round-2 reference where cached)", "",
              "| plane | w_meas3d_noVnorm (this run) | nominal (a1r2) | w_meas3d (a1r2) |", "|---|---|---|---|"]
        nz = {s: npz_planes(os.path.join(nv, tag(m, s)), "w_meas3d_noVnorm") for s, _ in rs}
        r2 = {}
        for s, _ in rs:
            p = os.path.join(ref, tag(m, s), "t1_npz.json")
            r2[s] = json.load(open(p)) if os.path.exists(p) else {}
        for p in PL:
            x = [nz[s]["by_plane"][p]["evt_med"] for s, _ in rs if nz[s]]
            n0 = [r2[s]["nominal"]["by_plane"][p]["evt_med"] for s, _ in rs if "nominal" in r2[s]]
            n1 = [r2[s]["w_meas3d"]["by_plane"][p]["evt_med"] for s, _ in rs if "w_meas3d" in r2[s]]
            L.append(f"| {p} | {ms(x, '{:.4f}')} | {ms(n0, '{:.4f}')} | {ms(n1, '{:.4f}')} |")
        L.append("")
    # gate 5
    L += ["## Gate 5: w_meas3d reproduces a1r2/full", "", f"Tolerances: T1 charge-wtd {GATE_T1}, T2 {GATE_T2} (absolute score).", "",
          "| run | T1 now | T1 a1r2 | |d| | T2 now | T2 a1r2 | |d| | ok |", "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for s in SEEDS:
            t = tag(m, s)
            old = load(os.path.join(ref, t, "metrics.json"))
            if (m, s) not in runs or old is None:
                gate.append(False)
                L.append(f"| {t} | missing | | | | | | FAIL |")
                continue
            new, old = runs[m, s], old["results"]
            try:
                a, b = new["hit"]["w_meas3d"]["score"], old["hit"]["w_meas3d"]["score"]
                c, e = new["evt_nu"]["w_meas3d"]["score"], old["evt_nu"]["w_meas3d"]["score"]
                nv_ok = all(np.isfinite(new[k]["w_meas3d_noVnorm"]["score"]) for k in ("hit", "evt_nu"))
            except KeyError as ex:
                errors.append(f"{t}: missing key {ex}")
                gate.append(False)
                L.append(f"| {t} | missing key {ex} | | | | | | FAIL |")
                continue
            ok = abs(a - b) <= GATE_T1 and abs(c - e) <= GATE_T2 and nv_ok
            gate.append(ok)
            L.append(f"| {t} | {a:.4f} | {b:.4f} | {abs(a - b):.4f} | {c:.4f} | {e:.4f} | {abs(c - e):.4f} | {'ok' if ok else 'FAIL'} |")
    passed = len(gate) == 6 and all(gate)
    L += ["", f"GATE 5 {'PASS' if passed else 'FAIL'} ({sum(gate)}/6 runs ok)", ""]
    if errors:
        L += ["## ERRORS / missing inputs", ""] + [f"- {x}" for x in errors] + [""]
        print("ERRORS:\n" + "\n".join(errors), file=sys.stderr)
    write(os.path.join(nv, "summary.md"), "\n".join(L))
    print(f"wrote {os.path.join(nv, 'summary.md')} ({len(runs)}/6 runs); GATE 5 {'PASS' if passed else 'FAIL'}")
    if check_gate5:
        return 0 if passed else 1
    return 1 if errors else 0


# ---------------------------------------------------------------- part 2
NEW = re.compile(r"^(?P<m>cl|cxf)_(?P<pre>sim|real)2(?P<down>sim|real)_(?P<mode>frozen|finetune)_s(?P<s>\d+)$")
BASE = [re.compile(r"^(?:base|baseline)_(?P<m>cl|cxf)_(?P<mode>frozen|finetune)_s(?P<s>\d+)$"),
        re.compile(r"^s3_mt_qlfm_M_(?P<m>CL|C_xf)_(?P<mode>frozen|finetune)_f0\.01_s(?P<s>\d+)$")]
MNAME = {"cl": "CL", "cxf": "C + XF", "CL": "CL", "C_xf": "C + XF"}
METRICS = ["t1_cw", "t1_med", "t2_all", "t2_bright"]


def parse(name):
    m = NEW.match(name)
    if m:
        return MNAME[m["m"]], m["pre"], m["down"], m["mode"], int(m["s"])
    for r in BASE:
        m = r.match(name)
        if m:
            return MNAME[m["m"]], "existing", "sim", m["mode"], int(m["s"])
    return None


def get_metrics(res, v):
    h, e = res["hit"][v], res["evt_nu"][v]
    return dict(t1_cw=h["score"], t1_med=h["evt_med"]["score"], t2_all=e["score"], t2_bright=e["bright"]["score"])


def check_lists(a1r3):
    """Re-check disjointness from the event_lists.npz files; returns (lines, n_pre, n_down)."""
    import glob
    L, bad = [], []
    pre = {os.path.basename(os.path.dirname(p)): np.load(p) for p in sorted(glob.glob(os.path.join(a1r3, "pre/*/event_lists.npz")))}
    down = {os.path.basename(os.path.dirname(p)): np.load(p) for p in sorted(glob.glob(os.path.join(a1r3, "down/*/event_lists.npz")))}
    inter = lambda a, b: len(np.intersect1d(a, b))
    pre_sets = {k: np.asarray(z["pretrain"]) for k, z in pre.items()}
    first = next(iter(pre_sets.values()), None)
    for k, v in pre_sets.items():
        if len(np.unique(v)) != len(v):
            bad.append(f"pre/{k}: duplicate indices")
        if not np.array_equal(v, first):
            bad.append(f"pre/{k}: pretrain list differs from the first pre run")
    allpre = first if first is not None else np.zeros(0, np.int64)
    ref = None
    for k, z in down.items():
        tr = np.asarray(z["train"])
        vals = {n[4:]: np.asarray(z[n]) for n in z.files if n.startswith("val_")}
        tests = {n[5:]: np.asarray(z[n]) for n in z.files if n.startswith("test_")}
        if inter(tr, allpre):
            bad.append(f"down/{k}: train overlaps pretrain ({inter(tr, allpre)})")
        for t, v in vals.items():
            for what, x in (("train", tr), ("pretrain", allpre)):
                if inter(v, x):
                    bad.append(f"down/{k}: val_{t} overlaps {what} ({inter(v, x)})")
        for t, v in tests.items():
            for what, x in (("train", tr), ("pretrain", allpre)):
                if inter(v, x):
                    bad.append(f"down/{k}: test_{t} overlaps {what} ({inter(v, x)})")
            for t2, v2 in vals.items():
                if inter(v, v2):
                    bad.append(f"down/{k}: test_{t} overlaps val_{t2} ({inter(v, v2)})")
        sig = {n: z[n] for n in z.files}
        if ref is None:
            ref = (k, sig)
        else:                                          # every downstream run uses the same lists
            for n in sig:
                if n not in ref[1] or not np.array_equal(sig[n], ref[1][n]):
                    bad.append(f"down/{k}: {n} differs from down/{ref[0]}")
    L.append(f"{len(pre)} pretrain and {len(down)} downstream event_lists.npz checked: "
             + (f"{len(bad)} PROBLEMS" if bad else "pretrain / downstream-train / val / test pairwise disjoint, lists identical across runs"))
    if first is not None and ref is not None:
        L.append(f"sizes: pretrain {len(first)}, train {len(ref[1]['train'])}, "
                 + ", ".join(f"{n} {len(v)}" for n, v in ref[1].items() if n != "train"))
    if not pre or not down:
        bad.append("no event_lists.npz found under pre/ or down/")
    return L + [f"- PROBLEM: {b}" for b in bad], bad


def part2(a1r3):
    ev = os.path.join(a1r3, "eval")
    runs, rows = {}, {}
    names = sorted(os.listdir(ev)) if os.path.isdir(ev) else []
    for n in names:
        k = parse(n)
        if k is None:
            errors.append(f"unrecognised eval dir name: {n}")
            continue
        r = load(os.path.join(ev, n, "metrics.json"))
        if r is None:
            continue
        try:
            res = r["results"]
            x = dict(real=get_metrics(res, "d_real"), sim=get_metrics(res, "nominal"),
                     w_only=get_metrics(res, "w_meas3d"), l_only=get_metrics(res, "l_comb80"),
                     real_noVnorm=get_metrics(res, "d_real_noVnorm"), name=n,
                     d1=dict(w=res["hit"]["w_meas3d"]["delta"], l=res["hit"]["l_comb80"]["delta"],
                             real=res["hit"]["d_real"]["delta"], ci95=res["hit"]["d_real"]["delta_ci95"]))
        except KeyError as ex:
            errors.append(f"{n}: missing key {ex} in metrics.json")
            continue
        try:
            z = npz_planes(os.path.join(ev, n), "d_real")
            x["planes"] = {p: z["by_plane"][p]["evt_med"] for p in PL} if z else {}
            if z and abs(z["all"]["evt_med"] - x["real"]["t1_med"]) > 2e-3:
                errors.append(f"{n}: npz evt-median {z['all']['evt_med']:.4f} != metrics.json {x['real']['t1_med']:.4f}")
        except Exception:
            errors.append(f"npz postprocessing failed for {n}:\n{traceback.format_exc()}")
            x["planes"] = {}
        runs[n] = x
        rows.setdefault(k[:4], {})[k[4]] = x
    order = lambda k: (k[0], {"existing": 0, "sim": 1, "real": 2}[k[1]], k[2], k[3])
    keys = sorted(rows, key=order)
    L = ["# A1 round 3, part 2: cross-domain pretrain -> downstream", "",
         f"Source: `{a1r3}`. All evaluations use `--absolute-domain`: real = d_real (w_meas3d widths on sim nu hits + l_comb80 light), "
         "sim = nominal. Mean ± seed SD (ddof=1) over the seeds present (single seed: SD 0). T1 cw = charge-weighted F1, "
         "T1 med = per-event median F1 (paper definition), T2 = AUC in-AV nu vs no-nu (all events) and on the bright-flash subset "
         "(native 3-6 us PE >= 200; not the paper truth.npz 200-PE cut). Domain gap = real - sim (per seed). "
         "Control = the real2real row of the same model and mode.", ""]
    summ = {}

    def agg(k, dom, m):
        return [x[dom][m] for x in rows[k].values()]

    def ctrl(k, m):
        c = rows.get((k[0], "real", "real", k[3]))
        return [x["real"][m] for x in c.values()] if c else []
    for model in ("CL", "C + XF"):
        for mode in ("frozen", "finetune"):
            ks = [k for k in keys if k[0] == model and k[3] == mode]
            if not ks:
                continue
            L += [f"## {model}, {mode}", "",
                  "| pretrain -> downstream | n seeds | " + " | ".join(f"{m} real" for m in METRICS) + " | "
                  + " | ".join(f"{m} sim" for m in METRICS) + " | gap " + " | gap ".join(METRICS) + " | d vs control T1 cw | d vs control T2 |",
                  "|---|---|" + "---|" * (4 * 3 + 2)]
            for k in ks:
                lab = "existing (Y) -> sim" if k[1] == "existing" else f"{k[1]} -> {k[2]}"
                cells = [ms(agg(k, "real", m), "{:.4f}") for m in METRICS] + [ms(agg(k, "sim", m), "{:.4f}") for m in METRICS]
                gaps = [ms([x["real"][m] - x["sim"][m] for x in rows[k].values()]) for m in METRICS]
                dv = []
                for m in ("t1_cw", "t2_all"):
                    c = ctrl(k, m)
                    dv.append(ms([np.mean(agg(k, "real", m)) - np.mean(c)]) if c and k[1] != "existing" else "n/a")
                L.append(f"| {lab} | {len(rows[k])} | " + " | ".join(cells + gaps + dv) + " |")
                summ[f"{k[0]}|{k[3]}|{k[1]}2{k[2]}"] = dict(
                    n_seeds=len(rows[k]), seeds=sorted(rows[k]),
                    real={m: [x["real"][m] for x in rows[k].values()] for m in METRICS},
                    sim={m: [x["sim"][m] for x in rows[k].values()] for m in METRICS},
                    gap={m: [x["real"][m] - x["sim"][m] for x in rows[k].values()] for m in METRICS},
                    d_vs_control={m: (float(np.mean(agg(k, "real", m)) - np.mean(ctrl(k, m))) if ctrl(k, m) and k[1] != "existing" else None)
                                  for m in ("t1_cw", "t1_med", "t2_all", "t2_bright")},
                    delta_t1_cw_vs_nominal=dict(w_meas3d=[x["d1"]["w"] for x in rows[k].values()], l_comb80=[x["d1"]["l"] for x in rows[k].values()],
                                                d_real=[x["d1"]["real"] for x in rows[k].values()]),
                    real_noVnorm={m: [x["real_noVnorm"][m] for x in rows[k].values()] for m in METRICS},
                    t1_med_real_by_plane={p: [x["planes"].get(p) for x in rows[k].values()] for p in PL},
                    runs=[x["name"] for x in rows[k].values()])
            L += ["", "Decomposition of the real-domain loss on T1 cw (delta vs sim, per-run bootstrap-paired) and d_real_noVnorm:", "",
                  "| pretrain -> downstream | w_meas3d only | l_comb80 only | d_real | d_real_noVnorm T1 cw | per-seed d_real CI95 |", "|---|---|---|---|---|---|"]
            for k in ks:
                lab = "existing (Y) -> sim" if k[1] == "existing" else f"{k[1]} -> {k[2]}"
                d = [x["d1"] for x in rows[k].values()]
                L.append(f"| {lab} | {ms([x['w'] for x in d])} | {ms([x['l'] for x in d])} | {ms([x['real'] for x in d])} | "
                         f"{ms([x['real_noVnorm']['t1_cw'] for x in rows[k].values()], '{:.4f}')} | "
                         + "; ".join(f"[{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" for x in d) + " |")
            L += ["", "T1 per-event-median F1 by plane on the real domain (npz):", "", "| pretrain -> downstream | U | V | Y |", "|---|---|---|---|"]
            for k in ks:
                lab = "existing (Y) -> sim" if k[1] == "existing" else f"{k[1]} -> {k[2]}"
                L.append(f"| {lab} | " + " | ".join(ms([x["planes"].get(p) for x in rows[k].values()], "{:.4f}") for p in PL) + " |")
            L.append("")
    ll, bad = check_lists(a1r3)
    L += ["## Event lists (disjointness re-check)", ""] + [f"- {x}" for x in ll] + [""]
    errors.extend(bad)
    L += ["## Caveats", "", "- Real is a modelled data-like domain (round-2 width factors, l_comb80 light), not detector data; widths corrected on "
          "simulated nu hits only, light on all ophits of flash events.",
          "- C + XF pretraining sees only the width part of the shift (charge-only backbone).",
          "- T5 (light) targets stay native in the real-domain downstream runs; only T1/T2 are reported.",
          "- One pretraining seed per backbone; seed spread is downstream only.", ""]
    if errors:
        L += ["## ERRORS / missing inputs", ""] + [f"- {x}" for x in errors] + [""]
        print("ERRORS:\n" + "\n".join(errors), file=sys.stderr)
    write(os.path.join(a1r3, "summary.md"), "\n".join(L))
    write(os.path.join(a1r3, "summary.json"), json.dumps(dict(rows=summ, event_lists=ll, errors=errors), indent=1))
    print(f"wrote {os.path.join(a1r3, 'summary.md')} ({len(runs)} eval runs, {len(errors)} errors)")
    return 1 if errors else 0


def main(argv=None):
    from qlfm.paths import WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, required=True, choices=[1, 2])
    ap.add_argument("--a1r3", default=os.path.join(WORK, "a1r3"))
    ap.add_argument("--work", default=WORK, help="holds a1r2/full (part 1 reference)")
    ap.add_argument("--check-gate5", action="store_true", help="part 1: exit code = gate 5 only (0 pass / 1 fail)")
    a = ap.parse_args(argv)
    return part1(a.a1r3, a.work, a.check_gate5) if a.part == 1 else part2(a.a1r3)


if __name__ == "__main__":
    sys.exit(main())
