"""A1 round-2 postprocessing: python a1r2_summarize.py FULL_DIR  ->  FULL_DIR/summary.md (+ t1_npz.json per run).
Per-event-median T1 and per-plane T1 come from the preds npz files (paper_metrics.t1 / t1_by_plane); charge-weighted
T1, T2 and their bootstrap CIs come from metrics.json. Missing runs/files are reported in the summary, not fatal."""
import glob
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS = {"qlfm_M_CL": "qlfm-M CL", "qlfm_M_C_xf": "qlfm-M C + XF"}
SEEDS = [0, 1, 2]
VARS = ["nominal", "w_meas", "w_meas3d", "w_meas3d_Y", "w_meas3d_UV", "w_diff", "w_diff_vd"]
NPZ_VARS = ["nominal", "w_meas3d"]
PL = "UVY"
errors = []


def tag(m, s):
    return f"s3_mt_{m}_finetune_f0.01_s{s}"


def npz_stats(d):
    """Per-event-median / charge-weighted T1, overall and per plane, for each saved variant; cached in t1_npz.json."""
    cache = os.path.join(d, "t1_npz.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    from qlfm import paper_metrics as PM
    out = {}
    for v in NPZ_VARS:
        p = os.path.join(d, f"preds_{v}.npz")
        if not os.path.exists(p):
            errors.append(f"missing {p}")
            continue
        z = np.load(p)
        g, bp = PM.t1(z), PM.t1_by_plane(z)
        out[v] = dict(all=dict(evt_med=g["f1_evt_med"], f1_q=g["f1_q"]),
                      by_plane={k: dict(evt_med=x["f1_evt_med"], f1_q=x["f1_q"]) for k, x in bp.items()})
        print(f"  {os.path.basename(d)} {v}: evt-med {g['f1_evt_med']:.4f}", flush=True)
    if len(out) == len(NPZ_VARS):
        json.dump(out, open(cache + ".tmp", "w"), indent=1)
        os.replace(cache + ".tmp", cache)
    return out


def ms(x, f="{:+.4f}"):
    x = [a for a in x if a is not None]
    if not x:
        return "n/a"
    return (f + " ± " + f.replace("+", "")).format(np.mean(x), np.std(x, ddof=1) if len(x) > 1 else 0.0)


def ci(x):
    return "[" + ", ".join(f"{a[0]:+.3f},{a[1]:+.3f}" for a in x) + "]" if x else ""


def main(full):
    L = ["# A1 round 2 summary", "", f"Source: `{full}`. Mean ± seed SD (ddof=1) over the seeds present; "
         "bracket = per-seed bootstrap 95% CI of the delta (n-boot 1000). T1 (charge-wtd) and T2 are the eval_perturb "
         "in-run scores; T1 (evt-median) is the paper definition, from the npz files (saved for nominal and w_meas3d only).", ""]
    runs = {}
    for m in MODELS:
        for s in SEEDS:
            d = os.path.join(full, tag(m, s))
            mp = os.path.join(d, "metrics.json")
            if not os.path.exists(mp):
                errors.append(f"missing metrics.json: {tag(m, s)}")
                continue
            r = dict(m=json.load(open(mp))["results"])
            try:
                r["npz"] = npz_stats(d)
            except Exception:
                errors.append(f"npz postprocessing failed for {tag(m, s)}:\n{traceback.format_exc()}")
                r["npz"] = {}
            runs[m, s] = r
    for m, name in MODELS.items():
        rs = [runs[m, s] for s in SEEDS if (m, s) in runs]
        if not rs:
            continue
        L += [f"## {name} ({len(rs)} seeds)", "", "### Main table (delta vs nominal; nominal row = absolute score)", "",
              "| variant | T1 charge-wtd | per-seed 95% CI of delta | T2 | per-seed 95% CI of delta | T1 evt-median |",
              "|---|---|---|---|---|---|"]
        for v in VARS:
            h = [r["m"]["hit"][v] for r in rs if v in r["m"]["hit"]]
            e = [r["m"]["evt_nu"][v] for r in rs if v in r["m"]["evt_nu"]]
            if v == "nominal":
                t1s, t2s, c1, c2 = ms([x["score"] for x in h], "{:.4f}"), ms([x["score"] for x in e], "{:.4f}"), "", ""
            else:
                t1s, t2s = ms([x["delta"] for x in h]), ms([x["delta"] for x in e])
                c1, c2 = ci([x["delta_ci95"] for x in h]), ci([x["delta_ci95"] for x in e])
            if v in NPZ_VARS:
                ev = [r["npz"][v]["all"]["evt_med"] for r in rs if v in r["npz"]]
                if v == "nominal":
                    evs = ms(ev, "{:.4f}")
                else:
                    nom = [r["npz"]["nominal"]["all"]["evt_med"] - 0 for r in rs if v in r["npz"] and "nominal" in r["npz"]]
                    evs = ms([a - b for a, b in zip(ev, nom)]) if len(nom) == len(ev) else "n/a"
            else:
                evs = "n/a"
            L.append(f"| {v} | {t1s} | {c1} | {t2s} | {c2} | {evs} |")
        # additivity
        dy, duv, da = ([r["m"]["hit"][v]["delta"] for r in rs if v in r["m"]["hit"]] for v in ("w_meas3d_Y", "w_meas3d_UV", "w_meas3d"))
        if dy and len(dy) == len(duv) == len(da):
            add = [a + b for a, b in zip(dy, duv)]
            L += ["", f"Additivity (T1 charge-wtd): ΔY + ΔUV = {ms(add)} vs Δall = {ms(da)}; "
                      f"per seed ΔY+ΔUV-Δall = {', '.join(f'{a - c:+.4f}' for a, c in zip(add, da))}"]
        # per plane
        L += ["", "### Per-plane T1 (seed-averaged)", "",
              "Charge-weighted (in-run, all variants; delta vs nominal on the same plane):", "",
              "| plane | nominal | w_meas3d | Δ | w_meas3d_Y Δ | w_meas3d_UV Δ |", "|---|---|---|---|---|---|"]
        for p in PL:
            g = lambda v, k: [r["m"]["hit"][v]["by_plane"][p][k] for r in rs if v in r["m"]["hit"] and "by_plane" in r["m"]["hit"][v]]
            L.append(f"| {p} | {ms(g('nominal', 'score'), '{:.4f}')} | {ms(g('w_meas3d', 'score'), '{:.4f}')} | {ms(g('w_meas3d', 'delta'))} "
                     f"| {ms(g('w_meas3d_Y', 'delta'))} | {ms(g('w_meas3d_UV', 'delta'))} |")
        L += ["", "Per-event median F1 (npz; paper definition):", "", "| plane | nominal | w_meas3d | Δ |", "|---|---|---|---|"]
        for p in PL:
            nom = [r["npz"]["nominal"]["by_plane"][p]["evt_med"] for r in rs if "nominal" in r["npz"]]
            w3 = [r["npz"]["w_meas3d"]["by_plane"][p]["evt_med"] for r in rs if "w_meas3d" in r["npz"]]
            dd = [b - a for a, b in zip(nom, w3)] if len(nom) == len(w3) else []
            L.append(f"| {p} | {ms(nom, '{:.4f}')} | {ms(w3, '{:.4f}')} | {ms(dd)} |")
        L.append("")
    L += ["## Bracket definition", "", "w_diff: both terms use v = 0.1098 cm/us (0.8745 at full drift). w_diff_vd: data term uses "
          "v_data = 0.1076 cm/us (cm2 -> us2 conversion only), drift time t and clamp on the sim timescale (0.8823).", ""]
    if errors:
        L += ["## ERRORS / missing inputs", ""] + [f"- {e}" for e in errors] + [""]
        print("ERRORS:\n" + "\n".join(errors), file=sys.stderr)
    p = os.path.join(full, "summary.md")
    open(p + ".tmp", "w").write("\n".join(L))
    os.replace(p + ".tmp", p)
    print(f"wrote {p} ({len(runs)}/6 runs, {len(errors)} errors)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
