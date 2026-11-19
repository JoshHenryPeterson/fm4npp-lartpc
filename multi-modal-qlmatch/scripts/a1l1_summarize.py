"""A1 light check round 1 postprocessing: python a1l1_summarize.py FULL_DIR  ->  FULL_DIR/summary.md
Everything comes from the per-run metrics.json files (T1 both definitions, T2 all/bright/P_b classes, T5 target test,
light diagnostics). Missing runs are reported in the summary, not fatal."""
import json
import os
import sys

import numpy as np

MODELS = {"qlfm_M_CL": "qlfm-M CL", "qlfm_M_C_xf": "qlfm-M C + XF"}
SEEDS = [0, 1, 2]
LV = ["l_scale80", "l_scale82", "l_slow", "l_tail", "l_comb80", "l_comb82", "l_prompt"]
TV = ["t_scale80", "t_scale82", "t_slow", "t_tail", "t_comb80", "t_comb82"]
GROUPS = ["nu", "out_av", "nonu_flash", "nonu_dark"]
PB_VARS = ["nominal", "l_slow", "l_tail", "l_comb80", "l_comb82", "l_prompt"]
WIN = ["[-0.25,0.5)", "[0.5,3)", "[3,8)", ">=8"]
errors = []


def tag(m, s):
    return f"s3_mt_{m}_finetune_f0.01_s{s}"


def ms(x, f="{:+.4f}"):
    x = [a for a in x if a is not None and np.isfinite(a)]
    if not x:
        return "n/a"
    return (f + " ± " + f.replace("+", "")).format(np.mean(x), np.std(x, ddof=1) if len(x) > 1 else 0.0)


def ci(x, f="{:+.3f}"):
    return "; ".join("[" + f.format(a[0]) + ", " + f.format(a[1]) + "]" for a in x) if x else ""


def main(full):
    L = ["# A1 light check round 1 summary", "",
         f"Source: `{full}`. Mean ± seed SD (ddof=1) over the seeds present; per-seed brackets are bootstrap 95% CIs "
         "(n-boot 1000, paired with nominal for deltas). Tasks: T1 (hit) both definitions, T2 (evt_nu) all-event and "
         "bright-subset (native 3-6 us PE >= 200), T5 target test. Hit width native, no charge change.", ""]
    runs = {}
    for m in MODELS:
        for s in SEEDS:
            mp = os.path.join(full, tag(m, s), "metrics.json")
            if os.path.exists(mp):
                runs[m, s] = json.load(open(mp))
            else:
                errors.append(f"missing metrics.json: {tag(m, s)}")
    # ---------------- 1. main table
    L += ["## 1. Main table: input-light variants (delta vs nominal; nominal row = absolute score)", ""]
    for m, name in MODELS.items():
        rs = [runs[m, s] for s in SEEDS if (m, s) in runs]
        if not rs:
            continue
        L += [f"### {name} ({len(rs)} seeds)", "",
              "| variant | T1 charge-wtd | T1 evt-median | per-seed CI (evt-median) | T2 all | T2 bright | per-seed CI (bright) |",
              "|---|---|---|---|---|---|---|"]
        for v in ["nominal"] + LV:
            h = [r["results"]["hit"][v] for r in rs if v in r["results"]["hit"]]
            e = [r["results"]["evt_nu"][v] for r in rs if v in r["results"]["evt_nu"]]
            if v == "nominal":
                row = [ms([x["score"] for x in h], "{:.4f}"), ms([x["evt_med"]["score"] for x in h], "{:.4f}"), "",
                       ms([x["score"] for x in e], "{:.4f}"), ms([x["bright"]["score"] for x in e], "{:.4f}"), ""]
            else:
                row = [ms([x["delta"] for x in h]), ms([x["evt_med"]["delta"] for x in h]), ci([x["evt_med"]["delta_ci95"] for x in h]),
                       ms([x["delta"] for x in e]), ms([x["bright"]["delta"] for x in e]), ci([x["bright"]["delta_ci95"] for x in e])]
            L.append(f"| {v} | " + " | ".join(row) + " |")
        L.append("")
    # ---------------- 2. T5 target table
    L += ["## 2. T5 target test", "",
          "Targets of has_flash events (outside nonu_dark) corrected with the light components; predictions fixed "
          "(nominal input, CL light-off pass kept). MSE over 32 x 17 features. gap = MSE(nu) - MSE(nonu_dark); "
          "closed = (gap_nominal - gap) / gap_nominal. Mean ± seed SD, per-seed paired-bootstrap 95% CIs in brackets. "
          "Bias = mean over PMTs of (pred - target) of the log1p per-PMT total.", ""]
    for m, name in MODELS.items():
        rs = [runs[m, s] for s in SEEDS if (m, s) in runs and runs[m, s].get("t5")]
        if not rs:
            continue
        t5 = [r["t5"] for r in rs]
        L += [f"### {name} ({len(rs)} seeds)", "",
              "nonu_dark definition per seed: " + ", ".join(t["dark_definition"]["definition"] for t in t5)
              + ". Group sizes (seed 0): " + str(t5[0]["group_counts"]) + ".", ""]
        nd = [t["variants"]["nominal"]["groups"]["nonu_dark"] for t in t5]
        L += [f"**MSE(nonu_dark)** (same under every target variant; identical check: "
              f"{', '.join(str(t['nonu_dark']['identical']) for t in t5)}): {ms([x['mse']['score'] for x in nd], '{:.4f}')}; "
              f"per seed {', '.join('%.4f' % x['mse']['score'] for x in nd)} CI {ci([x['mse']['ci95'] for x in nd], '{:.4f}')}; "
              f"n = {[x['n'] for x in nd]}; per-PMT-total MSE {ms([x['mse_tot']['score'] for x in nd], '{:.4f}')}; "
              f"signed total bias {ms([x['bias_tot']['score'] for x in nd])}.", ""]
        L += ["| variant | " + " | ".join(f"MSE {g}" for g in GROUPS[:3]) + " | gap nu-dark | change | per-seed CI of change | closed | "
              "gap out_av-dark change | gap nonu_flash-dark change | bias nu |", "|" + "---|" * 11]
        for v in ["nominal"] + TV:
            gs = [t["variants"][v] for t in t5]
            cells = [ms([x["groups"][g]["mse"]["score"] for x in gs], "{:.4f}") + f" (n={gs[0]['groups'][g]['n']})" for g in GROUPS[:3]]
            gp = lambda g, k: [x["gap"][g][k] for x in gs]
            cells += [ms(gp("nu", "gap"), "{:.4f}"), ms(gp("nu", "change")), ci(gp("nu", "change_ci95")),
                      ms(gp("nu", "frac_closed"), "{:.3f}"), ms(gp("out_av", "change")), ms(gp("nonu_flash", "change")),
                      ms([x["groups"]["nu"]["bias_tot"]["score"] for x in gs])]
            L.append(f"| {v} | " + " | ".join(cells) + " |")
        L += ["", "Per-PMT-total MSE (nu / nonu_dark, nominal): "
              + ms([t["variants"]["nominal"]["groups"]["nu"]["mse_tot"]["score"] for t in t5], "{:.4f}") + " / "
              + ms([x["mse_tot"]["score"] for x in nd], "{:.4f}"), ""]
        c0 = [t["class0_pb"] for t in t5]
        L += ["Class-0 P_b distribution (seed 0): frac with flash " + f"{c0[0]['frac_flash']:.3f}" + ", quantiles "
              + str({k: round(v, 1) for k, v in c0[0]["quantiles"].items()}) + ", fraction below 1/5/20/50/100/200 PE: "
              + str({k: round(v, 3) for k, v in c0[0]["frac_below"].items()}) + " (open point 6: nonu_dark definition).", ""]
    # ---------------- 3. T2 by P_b
    L += ["## 3. T2 by native P_b class", "", "AUC (nominal) / delta (variants), mean ± seed SD.", ""]
    for m, name in MODELS.items():
        rs = [runs[m, s] for s in SEEDS if (m, s) in runs]
        if not rs:
            continue
        L += [f"### {name}", "", "| variant | P_b < 600 | P_b >= 600 |", "|---|---|---|"]
        for v in PB_VARS:
            row = []
            for k in ("pb_lt600", "pb_ge600"):
                xs = [r["results"]["evt_nu"][v][k] for r in rs if v in r["results"]["evt_nu"]]
                row.append(ms([x["score"] for x in xs], "{:.4f}") if v == "nominal" else ms([x["delta"] for x in xs]) + "  " + ci([x["delta_ci95"] for x in xs]))
            L.append(f"| {v} | " + " | ".join(row) + " |")
        L.append("")
    # ---------------- 4. light diagnostics
    L += ["## 4. Light diagnostics", ""]
    rs = [runs[k] for k in sorted(runs)]
    if rs:
        d = rs[0]["light_diag"]
        L += [f"Same test events for all runs of a model. Events: {d['n_events']}; flash found in 3.4-5.6 us: {d['frac_flash']:.4f}; "
              f"t_b quantiles {d['tb_quantiles']}; P_b quantiles {d['pb_quantiles']}; P_b >= 600: {d['frac_pb_ge600']:.4f} of events "
              f"({d['frac_pb_ge600_of_flash']:.4f} of flash events); bright (3-6 us >= 200 PE): {d['frac_bright_pe_win']:.4f}.", "",
              f"t_b histogram (22 x 0.1 us from 3.4 us): {d['tb_hist']}", "",
              "Summed PE after/before per dt window (flash events; ratio), seed-0 run of the first model:", "",
              "| variant | class | " + " | ".join(WIN) + " |", "|---|---|" + "---|" * 4]
        for v in LV:
            for cls in ("dim", "bright"):
                x = d["pe_by_variant"].get(v, {}).get(cls)
                if x:
                    L.append(f"| {v} | {cls} (P_b {'<' if cls == 'dim' else '>='} 600) | "
                             + " | ".join(f"{a / b:.4f}" if b > 0 else "n/a" for a, b in zip(x["after"], x["before"])) + " |")
        L.append("")
    # ---------------- 5. write-up scaffold
    L += ["## 5. Write-up", "",
          "Answers to be filled from the tables above: (1) total scale 0.80/0.82 vs seed spread and between the two; "
          "(2) l_slow vs l_tail, and l_comb vs the sum of parts; (3) l_prompt (does the model use the time profile beyond "
          "the prompt light); (4) CL vs C + XF; (5) T5 target: does the nu - nonu_dark gap shrink, under which correction, "
          "does the nu bias move toward 0 (lower bound on the light-mismodelling share).", ""]
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
