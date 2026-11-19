"""A1 light check round 1 smoke acceptance (real data):
python a1l1_check_smoke.py SMOKE_RUN_OUT --ref ROUND2_METRICS.json [--ref ROUND1_METRICS.json]
Exit code 0 only if every hard check passes (WARN lines are informational)."""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LV = ["l_scale80", "l_scale82", "l_slow", "l_tail", "l_comb80", "l_comb82", "l_prompt"]
TV = ["t_scale80", "t_scale82", "t_slow", "t_tail", "t_comb80", "t_comb82"]
GROUPS = ["nu", "out_av", "nonu_flash", "nonu_dark"]
NOM_TOL, PE_RTOL, MIN_GROUP = 1e-3, 1e-3, 100
SCALE = {"l_scale80": 0.80, "l_scale82": 0.82, "l_comb80": 0.80, "l_comb82": 0.82}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ref", action="append", required=True)
    args = ap.parse_args()
    fails = []

    def check(name, cond, info="", hard=True):
        print(f"{'PASS' if cond else ('FAIL' if hard else 'WARN')}  {name} {info}", flush=True)
        if not cond and hard:
            fails.append(name)

    mp = os.path.join(args.out, "metrics.json")
    check("smoke metrics.json exists", os.path.exists(mp), mp)
    if fails:
        return 1
    m = json.load(open(mp))
    h, e, ld, t5 = m["results"]["hit"], m["results"]["evt_nu"], m["light_diag"], m["t5"]
    check("n-boot 200", m.get("n_boot") == 200, f"(n_boot={m.get('n_boot')})")
    check("tasks hit,evt_nu,light", m["tasks"] == ["hit", "evt_nu", "light"], str(m["tasks"]))
    check("variants = nominal + 7 light", m["variants"] == ["nominal"] + LV, str(m["variants"]))
    # --- nominal vs reference (T1 charge-weighted, T2 all-event)
    for ref in args.ref:
        r = json.load(open(ref))
        for t, a, b in (("hit", h, r["results"]["hit"]), ("evt_nu", e, r["results"]["evt_nu"])):
            d = abs(a["nominal"]["score"] - b["nominal"]["score"])
            check(f"{t} nominal == {ref.split('/')[-4]}/{ref.split('/')[-3]}", d < NOM_TOL,
                  f"({a['nominal']['score']:.5f} vs {b['nominal']['score']:.5f}, |d|={d:.1e}, tol {NOM_TOL})")
    # nominal per-event median T1 vs the round-2 npz-derived value (cached next to the first reference)
    c = os.path.join(os.path.dirname(args.ref[0]), "t1_npz.json")
    if os.path.exists(c):
        ref_med = json.load(open(c))["nominal"]["all"]["evt_med"]
        d = abs(h["nominal"]["evt_med"]["score"] - ref_med)
        check("T1 evt-median nominal == round 2 (paper_metrics.t1)", d < NOM_TOL,
              f"({h['nominal']['evt_med']['score']:.5f} vs {ref_med:.5f}, |d|={d:.1e})")
    else:
        check("round-2 t1_npz.json available for the evt-median reference", False, c, hard=False)
    # --- finite everything
    for v in m["variants"]:
        ok = all(math.isfinite(h[v][k]) for k in ("score", "delta")) and math.isfinite(h[v]["evt_med"]["score"]) \
            and math.isfinite(e[v]["score"]) and all(math.isfinite(e[v][s]["score"]) for s in ("bright", "pb_lt600", "pb_ge600"))
        check(f"{v}: T1 (both), T2 (all, bright, P_b classes) finite", ok)
        if v != "nominal":
            print(f"      {v:10s} T1q d={h[v]['delta']:+.4f}  T1med d={h[v]['evt_med']['delta']:+.4f}  T2 d={e[v]['delta']:+.4f}  "
                  f"T2bright d={e[v]['bright']['delta']:+.4f}  T2 P_b<600 d={e[v]['pb_lt600']['delta']:+.4f}  "
                  f"P_b>=600 d={e[v]['pb_ge600']['delta']:+.4f}", flush=True)
    n = e["nominal"]
    print(f"      nominal T2 all {n['score']:.4f} bright {n['bright']['score']:.4f} (n={n['bright']['n_events']}) "
          f"P_b<600 {n['pb_lt600']['score']:.4f} P_b>=600 {n['pb_ge600']['score']:.4f} (n={n['pb_ge600']['n_events']})", flush=True)
    # --- flash finding
    check("flash found in >= 95% of events", ld["frac_flash"] >= 0.95, f"({ld['frac_flash']:.4f})")
    mode = 3.4 + 0.1 * int(np.array(ld["tb_hist"]).argmax())
    check("t_b distribution peaks in 3.6-5.2 us", 3.6 <= mode < 5.2, f"(mode bin {mode:.1f} us; quantiles {ld['tb_quantiles']})")
    check("P_b >= 600 fraction > 0 (l_tail not empty)", ld["frac_pb_ge600"] > 0,
          f"({ld['frac_pb_ge600']:.4f} of events, {ld['frac_pb_ge600_of_flash']:.4f} of flash events); P_b quantiles {ld['pb_quantiles']}")
    # --- PE diagnostics
    pe = ld["pe_by_variant"]
    check("PE diagnostics for all 7 light variants", sorted(pe) == sorted(LV))
    near = lambda a, b: abs(a - b) <= PE_RTOL * max(abs(b), 1e-12) + 1e-9
    for v in LV:
        if v not in pe:
            continue
        for cls in ("dim", "bright"):
            bf, af = np.array(pe[v][cls]["before"]), np.array(pe[v][cls]["after"])
            if bf.sum() == 0:
                check(f"{v} {cls}: class has PE", False, "(none)")
                continue
            ratio = [a / b if b > 0 else float("nan") for a, b in zip(af, bf)]
            msg = "after/before in [-0.25,0.5) [0.5,3) [3,8) >=8 us: " + " ".join(f"{x:.4f}" for x in ratio)
            tail = (lambda: near(af[3], bf[3])) if cls == "dim" else (lambda: near(af[3], 0.37 * bf[3]))
            slow = (0.45, 0.83) if cls == "bright" else (0.61, 0.83)
            if v in ("l_scale80", "l_scale82"):
                s = SCALE[v]
                ok = near(af[0], s * bf[0]) and near(af[1], s * bf[1]) and near(af[2], bf[2]) and near(af[3], bf[3])
            elif v in ("l_comb80", "l_comb82"):
                s = SCALE[v]
                ok = near(af[0], s * bf[0]) and s * slow[0] * (1 - 1e-3) <= ratio[1] <= s * slow[1] * (1 + 1e-3) \
                    and near(af[2], bf[2]) and tail()
            elif v == "l_slow":
                ok = near(af[0], bf[0]) and slow[0] * (1 - 1e-3) <= ratio[1] <= slow[1] * (1 + 1e-3) \
                    and near(af[2], bf[2]) and near(af[3], bf[3])
            elif v == "l_tail":
                ok = all(near(a, b) for a, b in zip(af[:3], bf[:3])) and tail()
            else:   # l_prompt
                ok = near(af[0], bf[0]) and all(abs(x) <= PE_RTOL * bf.sum() for x in af[1:])
            check(f"{v} {cls}: PE change by definition", ok, msg)
    # --- native event info vs the paper truth (bright-cut variable)
    ep = os.path.join(args.out, "event_info.npz")
    check("event_info.npz exists", os.path.exists(ep))
    if os.path.exists(ep):
        from qlfm import paper_metrics as PM
        z = np.load(ep)
        tr = PM.truth()
        pw = tr["pe_win"][z["gidx"]]
        agree = float(((z["pe_win"] >= 200.0) == (pw >= tr["t2_cut"])).mean())
        check("paper_metrics t2 cut is 200 PE", tr["t2_cut"] == 200.0, f"({tr['t2_cut']})")
        check("native pe_win bright flag == paper_metrics truth (>= 99.9% of events)", agree >= 0.999,
              f"(agreement {agree:.5f}; max rel diff of pe_win {float(np.max(np.abs(z['pe_win'] - pw) / np.maximum(pw, 1))):.2e})")
    # --- T5
    d = t5["nominal_mse"] - t5["stored_mse"]
    check("T5 nominal MSE == stored test MSE", abs(d) < NOM_TOL, f"({t5['nominal_mse']:.5f} vs {t5['stored_mse']:.5f}, d={d:+.1e}; light-off pass kept)")
    iv = t5["input_variants"]
    worst = max(x["max_abs_pred_diff"] for x in iv.values())
    # Inference runs under bf16 autocast, so separate passes differ by a few bf16 ulps per feature (max |pred diff| ~0.1-0.3
    # on log1p features up to ~10). Leakage of input light into T5 would move the MSE systematically (most under l_prompt);
    # bf16 noise moves it by ~1e-5. Test the MSE at 1e-4 (~200x below the T5 target-test effects); report the max diff.
    dmse = max(abs(x["mse"] - iv["nominal"]["mse"]) for x in iv.values())
    check("input-light variants leave T5 unchanged (|dMSE| < 1e-4; bf16 noise)", dmse < 1e-4,
          f"(max |dMSE| {dmse:.1e}; max |pred diff| {worst:.2e}, bf16 rounding)")
    print(f"      nonu_dark definition: {t5['dark_definition']}; class-0 P_b: {t5['class0_pb']}", flush=True)
    print(f"      group counts: {t5['group_counts']}", flush=True)
    check("all 6 target variants present", sorted(t5["variants"]) == sorted(["nominal"] + TV))
    check("each group has >= 100 events", all(t5["group_counts"][g] >= MIN_GROUP for g in GROUPS), str({g: t5["group_counts"][g] for g in GROUPS}))
    nd = t5["nonu_dark"]
    check("nonu_dark MSE identical across target variants", nd["identical"], f"(max abs diff {nd['max_abs_diff']:.1e})")
    g0 = t5["variants"]["nominal"]["groups"]
    print(f"      nominal MSE nu {g0['nu']['mse']['score']:.4f} out_av {g0['out_av']['mse']['score']:.4f} "
          f"nonu_flash {g0['nonu_flash']['mse']['score']:.4f} nonu_dark {g0['nonu_dark']['mse']['score']:.4f}; "
          f"nu total bias {g0['nu']['bias_tot']['score']:+.4f}", flush=True)
    for v in TV:
        x = t5["variants"][v]["gap"]["nu"]
        print(f"      {v:10s} gap(nu-dark) {x['gap']:+.4f} change {x['change']:+.4f} {x['change_ci95']} closed {x['frac_closed']:+.3f}  "
              f"nu bias {t5['variants'][v]['groups']['nu']['bias_tot']['score']:+.4f}", flush=True)
        check(f"{v}: finite", all(math.isfinite(t5["variants"][v]["gap"][g]["gap"]) for g in ("nu", "out_av", "nonu_flash")))
    stray = [f for f in os.listdir(args.out) if ".tmp" in f]
    check("no stray tmp files", not stray, str(stray))
    print("SMOKE ACCEPTANCE " + ("FAILED: " + "; ".join(fails) if fails else "PASSED"), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
