"""A1 light check round 1 unit checks (CPU only): python tests/test_a1_light.py  (qlfm env)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qlfm import eval_perturb as E  # noqa: E402
from qlfm.tokenize import TokCfg, tokenize_light  # noqa: E402
from qlfm.train_downstream import light_target  # noqa: E402


def make_batch(events, rng=None):
    """events: list of lists of (t, pe[, ch]) ophits. Hits get random tensors so side effects are visible."""
    rng = rng or np.random.default_rng(0)
    ev, ch, t, pe = [], [], [], []
    for i, e in enumerate(events):
        for o in e:
            ev.append(i); t.append(o[0]); pe.append(o[1]); ch.append(o[2] if len(o) > 2 else int(rng.integers(0, 32)))
    nh = 50
    return dict(n_events=len(events), oph_evt=torch.tensor(ev), oph_ch=torch.tensor(ch), oph_t=torch.tensor(t, dtype=torch.float32),
                oph_pe=torch.tensor(pe, dtype=torch.float32), gidx=torch.arange(len(events)),
                hit_plane=torch.from_numpy(rng.integers(0, 3, nh)), hit_time=torch.from_numpy(rng.uniform(0, 4800, nh).astype(np.float32)),
                hit_rms=torch.from_numpy(rng.uniform(2, 6, nh).astype(np.float32)), hit_q=torch.from_numpy(rng.uniform(1, 50, nh).astype(np.float32)),
                hit_nu_edep=torch.from_numpy(rng.uniform(0, 1, nh).astype(np.float32)))


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name} {info}")

    # ---- t_b, P_b ----
    # event 0: peak 300 PE at 4.05 (bin [4.0, 4.1)); bright light outside the window (2.0, 7.0) must be ignored;
    #          P_b window [3.75, 4.5): 3.7 out, 3.8 in, 4.4 in, 4.6 out
    ev0 = [(4.05, 300.0), (2.0, 5000.0), (7.0, 5000.0), (3.7, 7.0), (3.8, 11.0), (4.4, 13.0), (4.6, 17.0), (5.0, 40.0)]
    ev1 = [(2.0, 100.0), (8.0, 100.0)]                                         # no PE in 3.4-5.6: no flash
    ev2 = [(5.55, 900.0), (3.45, 10.0), (5.2, 700.0, 3)]                       # peak is the last bin (5.5-5.6)
    b = make_batch([ev0, ev1, ev2])
    tb, pb, has = E.beam_flash(b)
    check("t_b = lower edge of the argmax bin", abs(float(tb[0]) - 4.0) < 1e-5 and abs(float(tb[2]) - 5.5) < 1e-5, f"({tb.tolist()})")
    check("t_b ignores bright light outside 3.4-5.6", abs(float(tb[0]) - 4.0) < 1e-5)
    check("P_b sums exactly [t_b-0.25, t_b+0.5)", abs(float(pb[0]) - (300 + 11 + 13)) < 1e-3, f"({float(pb[0])})")
    check("has_flash", has.tolist() == [True, False, True])
    check("no-flash event has P_b = 0", float(pb[1]) == 0)
    check("peak in last bin: P_b window", abs(float(pb[2]) - 900.0) < 1e-3, f"({float(pb[2])})")
    pw = E.pe_win_native(b)
    check("pe_win = PE in [3, 6)", abs(float(pw[0]) - (300 + 7 + 11 + 13 + 17 + 40)) < 1e-3 and abs(float(pw[1])) < 1e-6, f"({pw.tolist()})")

    # ---- factors ----
    F = E.light_factor_dt
    dt = torch.tensor([-0.26, -0.25, 0.0, 2.99, 3.0, 8.0])
    for k, f in (("scale80", 0.80), ("scale82", 0.82)):
        r = F(dt, torch.full_like(dt, 10.0), (k,)).tolist()
        check(f"{k} window edges", np.allclose(r, [1, f, f, f, 1, 1], atol=1e-6), str(r))
    edges = torch.tensor([0.49, 0.5, 0.99, 1.0, 1.49, 1.5, 1.99, 2.0, 2.99, 3.0, 5.0])
    exp_dim = [1, .83, .83, .65, .65, .61, .61, .64, .64, 1, 1]
    exp_br = [1, .82, .82, .63, .63, .48, .48, .45, .45, 1, 1]
    for pbv, exp, nm in ((599.9, exp_dim, "dim"), (600.0, exp_br, "bright"), (2000.0, exp_br, "bright")):
        r = F(edges, torch.full_like(edges, pbv), ("slow",)).tolist()
        check(f"slow factors, P_b={pbv} ({nm})", np.allclose(r, exp, atol=1e-6), str(r))
    d2 = torch.tensor([7.99, 8.0, 15.0])
    check("tail only for P_b >= 600", F(d2, torch.full_like(d2, 600.0), ("tail",)).tolist() == [1.0, np.float32(0.37), np.float32(0.37)]
          and F(d2, torch.full_like(d2, 599.9), ("tail",)).tolist() == [1, 1, 1])
    d3 = torch.tensor([-1.0, 0.49, 0.5, 3.0, 10.0])
    check("prompt zeroes dt >= 0.5", F(d3, torch.full_like(d3, 10.0), ("prompt",)).tolist() == [1, 1, 0, 0, 0])
    grid = torch.linspace(-2, 20, 2201)
    for pbv in (50.0, 1000.0):
        p = torch.full_like(grid, pbv)
        for comb, parts in (("l_comb80", ("scale80", "slow", "tail")), ("l_comb82", ("scale82", "slow", "tail"))):
            prod = torch.ones_like(grid)
            for q in parts:
                prod = prod * F(grid, p, (q,))
            check(f"{comb} == product of parts (P_b={pbv})", torch.allclose(F(grid, p, E.VARIANTS[comb][2]), prod))
    check("combined variants have the documented components", E.VARIANTS["l_comb80"][2] == ("scale80", "slow", "tail")
          and E.VARIANTS["l_comb82"][2] == ("scale82", "slow", "tail"))
    check("7 light variants", len(E.LIGHT_VARIANTS) == 7 and all(E.VARIANTS[v][:2] == (None, 0.0) for v in E.LIGHT_VARIANTS))
    check("existing variants have empty light", all(E.VARIANTS[v][2] == () for v in E.WIDTH_VARIANTS) and "nominal" in E.WIDTH_VARIANTS)

    # ---- perturb(): flash-less events untouched, l_prompt, no side effects ----
    pe0 = b["oph_pe"].clone()
    for v in E.LIGHT_VARIANTS:
        pb_ = E.perturb(b, v, 0.0)
        check(f"{v}: events without a flash untouched", bool((pb_["oph_pe"][b["oph_evt"] == 1] == pe0[b["oph_evt"] == 1]).all()))
        check(f"{v}: hit_* tensors bitwise unchanged", all(torch.equal(pb_[k], b[k]) for k in b if k.startswith("hit_")))
        check(f"{v}: input batch not modified in place", torch.equal(b["oph_pe"], pe0))
        check(f"{v}: oph_t/ch/evt unchanged", all(torch.equal(pb_[k], b[k]) for k in ("oph_t", "oph_ch", "oph_evt")))
    pp = E.perturb(b, "l_prompt", 0.0)
    dt_all = b["oph_t"] - tb[b["oph_evt"]]
    fl = has[b["oph_evt"]]
    check("l_prompt: PE > 0 only where dt < 0.5 (flash events)", bool(((pp["oph_pe"] > 0) & fl & (dt_all >= 0.5)).sum() == 0)
          and bool((pp["oph_pe"][fl & (dt_all < 0.5)] > 0).all()))
    ps = E.perturb(b, "l_scale80", 0.0)["oph_pe"]
    m = fl & (dt_all >= -0.25) & (dt_all < 3.0)
    check("l_scale80: exactly 0.80 in window, 1 outside", bool(torch.allclose(ps[m], 0.8 * pe0[m]) and torch.equal(ps[fl & ~m], pe0[fl & ~m])))
    pt = E.perturb(b, "l_tail", 0.0)["oph_pe"]
    check("l_tail: P_b < 600 events unchanged", torch.equal(pt[b["oph_evt"] == 0], pe0[b["oph_evt"] == 0]))
    for v in E.WIDTH_VARIANTS:
        pw_ = E.perturb(b, v, 0.0)
        check(f"{v}: oph_* unchanged", all(torch.equal(pw_[k], b[k]) for k in b if k.startswith("oph_")))
    # diagnostics windows
    f = E.light_factor(b, ("scale80",))
    d = E.pe_window_sums(b, (tb, pb, has), f)
    check("pe_window_sums: before = sums, window [-0.25,0.5) scaled by 0.8", abs(d[0, 1, 0] / d[0, 0, 0] - 0.8) < 1e-6 and d[1, 0].sum() > 0, str(d.tolist()))

    # ---- tokenization: zero-PE ophit == removed ophit ----
    cfg = TokCfg()
    b2 = dict(b)
    keep = torch.ones(len(b["oph_pe"]), dtype=torch.bool)
    keep[2] = False
    z = dict(b); z["oph_pe"] = b["oph_pe"].clone(); z["oph_pe"][2] = 0
    r = {k: (v[keep] if k.startswith("oph_") else v) for k, v in b.items()}
    a1 = tokenize_light(z["oph_ch"], z["oph_t"], z["oph_pe"], z["oph_evt"], 3, cfg).feat
    a2 = tokenize_light(r["oph_ch"], r["oph_t"], r["oph_pe"], r["oph_evt"], 3, cfg).feat
    check("zero-PE ophit tokenizes like a removed ophit", torch.equal(a1, a2))
    del b2

    # ---- T5 target test ----
    rng = np.random.default_rng(1)
    evs = []
    for i in range(12):
        if i % 4 == 3:                                   # no flash
            evs.append([(float(rng.uniform(6, 20)), float(rng.uniform(1, 5)), int(rng.integers(0, 32))) for _ in range(10)])
        else:
            pk = float(rng.uniform(4.0, 5.0))
            evs.append([(pk, 800.0 if i % 2 else 100.0), (pk + 1.0, 20.0), (pk + 9.0, 10.0), (pk + 2.5, 15.0)]
                       + [(float(rng.uniform(0, 24)), float(rng.uniform(1, 5))) for _ in range(8)])
    bb = make_batch(evs)
    flash = E.beam_flash(bb)
    nom = light_target(bb)
    for name, kinds in E.T5_TARGETS.items():
        b3 = dict(bb)
        b3["oph_pe"] = bb["oph_pe"] * E.light_factor(bb, kinds, flash)
        tg = light_target(b3)
        diff = (tg != nom).flatten(1).any(1)
        check(f"{name}: corrected target differs only for has_flash events", bool((diff <= flash[2]).all()) and bool(diff[flash[2]].any()),
              f"(changed {int(diff.sum())} of {int(flash[2].sum())} flash events)")
    b3 = dict(bb)
    b3["oph_pe"] = bb["oph_pe"] * E.light_factor(bb, (), flash)
    check("factor 1: target bitwise unchanged", torch.equal(light_target(b3), nom))
    # groups
    n = 3000
    y = rng.integers(0, 3, n)
    elig = rng.uniform(size=n) > 0.1
    has_f = rng.uniform(size=n) > 0.3
    pbv = np.where(has_f, rng.uniform(5, 800, n), 0.0)
    lab, dfn = E.t5_groups(elig, y, has_f, pbv)
    ok_g = (lab[elig & (y == 1)] == "nu").all() and (lab[elig & (y == 2)] == "out_av").all() and (lab[~elig] == "other").all() \
        and (lab[elig & (y == 0) & has_f] == "nonu_flash").all() and (lab[elig & (y == 0) & ~has_f] == "nonu_dark").all()
    check("group assignment: evt_nu class and has_flash", ok_g and dfn["definition"] == "no_flash", str(dfn))
    lab2, dfn2 = E.t5_groups(elig, y, np.ones(n, bool), pbv)             # no class-0 without flash -> fallback to P_b < 20
    check("fallback to P_b < 20 PE when < 100 no-flash class-0 events", dfn2["definition"] == "pb_cut_fallback"
          and bool((lab2[elig & (y == 0) & (pbv < 20)] == "nonu_dark").all()) and bool((lab2[elig & (y == 0) & (pbv >= 20)] == "nonu_flash").all()))
    lab3, dfn3 = E.t5_groups(elig, y, has_f, pbv, nonu_dark_pb=50.0)
    check("--nonu-dark-pb overrides", dfn3["definition"] == "pb_cut" and bool((lab3[elig & (y == 0) & (pbv < 50)] == "nonu_dark").all()))
    # paired bootstrap on synthetic per-event stats
    n = 600
    y = rng.integers(0, 3, n); elig = np.ones(n, bool); has_f = rng.uniform(size=n) > 0.3
    pbv = np.where(has_f, rng.uniform(5, 800, n), 0.0)
    pred = rng.normal(size=(n, 32, 17)).astype(np.float32)
    tgt = pred + rng.normal(scale=0.5, size=pred.shape).astype(np.float32)
    info = dict(gidx=np.arange(n), has=has_f, pb=pbv)
    res = E.t5_analysis(pred, {"nominal": tgt, "ident": tgt.copy(), "shift": np.where(has_f[:, None, None], tgt * 0.8, tgt)},
                        info, (elig, y), 200)
    g = res["variants"]
    check("identity target: gap change is exactly 0 (point and CI)", all(g["ident"]["gap"][k]["change"] == 0 and g["ident"]["gap"][k]["change_ci95"] == [0, 0]
                                                                         and g["ident"]["gap"][k]["frac_closed"] == 0 for k in ("nu", "out_av", "nonu_flash")))
    check("nonu_dark MSE identical across variants", res["nonu_dark"]["identical"], str(res["nonu_dark"]["mse_by_variant"]))
    check("shift target changes a group MSE", g["shift"]["groups"]["nu"]["mse"]["score"] != g["nominal"]["groups"]["nu"]["mse"]["score"])
    grp = res["group_counts"]
    check("group counts sum to n", sum(grp.values()) == n, str(grp))
    check("nominal group MSE matches direct computation", abs(g["nominal"]["groups"]["nu"]["mse"]["score"]
                                                              - float(((pred - tgt) ** 2).mean((1, 2))[y == 1].mean())) < 1e-6)
    # T1 median + T2 subset bootstrap run on synthetic predictions
    ne, nh = 80, 4000
    evid = np.sort(rng.integers(0, ne, nh))
    r = dict(pred=rng.uniform(size=nh).astype(np.float32), hit_tgt=rng.uniform(size=nh) < 0.4, hit_q=rng.uniform(1, 5, nh).astype(np.float32),
             hit_evt=evid + 10, gidx=np.arange(ne) + 10)
    bm = E.bootstrap_t1_median({"nominal": r, "x": {**r, "pred": 1 - r["pred"]}}, 50)
    from qlfm import paper_metrics as PM
    pm = PM.t1({**r, "pred": r["pred"]})
    check("T1 median point == paper_metrics.t1", abs(bm["nominal"]["score"] - pm["f1_evt_med"]) < 1e-9 and bm["nominal"]["delta"] == 0,
          f"({bm['nominal']['score']:.5f} vs {pm['f1_evt_med']:.5f})")
    ge = np.arange(ne) + 10
    ye = rng.integers(0, 3, ne)
    logits = rng.normal(size=(ne, 3)) + 2 * np.eye(3)[ye]
    re = dict(gidx=ge, y=ye, pred=logits)
    msk = np.arange((ye != 2).sum()) % 3 != 0
    bs = E.bootstrap_auc_subsets({"nominal": re, "x": {**re, "pred": -logits}}, 50, {"sub": msk})
    gg, yy, ss = E.evt_scores(re)
    from sklearn.metrics import roc_auc_score
    check("T2 subset AUC point == roc_auc_score on the subset", abs(bs["sub"]["nominal"]["score"] - roc_auc_score(yy[msk], ss[msk])) < 1e-12
          and bs["sub"]["nominal"]["delta"] == 0)

    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
