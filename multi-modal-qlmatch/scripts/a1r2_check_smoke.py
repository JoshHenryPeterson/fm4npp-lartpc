"""A1 round-2 smoke acceptance (real data): python a1r2_check_smoke.py SMOKE_RUN_OUT --ref ROUND1_METRICS.json
Exit code 0 only if every hard check passes (WARN lines are informational)."""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEW = ["w_meas", "w_meas3d", "w_meas3d_Y", "w_meas3d_UV", "w_diff", "w_diff_vd"]
NOM_TOL, DELTA_TOL, T1_NPZ_TOL = 1e-3, 0.01, 2e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ref", required=True)
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
    m, r1 = json.load(open(mp)), json.load(open(args.ref))
    h, h1 = m["results"]["hit"], r1["results"]["hit"]
    e, e1 = m["results"]["evt_nu"], r1["results"]["evt_nu"]
    for t, a, b in (("hit", h, h1), ("evt_nu", e, e1)):
        d = abs(a["nominal"]["score"] - b["nominal"]["score"])
        check(f"{t} nominal == round 1", d < NOM_TOL, f"({a['nominal']['score']:.5f} vs {b['nominal']['score']:.5f}, |d|={d:.1e}, tol {NOM_TOL})")
    for t, a, b in (("hit", h, h1), ("evt_nu", e, e1)):
        d = abs(a["w_meas"]["delta"] - b["w_meas"]["delta"])
        check(f"{t} w_meas delta == round 1", d < DELTA_TOL, f"({a['w_meas']['delta']:+.4f} vs {b['w_meas']['delta']:+.4f}, tol {DELTA_TOL})")
    check("w_diff delta == round 1", abs(h["w_diff"]["delta"] - h1["w_diff"]["delta"]) < DELTA_TOL,
          f"({h['w_diff']['delta']:+.4f} vs {h1['w_diff']['delta']:+.4f})")
    for v in NEW:
        check(f"variant {v} present, finite", v in h and all(math.isfinite(h[v][k]) for k in ("score", "delta")))
        bp = h.get(v, {}).get("by_plane", {})
        check(f"{v} by_plane U,V,Y", sorted(bp) == ["U", "V", "Y"] and all(math.isfinite(bp[p]["score"]) for p in bp))
        print(f"      {v:12s} T1 {h[v]['score']:.4f} d={h[v]['delta']:+.4f}  T2 d={e[v]['delta']:+.4f}  by plane d: "
              + " ".join(f"{p}={bp[p]['delta']:+.4f}" for p in sorted(bp)), flush=True)
    # sanity heuristics (informational)
    check("w_meas3d drop >= w_meas drop (3D factors are smaller)", h["w_meas3d"]["delta"] <= h["w_meas"]["delta"] + 0.005, hard=False)
    check("w_diff_vd drop <= w_diff drop (0.882 vs 0.875 at the cathode)", h["w_diff_vd"]["delta"] >= h["w_diff"]["delta"] - 0.005, hard=False)
    nb = m.get("n_boot")
    check("smoke n-boot 200", nb == 200, f"(n_boot={nb})")
    # npz files
    from qlfm import paper_metrics as PM
    sizes = {}
    for v in ("nominal", "w_meas3d"):
        p = os.path.join(args.out, f"preds_{v}.npz")
        ok = os.path.exists(p)
        check(f"preds_{v}.npz exists", ok)
        if not ok:
            continue
        sizes[v] = os.path.getsize(p)
        print(f"      {p}: {sizes[v] / 1e9:.2f} GB", flush=True)
        z = np.load(p)
        check(f"{v} columns", set(z.files) >= {"pred", "hit_tgt", "hit_q", "hit_evt", "gidx", "hit_plane", "hit_sim"}, str(z.files))
        pl = np.asarray(z["hit_plane"])
        u = set(np.unique(pl).tolist())
        check(f"{v} hit_plane in {{0,1,2}}, all present", u == {0, 1, 2}, f"({sorted(u)}, n_hits={len(pl)})")
        sim = np.asarray(z["hit_sim"])
        check(f"{v} hit_sim fraction plausible", 0.0 < sim.mean() < 0.5, f"({sim.mean():.4f})", hard=False)
        del pl, sim
        g = PM.t1(z)
        bpn = PM.t1_by_plane(z)
        print(f"      {v} t1: evt-median {g['f1_evt_med']:.4f}  f1_q {g['f1_q']:.4f}; by plane evt-median "
              + " ".join(f"{p}={bpn[p]['f1_evt_med']:.4f}" for p in "UVY"), flush=True)
        check(f"{v} npz T1 f1_q == in-run score", abs(g["f1_q"] - h[v]["score"]) < T1_NPZ_TOL, f"({g['f1_q']:.5f} vs {h[v]['score']:.5f})")
        for p in "UVY":
            check(f"{v} npz plane {p} f1_q == in-run by_plane", abs(bpn[p]["f1_q"] - h[v]["by_plane"][p]["score"]) < T1_NPZ_TOL,
                  f"({bpn[p]['f1_q']:.5f} vs {h[v]['by_plane'][p]['score']:.5f})")
    stray = glob.glob(os.path.join(args.out, "*.tmp*"))
    check("no stray tmp files", not stray, str(stray))
    if sizes:
        print(f"      projected npz total for 6 runs: {6 * sum(sizes.values()) / 1e9:.1f} GB", flush=True)
    print("SMOKE ACCEPTANCE " + ("FAILED: " + "; ".join(fails) if fails else "PASSED"), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
