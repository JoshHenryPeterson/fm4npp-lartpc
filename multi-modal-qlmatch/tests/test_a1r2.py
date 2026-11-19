"""A1 round-2 unit checks (CPU only): python tests/test_a1r2.py  (run from the repo root, qlfm env)."""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qlfm import eval_perturb as E  # noqa: E402
from qlfm import paper_metrics as PM  # noqa: E402


def ticks(d):
    return torch.tensor(d, dtype=torch.float32) + E.ANODE_TICK


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name} {info}")

    dc = ticks(E.D_C)
    for kind, tab in (("meas", E.F_MEAS_2D), ("meas3d", E.F_MEAS_3D)):
        for p, fp in tab.items():
            f = E.width_factor(torch.full((6,), p), dc, kind)
            check(f"{kind} table at D_C, plane {p}", torch.allclose(f, torch.tensor(fp), atol=1e-6, rtol=0),
                  f"max|diff|={float((f - torch.tensor(fp)).abs().max()):.2e}")
    check("F_MEAS alias is the 2D table", E.F_MEAS is E.F_MEAS_2D)
    # masks
    rng = np.random.default_rng(0)
    n = 20000
    plane = torch.from_numpy(rng.integers(0, 3, n))
    tick = torch.from_numpy(rng.uniform(0, 4800, n).astype(np.float32))
    full = E.width_factor(plane, tick, "meas3d")
    fy, fuv = E.width_factor(plane, tick, "meas3d@2"), E.width_factor(plane, tick, "meas3d@0,1")
    check("@2 exactly 1 on U,V", bool((fy[plane != 2] == 1).all()))
    check("@0,1 exactly 1 on Y", bool((fuv[plane == 2] == 1).all()))
    cy, cuv, call = fy != 1, fuv != 1, full != 1
    check("Y and UV change disjoint hits", not bool((cy & cuv).any()))
    check("Y union UV == all changed", bool(((cy | cuv) == call).all()), f"({int(call.sum())} of {n} hits)")
    check("masked values equal full-table values", bool(torch.equal(fy[plane == 2], full[plane == 2])
                                                      and torch.equal(fuv[plane != 2], full[plane != 2])))
    # diffusion
    far = torch.tensor([4271.0 + 4000.0])
    pl = torch.zeros(1, dtype=torch.long)
    fd, fv = float(E.width_factor(pl, far, "diff")), float(E.width_factor(pl, far, "diff_vd"))
    check("diff at full drift = 0.8745", abs(fd - 0.8745) < 1e-4, f"({fd:.4f})")
    check("diff_vd at full drift = 0.8823", abs(fv - 0.8823) < 1e-4, f"({fv:.4f})")
    check("diff_vd == diff at the anode", abs(float(E.width_factor(pl, ticks([0.0]), "diff_vd")) - 1) < 1e-6)
    # inverse controls
    check("inv = 1/meas", torch.allclose(E.width_factor(plane, tick, "inv"), 1 / E.width_factor(plane, tick, "meas")))
    check("inv3d = 1/meas3d", torch.allclose(E.width_factor(plane, tick, "inv3d"), 1 / full))
    # perturb() paths, incl. _all and variants registry
    b = dict(hit_plane=plane, hit_time=tick, hit_rms=torch.ones(n), hit_q=torch.ones(n),
             hit_nu_edep=torch.from_numpy(rng.uniform(-1, 1, n).astype(np.float32)).clamp(min=0))
    sim = b["hit_nu_edep"] > 0
    pm = E.perturb(b, "w_meas_all", 0.0)["hit_rms"]
    check("w_meas_all changes real hits too", bool((pm[~sim] != 1).any()))
    pm = E.perturb(b, "w_meas", 0.0)["hit_rms"]
    check("w_meas leaves real hits untouched", bool((pm[~sim] == 1).all()))
    ry, ruv, ra = (E.perturb(b, v, 0.0)["hit_rms"] != 1 for v in ("w_meas3d_Y", "w_meas3d_UV", "w_meas3d"))
    check("perturb: Y/UV disjoint, union == all", not bool((ry & ruv).any()) and bool(((ry | ruv) == ra).all()))
    check("perturb: inv still runs", E.perturb(b, "w_inv", 0.0)["hit_rms"].shape == (n,))
    check("perturb: const still runs", bool((E.perturb(b, "w_const", 0.0)["hit_rms"] == E.WIDTH_CONST).all()))
    check("all variants resolve", all(E.perturb(b, v, 0.0) is not None for v in E.WIDTH_VARIANTS))
    # save_preds / t1_by_plane / bootstrap_by_plane on synthetic hits
    ne, nh = 60, 6000
    ev = np.sort(rng.integers(0, ne, nh))
    gidx = np.arange(ne) + 100
    r = dict(pred=rng.uniform(size=nh).astype(np.float32), hit_tgt=rng.uniform(size=nh) < 0.4,
             hit_q=rng.uniform(1, 5, nh).astype(np.float32), hit_evt=gidx[ev], gidx=gidx,
             hit_plane=rng.integers(0, 3, nh).astype(np.uint8), hit_sim=rng.uniform(size=nh) < 0.1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "preds_nominal.npz")
        E.save_preds(path, r)
        z = np.load(path)
        check("npz columns/dtypes", {k: z[k].dtype.name for k in z.files} == dict(
            pred="float16", hit_tgt="uint8", hit_q="float32", hit_evt="int32", gidx=r["gidx"].dtype.name,
            hit_plane="uint8", hit_sim="bool"), str({k: z[k].dtype.name for k in z.files}))
        g = PM.t1(z)
        bp = PM.t1_by_plane(z)
        check("t1 reads the npz", 0 <= g["f1_q"] <= 1)
        # charge-weighted global F1 from eval_perturb sums == t1 f1_q
        ev_ids, sums = E.hit_event_sums({**r, "pred": r["pred"].astype(np.float16).astype(np.float32)})
        check("hit_event_sums F1 == t1 f1_q", abs(float(E.f1_from_sums(sums.sum(0))) - g["f1_q"]) < 1e-6)
        bs = E.bootstrap_by_plane({"nominal": r, "x": {**r, "pred": 1 - r["pred"]}}, r["hit_plane"], 50)
        for p, nm in E.PLANE_NAMES.items():
            m = r["hit_plane"] == p
            sub = {k: r[k][m] for k in ("pred", "hit_tgt", "hit_q", "hit_evt")}
            _, s = E.hit_event_sums(sub)
            check(f"by-plane bootstrap point == hit_event_sums, plane {nm}",
                  abs(bs["nominal"][nm]["score"] - float(E.f1_from_sums(s.sum(0)))) < 1e-9
                  and abs(bs["nominal"][nm]["delta"]) == 0)
            check(f"t1_by_plane {nm} f1_q consistent", abs(bp[nm]["f1_q"] - bs["nominal"][nm]["score"]) < 2e-3,
                  f"({bp[nm]['f1_q']:.4f} vs {bs['nominal'][nm]['score']:.4f}, fp16 rounding)")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
