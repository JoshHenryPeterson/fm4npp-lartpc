"""A1 round-3 unit checks (CPU only): pytest tests/test_a1r3.py  or  python tests/test_a1r3.py  (run from the repo root, qlfm env)."""
import contextlib
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qlfm import eval_perturb as E  # noqa: E402


@contextlib.contextmanager
def raises(exc):   # pytest-free replacement for pytest.raises
    try:
        yield
    except exc:
        return
    raise AssertionError(f"did not raise {exc.__name__}")


def ticks(d):
    return torch.tensor(d, dtype=torch.float32) + E.ANODE_TICK


def make_batch(seed=0, nh=400):
    """3 events: 0 and 2 have a beam flash (PE in 3.4-5.6 us, a late tail), 1 has none. Hits: ~half simulated nu hits."""
    rng = np.random.default_rng(seed)
    oph = {0: [(4.05, 300.0), (4.4, 50.0), (6.0, 30.0), (13.0, 40.0)], 1: [(2.0, 100.0), (8.0, 100.0)],
           2: [(5.2, 900.0), (4.0, 20.0), (6.5, 60.0), (14.0, 80.0)]}
    ev, t, pe = [], [], []
    for i, o in oph.items():
        for tt, p in o:
            ev.append(i); t.append(tt); pe.append(p)
    edep = rng.uniform(0, 1, nh).astype(np.float32) * (rng.uniform(size=nh) < 0.5)
    return dict(n_events=3, oph_evt=torch.tensor(ev), oph_ch=torch.zeros(len(ev), dtype=torch.long),
                oph_t=torch.tensor(t, dtype=torch.float32), oph_pe=torch.tensor(pe, dtype=torch.float32),
                gidx=torch.arange(3), hit_evt=torch.from_numpy(np.sort(rng.integers(0, 3, nh))),
                hit_plane=torch.from_numpy(rng.integers(0, 3, nh)),
                hit_time=torch.from_numpy(rng.uniform(0, 4800, nh).astype(np.float32)),
                hit_rms=torch.from_numpy(rng.uniform(2, 6, nh).astype(np.float32)),
                hit_q=torch.from_numpy(rng.uniform(1, 50, nh).astype(np.float32)), hit_nu_edep=torch.from_numpy(edep))


def test_novnorm_table():
    dc = ticks(E.D_C)
    for p in (0, 1, 2):
        pl = torch.full((6,), p)
        f_old, f_new = E.width_factor(pl, dc, "meas3d"), E.width_factor(pl, dc, "meas3d_noVnorm")
        assert torch.allclose(f_new, torch.tensor(E.F_MEAS_3D_NOVNORM[p]), atol=1e-6, rtol=0)
        if p == 1:
            assert abs(float(f_new[0]) - 0.900) < 1e-6 and abs(float(f_old[0]) - 1.000) < 1e-6
        else:
            assert torch.equal(f_old, f_new)                              # U and Y identical between the tables
    assert E.TABLES["meas3d_noVnorm"] is E.F_MEAS_3D_NOVNORM and E.TABLES["meas3d"] is E.F_MEAS_3D


def test_kind_resolution():
    b = make_batch()
    pl, tk = b["hit_plane"], b["hit_time"]
    full, nov = E.width_factor(pl, tk, "meas3d"), E.width_factor(pl, tk, "meas3d_noVnorm")
    assert not torch.equal(full, nov)                                      # not resolved to meas3d
    for kind, tab in (("meas3d_noVnorm@1", "meas3d_noVnorm"), ("meas3d@1", "meas3d")):
        f = E.width_factor(pl, tk, kind)
        ref = nov if tab == "meas3d_noVnorm" else full
        assert torch.equal(f[pl == 1], ref[pl == 1]) and bool((f[pl != 1] == 1).all()), kind
    assert not torch.equal(E.width_factor(pl, tk, "meas3d_noVnorm@1"), E.width_factor(pl, tk, "meas3d@1"))
    # perturb(): V hits of simulated nu change by the factor, non-nu hits untouched, U and Y same as meas3d
    r1, r0 = E.perturb(b, "w_meas3d_noVnorm", 0.0)["hit_rms"], E.perturb(b, "w_meas3d", 0.0)["hit_rms"]
    sim = b["hit_nu_edep"] > 0
    assert bool((r1[~sim] == b["hit_rms"][~sim]).all())
    assert torch.equal(r1[pl != 1], r0[pl != 1]) and not torch.equal(r1[pl == 1], r0[pl == 1])
    assert torch.allclose(r1[sim], b["hit_rms"][sim] * nov[sim])
    for v in ("w_meas3d_V", "w_meas3d_noVnorm_V"):
        r = E.perturb(b, v, 0.0)["hit_rms"]
        assert bool((r[pl != 1] == b["hit_rms"][pl != 1]).all()), v


def test_variant_registry():
    assert E.DOMAINS == {"sim": "nominal", "real": "d_real", "real_noVnorm": "d_real_noVnorm"}
    assert E.VARIANTS["d_real"] == ("meas3d", 0.0, ("scale80", "slow", "tail"))
    assert E.VARIANTS["d_real_noVnorm"] == ("meas3d_noVnorm", 0.0, ("scale80", "slow", "tail"))
    assert E.VARIANTS["l_comb80"][2] == ("scale80", "slow", "tail")
    for v in ("w_meas3d_noVnorm", "w_meas3d_V", "w_meas3d_noVnorm_V"):
        assert v in E.VARIANTS and not E.VARIANTS[v][2]
    # `--variants all` keeps its round-2 meaning: no round-3 variant, no light variant
    assert not set(E.WIDTH_VARIANTS) & set(E.ROUND3_VARIANTS) and not set(E.WIDTH_VARIANTS) & set(E.LIGHT_VARIANTS)
    assert "w_meas3d" in E.WIDTH_VARIANTS and "d_real" not in E.LIGHT_VARIANTS and "d_real_noVnorm" not in E.LIGHT_VARIANTS
    assert E.LIGHT_VARIANTS == ["l_scale80", "l_scale82", "l_slow", "l_tail", "l_comb80", "l_comb82", "l_prompt"]
    b = make_batch()
    assert all(E.perturb(b, v, 0.0) is not None for v in E.VARIANTS)


def test_d_real_perturb():
    b = make_batch()
    sim = b["hit_nu_edep"] > 0
    o = E.perturb(b, "d_real", 0.0)
    assert bool((o["hit_rms"][~sim] == b["hit_rms"][~sim]).all())          # width only on nu hits
    assert bool((o["hit_rms"][sim] != b["hit_rms"][sim]).any())
    assert torch.allclose(o["hit_rms"], E.perturb(b, "w_meas3d", 0.0)["hit_rms"])
    ev = b["oph_evt"]
    assert bool((o["oph_pe"][ev == 1] == b["oph_pe"][ev == 1]).all())      # no-flash event untouched
    assert bool((o["oph_pe"][ev != 1] != b["oph_pe"][ev != 1]).any())      # flash events rescaled
    assert torch.allclose(o["oph_pe"], E.perturb(b, "l_comb80", 0.0)["oph_pe"])
    assert torch.equal(o["hit_q"], b["hit_q"]) and torch.equal(o["oph_t"], b["oph_t"])


def test_nominal_identity():
    b = make_batch()
    o = E.perturb(b, "nominal", 0.0)
    assert o is b or all(torch.equal(o[k], b[k]) for k in b if torch.is_tensor(b[k]))
    assert E.DOMAINS["sim"] == "nominal"


class _Tiny(nn.Module):
    d, hit_dim = 8, 4

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.seen = None

    def forward(self, b):
        self.seen = b
        return dict(z=torch.zeros(1, self.d), evt=torch.zeros(1, dtype=torch.long))


def test_multitask_domain_forward():
    from qlfm.train_multitask import MultiTaskModel
    b = make_batch()
    m = MultiTaskModel(_Tiny(), [], "frozen")
    m(b)
    assert m.backbone.seen is b or torch.equal(m.backbone.seen["hit_rms"], b["hit_rms"])    # default: sim, no perturbation
    m.domain = "real"
    m(b)
    s = m.backbone.seen
    assert torch.allclose(s["hit_rms"], E.perturb(b, "d_real", 0.0)["hit_rms"])
    assert torch.allclose(s["oph_pe"], E.perturb(b, "d_real", 0.0)["oph_pe"])
    assert not torch.equal(b["oph_pe"], s["oph_pe"])                       # the input batch itself is not modified in place


def test_width_and_domain_exclusive():
    from qlfm.train_multitask import get_args
    base = ["--mode", "frozen", "--fm", "none", "--out", "/tmp/x_unused"]
    assert get_args(base + ["--domain", "real"]).domain == "real"
    assert get_args(base + ["--width", "meas"]).width == "meas"
    assert get_args(base).domain == "sim"
    with raises(SystemExit):
        get_args(base + ["--width", "meas", "--domain", "real"])


def test_absolute_domain_load_run(monkeypatch):
    from qlfm.train_multitask import MultiTaskModel
    monkeypatch.setattr(E, "build_backbone", lambda name, ns, ckpt=None: _Tiny())
    with tempfile.TemporaryDirectory() as d:
        json.dump(dict(tasks=[], mode="frozen", fm="none", backbone="qlfm", width="native", domain="real"),
                  open(os.path.join(d, "metrics.json"), "w"))
        torch.save({}, os.path.join(d, "best.pt"))
        _, ns, model, mt = E.load_run(d, "cpu")
        assert mt and isinstance(model, MultiTaskModel) and ns.domain == "real"
        assert model.domain == "real" and model.width == "native"           # trained domain kept by default
        _, _, model, _ = E.load_run(d, "cpu", absolute_domain=True)
        assert model.domain == "sim" and model.width == "native"
        json.dump(dict(tasks=[], mode="frozen", fm="none", backbone="qlfm", width="meas"), open(os.path.join(d, "metrics.json"), "w"))
        _, _, model, _ = E.load_run(d, "cpu")
        assert model.width == "meas" and model.domain == "sim"             # old runs without a domain field
        _, _, model, _ = E.load_run(d, "cpu", absolute_domain=True)
        assert model.width == "native" and model.domain == "sim"


def test_check_disjoint():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from a1r3_make_splits import check_disjoint
    run, sub = np.array([1, 1, 2, 2, 3, 3]), np.array([0, 0, 0, 1, 0, 1])
    check_disjoint({"pre_r3": np.array([0, 1]), "lab_f0.01": np.array([2]), "val": np.array([4]), "test": np.array([5])}, run, sub)
    with raises(AssertionError):                                     # index overlap
        check_disjoint({"pre_r3": np.array([0, 1]), "val": np.array([1])}, run, sub)
    with raises(AssertionError):                                     # same (run, subrun) across partitions
        check_disjoint({"pre_r3": np.array([0]), "test": np.array([1])}, run, sub)


def main():
    class MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)
    ok = True
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(MP()) if "monkeypatch" in fn.__code__.co_varnames[:fn.__code__.co_argcount] else fn()
                print(f"PASS  {name}")
            except Exception as ex:
                ok = False
                print(f"FAIL  {name}: {ex!r}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
