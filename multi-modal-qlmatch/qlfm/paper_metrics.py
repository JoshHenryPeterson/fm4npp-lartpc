"""Paper metrics (issue #8, E1.1), computed from saved test predictions (preds_test.npz), so no retraining.

  T1  per-event neutrino-hit F1 from hit counts (unweighted), median over test events ("f1_evt_med");
      the charge-weighted F1 over all hits (round 1-4 headline) is kept as "f1_q"
  T2  AUC in-AV nu vs no-nu on the bright-flash sample: beam-window PE (scripts/r4_light_stats.py, 3-6 us) >= cut.
      The cut maximises the light-only balanced accuracy on the VALIDATION split (audit #4; round 4 chose it on
      test), snapped to the round-4 grid. All-event AUC kept as "auc_all"
  T3  macro-F1 (unchanged)
  T4  resolution on the truth-clean subset: nu share of the event's collection-plane charge >= 5% and visible
      fraction (nu MC deposit on collection-plane hits / E_nu) >= 0.3, a containment proxy. All-eligible
      resolution kept as "res_all"
  T5  test MSE of the log1p PMT features (unchanged)

python -m qlfm.paper_metrics truth           # one-off: $QLFM_WORK/paper/truth.npz (val light cut + T4 truth)
python -m qlfm.paper_metrics RUN_DIR ...     # print the metrics of finished runs (multi-task or single-task)
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from qlfm.evaluate import _softmax, metrics_for

W = os.environ.get("QLFM_WORK", "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work")
TRUTH = os.path.join(W, "paper", "truth.npz")
CUTS = (10, 20, 50, 100, 200, 500)          # round-4 grid (r4_light_stats.CUTS)
T4_SHARE, T4_VIS = 0.05, 0.3


def build_truth(corpus="$QLFM_WORK/cache/bnb,$QLFM_WORK/cache/nue", splits="$QLFM_WORK/cache/splits_all.npz"):
    from qlfm.data.dataset import Corpus
    from qlfm.data.labels import EV_KEYS, event_labels
    c, sp = Corpus(os.path.expandvars(corpus)), np.load(os.path.expandvars(splits))
    lab = event_labels({k: c.ev_array(k) for k in EV_KEYS})
    ls = np.load(os.path.join(W, "r4", "light", "light_stats.npz"))
    pe_win = ls["pe_win"]
    # T2 light cut on val (same procedure as r4_light_stats, which used test)
    elig, y = lab["evt_nu"]
    va = sp["val"][elig[sp["val"]]]
    yv, pv = y[va], pe_win[va]
    grid = np.unique(np.round(np.logspace(0, 3.5, 200)))
    ba = [0.5 * ((pv[yv == 1] >= g).mean() + (pv[yv == 0] < g).mean()) for g in grid]
    g_best = float(grid[int(np.argmax(ba))])
    cut = float(min(CUTS, key=lambda x: abs(np.log(x) - np.log(g_best))))
    # T4 truth on test events eligible for the energy task
    te4 = np.sort(sp["test"][lab["energy"][0][sp["test"]]])
    nu_q, nu_edep, e_nu = c.ev_array("nu_q")[:, 2], c.ev_array("nu_edep"), c.ev_array("nu_energy")
    q_y = np.zeros(len(te4))
    s_of, l_of = c.locate(te4)
    for s in np.unique(s_of):
        sh = c.shards[s]
        off, cnt = np.asarray(sh.ev("hit_off")).astype(np.int64), np.asarray(sh.ev("hit_cnt")).astype(np.int64)
        pl, q = sh.hit("plane"), sh.hit("integral")
        for i in np.nonzero(s_of == s)[0]:
            a, n = off[l_of[i]], cnt[l_of[i]]
            m = np.asarray(pl[a:a + n]) == 2
            q_y[i] = np.asarray(q[a:a + n], dtype=np.float64)[m].clip(min=0).sum()
    share = nu_q[te4] / np.maximum(q_y, 1e-9)
    vis = nu_edep[te4] / np.maximum(1000.0 * e_nu[te4], 1e-9)
    os.makedirs(os.path.dirname(TRUTH), exist_ok=True)
    np.savez_compressed(TRUTH, pe_win=pe_win.astype(np.float32), t2_cut=cut, t2_cut_best=g_best,
                        t2_ba_val=float(max(ba)), t4_gidx=te4, t4_share=share.astype(np.float32),
                        t4_vis=vis.astype(np.float32))
    clean = (share >= T4_SHARE) & (vis >= T4_VIS)
    print(json.dumps(dict(t2_cut_val=cut, best_val=g_best, ba_val=float(max(ba)), t2_cut_round4_test=200.0,
                          t4_n=int(len(te4)), t4_keep=float(clean.mean()),
                          t4_keep_share=float((share >= T4_SHARE).mean()), t4_keep_vis=float((vis >= T4_VIS).mean()),
                          share_q=np.quantile(share, [0.1, 0.5, 0.9]).round(3).tolist(),
                          vis_q=np.quantile(vis, [0.1, 0.5, 0.9]).round(3).tolist())))


_truth = None


def truth():
    global _truth
    if _truth is None:
        _truth = dict(np.load(TRUTH))
    return _truth


def t1(z, thr=0.5):
    """z: hit preds (pred, hit_tgt, hit_q, hit_evt; an npz is read one column at a time, ~3 GB for 160M hits).
    Per-event unweighted F1 (events with no true and no selected hit are skipped), median and mean; plus the
    charge-weighted global F1 (as qlfm.evaluate.hit_metrics)."""
    sel = z["pred"] > thr
    t = z["hit_tgt"].astype(bool)
    ev = z["gidx"] if "gidx" in z else None
    he = z["hit_evt"]
    e = (np.searchsorted(ev, he) if ev is not None else np.unique(he, return_inverse=True)[1]).astype(np.int32)
    del he
    n = int(e.max()) + 1
    tp, nt, ns = (np.bincount(e, w, n).astype(np.float64) for w in (sel & t, t, sel))
    ok = nt + ns > 0
    f1 = 2 * tp[ok] / (nt[ok] + ns[ok])
    q = z["hit_q"].clip(min=0)
    tpq, ntq, nsq = (q * (sel & t)).sum(dtype=np.float64), (q * t).sum(dtype=np.float64), (q * sel).sum(dtype=np.float64)
    eff, pur = tpq / max(ntq, 1e-9), tpq / max(nsq, 1e-9)
    return dict(score=float(np.median(f1)), f1_evt_med=float(np.median(f1)), f1_evt_mean=float(f1.mean()),
                n_events=int(ok.sum()), n_no_true=int((nt == 0).sum()), f1_q=float(2 * eff * pur / max(eff + pur, 1e-9)))


def t2(r):
    tr = truth()
    y, p = r["y"].astype(int), _softmax(r["pred"])[:, 1]
    bright = tr["pe_win"][r["gidx"]] >= tr["t2_cut"]
    m = y != 2
    auc = lambda k: float(roc_auc_score(y[k] == 1, p[k])) if (y[k] == 1).any() and (y[k] == 0).any() else float("nan")
    return dict(score=auc(m & bright), auc_bright=auc(m & bright), auc_dim=auc(m & ~bright), auc_all=auc(m),
                cut_pe=float(tr["t2_cut"]), frac_bright=float(bright[m].mean()),
                n_bright=np.bincount(y[m & bright], minlength=2).tolist())


def t4(r):
    tr = truth()
    i = np.searchsorted(tr["t4_gidx"], r["gidx"])
    assert (tr["t4_gidx"][i.clip(max=len(tr["t4_gidx"]) - 1)] == r["gidx"]).all(), "T4 test events missing from truth.npz"
    clean = (tr["t4_share"][i] >= T4_SHARE) & (tr["t4_vis"][i] >= T4_VIS)
    rr = {k: v[clean] for k, v in r.items() if k in ("y", "pred")}
    m, a = metrics_for("energy", rr), metrics_for("energy", r)
    return dict(score=m["score"], res=m["resolution"], bias=m["bias"], res_all=a["resolution"], bias_all=a["bias"],
                n_clean=int(clean.sum()), frac_clean=float(clean.mean()))


def paper_metrics(task, r):
    if task == "hit":
        return t1(r)
    if task == "evt_nu":
        return t2(r)
    if task == "energy":
        return t4(r)
    m = metrics_for(task, r)
    if task == "light":
        m = dict(score=m["score"], mse=m["mse"], tot_resolution=m.get("tot_resolution"))
    return m


def headline(task, m):
    return {"hit": m["score"], "evt_nu": m["score"], "flavor": m["score"], "energy": m["res"],
            "light": m["mse"]}[task]


def run_metrics(run, refresh=False):
    """{task: metrics} for a finished multi-task run (RUN/<task>/preds_test.npz) or single-task run.
    Cached in RUN/paper_metrics.json (the hit predictions take minutes to read); refresh=True recomputes."""
    cache = os.path.join(run, "paper_metrics.json")
    a = json.load(open(os.path.join(run, "metrics.json")))
    tasks = a["tasks"] if "tasks" in a else [a["task"]]
    out = {} if refresh or not os.path.exists(cache) else json.load(open(cache))
    for t in tasks:
        f = os.path.join(run, t, "preds_test.npz") if "tasks" in a else os.path.join(run, "preds_test.npz")
        if t not in out and os.path.exists(f):
            z = np.load(f)
            out[t] = t1(z) if t == "hit" else paper_metrics(t, {k: z[k] for k in z.files})
            json.dump(out, open(cache + ".tmp", "w"), indent=1)
            os.replace(cache + ".tmp", cache)
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["truth"]:
        build_truth()
    else:
        for run in sys.argv[1:]:
            for t, m in run_metrics(run).items():
                print(run, t, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}),
                      flush=True)
