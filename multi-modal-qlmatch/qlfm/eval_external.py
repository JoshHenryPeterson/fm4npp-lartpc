"""Run a finished multi-task model on an external corpus (issue #10: MicroBooNE round-1 overlay / beam-on / EXT with
Wire-Cell labels), no retraining. Every event of the corpus is predicted; truth-free (real data) corpora work too.

python -m qlfm.eval_external --run RUN_DIR --corpus DIRS --out OUT_DIR [--tasks hit,evt_nu,flavor]

Writes OUT_DIR/preds_<task>.npz (as train_multitask's preds_test.npz: per-hit pred, hit_q, hit_tgt, hit_pnd, hit_evt;
per-event pred) plus per-hit hit_wc (Wire-Cell in-time main cluster, hit_wc_main) when the corpus has it, and
OUT_DIR/events.npz with the per-event Wire-Cell / Pandora / flash columns and the event labels, indexed by gidx.
Metrics: python -m qlfm.eval_external report OUT_DIR ... (see report()).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from qlfm.data.dataset import Corpus, EventDataset, collate
from qlfm.data.labels import EV_KEYS, event_labels
from qlfm.eval_perturb import load_run

EV_EXTRA = ("run", "subrun", "event", "is_data", "pnd_score", "pnd_fm", "fl_pe", "fl_t", "nu_nhits", "nu_q")


def predict_corpus(run, corpus_dirs, out, tasks, bs=16, workers=6):
    from qlfm.train_multitask import predict
    t0 = time.time()
    dev = torch.device("cuda")
    a, ns, model, multitask = load_run(run, dev)
    assert multitask, "multi-task runs only"
    tasks = [t for t in tasks if t in a["tasks"]]
    model.tasks, model.light_off_pass = tasks, False
    corpus = Corpus(corpus_dirs)
    n = corpus.n
    allg = np.arange(n)
    lab = event_labels({k: corpus.ev_array(k) for k in EV_KEYS})
    lab_np = {t: (np.ones(n, bool), lab[t][1] if lab[t][1] is not None else np.zeros(n, np.int64)) for t in tasks}
    loader = torch.utils.data.DataLoader(EventDataset(corpus, allg, truth=True), batch_size=bs, shuffle=False,
                                         collate_fn=collate, num_workers=workers, pin_memory=True)
    torch.manual_seed(0)
    r = predict(model, loader, dev, {t: allg for t in tasks}, lab_np)
    os.makedirs(out, exist_ok=True)
    has_wc = os.path.exists(os.path.join(corpus.shards[0].dir, "hit_wc_main.npy"))
    for t in tasks:
        rt = r[t]
        if t == "hit" and has_wc:
            # predict keeps every hit of each event in shard order (checked against the stored charge)
            g = rt["gidx"]
            s_of, l_of = corpus.locate(g)
            wc, q = [], []
            for s, l in zip(s_of, l_of):
                sh = corpus.shards[s]
                o, c = int(sh.ev("hit_off")[l]), int(sh.ev("hit_cnt")[l])
                wc.append(np.asarray(sh.hit("wc_main")[o:o + c]))
                q.append(np.asarray(sh.hit("integral")[o:o + c], np.float32))
            q = np.concatenate(q)
            assert len(q) == len(rt["hit_q"]) and np.allclose(q, rt["hit_q"], rtol=1e-2), "hit order mismatch"
            rt["hit_wc"] = np.concatenate(wc)
        np.savez(os.path.join(out, f"preds_{t}.npz"), **rt)
    ev = {k: corpus.ev_array(k) for k in EV_EXTRA}
    ev["sample"] = np.repeat([os.path.basename(sh.dir).removesuffix("_p0") for sh in corpus.shards], corpus.sizes)
    shard0 = corpus.shards[0].dir
    for f in sorted(os.listdir(shard0)):
        if f.startswith("ev_wc_"):
            ev[f[3:-4]] = corpus.ev_array(f[3:-4])
    for t in ("hit", "evt_nu", "flavor", "energy"):
        ev[f"elig_{t}"] = lab[t][0]
        if lab[t][1] is not None:
            ev[f"y_{t}"] = lab[t][1]
    np.savez(os.path.join(out, "events.npz"), **{k: np.asarray(v) for k, v in ev.items()})
    meta = dict(run=run, corpus=corpus_dirs, tasks=tasks, n_events=int(n), backbone=ns.backbone,
                width=a.get("width", "native"), minutes=(time.time() - t0) / 60)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)


# ---------------- report (round-1 samples: bnb_overlay, nue_overlay, beam_on, ext) ----------------
def _hit_scores(q, t, s, he, n_ev):
    """Charge-weighted eff / purity / F1 over all hits, and the median per-event (hit-count) F1."""
    tp, tt, ss = (q * (s & t)).sum(), (q * t).sum(), (q * s).sum()
    eff, pur = tp / max(tt, 1e-9), tp / max(ss, 1e-9)
    ctp, ct, cs = (np.bincount(he, w, n_ev) for w in (s & t, t, s))
    ok = ct + cs > 0
    f1e = 2 * ctp[ok] / (ct[ok] + cs[ok])
    return dict(f1_q=float(2 * eff * pur / max(eff + pur, 1e-9)), eff_q=float(eff), pur_q=float(pur),
                f1_evt_med=float(np.median(f1e)) if ok.any() else float("nan"), n_events=int(ok.sum()))


def report(out):
    from sklearn.metrics import roc_auc_score
    from qlfm.evaluate import _softmax
    meta = json.load(open(os.path.join(out, "meta.json")))
    E = dict(np.load(os.path.join(out, "events.npz"), allow_pickle=True))
    tag = np.load(os.path.join(meta["corpus"].split(",")[0], "split_tag.npz"))["tag"]
    smp, n_ev = E["sample"], len(E["sample"])
    overlay = np.isin(smp, ["bnb_overlay", "nue_overlay"])
    wc_ok = E["wc_in_nusel"] == 1
    gen = E["wc_generic_nu"] == 1
    subsets = {"all": np.ones(n_ev, bool), "clean": tag != 1}      # clean = not in our train split
    res = dict(run=meta["run"], width=meta["width"])
    # T1: overlay events with a visible in-AV nu and WC labels
    z = np.load(os.path.join(out, "preds_hit.npz"))
    he = z["hit_evt"].astype(np.int64)
    q, t = z["hit_q"].clip(min=0).astype(np.float64), z["hit_tgt"].astype(bool)
    sel = {"model": z["pred"] > 0.5, "pandora": z["hit_pnd"] > 0.5, "wc_main": z["hit_wc"] > 0,
           "wc_main&generic": (z["hit_wc"] > 0) & gen[he]}
    for name, m_ev in subsets.items():
        for s_name in ("bnb_overlay", "nue_overlay"):
            em = m_ev & (smp == s_name) & E["elig_hit"] & wc_ok
            hm = em[he]
            res[f"T1/{s_name}/{name}"] = {k: _hit_scores(q[hm], t[hm], v[hm], he[hm], n_ev) for k, v in sel.items()}
    # T1 on data: agreement of model hits with WC hits on WC-selected events (no truth), vs the same on overlay
    for s_name in ("beam_on", "bnb_overlay"):
        em = (smp == s_name) & wc_ok & gen
        hm = em[he]
        a = _hit_scores(q[hm], sel["wc_main&generic"][hm], sel["model"][hm], he[hm], n_ev)
        res[f"T1agree/{s_name}"] = dict(f1_q_vs_wc=a["f1_q"], model_in_wc=a["pur_q"], wc_in_model=a["eff_q"],
                                       n_events=int(em.sum()))
    for s_name in ("beam_on", "ext"):     # charge share the model calls nu, per sample
        hm = (smp == s_name)[he]
        res[f"nu_charge_share/{s_name}"] = float((q[hm] * sel["model"][hm]).sum() / q[hm].sum())
    # T2: nu (class 1) vs no-nu (class 0) on overlay; WC generic nu selection as a working point
    r = np.load(os.path.join(out, "preds_evt_nu.npz"))
    p = np.full(n_ev, np.nan)
    p[r["gidx"]] = _softmax(r["pred"])[:, 1]
    y = E["y_evt_nu"]
    for name, m_ev in subsets.items():
        em = m_ev & overlay & E["elig_evt_nu"] & (y != 2) & wc_ok
        yy = y[em] == 1
        d = dict(n=[int((~yy).sum()), int(yy.sum())])
        if yy.any() and (~yy).any():
            fpr_wc, tpr_wc = float(gen[em][~yy].mean()), float(gen[em][yy].mean())
            thr = np.quantile(p[em][~yy], 1 - fpr_wc)
            d.update(auc_model=float(roc_auc_score(yy, p[em])), auc_pandora=float(roc_auc_score(yy, E["pnd_score"][em])),
                     wc_tpr=tpr_wc, wc_fpr=fpr_wc, model_tpr_at_wc_fpr=float((p[em][yy] > thr).mean()))
        res[f"T2/overlay/{name}"] = d
    # T2 on data: pass rates at the threshold giving WC's EXT pass rate
    on, ext = (smp == "beam_on") & wc_ok, (smp == "ext") & wc_ok
    wc_on, wc_ext = float(gen[on].mean()), float(gen[ext].mean())
    thr = np.quantile(p[ext], 1 - wc_ext)
    res["T2/data"] = dict(n_on=int(on.sum()), n_ext=int(ext.sum()), wc_on=wc_on, wc_ext=wc_ext,
                          model_on_at_wc_ext=float((p[on] > thr).mean()), model_ext=float((p[ext] > thr).mean()),
                          model_on_p50=float((p[on] > 0.5).mean()), model_ext_p50=float((p[ext] > 0.5).mean()),
                          thr=float(thr))
    json.dump(res, open(os.path.join(out, "report.json"), "w"), indent=1)
    return res


def main(argv=None):
    if argv and argv[0] == "report":
        for out in argv[1:]:
            print(json.dumps(report(out)), flush=True)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="hit,evt_nu,flavor")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=None, help="accepted for run_queue.sh; unused")
    a = ap.parse_args(argv)
    predict_corpus(a.run, a.corpus, a.out, a.tasks.split(","), a.bs, a.workers)
    json.dump({"done": True}, open(os.path.join(a.out, "metrics.json"), "w"))     # run_queue.sh completion marker


if __name__ == "__main__":
    main(sys.argv[1:])
