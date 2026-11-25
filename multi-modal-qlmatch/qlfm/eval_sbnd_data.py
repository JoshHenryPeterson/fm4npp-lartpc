"""SBND models vs truth and Wire-Cell on simulation and real data (issue #14). Runs a finished multi-task model on a corpus
(beam-on / off-beam data, or simulation test events) and keeps per-event aggregates instead of per-hit predictions
(the data hold ~1.8B hits). No retraining.

python -m qlfm.eval_sbnd_data --run RUN_DIR --corpus DIRS [--pattern GLOB | --events IDX.npy] [--perturb w_sbnd]
                              --out OUT_ROOT/<tag>/<chunk>                         (QLFM_DETECTOR=sbnd)
python -m qlfm.eval_sbnd_data report OUT_ROOT       (each OUT_ROOT/<tag>: its chunks are concatenated, then scored)

Chunks: on_* / off_* = real data; sim_nom = simulation test events; sim_w = the same with every hit width rescaled to
data (eval_perturb.F_SBND). The model's own training-time width option is switched off (model.width = native): real
data are not rescaled, and simulation is rescaled only by --perturb.

Per event (events.npz): the evt_nu logits (score = log-odds of class 1, which does not saturate like the softmax),
truth labels (event_labels: elig_hit, elig_evt_nu, y_evt_nu; all class 0 for data), and charge-weighted (q_*) and
hit-count (n_*) sums of: all hits, model nu hits (sigmoid > 0.5), truth nu hits (hit_nu_edep > 0.05 MeV, simulation
only), Wire-Cell nu-candidate footprint hits (hit_pnd_nu, charge-light matched after the TGM/STM/light-mismatch veto),
the same restricted to candidates passing the PR cosmic tagger (hit_pnd_nu_ct0), and their overlaps. Plus
brightest-flash time/PE and run/subrun/event.

The report compares, per run:
  * data, event level: WC selects an event if it has any WC nu hit; each model's cut is set to pass the same off-beam
    fraction as WC, and the beam-on pass rates are compared (also on events whose brightest flash is in the beam
    window), on all events and on the held-out test files of the real-data fine-tune split (heldout());
  * data, hit level: on beam-on events with a WC candidate, the share of WC nu charge the model also tags (wc_in_model),
    the share of model nu charge inside the WC footprint (model_in_wc); the model's nu share of all charge;
  * simulation (sim_nom, sim_w): T1 charge-weighted F1 vs truth on T1-eligible events, T2 AUC (visible in-AV nu vs
    no nu), and the same WC agreement shares on T1-eligible events with a WC candidate.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

from qlfm import geometry as G
from qlfm.data.dataset import Corpus, EventDataset, collate, to_device
from qlfm.data.labels import EV_KEYS, HIT_NU_EDEP_MEV, event_labels
from qlfm.eval_perturb import load_run, perturb

SUMS = ("tot", "model", "true", "model_true", "wc", "wc0", "model_wc", "model_wc0", "wc_true")
EV_KEEP = ("run", "subrun", "event", "fl_t", "fl_pe", "wc_n_bundle", "wc_n_bundle_cosmic")


def heldout(run, subrun, event):
    """True for events in the test split of the real-data fine-tune (split by source file, 75/5/20: ev_fsplit in the
    data cache, cache/splits_sbnd_data.npz); matched by (run, subrun, event) so it also applies to older outputs."""
    keys = np.load(os.path.join(os.environ.get("QLFM_WORK", "."), "cache", "sbnd_data_test_keys.npy"))
    k = (np.asarray(run, np.int64) << 40) | (np.asarray(subrun, np.int64) << 20) | np.asarray(event, np.int64)
    return np.isin(k, keys)


class DS(EventDataset):
    def __getitem__(self, i):
        out = super().__getitem__(i)
        s, j = self.c.locate(out["gidx"])
        sh = self.c.shards[s]
        h0, hn = int(sh.ev("hit_off")[j]), int(sh.ev("hit_cnt")[j])
        out["hit_ct0"] = np.asarray(sh.hit("pnd_nu_ct0")[h0:h0 + hn])
        return out


def collate_ct0(items):
    b = collate(items)
    b["hit_ct0"] = torch.from_numpy(np.concatenate([it["hit_ct0"] for it in items]).astype(np.uint8))
    return b


@torch.no_grad()
def predict_corpus(run, corpus_dirs, out, bs=16, workers=6, pattern="*", max_events=None, events=None, pert=None):
    t0 = time.time()
    dev = torch.device("cuda")
    a, ns, model, multitask = load_run(run, dev)
    assert multitask and G.DETECTOR == "sbnd"
    model.tasks, model.light_off_pass = ["hit", "evt_nu"], False
    model.width = "native"                 # widths only through --perturb (never on real data)
    corpus = Corpus(corpus_dirs, pattern)
    idx = np.load(events) if events else np.arange(corpus.n if max_events is None else min(corpus.n, max_events))
    n = len(idx)
    pos = np.full(corpus.n, -1, np.int64)
    pos[idx] = np.arange(n)
    loader = torch.utils.data.DataLoader(DS(corpus, idx, truth=True), batch_size=bs, shuffle=False,
                                         collate_fn=collate_ct0, num_workers=workers, pin_memory=True)
    agg = {f"{w}_{k}": np.zeros(n) for w in ("q", "n") for k in SUMS}
    logit = np.full((n, 3), np.nan, np.float32)
    torch.manual_seed(0)
    model.eval()
    for b in loader:
        b = to_device(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(perturb(b, pert, 0.0) if pert else b)
        g = pos[b["gidx"].cpu().numpy()]
        he, ne = b["hit_evt"], b["n_events"]
        s = torch.sigmoid(o["hit"].float()).reshape(-1) > 0.5
        w, w0 = b["hit_pnd_nu"] > 0, b["hit_ct0"] > 0
        t = b["hit_nu_edep"] > HIT_NU_EDEP_MEV
        masks = {"tot": torch.ones_like(s), "model": s, "true": t, "model_true": s & t, "wc": w, "wc0": w0,
                 "model_wc": s & w, "model_wc0": s & w0, "wc_true": w & t}
        q = b["hit_q"].clamp(min=0)
        for k, m in masks.items():
            for pre, val in (("q", q * m), ("n", m.float())):
                acc = torch.zeros(ne, device=dev, dtype=torch.float64).index_add_(0, he, val.double())
                agg[f"{pre}_{k}"][g] = acc.cpu().numpy()
        logit[g] = o["evt_nu"].float().cpu().numpy()
    assert not np.isnan(logit).any(), "events without a prediction"
    have = set(f[3:-4] for f in os.listdir(corpus.shards[0].dir) if f.startswith("ev_"))
    ev = {k: corpus.ev_array(k)[idx] for k in EV_KEEP if k in have}
    ev["sample"] = np.repeat([os.path.basename(os.path.dirname(sh.dir)) for sh in corpus.shards], corpus.sizes)[idx]
    lab = event_labels({k: corpus.ev_array(k) for k in EV_KEYS})
    ev.update(gidx=idx, elig_hit=lab["hit"][0][idx], elig_evt_nu=lab["evt_nu"][0][idx], y_evt_nu=lab["evt_nu"][1][idx])
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "events.npz"), logit=logit, **agg, **{k: np.asarray(v) for k, v in ev.items()})
    meta = dict(run=run, corpus=corpus_dirs, pattern=pattern, events=events, perturb=pert, n_events=int(n),
                backbone=ns.backbone, minutes=(time.time() - t0) / 60)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)


# ---------------- report ----------------
def _share(num, den):
    return float(num.sum() / max(den.sum(), 1e-9))


def _score(E):
    lg = E["logit"].astype(np.float64)
    return lg[:, 1] - np.logaddexp(lg[:, 0], lg[:, 2])           # nu log-odds (> 0  <=>  softmax p_nu > 0.5)


def _load(chunks):
    parts = [dict(np.load(os.path.join(c, "events.npz"), allow_pickle=True)) for c in chunks]
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}


def data_report(E, keep=None):
    smp = E["sample"]
    keep = np.ones(len(smp), bool) if keep is None else keep
    on, off = (smp == "bnb-opendata") & keep, (smp == "offbeam-intime-run1") & keep
    inwin = (E["fl_t"] >= G.BEAM_WINDOW_US[0]) & (E["fl_t"] < G.BEAM_WINDOW_US[1])
    p = _score(E)
    res = dict(n_on=int(on.sum()), n_off=int(off.sum()))
    for lab in ("wc", "wc0"):
        sel = E[f"n_{lab}"] > 0
        for cut_name, cut in (("all", np.ones_like(on)), ("beamwin", inwin)):
            o, f = on & cut, off & cut
            wc_on, wc_off = float(sel[o].mean()), float(sel[f].mean())
            thr = float(np.quantile(p[f], 1 - wc_off))
            res[f"evt/{lab}/{cut_name}"] = dict(n_on=int(o.sum()), n_off=int(f.sum()), wc_on=wc_on, wc_off=wc_off,
                                                model_on=float((p[o] > thr).mean()), model_off=float((p[f] > thr).mean()),
                                                thr=thr, model_on_p50=float((p[o] > 0).mean()),
                                                model_off_p50=float((p[f] > 0).mean()))
        m = on & sel                                       # hit level on beam-on events with a WC candidate
        res[f"hit/{lab}/beam_on"] = dict(n_events=int(m.sum()),
                                         wc_in_model=_share(E[f"q_model_{lab}"][m], E[f"q_{lab}"][m]),
                                         model_in_wc=_share(E[f"q_model_{lab}"][m], E["q_model"][m]))
        # event-level AUC of the model score against the WC decision, beam-on + off-beam together
        from sklearn.metrics import roc_auc_score
        both = on | off
        res[f"evt/{lab}/auc_vs_wc"] = float(roc_auc_score(sel[both], p[both]))
    for nm, msk in (("beam_on", on), ("off_beam", off)):
        res[f"nu_charge_share/{nm}"] = dict(model=_share(E["q_model"][msk], E["q_tot"][msk]),
                                            wc=_share(E["q_wc"][msk], E["q_tot"][msk]),
                                            wc0=_share(E["q_wc0"][msk], E["q_tot"][msk]))
    return res


def sim_report(E):
    from sklearn.metrics import roc_auc_score
    h = E["elig_hit"].astype(bool)
    tp, tt, ss = E["q_model_true"][h].sum(), E["q_true"][h].sum(), E["q_model"][h].sum()
    eff, pur = tp / max(tt, 1e-9), tp / max(ss, 1e-9)
    wtp, wss = E["q_wc_true"][h].sum(), E["q_wc"][h].sum()
    weff, wpur = wtp / max(tt, 1e-9), wtp / max(wss, 1e-9)
    e2 = E["elig_evt_nu"].astype(bool) & (E["y_evt_nu"] != 2)
    m = h & (E["n_wc"] > 0)
    return dict(n_hit=int(h.sum()), n_evt=int(e2.sum()),
                t1_f1=float(2 * eff * pur / max(eff + pur, 1e-9)), t1_eff=float(eff), t1_pur=float(pur),
                wc_f1=float(2 * weff * wpur / max(weff + wpur, 1e-9)),
                t2_auc=float(roc_auc_score(E["y_evt_nu"][e2] == 1, _score(E)[e2])),
                n_wc_events=int(m.sum()), wc_in_model=_share(E["q_model_wc"][m], E["q_wc"][m]),
                model_in_wc=_share(E["q_model_wc"][m], E["q_model"][m]),
                wc0_in_model=_share(E["q_model_wc0"][m], E["q_wc0"][m]))


def report_one(out):
    chunks = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(out, "*", "events.npz")))
    metas = {os.path.basename(c): json.load(open(os.path.join(c, "meta.json"))) for c in chunks}
    assert len({m["run"] for m in metas.values()}) == 1, out
    res = dict(run=next(iter(metas.values()))["run"], chunks=len(chunks))
    data = [c for c in chunks if os.path.basename(c).split("_")[0] in ("on", "off")]
    if data:
        E = _load(data)
        res["data"] = data_report(E)
        res["data_test"] = data_report(E, heldout(E["run"], E["subrun"], E["event"]))
    for c in chunks:
        if os.path.basename(c).startswith("sim_"):
            res[os.path.basename(c)] = sim_report(_load([c]))
    json.dump(res, open(os.path.join(out, "report.json"), "w"), indent=1)
    return res


def main(argv=None):
    if argv and argv[0] == "report":
        rows = []
        for out in sorted(d for d in glob.glob(os.path.join(argv[1], "*")) if os.path.isdir(d)):
            r = report_one(out)
            rows.append(r)
            print(json.dumps(r), flush=True)
        with open(os.path.join(argv[1], "report_all.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="*", help="shard glob inside each corpus dir (split a sample into chunks)")
    ap.add_argument("--max-events", type=int, default=None, help="smoke tests")
    ap.add_argument("--events", default=None, help=".npy of corpus indices to predict (simulation test events)")
    ap.add_argument("--perturb", default=None, help="eval_perturb variant applied to every batch, e.g. w_sbnd")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=None, help="accepted for run_queue.sh; unused")
    a = ap.parse_args(argv)
    predict_corpus(a.run, a.corpus, a.out, a.bs, a.workers, a.pattern, a.max_events, a.events, a.perturb)
    json.dump({"done": True}, open(os.path.join(a.out, "metrics.json"), "w"))     # run_queue.sh completion marker


if __name__ == "__main__":
    main(sys.argv[1:])
