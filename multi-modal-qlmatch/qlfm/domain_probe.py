"""Domain-shift diagnostics (#13): which backbone features are sim-only (width-sensitive) and which are data/sim
consistent, layer by layer. Uses the paired simulated domains of docs/domain-shift-fm.md: the same overlay events
with nominal simulated nu-hit widths (S) and with the data-like widths (D, eval_perturb w_<width>).

Per layer (charge-token embeddings, then each Transformer block):
  sens_nu / sens_cos  paired sensitivity |z_D - z_S| / |z_S|, mean over nu tokens / cosmic tokens (method a)
  dom_auc             AUC of a linear probe telling D from S on nu tokens, split by event (method b; 0.5 = invariant)
  task_SS / task_SD   AUC of a linear nu-vs-cosmic token probe trained on S, tested on S / on D (transfer of a
                      sim-trained linear head to the data-like domain)
nu token: >= 50% of its charge from hits with a simulated nu deposit; cosmic token: none.

python -m qlfm.domain_probe (--fm PRE/fm.pt | --run FINETUNE_RUN) --out OUT_DIR [--width meas3d] [--n-events 400]
(writes OUT_DIR/metrics.json, the run_queue.sh completion marker)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from qlfm.data.dataset import Corpus, EventDataset, collate, to_device
from qlfm.data.labels import EV_KEYS, event_labels
from qlfm.eval_perturb import perturb

NU_FRAC = 0.5


def load_fm(a, dev):
    if a.run:
        from qlfm.eval_perturb import load_run
        _, _, model, _ = load_run(a.run, dev)
        return model.backbone.fm
    from qlfm.ref import build_backbone
    from qlfm.train_multitask import get_args
    ns = get_args(["--mode", "frozen", "--fm", a.fm, "--out", "/tmp/unused"])
    return build_backbone("qlfm", ns, a.fm).fm.to(dev)


@torch.no_grad()
def layer_feats(fm, b, seed):
    """[embedding, block 1..L] charge-token features and per-token nu-charge fraction for one batch."""
    g = torch.Generator(device=b["hit_q"].device).manual_seed(seed)      # same token cap in S and D
    ct, lt = fm.tokenize(b, generator=g)
    xc, xl, _ = fm.embed(ct, lt)
    nl = len(fm.encoder.blocks)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, layers = fm.encode(xc, ct.tok_evt, xl, lt.evt if lt is not None else None, b["n_events"],
                              return_layers=tuple(range(nl)))
    feats = [xc.float()] + [zc.float() for zc, _ in layers]
    m = ct.hit2tok >= 0
    nt = xc.shape[0]
    q = b["hit_q"].clamp(min=0)[m]
    qs = torch.zeros(nt, device=q.device).index_add_(0, ct.hit2tok[m], q)
    qn = torch.zeros(nt, device=q.device).index_add_(0, ct.hit2tok[m], q * (b["hit_nu_edep"][m] > 0))
    return feats, (qn / qs.clamp(min=1e-6)), ct.tok_evt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", default="meas3d")
    ap.add_argument("--corpus", default=os.path.expandvars("$QLFM_WORK/cache/bnb,$QLFM_WORK/cache/nue"))
    ap.add_argument("--splits", default=os.path.expandvars("$QLFM_WORK/cache/splits_all.npz"))
    ap.add_argument("--n-events", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-tok", type=int, default=20000, help="tokens per class per probe")
    ap.add_argument("--time-limit", type=float, default=None, help="accepted for run_queue.sh; unused")
    a = ap.parse_args()
    assert bool(a.fm) != bool(a.run), "give --fm or --run"
    t0 = time.time()
    dev = torch.device("cuda")
    fm = load_fm(a, dev).eval()
    c, sp = Corpus(a.corpus), np.load(a.splits)
    lab = event_labels({k: c.ev_array(k) for k in EV_KEYS})
    te = sp["test"][lab["hit"][0][sp["test"]]]                        # events with a visible in-AV nu
    ev = np.sort(np.random.default_rng(5).permutation(te)[:a.n_events])
    dl = torch.utils.data.DataLoader(EventDataset(c, ev, truth=True), batch_size=a.bs, collate_fn=collate, num_workers=4)
    acc = None
    for i, b in enumerate(dl):
        b = to_device(b, dev)
        fS, frac, tev = layer_feats(fm, b, i)
        fD, frac2, _ = layer_feats(fm, perturb(b, "w_" + a.width, 0.0), i)
        assert torch.equal(frac, frac2)
        g = b["gidx"][tev]
        if acc is None:
            acc = {"S": [[] for _ in fS], "D": [[] for _ in fS], "frac": [], "evt": []}
        for l in range(len(fS)):
            acc["S"][l].append(fS[l].cpu().numpy())
            acc["D"][l].append(fD[l].cpu().numpy())
        acc["frac"].append(frac.cpu().numpy())
        acc["evt"].append(g.cpu().numpy())
    frac, evt = np.concatenate(acc["frac"]), np.concatenate(acc["evt"])
    nu, cos = frac >= NU_FRAC, frac == 0
    rng = np.random.default_rng(0)
    test_ev = set(rng.permutation(np.unique(evt))[: len(np.unique(evt)) * 3 // 10].tolist())
    is_te = np.isin(evt, list(test_ev))
    cap = lambda m: np.flatnonzero(m)[rng.permutation(int(m.sum()))[:a.max_tok]]
    res = []
    for l in range(len(acc["S"])):
        S, D = np.concatenate(acc["S"][l]), np.concatenate(acc["D"][l])
        rel = np.linalg.norm(D - S, axis=1) / np.maximum(np.linalg.norm(S, axis=1), 1e-6)
        r = dict(layer=l, sens_nu=float(rel[nu].mean()), sens_cos=float(rel[cos].mean()))
        # (b) domain probe on nu tokens: S (0) vs D (1), split by event
        i_tr, i_te = cap(nu & ~is_te), cap(nu & is_te)
        X = np.concatenate([S[i_tr], D[i_tr]]); y = np.r_[np.zeros(len(i_tr)), np.ones(len(i_tr))]
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(X), y)
        Xt = np.concatenate([S[i_te], D[i_te]]); yt = np.r_[np.zeros(len(i_te)), np.ones(len(i_te))]
        r["dom_auc"] = float(roc_auc_score(yt, clf.predict_proba(sc.transform(Xt))[:, 1]))
        # task probe: nu vs cosmic tokens, trained on S, tested on S and on D (same test tokens)
        j_tr = np.concatenate([cap(nu & ~is_te), cap(cos & ~is_te)])
        j_te = np.concatenate([cap(nu & is_te), cap(cos & is_te)])
        sc2 = StandardScaler().fit(S[j_tr])
        clf2 = LogisticRegression(max_iter=300, C=1.0).fit(sc2.transform(S[j_tr]), nu[j_tr])
        r["task_SS"] = float(roc_auc_score(nu[j_te], clf2.predict_proba(sc2.transform(S[j_te]))[:, 1]))
        r["task_SD"] = float(roc_auc_score(nu[j_te], clf2.predict_proba(sc2.transform(D[j_te]))[:, 1]))
        res.append(r)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    out = dict(fm=a.fm, run=a.run, width=a.width, n_events=len(ev), n_tokens=int(len(frac)), n_nu_tok=int(nu.sum()),
               n_cos_tok=int(cos.sum()), layers=res, minutes=(time.time() - t0) / 60)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
