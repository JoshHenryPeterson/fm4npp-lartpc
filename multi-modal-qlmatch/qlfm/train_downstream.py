"""Downstream training/evaluation: scratch | frozen | finetune | linear, one task per run (single GPU).

python -m qlfm.train_downstream --task evt_nu --mode frozen --fm RUN/fm.pt --label-frac 0.01 --out DIR
Writes best-on-val head/model, test predictions (preds_test.npz) and metrics (metrics.json).
"""
import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qlfm.data.dataset import Corpus, EventDataset, collate, to_device
from qlfm.data.labels import EV_KEYS, HIT_NU_EDEP_MEV, N_OUT, event_labels
from qlfm.evaluate import metrics_for
from qlfm.models.heads import AttnPool, HitHead, LateLight, LightHead, MeanLinear, RawHit, XFusion
from qlfm.ref import build_backbone


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.expandvars("$QLFM_WORK/cache/bnb,$QLFM_WORK/cache/nue"))
    ap.add_argument("--splits", default=os.path.expandvars("$QLFM_WORK/cache/splits_all.npz"))
    ap.add_argument("--task", required=True, choices=list(N_OUT))
    ap.add_argument("--mode", required=True, choices=["scratch", "frozen", "finetune", "linear"])
    ap.add_argument("--fm", default=None, help="pretrained checkpoint (frozen/finetune/linear)")
    ap.add_argument("--backbone", default="qlfm", choices=["qlfm", "fm4npp", "fm4npp_l", "panda", "panda_l", "polarmae"])
    ap.add_argument("--late-light", type=int, default=0,
                    help="late-fused PMT light: 0 none, 1 PMT tokens in the pooled set (R2P), "
                         "2 strong fusion: 2 cross-attention layers (tokens <- PMT tokens) at a common width (round 3)")
    ap.add_argument("--hit-feat", default="native", choices=["native", "raw", "none"],
                    help="T1 per-hit input: backbone's own hit embedding (qlfm only), a common trainable raw-hit "
                         "adapter for every backbone, or none")
    ap.add_argument("--pimm-prep", default="matched", choices=["matched", "legacy"],
                    help="Panda/PTv3 input preprocessing: matched to the pretraining views (audit #6) or rounds 2-4")
    ap.add_argument("--max-hits", type=int, default=None, help="FM4NPP: per-event hit cap override (default: checkpoint)")
    ap.add_argument("--light", type=int, default=1, help="scratch only; otherwise taken from the FM")
    ap.add_argument("--label-frac", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation: micro-batch = bs / accum")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=float, default=30)
    ap.add_argument("--min-steps", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=15000)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=50.0)
    ap.add_argument("--resume", type=int, default=1,
                    help="at the time limit save OUT/state.pt and exit without testing; a later call resumes from it "
                         "(0: test the best checkpoint so far; metrics.json then has truncated=true)")
    ap.add_argument("--state-every", type=float, default=20.0,
                    help="with --resume, also save OUT/state.pt at an eval step every this many minutes (preemptable jobs)")
    ap.add_argument("--layer-decay", type=float, default=0.75)
    # architecture (scratch); for pretrained models these come from the checkpoint
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--pos-mode", default="centroid")
    ap.add_argument("--pos-freq", type=int, default=10)
    ap.add_argument("--cell-wires", type=int, default=16)
    ap.add_argument("--cell-ticks", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="qlfm token cap; default: the pretraining value from the checkpoint (8192 for scratch)")
    return ap.parse_args(argv)


class DownstreamModel(nn.Module):
    """Backbone (common interface, qlfm/ref) + identical heads; optional late-fused PMT light."""

    def __init__(self, backbone, task, mode, late_light=0, hit_feat="native"):
        super().__init__()
        self.backbone, self.task, self.mode = backbone, task, mode
        late_light = int(late_light)
        if task == "light":                                # T5: charge only, also for light-pretrained encoders
            late_light = 0
            if getattr(backbone, "light_on", False):
                backbone.light_on = False
        self.fusion = XFusion(backbone.d) if late_light == 2 else None
        d = self.fusion.d if self.fusion is not None else backbone.d
        self.light = LateLight(d) if late_light else None
        self.hit_feat = hit_feat
        self.raw_hit = RawHit() if (task == "hit" and hit_feat == "raw") else None
        hit_dim = {"native": backbone.hit_dim, "raw": 64, "none": 0}[hit_feat]
        if task == "hit":
            self.head = HitHead(d, hit_dim, extra_dim=d if late_light else 0)
        elif task == "light":
            self.head = LightHead(d)
        elif mode == "linear":
            self.head = MeanLinear(d, N_OUT[task])
        else:
            self.head = AttnPool(d, out_dim=N_OUT[task])

    def train(self, mode=True):
        super().train(mode)
        if self.mode in ("frozen", "linear"):               # frozen backbones never use dropout / drop-path
            self.backbone.eval()
        return self

    def forward(self, b):
        # frozen: no graph through the backbone, unless it has trainable adapters (e.g. FM4NPP's layer mix)
        frozen_all = self.mode in ("frozen", "linear") and not any(p.requires_grad for p in self.backbone.parameters())
        with torch.no_grad() if frozen_all else contextlib.nullcontext():
            o = self.backbone(b)
        z, evt = o["z"], o["evt"]
        if self.light is not None:
            zl, levt, lpool = self.light(b)
        if self.fusion is not None:
            z = self.fusion(z, evt, zl, b["n_events"])
        if self.task == "hit":
            extra = lpool[b["hit_evt"]] if self.light is not None else None
            h_hit = {"native": o["h_hit"], "none": None}.get(self.hit_feat) if self.raw_hit is None else self.raw_hit(b)
            return self.head(h_hit, z, o["hit2tok"], extra), o
        if self.light is not None:
            z, evt = torch.cat([z.to(zl.dtype), zl]), torch.cat([evt, levt])
        return self.head(z, evt, b["n_events"]), o


def light_target(b):
    """T5 target: the 32 x 17 light-token features (log1p PE per time bin, log1p total) of each event."""
    from qlfm.tokenize import TokCfg, tokenize_light
    lt = tokenize_light(b["oph_ch"], b["oph_t"], b["oph_pe"], b["oph_evt"], b["n_events"], TokCfg())
    return lt.feat.view(b["n_events"], -1, lt.feat.shape[1])


def loss_fn(task, out, b, y, class_w):
    if task == "light":
        t = light_target(b)
        return F.mse_loss(out, t) + F.mse_loss(out[..., -1], t[..., -1])     # extra weight on the per-PMT total
    if task == "hit":
        tgt = (b["hit_nu_edep"] > HIT_NU_EDEP_MEV).float()
        w = b["hit_q"].clamp(min=1.0).log1p()            # mild charge weighting
        return (F.binary_cross_entropy_with_logits(out, tgt, reduction="none") * w).sum() / w.sum()
    if task in ("evt_nu", "flavor"):
        return F.cross_entropy(out, y, weight=class_w)
    return F.huber_loss(out.squeeze(1), y.float(), delta=0.5)


def layer_groups(model, base_lr, decay):
    """Layer-wise lr decay for fine-tuning: head/last block get base_lr, embeddings the smallest."""
    bb = model.backbone
    depth = bb.n_layers - 1
    groups = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not n.startswith("backbone."):                 # heads and late light
            lid = depth + 1
        else:
            lid = bb.layer_id(n[len("backbone."):])
        groups.append({"params": [p], "lr_scale": decay ** (depth + 1 - lid),
                       "weight_decay": 0.05 if p.ndim >= 2 else 0.0})
    for g in groups:
        g["lr"] = base_lr * g["lr_scale"]
    return groups


@torch.no_grad()
def predict(model, loader, dev, task, labels_of):
    model.eval()
    preds, ys, gid, extra = [], [], [], {"hit_q": [], "hit_pnd": [], "hit_evt": [], "hit_tgt": [], "light_tgt": []}
    for b in loader:
        b = to_device(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = model(b)
        out = out.float()
        if task == "hit":
            preds.append(torch.sigmoid(out).cpu())
            extra["hit_q"].append(b["hit_q"].cpu())
            extra["hit_pnd"].append(b["hit_pnd_nu"].cpu())
            extra["hit_tgt"].append((b["hit_nu_edep"] > HIT_NU_EDEP_MEV).cpu())
            extra["hit_evt"].append(b["gidx"][b["hit_evt"]].cpu())
        elif task == "light":
            preds.append(out.half().cpu())
            extra["light_tgt"].append(light_target(b).half().cpu())
        else:
            preds.append(out.cpu())
        gid.append(b["gidx"].cpu())
    model.train()
    g = torch.cat(gid).numpy()
    res = {"gidx": g, "pred": torch.cat(preds).numpy()}
    if task in ("hit", "light"):
        res.update({k: torch.cat(v).numpy() for k, v in extra.items() if v})
    else:
        res["y"] = labels_of(g)
    return res


def main(argv=None):
    args = get_args(argv)
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    corpus = Corpus(args.corpus)
    sp = np.load(args.splits)
    ev = {k: corpus.ev_array(k) for k in EV_KEYS}
    lab = event_labels(ev)
    elig, target = lab[args.task]
    labels_of = (lambda g: target[g]) if target is not None else (lambda g: None)

    pool = sp[f"lab_f{args.label_frac:g}"]
    tr = pool[elig[pool]]
    # fixed random (not index-ordered) val/test subsets: samples are concatenated in index order
    pick = lambda idx, n, seed: np.sort(np.random.default_rng(seed).permutation(idx)[:n])
    va = pick(sp["val"][elig[sp["val"]]], args.n_val, 11)
    te = pick(sp["test"][elig[sp["test"]]], args.n_test, 12)

    # ---- model ----
    backbone = build_backbone(args.backbone, args, None if args.mode == "scratch" else args.fm)
    model = DownstreamModel(backbone, args.task, args.mode, args.late_light, args.hit_feat).to(dev)
    if args.mode in ("frozen", "linear"):
        for n_, p in backbone.named_parameters():
            p.requires_grad_("adapter_" in n_)            # reference adapters (e.g. FM4NPP layer mix) stay trainable
        backbone.eval()

    lr = args.lr or {"scratch": 3e-4, "finetune": 2e-4, "frozen": 1e-3, "linear": 3e-3}[args.mode]
    if args.mode == "finetune":
        opt = torch.optim.AdamW(layer_groups(model, lr, args.layer_decay), lr=lr)
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05)
        for g in opt.param_groups:
            g["lr_scale"] = 1.0
    steps = int(np.clip(args.epochs * len(tr) / args.bs, args.min_steps, args.max_steps))
    class_w = None
    if args.task in ("evt_nu", "flavor"):
        cnt = np.bincount(target[tr], minlength=3).astype(np.float64)
        class_w = torch.tensor((cnt.sum() / np.maximum(cnt, 1) / 3) ** 0.5, dtype=torch.float32, device=dev)

    mk = lambda idx, shuffle, bs: torch.utils.data.DataLoader(
        EventDataset(corpus, idx, truth=(args.task == "hit")), batch_size=bs, shuffle=shuffle,
        collate_fn=collate, num_workers=args.workers, pin_memory=True, drop_last=shuffle,
        persistent_workers=shuffle)
    if len(tr) < 4:
        raise SystemExit(f"too few labeled training events ({len(tr)})")
    trl = mk(tr, True, max(1, min(args.bs, max(2, len(tr) // 2)) // args.accum))
    val = mk(va, False, max(1, args.bs // args.accum))
    tel = mk(te, False, max(1, args.bs // args.accum))
    yt = torch.from_numpy(target).to(dev) if target is not None else None
    info = dict(vars(args), n_train=len(tr), n_val=len(va), n_test=len(te), steps=steps,
                n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(json.dumps(info), flush=True)

    best, best_step, step, logs = -1e9, 0, 0, []
    ck = os.path.join(args.out, "best.pt")
    st = os.path.join(args.out, "state.pt")
    if args.resume and os.path.exists(st):
        s = torch.load(st, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step, best, best_step, logs = s["step"], s["best"], s["best_step"], s["logs"]
        print(f"resumed at step {step}", flush=True)
    t_state = time.time()

    def save_state():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best=best,
                        best_step=best_step, logs=logs), st + ".tmp")
        os.replace(st + ".tmp", st)
    model.train()
    if args.mode in ("frozen", "linear"):
        backbone.eval()
    done = False
    while not done:
        micro = 0
        for b in trl:
            b = to_device(b, dev)
            if micro == 0:
                for g in opt.param_groups:
                    w = min(1.0, (step + 1) / max(1, steps // 20))
                    g["lr"] = lr * g["lr_scale"] * w * 0.5 * (1 + np.cos(np.pi * step / steps))
                opt.zero_grad(set_to_none=True)
            y = yt[b["gidx"]] if yt is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = model(b)
            loss = loss_fn(args.task, out.float(), b, y, class_w)
            (loss / args.accum).backward()
            micro += 1
            if micro < args.accum:
                continue
            micro = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.eval_every == 0 or step == steps:
                m = metrics_for(args.task, predict(model, val, dev, args.task, labels_of))
                logs.append({"step": step, "loss": float(loss), **m})
                print(json.dumps(logs[-1]), flush=True)
                # NaN scores (e.g. a one-class val batch) never count as best, but a first checkpoint always exists
                if np.isfinite(m["score"]) and m["score"] > best or not os.path.exists(ck):
                    if np.isfinite(m["score"]):
                        best, best_step = m["score"], step
                    torch.save({k: v for k, v in model.state_dict().items()
                                if args.mode not in ("frozen", "linear") or not k.startswith("backbone.") or "adapter_" in k}, ck)
                if args.resume and step < steps and (time.time() - t_state) / 60 > args.state_every:
                    save_state()
                    t_state = time.time()
            if step >= steps or (time.time() - t0) / 60 > args.time_limit:
                if args.resume and step < steps:
                    save_state()
                    print(f"time limit at step {step}/{steps}: state saved, exiting for resume", flush=True)
                    return
                done = True
                break
    if not os.path.exists(ck):                        # stopped before the first evaluation: keep the last state
        torch.save({k: v for k, v in model.state_dict().items()
                    if args.mode not in ("frozen", "linear") or not k.startswith("backbone.") or "adapter_" in k}, ck)
    sd = torch.load(ck, map_location=dev)
    model.load_state_dict(sd, strict=False)
    res = predict(model, tel, dev, args.task, labels_of)
    m = metrics_for(args.task, res)
    np.savez_compressed(os.path.join(args.out, "preds_test.npz"), **{k: v for k, v in res.items() if v is not None})
    out = dict(info, best_val=best, best_step=best_step, final_step=step, truncated=step < steps, test=m,
               minutes=(time.time() - t0) / 60, val_log=logs)
    json.dump(out, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    if os.path.exists(st):      # resume state is no longer needed (fine-tune states are ~0.25 GB)
        os.remove(st)
    print("TEST", json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
