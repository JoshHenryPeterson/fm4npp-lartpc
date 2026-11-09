"""Multi-task downstream training: ONE backbone, one head per task (T1-T5), trained together (single GPU).

python -m qlfm.train_multitask --mode finetune --fm RUN/fm.pt --label-frac 0.01 --out DIR

This is how a foundation model is deployed: one backbone shared by all tasks. Heads, losses, val/test subsets,
predictions and metrics are those of qlfm.train_downstream, so every task can be compared with its single-task run:
  * training events: the labelled pool; each task's loss uses only the events eligible for that task
  * T5 (light) sees charge only: a light-pretrained qlfm backbone runs a second, light-off pass for it
  * loss balancing: each task loss is divided by a running mean of itself, so all tasks start at weight ~1
  * model selection (--select): last = the final checkpoint of the cosine schedule (default; a per-task best
    would not be one backbone, and relative score changes are dominated by tasks that start near 0);
    rel = best mean over tasks of the val-score change since the first evaluation / max(|first|, 0.1)
  * test: the single-task test subset of every task (same seeds and sizes); OUT/<task>/preds_test.npz and
    OUT/<task>/metrics.json have the single-task format, OUT/metrics.json collects all tasks
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
from qlfm.models.heads import AttnPool, HitHead, LateLight, LightHead, XFusion
from qlfm.ref import build_backbone
from qlfm.train_downstream import layer_groups, light_target

TASKS = ("hit", "evt_nu", "flavor", "energy", "light")


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.expandvars("$QLFM_WORK/cache/bnb,$QLFM_WORK/cache/nue"))
    ap.add_argument("--splits", default=os.path.expandvars("$QLFM_WORK/cache/splits_all.npz"))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--mode", required=True, choices=["frozen", "finetune"])
    ap.add_argument("--fm", required=True, help="pretrained checkpoint")
    ap.add_argument("--backbone", default="qlfm", choices=["qlfm", "fm4npp", "fm4npp_l", "panda", "panda_l", "polarmae"])
    ap.add_argument("--late-light", type=int, default=0, help="as train_downstream (2 = strong late fusion)")
    ap.add_argument("--hit-feat", default="native", choices=["native", "raw", "none"])
    ap.add_argument("--pimm-prep", default="matched", choices=["matched", "legacy"])
    ap.add_argument("--width", default="native", choices=["native", "meas", "const"],
                    help="hit widths in training and evaluation (#9): meas = simulated nu hits rescaled to data "
                         "(qlfm.eval_perturb w_meas), const = every hit the same width (no width information)")
    ap.add_argument("--max-hits", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--label-frac", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--layer-decay", type=float, default=0.9)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=2000, help="val events per task")
    ap.add_argument("--n-test", type=int, default=20000, help="test events per task (single-task default)")
    ap.add_argument("--n-test-light", type=int, default=10000, help="T5 test events (round-4 T5 runs)")
    ap.add_argument("--loss-ema", type=float, default=0.99)
    ap.add_argument("--select", default="last", choices=["last", "rel"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=50.0)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--state-every", type=float, default=20.0)
    # scratch-only architecture args read by build_backbone (unused for checkpoints)
    ap.add_argument("--light", type=int, default=1)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--pos-mode", default="centroid")
    ap.add_argument("--pos-freq", type=int, default=10)
    ap.add_argument("--cell-wires", type=int, default=16)
    ap.add_argument("--cell-ticks", type=int, default=64)
    return ap.parse_args(argv)


class MultiTaskModel(nn.Module):
    """Shared backbone (+ optional shared late light) and one head per task."""

    def __init__(self, backbone, tasks, mode, late_light=0, hit_feat="native"):
        super().__init__()
        self.backbone, self.tasks, self.mode, self.hit_feat = backbone, tasks, mode, hit_feat
        self.fusion = XFusion(backbone.d) if late_light == 2 else None
        d = self.fusion.d if self.fusion is not None else backbone.d
        self.light = LateLight(d) if late_light else None
        from qlfm.models.heads import RawHit
        self.raw_hit = RawHit() if ("hit" in tasks and hit_feat == "raw") else None
        hit_dim = {"native": backbone.hit_dim, "raw": 64, "none": 0}[hit_feat]
        heads = {}
        for t in tasks:
            if t == "hit":
                heads[t] = HitHead(d, hit_dim, extra_dim=d if late_light else 0)
            elif t == "light":
                heads[t] = LightHead(backbone.d)
            else:
                heads[t] = AttnPool(d, out_dim=N_OUT[t])
        self.heads = nn.ModuleDict(heads)
        # T5 must see charge only: a backbone with light in its input gets a separate light-off pass
        self.light_off_pass = "light" in tasks and bool(getattr(backbone, "light_on", False))

    def train(self, mode=True):
        super().train(mode)
        if self.mode == "frozen":
            self.backbone.eval()
        return self

    def _bb(self, b):
        frozen_all = self.mode == "frozen" and not any(p.requires_grad for p in self.backbone.parameters())
        with torch.no_grad() if frozen_all else contextlib.nullcontext():
            return self.backbone(b)

    def forward(self, b):
        if getattr(self, "width", "native") != "native":
            from qlfm.eval_perturb import perturb
            b = perturb(b, "w_" + self.width, 0.0)
        out = {}
        need_main = any(t != "light" for t in self.tasks)
        if need_main or not self.light_off_pass:
            o = self._bb(b)
            z, evt = o["z"], o["evt"]
            if "light" in self.tasks and not self.light_off_pass:      # charge-only backbone: before late light
                out["light"] = self.heads["light"](z, evt, b["n_events"])
        if need_main:
            if self.light is not None:
                zl, levt, lpool = self.light(b)
            if self.fusion is not None:
                z = self.fusion(z, evt, zl, b["n_events"])
            if "hit" in self.tasks:
                extra = lpool[b["hit_evt"]] if self.light is not None else None
                h_hit = self.raw_hit(b) if self.raw_hit is not None else (o["h_hit"] if self.hit_feat == "native" else None)
                out["hit"] = self.heads["hit"](h_hit, z, o["hit2tok"], extra)
            if self.light is not None:
                z, evt = torch.cat([z.to(zl.dtype), zl]), torch.cat([evt, levt])
            for t in self.tasks:
                if t not in ("hit", "light"):
                    out[t] = self.heads[t](z, evt, b["n_events"])
        if self.light_off_pass:
            self.backbone.light_on = False
            try:
                o2 = self._bb(b)
            finally:
                self.backbone.light_on = True
            out["light"] = self.heads["light"](o2["z"], o2["evt"], b["n_events"])
        return out


def task_losses(out, b, lab, class_w):
    """Per-task losses on the events of the batch eligible for each task (tasks without such events are skipped)."""
    g = b["gidx"]
    losses = {}
    for t, o in out.items():
        o = o.float()
        elig, y = lab[t]
        m = elig[g]
        if t == "light":
            tgt = light_target(b)
            losses[t] = F.mse_loss(o, tgt) + F.mse_loss(o[..., -1], tgt[..., -1])
        elif t == "hit":
            mh = m[b["hit_evt"]]
            if not mh.any():
                continue
            tgt = (b["hit_nu_edep"][mh] > HIT_NU_EDEP_MEV).float()
            w = b["hit_q"][mh].clamp(min=1.0).log1p()
            losses[t] = (F.binary_cross_entropy_with_logits(o[mh], tgt, reduction="none") * w).sum() / w.sum()
        else:
            if not m.any():
                continue
            if t in ("evt_nu", "flavor"):
                losses[t] = F.cross_entropy(o[m], y[g][m], weight=class_w[t])
            else:
                losses[t] = F.huber_loss(o[m].squeeze(1), y[g][m].float(), delta=0.5)
    return losses


@torch.no_grad()
def predict(model, loader, dev, sets, lab_np):
    """Predictions of every task on its own subset (sets[t]: sorted gidx array) of the union loader."""
    model.eval()
    acc = {t: {"pred": [], "gidx": [], "hit_q": [], "hit_pnd": [], "hit_evt": [], "hit_tgt": [], "light_tgt": []}
           for t in sets}
    for b in loader:
        b = to_device(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        g = b["gidx"]
        gn = g.cpu().numpy()
        for t in sets:
            sel = torch.from_numpy(np.isin(gn, sets[t])).to(dev)
            if not sel.any():
                continue
            o = out[t].float()
            a = acc[t]
            if t == "hit":
                mh = sel[b["hit_evt"]]
                a["pred"].append(torch.sigmoid(o[mh]).cpu())
                a["hit_q"].append(b["hit_q"][mh].cpu())
                a["hit_pnd"].append(b["hit_pnd_nu"][mh].cpu())
                a["hit_tgt"].append((b["hit_nu_edep"][mh] > HIT_NU_EDEP_MEV).cpu())
                a["hit_evt"].append(g[b["hit_evt"]][mh].cpu())
            elif t == "light":
                a["pred"].append(o[sel].half().cpu())
                a["light_tgt"].append(light_target(b)[sel].half().cpu())
            else:
                a["pred"].append(o[sel].cpu())
            a["gidx"].append(g[sel].cpu())
    model.train()
    res = {}
    for t, a in acc.items():
        gidx = torch.cat(a["gidx"]).numpy()
        r = {"gidx": gidx, "pred": torch.cat(a["pred"]).numpy()}
        if t in ("hit", "light"):
            r.update({k: torch.cat(v).numpy() for k, v in a.items() if v and k not in ("pred", "gidx")})
        else:
            r["y"] = lab_np[t][1][gidx]
        res[t] = r
    return res


def main(argv=None):
    args = get_args(argv)
    tasks = [t for t in args.tasks.split(",") if t]
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    corpus = Corpus(args.corpus)
    sp = np.load(args.splits)
    ev = {k: corpus.ev_array(k) for k in EV_KEYS}
    lab_np = event_labels(ev)
    lab_np["light"] = (lab_np["light"][0], None)
    lab_np["hit"] = (lab_np["hit"][0], None)
    lab = {t: (torch.from_numpy(lab_np[t][0]).to(dev),
               torch.from_numpy(lab_np[t][1]).to(dev) if lab_np[t][1] is not None else None) for t in tasks}

    pool = sp[f"lab_f{args.label_frac:g}"]
    any_elig = np.zeros(len(lab_np["light"][0]), bool)
    for t in tasks:
        any_elig |= lab_np[t][0]
    tr = pool[any_elig[pool]]
    # identical subsets to train_downstream (same seeds, per-task eligibility)
    pick = lambda idx, n, seed: np.sort(np.random.default_rng(seed).permutation(idx)[:n])
    va = {t: pick(sp["val"][lab_np[t][0][sp["val"]]], args.n_val, 11) for t in tasks}
    te = {t: pick(sp["test"][lab_np[t][0][sp["test"]]], args.n_test_light if t == "light" else args.n_test, 12)
          for t in tasks}

    backbone = build_backbone(args.backbone, args, args.fm)
    model = MultiTaskModel(backbone, tasks, args.mode, args.late_light, args.hit_feat).to(dev)
    model.width = args.width
    if args.mode == "frozen":
        for n_, p in backbone.named_parameters():
            p.requires_grad_("adapter_" in n_)
        backbone.eval()
    lr = args.lr or {"finetune": 2e-4, "frozen": 1e-3}[args.mode]
    if args.mode == "finetune":
        opt = torch.optim.AdamW(layer_groups(model, lr, args.layer_decay), lr=lr)
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05)
        for g in opt.param_groups:
            g["lr_scale"] = 1.0
    steps = args.max_steps
    class_w = {}
    for t in tasks:
        if t in ("evt_nu", "flavor"):
            trt = tr[lab_np[t][0][tr]]
            cnt = np.bincount(lab_np[t][1][trt], minlength=3).astype(np.float64)
            class_w[t] = torch.tensor((cnt.sum() / np.maximum(cnt, 1) / 3) ** 0.5, dtype=torch.float32, device=dev)

    mk = lambda idx, shuffle, bs: torch.utils.data.DataLoader(
        EventDataset(corpus, idx, truth=True), batch_size=bs, shuffle=shuffle, collate_fn=collate,
        num_workers=args.workers, pin_memory=True, drop_last=shuffle, persistent_workers=shuffle)
    mbs = max(1, args.bs // args.accum)
    trl = mk(tr, True, max(1, min(args.bs, max(2, len(tr) // 2)) // args.accum))
    val = mk(np.unique(np.concatenate(list(va.values()))), False, mbs)
    tel = mk(np.unique(np.concatenate(list(te.values()))), False, mbs)
    info = dict(vars(args), tasks=tasks, n_train=len(tr), n_train_task={t: int(lab_np[t][0][tr].sum()) for t in tasks},
                n_val={t: len(v) for t, v in va.items()}, n_test={t: len(v) for t, v in te.items()}, steps=steps,
                n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(json.dumps(info), flush=True)

    best, best_step, step, logs, ref, ema = -1e9, 0, 0, [], None, {}
    ck, st = os.path.join(args.out, "best.pt"), os.path.join(args.out, "state.pt")
    keep = lambda k: args.mode != "frozen" or not k.startswith("backbone.") or "adapter_" in k
    if args.resume and os.path.exists(st):
        s = torch.load(st, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step, best, best_step, logs, ref, ema = s["step"], s["best"], s["best_step"], s["logs"], s["ref"], s["ema"]
        print(f"resumed at step {step}", flush=True)
    t_state = time.time()

    def save_state():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best=best, best_step=best_step,
                        logs=logs, ref=ref, ema=ema), st + ".tmp")
        os.replace(st + ".tmp", st)

    model.train()
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b)
            losses = task_losses(out, b, lab, class_w)
            for t, l in losses.items():               # running loss scale per task
                v = float(l.detach())
                ema[t] = v if t not in ema else args.loss_ema * ema[t] + (1 - args.loss_ema) * v
            loss = sum(l / max(ema[t], 1e-6) for t, l in losses.items()) / max(len(losses), 1)
            (loss / args.accum).backward()
            micro += 1
            if micro < args.accum:
                continue
            micro = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.eval_every == 0 or step == steps:
                res = predict(model, val, dev, va, lab_np)
                m = {t: metrics_for(t, res[t]) for t in tasks}
                sc = {t: m[t]["score"] for t in tasks}
                if ref is None:
                    ref = sc
                rel = [(sc[t] - ref[t]) / max(abs(ref[t]), 0.1) for t in tasks if np.isfinite(sc[t]) and np.isfinite(ref[t])]
                sel = float(np.mean(rel)) if rel else float("nan")
                logs.append({"step": step, "sel": sel, **{f"loss_{t}": float(l.detach()) for t, l in losses.items()},
                             **{f"score_{t}": sc[t] for t in tasks}})
                print(json.dumps(logs[-1]), flush=True)
                if args.select == "last" or np.isfinite(sel) and sel > best or not os.path.exists(ck):
                    if np.isfinite(sel) or args.select == "last":
                        best, best_step = sel, step
                    torch.save({k: v for k, v in model.state_dict().items() if keep(k)}, ck)
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
    if not os.path.exists(ck) or args.select == "last":
        torch.save({k: v for k, v in model.state_dict().items() if keep(k)}, ck)
        best_step = step
    model.load_state_dict(torch.load(ck, map_location=dev), strict=False)
    res = predict(model, tel, dev, te, lab_np)
    allm = {}
    for t in tasks:
        m = metrics_for(t, res[t])
        allm[t] = m
        d = os.path.join(args.out, t)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "preds_test.npz"), **{k: v for k, v in res[t].items() if v is not None})
        json.dump(dict(task=t, multitask=True, test=m, best_step=best_step, final_step=step, n_test=len(te[t])),
                  open(os.path.join(d, "metrics.json"), "w"), indent=1)
    out = dict(info, best_sel=best, best_step=best_step, final_step=step, truncated=step < steps, test=allm,
               minutes=(time.time() - t0) / 60, val_log=logs)
    json.dump(out, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    if os.path.exists(st):
        os.remove(st)
    print("TEST", json.dumps({t: allm[t]["score"] for t in tasks}), flush=True)


if __name__ == "__main__":
    main()
