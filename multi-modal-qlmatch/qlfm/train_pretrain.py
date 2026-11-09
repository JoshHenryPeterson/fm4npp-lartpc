"""Multi-modal MAE pretraining (step-based, resumable, time-limited for 1 h debug jobs).

Launch: mpiexec -n NGPU --ppn 4 python -m qlfm.train_pretrain --out RUN_DIR [...]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from qlfm import dist
from qlfm.data.dataset import Corpus, EventDataset, collate, to_device
from qlfm.models.fm import FM, MAE, FMCfg, MAECfg
from qlfm.tokenize import TokCfg


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.expandvars("$QLFM_WORK/cache/bnb,$QLFM_WORK/cache/nue"))
    ap.add_argument("--splits", default=os.path.expandvars("$QLFM_WORK/cache/splits_all.npz"))
    ap.add_argument("--subset", default="train", help="key in the splits file (e.g. train, pre_f0.10)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--light", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32, help="events per GPU")
    ap.add_argument("--lr", type=float, default=1.5e-4, help="per 256 events")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="absolute warmup length; default warmup * steps, fixed at the first start (kept on resume, "
                         "so --budget-h cannot move it)")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--pos-mode", default="centroid")
    ap.add_argument("--pos-freq", type=int, default=10)
    ap.add_argument("--cell-wires", type=int, default=16)
    ap.add_argument("--cell-ticks", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--mask-ratio", type=float, default=0.6)
    ap.add_argument("--light-weight", type=float, default=1.0)
    ap.add_argument("--strip-frac", type=float, default=0.5)
    ap.add_argument("--target-norm", type=int, default=1)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--time-limit", type=float, default=50.0, help="minutes before checkpoint+exit")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=2048)
    ap.add_argument("--val-pick", default="random", choices=["random", "first"],
                    help="random: fixed random subset of the val split (BNB + nue); first: the first n-val indices "
                         "(rounds 1-4; BNB only, since the corpus is concatenated BNB then nue)")
    ap.add_argument("--seed", type=int, default=0)
    # reference methods (round 2)
    ap.add_argument("--method", default="mae", choices=["mae", "fm4npp", "fm4npp_l", "sd", "jepa"])
    ap.add_argument("--lr-abs", type=float, default=None, help="absolute lr (no batch scaling)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-hits", type=int, default=16384)
    ap.add_argument("--klen", type=int, default=30)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--keep-frac", default="",
                    help="comma- or colon-separated (qsub -v) fractions of the final step count at which fm.pt is also kept as "
                         "fm_s<step>.pt (convergence probes), e.g. 0.1,0.25,0.5")
    ap.add_argument("--jepa-layers", type=int, default=1, help="JEPA: targets = mean of the top K layer outputs")
    ap.add_argument("--jepa-ema", type=float, default=0.996, help="JEPA: initial EMA momentum (cosine to 1)")
    ap.add_argument("--jepa-tnorm", type=int, default=0, help="JEPA: standardise targets per dimension")
    ap.add_argument("--jepa-vreg", type=float, default=0.0, help="JEPA: VICReg variance/covariance weight")
    ap.add_argument("--jepa-pred-pos", default="centroid", choices=["centroid", "cell"],
                    help="JEPA: position input of masked charge tokens to the predictor")
    ap.add_argument("--budget-h", type=float, default=None,
                    help="training-hour budget: after 200 steps the measured rate fixes --steps (kept on resume)")
    return ap.parse_args(argv)


def build(args):
    if args.method == "fm4npp":
        from qlfm.ref.fm4npp import FM4NPPPretrain, MambaGPT
        net = MambaGPT(d=args.d, n_layers=args.depth, d_state=args.d_state, klen=args.klen, dropout=args.dropout)
        return net, FM4NPPPretrain(net, args.max_hits)
    if args.method == "fm4npp_l":
        from qlfm.ref.fm4npp_light import FM4NPPLightPretrain, MambaGPTLight
        net = MambaGPTLight(d=args.d, n_layers=args.depth, d_state=args.d_state, klen=args.klen, dropout=args.dropout)
        return net, FM4NPPLightPretrain(net, args.max_hits, light_weight=args.light_weight)
    tok = TokCfg(cell_wires=args.cell_wires, cell_ticks=args.cell_ticks, max_tokens=args.max_tokens)
    fm = FM(FMCfg(d=args.d, depth=args.depth, heads=args.heads, use_light=bool(args.light), tok=tok,
                  pos_mode=args.pos_mode, pos_freq=args.pos_freq))
    if args.method == "sd":
        from qlfm.models.sd import SelfDistill
        return fm, SelfDistill(fm)
    mcfg = MAECfg(mask_ratio=args.mask_ratio, light_weight=args.light_weight, strip_frac=args.strip_frac,
                  target_norm=bool(args.target_norm))
    if args.method == "jepa":
        from qlfm.models.jepa import JEPA
        return fm, JEPA(fm, mcfg, momentum=(args.jepa_ema, 1.0), target_layers=args.jepa_layers,
                        tnorm=bool(args.jepa_tnorm), vreg=args.jepa_vreg, pred_pos=args.jepa_pred_pos)
    return fm, MAE(fm, mcfg)


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, n, seed):
        self.n, self.seed = n, seed

    def __iter__(self):
        g = np.random.default_rng(self.seed)
        while True:
            yield from g.permutation(self.n).tolist()


def lr_at(step, args, base):
    w = args.warmup_steps
    if step < w:
        return base * (step + 1) / w
    p = (step - w) / max(1, args.steps - w)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def validate(mae, loader, dev, n_batches):
    mae.eval()
    state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(1234)                      # identical masks every validation
    acc, cnt = {}, {}                                  # per-key sums and counts: some losses are not in every batch
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mae(to_device(b, dev))
        for key, v in out.items():
            if key.startswith("loss"):
                acc[key] = acc.get(key, 0.0) + float(v)
                cnt[key] = cnt.get(key, 0) + 1
    torch.cuda.set_rng_state(state)
    mae.train()
    return dist.all_reduce_keyed_mean(acc, cnt, dev)


def main(argv=None):
    args = get_args(argv)
    t_start = time.time()
    rank, world, local = dist.init()
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed + rank)

    corpus = Corpus(args.corpus)
    sp = np.load(args.splits)
    train_idx = sp[args.subset]
    if args.val_pick == "random":
        val_idx = np.sort(np.random.default_rng(11).permutation(sp["val"])[: args.n_val])
    else:
        val_idx = sp["val"][: args.n_val]
    fm, mae = build(args)
    mae.to(dev)
    n_params = sum(p.numel() for p in fm.parameters())
    base_lr = args.lr_abs if args.lr_abs else args.lr * args.bs * world / 256
    decay = [p for n, p in mae.named_parameters() if p.requires_grad and p.ndim >= 2 and "pos" not in n]
    no_decay = [p for n, p in mae.named_parameters() if p.requires_grad and (p.ndim < 2 or "pos" in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=base_lr, betas=(0.9, 0.95))

    step = 0
    ck = os.path.join(args.out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)
        mae.load_state_dict(s["mae"])
        opt.load_state_dict(s["opt"])
        step = s["step"]
        if s.get("steps_final"):
            args.steps = s["steps_final"]
        if s["args"].get("warmup_steps"):
            args.warmup_steps = s["args"]["warmup_steps"]
    if not args.warmup_steps:                         # fixed once; later --budget-h changes of steps keep it
        args.warmup_steps = int(args.warmup * args.steps)
    if step >= args.steps:
        if dist.is_main():
            print(f"already done ({step} steps)", flush=True)
        return
    model = DDP(mae, device_ids=[dev.index], find_unused_parameters=True) if world > 1 else mae

    ds = EventDataset(corpus, train_idx)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.bs, sampler=InfiniteSampler(len(ds), args.seed * 1000 + rank * 7 + step),
        collate_fn=collate, num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    vds = EventDataset(corpus, val_idx[rank::world])
    vloader = torch.utils.data.DataLoader(vds, batch_size=args.bs, collate_fn=collate, num_workers=2)
    n_val_batches = len(vds) // args.bs

    if dist.is_main():
        print(f"params(FM)={n_params/1e6:.2f}M world={world} train={len(train_idx)} lr={base_lr:.2e} "
              f"start_step={step}", flush=True)
        json.dump(vars(args) | {"n_params": n_params, "world": world, "n_train": len(train_idx)},
                  open(os.path.join(args.out, "args.json"), "w"), indent=1)
    logf = open(os.path.join(args.out, "log.jsonl"), "a") if dist.is_main() else None

    def save():
        if dist.is_main():
            torch.save({"mae": mae.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args),
                        "steps_final": steps_final}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            # self-distillation: the EMA teacher is the downstream backbone (DINO / Sonata / Panda convention)
            fm_save = mae.teacher if args.method in ("sd", "jepa") else fm
            torch.save({"fm": fm_save.state_dict(), "args": vars(args), "step": step}, os.path.join(args.out, "fm.pt"))

    keep_steps = lambda: sorted({int(float(f) * args.steps) for f in args.keep_frac.replace(":", ",").split(",") if f})

    def keep_snapshot():
        if dist.is_main():
            fm_save = mae.teacher if args.method in ("sd", "jepa") else fm
            torch.save({"fm": fm_save.state_dict(), "args": vars(args), "step": step},
                       os.path.join(args.out, f"fm_s{step}.pt"))
            print(f"kept snapshot fm_s{step}.pt", flush=True)

    steps_final = args.steps if (os.path.exists(ck) and torch.load(ck, map_location="cpu", weights_only=False).get(
        "steps_final")) else None
    t_rate = None
    t_log, ev_log, acc, acc_n = time.time(), 0, {}, {}
    model.train()
    it = iter(loader)
    while step < args.steps:
        b = to_device(next(it), dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args, base_lr)
        if args.budget_h and steps_final is None:
            if step == 50:
                t_rate = time.time()
            elif step == 200:
                rate = torch.tensor(150.0 / (time.time() - t_rate), device=dev)
                if world > 1:
                    torch.distributed.broadcast(rate, 0)
                steps_final = int(float(rate) * args.budget_h * 3600)
                args.steps = steps_final
                if dist.is_main():
                    print(f"[budget] {float(rate):.2f} steps/s -> steps = {steps_final}", flush=True)
        mae.progress = step / args.steps
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(mae.parameters(), args.clip)
        opt.step()
        if hasattr(mae, "ema_update"):
            mae.ema_update()
        step += 1
        if args.keep_frac and (steps_final is not None or not args.budget_h) and step in keep_steps():
            keep_snapshot()
        ev_log += b["n_events"]
        for k, v in list(out.items()) + [("gnorm", gn)]:
            acc[k] = acc.get(k, 0.0) + float(v.detach())
            acc_n[k] = acc_n.get(k, 0) + 1             # per key: some losses are not in every batch

        if step % args.log_every == 0:
            dt = time.time() - t_log
            rec = {k: v / acc_n[k] for k, v in acc.items()}
            rec.update(step=step, lr=opt.param_groups[0]["lr"], ev_per_s=ev_log * world / dt,
                       mem_gb=torch.cuda.max_memory_allocated() / 1e9, wall_min=(time.time() - t_start) / 60)
            if dist.is_main():
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
            t_log, ev_log, acc, acc_n = time.time(), 0, {}, {}
        if step % args.val_every == 0 or step == args.steps:
            v = validate(mae, vloader, dev, n_val_batches)
            if dist.is_main():
                rec = {"step": step, **{"val_" + k: x for k, x in v.items()}}
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
            save()
        # time limit: decided on rank 0, broadcast so all ranks stop together
        stop = torch.tensor(float((time.time() - t_start) / 60 > args.time_limit), device=dev)
        if world > 1:
            torch.distributed.broadcast(stop, 0)
        if stop.item():
            break
    save()
    if dist.is_main():
        print(f"stopped at step {step}", flush=True)
        if step >= args.steps:
            open(os.path.join(args.out, "done"), "w").write(f"{step}\n")
    dist.barrier()


if __name__ == "__main__":
    main()
