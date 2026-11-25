#!/usr/bin/env python
"""Profile the P3 pretraining step (#16): data-loader throughput vs model step time, on one GPU.

Usage (QLFM_DETECTOR=sbnd, pimm venv, compute node): profile_p3.py CORPUS SPLITS [--bs 64]
"""
import argparse
import json
import time

import numpy as np
import torch

from qlfm.data.dataset import Corpus, EventDataset, collate, to_device
from qlfm.models.fm import FM, FMCfg, MAE, MAECfg
from qlfm.tokenize import TokCfg

ap = argparse.ArgumentParser()
ap.add_argument("corpus")
ap.add_argument("splits")
ap.add_argument("--bs", type=int, default=64)
a = ap.parse_args()
c = Corpus(a.corpus)
idx = np.load(a.splits)["pre_f0.90"]
rng = np.random.default_rng(0)
for workers, truth in ((7, True), (14, True), (7, False)):
    ds = EventDataset(c, rng.permutation(idx)[:a.bs * 12], truth=truth)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, collate_fn=collate, num_workers=workers, shuffle=False)
    it = iter(dl)
    next(it)
    t0 = time.time()
    n = sum(1 for _ in it)
    print(json.dumps(dict(test="loader", workers=workers, truth=truth, ev_per_s=n * a.bs / (time.time() - t0))), flush=True)

dev = torch.device("cuda")
ds = EventDataset(c, rng.permutation(idx)[:a.bs], truth=True)
b = to_device(collate([ds[i] for i in range(a.bs)]), dev)
configs = {
    "full": dict(charge_stem="cnn", pos3d=True, pmt_pos="v2", use_blobs=True),
    "no_cnn": dict(charge_stem="mlp", pos3d=True, pmt_pos="v2", use_blobs=True),
    "no_blobs": dict(charge_stem="cnn", pos3d=True, pmt_pos="v2", use_blobs=False),
    "baseline": dict(),
}
for name, extra in configs.items():
    fm = FM(FMCfg(d=384, depth=12, heads=6, tok=TokCfg(), pos_mode="centroid", pos_freq=10, **extra)).to(dev)
    mae = MAE(fm, MAECfg(aux3d_weight=1.0 if extra.get("use_blobs") or extra.get("pos3d") else 0.0)).to(dev)
    opt = torch.optim.AdamW(mae.parameters(), lr=1e-5)
    for i in range(13):
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mae(b)
        out["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 10
    print(json.dumps(dict(test="model", config=name, s_per_step=round(dt, 3), ev_per_s_per_gpu=round(a.bs / dt, 1),
                          mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))), flush=True)
    del fm, mae, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
