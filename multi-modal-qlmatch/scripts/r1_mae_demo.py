#!/usr/bin/env python
"""Round-1 MAE input/output demo (GPU): run the pretrained multi-modal MAE on one event and save the masks,
targets and reconstructions for the tech-note figures (scripts/r1_io_figures.py).

The event is batched with filler test events so that the per-batch target standardization matches training.
Charge masking follows the training recipe (60 %: cross-plane drift strips + random); the light of the demo
event is fully masked (charge -> light prediction). Predictions are mapped back to log1p units.

Usage: r1_mae_demo.py CKPT GIDX OUT.npz [N_FILLER]
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from qlfm.data.dataset import Corpus, EventDataset, collate, to_device
from qlfm.models.encoder import pack
from qlfm.models.fm import FM, MAE, FMCfg, MAECfg
from qlfm.tokenize import TokCfg
from qlfm.paths import WORK as W



def main():
    ckpt, gidx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    n_fill = int(sys.argv[4]) if len(sys.argv) > 4 else 31
    torch.manual_seed(7)
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = s["args"]
    tok = TokCfg(cell_wires=a["cell_wires"], cell_ticks=a["cell_ticks"], max_tokens=a["max_tokens"])
    fm = FM(FMCfg(d=a["d"], depth=a["depth"], heads=a["heads"], use_light=bool(a["light"]), tok=tok,
                  pos_mode=a["pos_mode"], pos_freq=a["pos_freq"]))
    mae = MAE(fm, MAECfg(mask_ratio=a["mask_ratio"], strip_frac=a["strip_frac"], light_weight=a["light_weight"],
                         target_norm=bool(a["target_norm"])))
    mae.load_state_dict(s["mae"])
    dev = torch.device("cuda")
    mae = mae.to(dev).eval()

    c = Corpus(f"{W}/cache/bnb,{W}/cache/nue")
    sp = np.load(f"{W}/cache/splits_all.npz")
    fill = np.random.default_rng(3).choice(sp["test"], n_fill, replace=False)
    ds = EventDataset(c, np.concatenate([[gidx], fill]))
    b = to_device(collate([ds[i] for i in range(len(ds))]), dev)
    n = b["n_events"]

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ct, lt = fm.tokenize(b)
        xc, xl, _ = fm.embed(ct, lt)
        mc = mae.charge_mask(ct, n)
        ml, _ = mae.light_mask(lt, n)
        ml = ml | (lt.evt == 0)                                   # demo event: all PMTs masked
        vis = ~mc
        (zc, zl), _ = fm.encode(xc[vis], ct.tok_evt[vis], xl[~ml], lt.evt[~ml], n)
        dc = torch.empty(xc.shape[0], mae.cfg.dec_d, device=dev, dtype=zc.dtype)
        dc[vis] = mae.enc2dec(zc).to(dc.dtype)
        dc[mc] = mae.mask_c.to(dc.dtype)
        dc = dc + mae.dec_pos_c(fm.charge.pos_embed(ct))
        dl = torch.empty(xl.shape[0], mae.cfg.dec_d, device=dev, dtype=zc.dtype)
        if zl is not None and zl.shape[0]:
            dl[~ml] = mae.enc2dec(zl).to(dl.dtype)
        dl[ml] = mae.mask_l.to(dl.dtype)
        dl = dl + mae.dec_pos_l(fm.light.pos_embed(lt))
        perm, cu, maxlen, _ = pack([ct.tok_evt, lt.evt], n)
        y = mae.decoder(torch.cat([dc, dl])[perm], cu, maxlen)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=dev)
        nc = dc.shape[0]
        yc, yl = y[inv[:nc]], y[inv[nc:]]
        tgt_all = torch.cat([ct.tok_target, ct.tok_stat], 1).float()
        mu, sd = tgt_all[mc].mean(0), tgt_all[mc].std(0) + 1e-3
        pred_c = mae.head_c(yc).float() * sd + mu                 # back to log1p units (all tokens; use masked)
        pred_l = mae.head_l(yl).float()
        print("batch masked-charge MSE (standardized):",
              float(F.mse_loss((pred_c[mc] - mu) / sd, (tgt_all[mc] - mu) / sd)),
              " mean-predictor:", float(((tgt_all[mc] - mu) / sd).pow(2).mean()))

    e = ct.tok_evt == 0
    le = lt.evt == 0
    np.savez_compressed(out, gidx=gidx, tok_cell=ct.tok_cell[e].cpu().numpy(), tok_plane=ct.tok_plane[e].cpu().numpy(),
                        tok_cpos=ct.tok_cpos[e].cpu().numpy(), tok_target=ct.tok_target[e].float().cpu().numpy(),
                        tok_stat=ct.tok_stat[e].float().cpu().numpy(), mask_c=mc[e].cpu().numpy(),
                        tok_time=ct.tok_time[e].float().cpu().numpy(),
                        pred_c=pred_c[e].cpu().numpy(), light_feat=lt.feat[le].float().cpu().numpy(),
                        mask_l=ml[le].cpu().numpy(), pred_l=pred_l[le].cpu().numpy(), light_pmt=lt.pmt[le].cpu().numpy(),
                        sub=np.array([tok.sub_w, tok.sub_t]), light_edges=np.array(tok.light_edges))
    print("saved", out, "tokens", int(e.sum()), "masked", int(mc[e].sum()))


if __name__ == "__main__":
    main()
