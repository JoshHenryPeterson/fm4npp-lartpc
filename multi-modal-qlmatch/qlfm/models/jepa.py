"""qlfm-JEPA: joint-embedding predictive pretraining (I-JEPA, arXiv:2301.08243) on qlfm charge + light tokens.

Same tokens, backbone and masks as the multi-modal MAE (qlfm.models.fm.MAE), so MAE vs JEPA isolates the objective:
  * context encoder = the FM, on the visible tokens (charge strips + random cells masked; PMTs partly or all masked)
  * target encoder = EMA copy of the FM (momentum 0.996 -> 1, cosine), on the FULL event; targets are its final
    token features, layer-normalised (I-JEPA)
  * predictor = the MAE decoder stack (narrow Transformer, mask tokens + positional embeddings of the masked
    tokens), followed by a linear map to the target width
  * loss: MSE between predicted and target features of the masked charge tokens and masked PMT tokens
    (light weight as in the MAE); "all PMTs masked" events make it a charge -> light prediction in latent space
  * downstream backbone: the EMA target encoder (I-JEPA / DINO convention), saved as fm.pt
Collapse monitor: per-dimension std of the targets and of the predictions (logged as tgt_std, pred_std).
Option target_layers = K > 1: targets are the mean of the top K block outputs, each layer-normalised (data2vec,
arXiv:2202.03555), which keeps the targets spread across tokens in small models.
Option tnorm: targets standardised per dimension over the masked tokens of the batch (as the MAE charge targets),
so near-constant target dimensions cannot make the loss small.
Option vreg = w: VICReg (arXiv:2105.04906) variance hinge (std >= 1 per dimension) + covariance penalty on the
context-encoder features, weight w; an explicit guard against (dimensional) collapse.
Option pred_pos = "cell": the predictor gets only the cell centre of each masked charge token, not its
charge-weighted centroid and in-cell offset (which describe where the token's charge is; Josh Peterson's audit #7),
so a target cannot be inferred from its position input.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qlfm.models.encoder import pack
from qlfm.models.fm import MAE


class JEPA(MAE):
    def __init__(self, fm, cfg, momentum=(0.996, 1.0), target_layers=1, tnorm=False, vreg=0.0, pred_pos="centroid"):
        super().__init__(fm, cfg)
        self.target_layers, self.tnorm, self.vreg, self.pred_pos = target_layers, tnorm, vreg, pred_pos
        del self.head_c, self.head_l                            # no pixel-space targets
        self.head_p = nn.Linear(cfg.dec_d, fm.cfg.d)
        self.teacher = copy.deepcopy(fm)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.momentum = momentum
        self.progress = 0.0                                     # set by the training loop (step / steps)

    @torch.no_grad()
    def ema_update(self):
        m = self.momentum[0] + (self.momentum[1] - self.momentum[0]) * 0.5 * (1 - math.cos(math.pi * self.progress))
        for ps, pt in zip(self.fm.parameters(), self.teacher.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)

    def forward(self, b):
        fm, n = self.fm, b["n_events"]
        ct, lt = fm.tokenize(b)
        d = fm.cfg.d
        # ---- targets: EMA encoder on the full event ----
        with torch.no_grad():
            xc_t, xl_t, _ = self.teacher.embed(ct, lt)
            depth = len(self.teacher.encoder.blocks)
            if self.target_layers > 1:
                _, layers = self.teacher.encode(xc_t, ct.tok_evt, xl_t, lt.evt if lt is not None else None, n,
                                                return_layers=tuple(range(depth - self.target_layers, depth)))
                ln = lambda t: F.layer_norm(t.float(), (d,))
                tc = sum(ln(c) for c, _ in layers) / len(layers)
                tl = sum(ln(l) for _, l in layers) / len(layers) if lt is not None else None
            else:
                (zc_t, zl_t), _ = self.teacher.encode(xc_t, ct.tok_evt, xl_t, lt.evt if lt is not None else None, n)
                tc = F.layer_norm(zc_t.float(), (d,))
                tl = F.layer_norm(zl_t.float(), (d,)) if zl_t is not None else None

        xc, xl, _ = fm.embed(ct, lt)
        mc = self.charge_mask(ct, n)
        ml, lmode = self.light_mask(lt, n) if lt is not None else (None, None)

        # ---- context encoder on visible tokens ----
        vis_c = ~mc
        xl_vis = xl[~ml] if xl is not None else None
        levt_vis = lt.evt[~ml] if lt is not None else None
        (zc, zl), _ = fm.encode(xc[vis_c], ct.tok_evt[vis_c], xl_vis, levt_vis, n)

        # ---- predictor: visible features + mask tokens at the masked positions ----
        dc = torch.empty(xc.shape[0], self.cfg.dec_d, device=xc.device, dtype=zc.dtype)
        dc[vis_c] = self.enc2dec(zc).to(dc.dtype)
        dc[mc] = self.mask_c.to(dc.dtype)
        pe = fm.charge.pos_embed(ct)
        if self.pred_pos == "cell":                     # masked tokens: cell centre only
            ch = fm.charge
            coarse = ch.plane(ct.tok_plane) + ch.pos(ct.tok_pos) + ch.crel(torch.zeros_like(ct.tok_crel))
            pe = torch.where(mc[:, None], coarse.to(pe.dtype), pe)
        dc = dc + self.dec_pos_c(pe)
        srcs, evts = [dc], [ct.tok_evt]
        if lt is not None:
            dl = torch.empty(xl.shape[0], self.cfg.dec_d, device=xl.device, dtype=zc.dtype)
            if zl is not None and zl.shape[0]:
                dl[~ml] = self.enc2dec(zl).to(dl.dtype)
            dl[ml] = self.mask_l.to(dl.dtype)
            dl = dl + self.dec_pos_l(fm.light.pos_embed(lt))
            srcs.append(dl)
            evts.append(lt.evt)
        perm, cu, maxlen, _ = pack(evts, n)
        y = self.decoder(torch.cat(srcs)[perm], cu, maxlen)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        nc = dc.shape[0]
        yc = y[inv[:nc]]

        std_norm = lambda t: (t - t.mean(0)) / (t.std(0) + 1e-3)
        pred_c = self.head_p(yc[mc]).float()
        tgt_c = std_norm(tc[mc]) if self.tnorm else tc[mc]
        loss_c = F.mse_loss(pred_c, tgt_c)
        loss = loss_c
        out = dict(loss_c=loss_c.detach(), n_tok=torch.tensor(float(nc), device=y.device),
                   tgt_std=tc.std(0).mean().detach(), pred_std=pred_c.std(0).mean().detach())
        if lt is not None and ml.any():
            yl = y[inv[nc:]]
            pred_l = self.head_p(yl[ml]).float()
            tgt_l = std_norm(tl[ml]) if self.tnorm else tl[ml]
            loss_l = F.mse_loss(pred_l, tgt_l)
            loss = loss + self.cfg.light_weight * loss_l
            out["loss_l"] = loss_l.detach()
            allm = (lmode[lt.evt] == 2)[ml]
            if allm.any():
                out["loss_l_all"] = F.mse_loss(pred_l[allm], tgt_l[allm]).detach()
        if self.vreg > 0:
            zs = zc.float()
            zs = zs - zs.mean(0)
            std = torch.sqrt(zs.var(0) + 1e-4)
            l_var = F.relu(1.0 - std).mean()
            cov = (zs.t() @ zs) / max(zs.shape[0] - 1, 1)
            l_cov = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / d
            loss = loss + self.vreg * (l_var + 0.04 * l_cov)
            out["loss_var"] = l_var.detach()
            out["loss_cov"] = l_cov.detach()
        out["loss"] = loss
        return out
