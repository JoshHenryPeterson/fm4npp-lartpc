"""Foundation model: tokenizers + embedders + encoder, and the multi-modal MAE pretraining wrapper."""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from qlfm.models.encoder import ChargeEmbed, Encoder, LightEmbed, mlp, pack
from qlfm.tokenize import TokCfg, tokenize_charge, tokenize_light


@dataclass
class FMCfg:
    d: int = 256
    depth: int = 8
    heads: int = 8
    use_light: bool = True
    drop_path: float = 0.0
    pos_mode: str = "centroid"  # centroid (default) | fourier | grid | both | cellrel
    pos_freq: int = 10
    tok: TokCfg = field(default_factory=TokCfg)


class FM(nn.Module):
    """Backbone shared by pretraining and all downstream modes."""

    def __init__(self, cfg: FMCfg):
        super().__init__()
        self.cfg = cfg
        self.charge = ChargeEmbed(cfg.d, pos_mode=cfg.pos_mode, n_freq=cfg.pos_freq,
                                  n_cells=(3 * (3456 // cfg.tok.cell_wires + 1), 8192 // cfg.tok.cell_ticks + 1))
        nb = len(cfg.tok.light_edges)          # n_bins + total
        self.light = LightEmbed(cfg.d, nb) if cfg.use_light else None
        self.modality = nn.Embedding(2, cfg.d)
        self.encoder = Encoder(cfg.d, cfg.depth, cfg.heads, drop_path=cfg.drop_path)

    def tokenize(self, b, generator=None):
        ct = tokenize_charge(b["hit_plane"], b["hit_wire"], b["hit_time"], b["hit_q"], b["hit_rms"],
                             b["hit_evt"], b["n_events"], self.cfg.tok, generator, tpc=b.get("hit_tpc"))
        lt = None
        if self.cfg.use_light:
            lt = tokenize_light(b["oph_ch"], b["oph_t"], b["oph_pe"], b["oph_evt"], b["n_events"], self.cfg.tok)
        return ct, lt

    def embed(self, ct, lt):
        xc, h_hit = self.charge(ct)
        xc = xc + self.modality.weight[0]
        xl = None
        if lt is not None:
            xl = self.light(lt) + self.modality.weight[1]
        return xc, xl, h_hit

    def encode(self, xc, c_evt, xl, l_evt, n_events, return_layers=()):
        """Encode packed tokens. Returns features split back to (charge, light) order."""
        srcs, evts = [xc], [c_evt]
        if xl is not None:
            srcs.append(xl)
            evts.append(l_evt)
        perm, cu, maxlen, evt_sorted = pack(evts, n_events)
        x = torch.cat(srcs)[perm]
        out = self.encoder(x, cu, maxlen, return_layers)
        z, layers = out if return_layers else (out, [])
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        nc = xc.shape[0]
        unpack = lambda t: (t[inv[:nc]], t[inv[nc:]] if xl is not None else None)
        return unpack(z), [unpack(t) for t in layers]

    def forward(self, b, return_layers=()):
        """Full (unmasked) forward for downstream use."""
        ct, lt = self.tokenize(b)
        xc, xl, h_hit = self.embed(ct, lt)
        (zc, zl), layers = self.encode(xc, ct.tok_evt, xl, lt.evt if lt is not None else None,
                                       b["n_events"], return_layers)
        return dict(ct=ct, lt=lt, zc=zc, zl=zl, h_hit=h_hit, layers=layers)


@dataclass
class MAECfg:
    mask_ratio: float = 0.6
    strip_frac: float = 0.5        # share of the masked budget from cross-plane drift-time strips
    strip_ticks: int = 256
    light_p_partial: float = 0.4   # mask ~light_ratio of PMTs
    light_p_all: float = 0.4       # mask all PMTs (charge -> light prediction)
    light_ratio: float = 0.5
    light_weight: float = 1.0
    dec_d: int = 192
    dec_depth: int = 3
    dec_heads: int = 6
    target_norm: bool = True       # standardize each charge-target dim (per batch) before the MSE


class MAE(nn.Module):
    def __init__(self, fm: FM, cfg: MAECfg):
        super().__init__()
        self.fm, self.cfg = fm, cfg
        d, dd = fm.cfg.d, cfg.dec_d
        tok = fm.cfg.tok
        self.enc2dec = nn.Linear(d, dd)
        self.mask_c = nn.Parameter(torch.zeros(dd))
        self.mask_l = nn.Parameter(torch.zeros(dd))
        self.dec_pos_c = nn.Linear(d, dd)          # reuse the backbone's positional embedding, projected
        self.dec_pos_l = nn.Linear(d, dd)
        self.decoder = Encoder(dd, cfg.dec_depth, cfg.dec_heads, final_norm=True)
        self.head_c = nn.Linear(dd, tok.sub_w * tok.sub_t + 2)
        self.head_l = nn.Linear(dd, len(tok.light_edges))
        nn.init.normal_(self.mask_c, std=0.02)
        nn.init.normal_(self.mask_l, std=0.02)

    def charge_mask(self, ct, n_events):
        c = self.cfg
        dev = ct.tok_evt.device
        n_strip = 8192 // c.strip_ticks + 1
        strip = (ct.tok_time / c.strip_ticks).long().clamp(0, n_strip - 1)
        strip_score = torch.rand(n_events, n_strip, device=dev)
        p_strip = c.mask_ratio * c.strip_frac
        m = strip_score[ct.tok_evt, strip] < p_strip
        p_rand = (c.mask_ratio - p_strip) / (1 - p_strip)
        m |= torch.rand(m.shape, device=dev) < p_rand
        return m

    def light_mask(self, lt, n_events):
        c = self.cfg
        dev = lt.evt.device
        u = torch.rand(n_events, device=dev)
        mode = torch.where(u < c.light_p_all, 2, torch.where(u < c.light_p_all + c.light_p_partial, 1, 0))
        part = torch.rand(lt.evt.shape, device=dev) < c.light_ratio
        m = (mode[lt.evt] == 2) | ((mode[lt.evt] == 1) & part)
        return m, mode

    def forward(self, b):
        fm, n = self.fm, b["n_events"]
        ct, lt = fm.tokenize(b)
        xc, xl, _ = fm.embed(ct, lt)
        mc = self.charge_mask(ct, n)
        ml, lmode = self.light_mask(lt, n) if lt is not None else (None, None)

        # ---- encoder on visible tokens ----
        vis_c = ~mc
        xl_vis = xl[~ml] if xl is not None else None
        levt_vis = lt.evt[~ml] if lt is not None else None
        (zc, zl), _ = fm.encode(xc[vis_c], ct.tok_evt[vis_c], xl_vis, levt_vis, n)

        # ---- decoder on visible + mask tokens ----
        dc = torch.empty(xc.shape[0], self.cfg.dec_d, device=xc.device, dtype=zc.dtype)
        dc[vis_c] = self.enc2dec(zc).to(dc.dtype)
        dc[mc] = self.mask_c.to(dc.dtype)
        dc = dc + self.dec_pos_c(fm.charge.pos_embed(ct))
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

        tgt_c = torch.cat([ct.tok_target, ct.tok_stat], 1)[mc]
        if self.cfg.target_norm:
            tgt_c = (tgt_c - tgt_c.mean(0)) / (tgt_c.std(0) + 1e-3)
        pred_c = self.head_c(yc[mc]).float()
        loss_c = F.mse_loss(pred_c, tgt_c)
        out = dict(loss_c=loss_c.detach(), n_tok=torch.tensor(float(nc), device=y.device))
        loss = loss_c
        if lt is not None and ml.any():
            yl = y[inv[nc:]]
            pred_l = self.head_l(yl[ml]).float()
            loss_l = F.mse_loss(pred_l, lt.feat[ml])
            loss = loss + self.cfg.light_weight * loss_l
            out["loss_l"] = loss_l.detach()
            # charge->light (all PMTs masked) loss reported separately
            allm = (lmode[lt.evt] == 2)[ml]
            if allm.any():
                out["loss_l_all"] = F.mse_loss(pred_l[allm], lt.feat[ml][allm]).detach()
        out["loss"] = loss
        return out
