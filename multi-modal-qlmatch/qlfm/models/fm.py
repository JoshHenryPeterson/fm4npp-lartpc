"""Foundation model: tokenizers + embedders + encoder, and the multi-modal MAE pretraining wrapper."""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from qlfm import geometry as G
from qlfm.models.encoder import BlobEmbed, ChargeEmbed, Encoder, FlashEmbed, LightEmbed, mlp, pack
from qlfm.tokenize import TokCfg, hit_xyz_from_blobs, tokenize_blobs, tokenize_charge, tokenize_flash, tokenize_light


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
    # #16 options; the defaults reproduce the models before #16
    charge_stem: str = "mlp"   # mlp | cnn (sparse-conv stem, needs spconv)
    pos3d: bool = False        # 3D position embedding of charge tokens from the Wire-Cell blobs of their hits
    light_tok: str = "pmt"     # pmt (16-bin profile per PMT) | flash (one in-beam OpFlash token) | pmt_flash
    pmt_pos: str = "v1"        # v1 | v2 (geometry scales + TPC-side embedding)
    use_blobs: bool = False    # 3D Wire-Cell blob voxel tokens


class FM(nn.Module):
    """Backbone shared by pretraining and all downstream modes."""

    def __init__(self, cfg: FMCfg):
        super().__init__()
        self.cfg = cfg
        self.charge = ChargeEmbed(cfg.d, pos_mode=cfg.pos_mode, n_freq=cfg.pos_freq,
                                  n_cells=(3 * (3456 // cfg.tok.cell_wires + 1), 8192 // cfg.tok.cell_ticks + 1),
                                  stem=cfg.charge_stem, pos3d=cfg.pos3d)
        nb = len(cfg.tok.light_edges)          # n_bins + total
        self.light_feat = {"pmt": nb, "pmt_flash": 3, "flash": G.N_PMT + 3}[cfg.light_tok]
        if not cfg.use_light:
            self.light = None
        elif cfg.light_tok == "flash":
            self.light = FlashEmbed(cfg.d, self.light_feat)
        else:
            self.light = LightEmbed(cfg.d, self.light_feat, pmt_pos=cfg.pmt_pos)
        self.modality = nn.Embedding(2, cfg.d)
        self.blob = BlobEmbed(cfg.d, self.charge.hit_dim, n_freq=cfg.pos_freq) if cfg.use_blobs else None
        self.mod_blob = nn.Parameter(torch.zeros(cfg.d)) if cfg.use_blobs else None
        self.encoder = Encoder(cfg.d, cfg.depth, cfg.heads, drop_path=cfg.drop_path)

    def tokenize(self, b, generator=None):
        xyz_hit = None
        if self.cfg.pos3d and "hb_hit" in b:
            xyz_hit = hit_xyz_from_blobs(b["hit_plane"].numel(), b["hb_hit"], b["hb_blob"], b["blob_xyz"], b["blob_q"])
        ct = tokenize_charge(b["hit_plane"], b["hit_wire"], b["hit_time"], b["hit_q"], b["hit_rms"],
                             b["hit_evt"], b["n_events"], self.cfg.tok, generator, tpc=b.get("hit_tpc"),
                             xyz_hit=xyz_hit, truth_xyz_hit=b.get("hit_txyz"))
        lt = None
        if self.cfg.use_light:
            if self.cfg.light_tok == "pmt":
                lt = tokenize_light(b["oph_ch"], b["oph_t"], b["oph_pe"], b["oph_evt"], b["n_events"], self.cfg.tok)
            else:
                lt = tokenize_flash(b["fl_pmt"], b["fl_t"], b["fl_pe"], self.cfg.light_tok)
        return ct, lt

    def tokenize_blob(self, b, generator=None):
        if not self.cfg.use_blobs:
            return None
        return tokenize_blobs(b["blob_xyz"], b["blob_q"], b["blob_ext"], b["blob_tpc"], b["blob_evt"],
                              b["n_events"], self.cfg.tok, generator)

    def stem_input(self, b):
        if self.charge.stem is None:
            return None
        tok = self.cfg.tok
        tpc = b.get("hit_tpc")
        img = (b["hit_evt"] * G.N_TPC + (tpc if tpc is not None and G.N_TPC > 1 else 0)) * 3 + b["hit_plane"]
        return (img, b["hit_wire"], b["hit_time"], torch.log1p(b["hit_q"].clamp(min=0) / tok.q_scale),
                b["hit_rms"] / tok.rms_scale)

    def embed(self, ct, lt, b=None):
        xc, h_hit = self.charge(ct, self.stem_input(b) if b is not None else None)
        xc = xc + self.modality.weight[0]
        xl = None
        if lt is not None:
            xl = self.light(lt) + self.modality.weight[1]
        return xc, xl, h_hit

    def embed_blob(self, bt, b, h_hit, hit_ok=None):
        """Blob tokens; hit_ok (bool [Nh]) restricts the nexus pooling, e.g. to hits of visible charge tokens."""
        hb_hit, hb_blob = b.get("hb_hit"), b.get("hb_blob")
        if hb_hit is not None and hit_ok is not None:
            m = hit_ok[hb_hit]
            hb_hit, hb_blob = hb_hit[m], hb_blob[m]
        if hb_hit is not None:
            m = bt.blob2tok[hb_blob] >= 0
            hb_hit, hb_blob = hb_hit[m], hb_blob[m]
        return self.blob(bt, h_hit, hb_hit, hb_blob) + self.mod_blob

    def encode(self, xc, c_evt, xl, l_evt, n_events, return_layers=(), xb=None, b_evt=None):
        """Encode packed tokens. Returns features split back to (charge, light) order, and (charge, light, blob)
        when blob tokens are given."""
        srcs, evts = [xc], [c_evt]
        if xl is not None:
            srcs.append(xl)
            evts.append(l_evt)
        if xb is not None:
            srcs.append(xb)
            evts.append(b_evt)
        perm, cu, maxlen, evt_sorted = pack(evts, n_events)
        x = torch.cat(srcs)[perm]
        out = self.encoder(x, cu, maxlen, return_layers)
        z, layers = out if return_layers else (out, [])
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        nc = xc.shape[0]
        nl = xl.shape[0] if xl is not None else 0

        def unpack(t):
            zc = t[inv[:nc]]
            zl = t[inv[nc:nc + nl]] if xl is not None else None
            if xb is None:
                return zc, zl
            return zc, zl, t[inv[nc + nl:]]
        return unpack(z), [unpack(t) for t in layers]

    def forward(self, b, return_layers=()):
        """Full (unmasked) forward for downstream use."""
        ct, lt = self.tokenize(b)
        xc, xl, h_hit = self.embed(ct, lt, b)
        bt = self.tokenize_blob(b)
        xb = self.embed_blob(bt, b, h_hit) if bt is not None else None
        zs, layers = self.encode(xc, ct.tok_evt, xl, lt.evt if lt is not None else None, b["n_events"], return_layers,
                                 xb=xb, b_evt=bt.tok_evt if bt is not None else None)
        zc, zl = zs[0], zs[1]
        zb = zs[2] if bt is not None else None
        return dict(ct=ct, lt=lt, bt=bt, zc=zc, zl=zl, zb=zb, h_hit=h_hit, layers=layers)


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
    # #16: blob tokens and the auxiliary 3D head
    blob_mask_ratio: float = 0.6
    blob_weight: float = 1.0
    aux3d_weight: float = 0.0      # > 0: predict each visible charge token's truth 3D position (P3a)
    # coordinates in the P3a loss. "yz" (default): x of out-of-time cosmics depends on their unknown t0, which charge
    # alone cannot give (v06 test: blob-vs-truth x residual 0.16 cm median for nu hits, 75 cm for cosmic hits);
    # "xyz" adds x (learnable only with the light)
    aux3d_dims: str = "yz"


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
        self.head_l = nn.Linear(dd, fm.light_feat)
        nn.init.normal_(self.mask_c, std=0.02)
        nn.init.normal_(self.mask_l, std=0.02)
        if fm.blob is not None:
            self.mask_b = nn.Parameter(torch.zeros(dd))
            self.dec_pos_b = nn.Linear(d, dd)
            self.head_b = nn.Linear(dd, tok.blob_sub ** 3 + 2)
            nn.init.normal_(self.mask_b, std=0.02)
        self.head_3d = mlp([d, d, 3]) if cfg.aux3d_weight > 0 else None
        self.register_buffer("pos3_min", torch.tensor(G.POS3D_MIN, dtype=torch.float32))
        self.register_buffer("pos3_scale", torch.tensor(G.POS3D_SCALE, dtype=torch.float32))

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
        bt = fm.tokenize_blob(b)
        xc, xl, h_hit = fm.embed(ct, lt, b)
        mc = self.charge_mask(ct, n)
        ml, lmode = self.light_mask(lt, n) if lt is not None else (None, None)

        # ---- blob tokens: the nexus term only pools hits of visible charge tokens (no leak of masked charge) ----
        xb = mb = None
        if bt is not None:
            mb = torch.rand(bt.tok_evt.shape, device=xc.device) < self.cfg.blob_mask_ratio
            hit_ok = (ct.hit2tok >= 0) & ~mc[ct.hit2tok.clamp(min=0)]
            xb = fm.embed_blob(bt, b, h_hit, hit_ok)

        # ---- encoder on visible tokens ----
        vis_c = ~mc
        xl_vis = xl[~ml] if xl is not None else None
        levt_vis = lt.evt[~ml] if lt is not None else None
        zs, _ = fm.encode(xc[vis_c], ct.tok_evt[vis_c], xl_vis, levt_vis, n,
                          xb=xb[~mb] if xb is not None else None, b_evt=bt.tok_evt[~mb] if bt is not None else None)
        zc, zl = zs[0], zs[1]
        zb = zs[2] if bt is not None else None

        # ---- decoder on visible + mask tokens ----
        dc = torch.empty(xc.shape[0], self.cfg.dec_d, device=xc.device, dtype=zc.dtype)
        dc[vis_c] = self.enc2dec(zc).to(dc.dtype)
        dc[mc] = self.mask_c.to(dc.dtype)
        dc = dc + self.dec_pos_c(fm.charge.pos_embed(ct))
        srcs, evts = [dc], [ct.tok_evt]
        nl = 0
        if lt is not None:
            dl = torch.empty(xl.shape[0], self.cfg.dec_d, device=xl.device, dtype=zc.dtype)
            if zl is not None and zl.shape[0]:
                dl[~ml] = self.enc2dec(zl).to(dl.dtype)
            dl[ml] = self.mask_l.to(dl.dtype)
            dl = dl + self.dec_pos_l(fm.light.pos_embed(lt))
            srcs.append(dl)
            evts.append(lt.evt)
            nl = dl.shape[0]
        if bt is not None:
            db = torch.empty(xb.shape[0], self.cfg.dec_d, device=xb.device, dtype=zc.dtype)
            if zb is not None and zb.shape[0]:
                db[~mb] = self.enc2dec(zb).to(db.dtype)
            db[mb] = self.mask_b.to(db.dtype)
            db = db + self.dec_pos_b(fm.blob.pos_embed(bt))
            srcs.append(db)
            evts.append(bt.tok_evt)
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
            yl = y[inv[nc:nc + nl]]
            pred_l = self.head_l(yl[ml]).float()
            loss_l = F.mse_loss(pred_l, lt.feat[ml])
            loss = loss + self.cfg.light_weight * loss_l
            out["loss_l"] = loss_l.detach()
            # charge->light (all PMTs masked) loss reported separately
            allm = (lmode[lt.evt] == 2)[ml]
            if allm.any():
                out["loss_l_all"] = F.mse_loss(pred_l[allm], lt.feat[ml][allm]).detach()
        if bt is not None and mb.any():
            yb = y[inv[nc + nl:]]
            tgt_b = torch.cat([bt.tok_target, bt.tok_stat], 1)[mb]
            if self.cfg.target_norm:
                tgt_b = (tgt_b - tgt_b.mean(0)) / (tgt_b.std(0) + 1e-3)
            loss_b = F.mse_loss(self.head_b(yb[mb]).float(), tgt_b)
            loss = loss + self.cfg.blob_weight * loss_b
            out["loss_b"] = loss_b.detach()
        if self.head_3d is not None and ct.tok_txyz is not None:
            ok = ct.tok_t3ok[vis_c]
            if ok.any():
                tgt3 = (ct.tok_txyz[vis_c][ok] - self.pos3_min) / self.pos3_scale
                dims = [1, 2] if self.cfg.aux3d_dims == "yz" else [0, 1, 2]
                loss_3d = F.smooth_l1_loss(self.head_3d(zc[ok]).float()[:, dims], tgt3[:, dims], beta=0.02)
                loss = loss + self.cfg.aux3d_weight * loss_3d
                out["loss_3d"] = loss_3d.detach()
        out["loss"] = loss
        return out
