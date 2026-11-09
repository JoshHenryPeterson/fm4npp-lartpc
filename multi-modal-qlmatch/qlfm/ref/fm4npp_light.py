"""Mamba-2 + light: the FM4NPP recipe (qlfm/ref/fm4npp.py) with the 32 PMT tokens in the pretraining sequence.

Per event the causal sequence is
    [32 PMT tokens (some masked)] + [wire hits, FM4NPP serialization] + [32 PMT query tokens]
  * PMT token = Linear(17 light features: log1p PE in 16 time bins + total) + fixed NeRF embedding of the PMT
    (y, z) + a learned modality vector; a masked PMT shows a learned mask vector instead of its features
  * prefix: the hits are read after the visible light (causal model: light conditions every hit)
  * suffix: one query per PMT after the last hit; it regresses the 17 features of the PMTs that were masked in
    the prefix, i.e. light from charge (+ the visible PMTs). Masking as in the qlfm MAE: per event all PMTs
    (p 0.4), about half of them (p 0.4) or none (p 0.2)
  * loss = FM4NPP k-NNN loss on the hits + light_weight * MSE on the masked PMTs
  * the sequence is packed per event (prefix, hits, suffix, then padding), so no padding sits inside a sequence
Downstream (FM4NPPLightBackbone): the full light in the prefix, no suffix; tokens = hits + PMT prefix tokens.
`light_on = False` masks every prefix PMT (charge-only input, used for T5).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qlfm import geometry as G
from qlfm.ref.fm4npp import MambaGPT, NerfFixed, knn_next_targets, serialize
from qlfm.tokenize import TokCfg, tokenize_light

N_PMT = 32


class MambaGPTLight(MambaGPT):
    def __init__(self, d=512, n_layers=12, d_state=32, klen=30, dropout=0.1, n_feat=17):
        super().__init__(d=d, n_layers=n_layers, d_state=d_state, klen=klen, dropout=dropout)
        self.n_feat = n_feat
        self.l_proj = nn.Linear(n_feat, d)
        self.l_pos = NerfFixed(2, d)
        self.mod = nn.Parameter(torch.zeros(3, d))        # hit, PMT, PMT query
        self.l_mask = nn.Parameter(torch.zeros(d))
        self.l_out = nn.Linear(d, n_feat)
        nn.init.normal_(self.mod, std=0.02)
        nn.init.normal_(self.l_mask, std=0.02)
        yz = torch.as_tensor(G.PMT_YZ, dtype=torch.float32)
        self.register_buffer("pmt_yz", (yz - yz.min(0).values) / (yz.max(0).values - yz.min(0).values))

    def embed_hits(self, x):
        return x[..., :1] @ self.e_proj + self.pos(x[..., 1:4]) + self.mod[0]

    def embed_light(self, feat, masked):
        """feat [B, 32, 17], masked [B, 32] -> prefix tokens [B, 32, d]."""
        pos = self.l_pos(self.pmt_yz)[None]
        x = torch.where(masked[..., None], self.l_mask.to(feat.dtype), self.l_proj(feat))
        return x + pos + self.mod[1]

    def queries(self, B):
        return (self.l_pos(self.pmt_yz) + self.mod[2])[None].expand(B, -1, -1)

    def run(self, h):
        feats = []
        for blk in self.layers:
            z = blk["drop"](blk["mixer"](blk["norm"](h)))
            feats.append(z)
            h = h + z
        return h, feats


def light_feats(b):
    lt = tokenize_light(b["oph_ch"], b["oph_t"], b["oph_pe"], b["oph_evt"], b["n_events"], TokCfg())
    return lt.feat.view(b["n_events"], N_PMT, -1)


def pack_sequence(net, s, feat, masked, with_queries):
    """Build packed per-event sequences. Returns seq [B, Lt, d], valid [B, Lt], and index maps:
    hit_pos [B, L] (position in seq of each serialized hit slot, -1 for padding), pre/qry position [B, 32]."""
    B, L = s["valid"].shape
    dev = feat.device
    nh = s["valid"].sum(1)                                     # hits per event
    nq = N_PMT if with_queries else 0
    Lt = int(N_PMT + nh.max() + nq) if B else 0
    d = net.d
    seq = torch.zeros(B, Lt, d, device=dev)
    valid = torch.zeros(B, Lt, dtype=torch.bool, device=dev)
    ar = torch.arange(N_PMT, device=dev)
    pre_pos = ar[None].expand(B, -1)
    seq[:, :N_PMT] = net.embed_light(feat, masked)
    valid[:, :N_PMT] = True
    # serialized hits are left-aligned in their padded rows: slot j of event e -> position 32 + j
    j = torch.arange(L, device=dev)[None].expand(B, -1)
    hit_pos = torch.where(s["valid"], N_PMT + j, torch.full_like(j, -1))
    hh = net.embed_hits(s["x"])
    bi, ji = torch.nonzero(s["valid"], as_tuple=True)
    seq[bi, N_PMT + ji] = hh[bi, ji].to(seq.dtype)
    valid[bi, N_PMT + ji] = True
    qry_pos = None
    if with_queries:
        qry_pos = N_PMT + nh[:, None] + ar[None]
        bq = torch.arange(B, device=dev)[:, None].expand(-1, N_PMT)
        seq[bq, qry_pos] = net.queries(B).to(seq.dtype)
        valid[bq, qry_pos] = True
    return seq, valid, hit_pos, pre_pos, qry_pos


class FM4NPPLightPretrain(nn.Module):
    """k-NNN + masked-light objective (train_pretrain --method fm4npp_l)."""

    def __init__(self, net: MambaGPTLight, max_hits=16384, light_weight=1.0, p_all=0.4, p_part=0.4, ratio=0.5):
        super().__init__()
        self.net, self.max_hits, self.light_weight = net, max_hits, light_weight
        self.p_all, self.p_part, self.ratio = p_all, p_part, ratio

    def forward(self, b):
        net = self.net
        s = serialize(b, self.max_hits)
        tgt = knn_next_targets(s, net.klen)
        feat = light_feats(b)
        B = feat.shape[0]
        u = torch.rand(B, device=feat.device)
        mode = torch.where(u < self.p_all, 2, torch.where(u < self.p_all + self.p_part, 1, 0))
        part = torch.rand(B, N_PMT, device=feat.device) < self.ratio
        masked = (mode[:, None] == 2) | ((mode[:, None] == 1) & part)
        seq, valid, hit_pos, _, qry_pos = pack_sequence(net, s, feat, masked, with_queries=True)
        h, _ = net.run(seq)
        with torch.autocast("cuda", enabled=False):
            hn = net.norm(h).float()
            bi = torch.arange(B, device=h.device)[:, None]
            # k-NNN on hits (hit_pos == -1 only for padding slots, masked out below)
            pred = net.out(hn[bi, hit_pos.clamp(min=0)])
            m = (tgt != -100.0) & s["valid"][..., None]
            loss_k = ((pred - tgt) ** 2)[m].mean()
            out = dict(loss_c=loss_k.detach(), n_tok=torch.tensor(float(s["valid"].sum()), device=h.device))
            loss = loss_k
            if masked.any():
                pl = net.l_out(hn[bi, qry_pos])                    # [B, 32, 17]
                loss_l = F.mse_loss(pl[masked], feat[masked].float())
                loss = loss + self.light_weight * loss_l
                out["loss_l"] = loss_l.detach()
                allm = (mode == 2)[:, None].expand(-1, N_PMT)[masked]
                if allm.any():
                    out["loss_l_all"] = F.mse_loss(pl[masked][allm], feat[masked][allm].float()).detach()
        out["loss"] = loss
        return out


class FM4NPPLightBackbone(nn.Module):
    """Downstream features: learned softmax mix of the layer outputs (as FM4NPPBackbone) at the hit and PMT
    prefix positions. Tokens = hits (one per hit, as FM4NPP) followed by the 32 PMT tokens of each event."""

    def __init__(self, net: MambaGPTLight, max_hits=16384):
        super().__init__()
        self.net, self.max_hits = net, max_hits
        self.d, self.hit_dim, self.n_layers = net.d, 0, len(net.layers) + 1
        self.adapter_mix = nn.Parameter(torch.zeros(len(net.layers)))
        self.adapter_norm = nn.LayerNorm(net.d)
        self.light_on = True

    def layer_id(self, name):
        if name.startswith("adapter_"):
            return self.n_layers
        if name.startswith("net.norm."):
            return self.n_layers - 1
        return int(name.split("layers.")[1].split(".")[0]) + 1 if "layers." in name else 0

    def forward(self, b):
        net = self.net
        s = serialize(b, self.max_hits)
        feat = light_feats(b)
        B, L = s["valid"].shape
        masked = torch.full((B, N_PMT), not self.light_on, dtype=torch.bool, device=feat.device)
        seq, valid, hit_pos, pre_pos, _ = pack_sequence(net, s, feat, masked, with_queries=False)
        _, feats = net.run(seq)
        w = torch.softmax(self.adapter_mix, 0)
        z = self.adapter_norm(sum(wi * f for wi, f in zip(w, feats)).float())     # [B, Lt, d]
        bi, ji = torch.nonzero(s["valid"], as_tuple=True)
        zh = z[bi, hit_pos[bi, ji]]
        # hit -> token: tokens are listed in (event, slot) order of the valid serialized slots
        flat_slot = bi * L + ji
        newidx = torch.full((B * L,), -1, dtype=torch.long, device=z.device)
        newidx[flat_slot] = torch.arange(len(flat_slot), device=z.device)
        hit2tok = torch.where(s["hit2pos"] >= 0, newidx[s["hit2pos"].clamp(min=0)], s["hit2pos"])
        if self.light_on:
            zl = z[:, :N_PMT].reshape(-1, self.d)
            levt = torch.arange(B, device=z.device).repeat_interleave(N_PMT)
            return dict(z=torch.cat([zh, zl]), evt=torch.cat([bi, levt]), h_hit=None, hit2tok=hit2tok)
        return dict(z=zh, evt=bi, h_hit=None, hit2tok=hit2tok)


def build(name, args, ckpt):
    if ckpt is None:
        net = MambaGPTLight(d=getattr(args, "d", 512), n_layers=getattr(args, "depth", 12))
        return FM4NPPLightBackbone(net, getattr(args, "max_hits", None) or 16384)
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = s["args"]
    net = MambaGPTLight(d=a["d"], n_layers=a["depth"], d_state=a.get("d_state", 32), klen=a.get("klen", 30))
    net.load_state_dict(s["fm"])
    cap = getattr(args, "max_hits", None)
    return FM4NPPLightBackbone(net, cap if cap else a.get("max_hits", 16384))
