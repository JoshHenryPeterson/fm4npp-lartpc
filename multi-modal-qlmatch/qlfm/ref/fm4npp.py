"""FM4NPP (arXiv:2508.14087) ported to MicroBooNE wire hits.

Faithful to the released MambaGPT (FM4NPP/PP_collision, branch downstream-reproducibility):
  fixed random energy projection + NeRF coordinate embedding (10 freqs, fixed projection), summed;
  12 x [RMSNorm -> Mamba2 (d_conv 4, expand 2) -> Dropout] with residuals; final RMSNorm; Linear(d, 3k);
  causal (unidirectional); k-next-nearest-neighbour (k-NNN) coordinate regression with MSE, k = 30,
  missing neighbours masked.
Adaptations forced by the data (documented in the round-2 note):
  * one token per wire hit; coordinates (wire/n_wires[plane], tick/6400, plane/2), energy log1p(q/200)
  * serialization: hierarchical time-major raster -- 8 drift slabs x 3 planes (interleaved per slab)
    x 8 wire boxes, then wire, then tick (FM4NPP: 6x8x8 (r, eta, phi) boxes, inner -> outer)
  * "next" neighbours: hits later in the sequence AND on the same plane (FM4NPP: larger r), distances in cm
  * hits per event capped at max_hits (random subsample, order kept); FM4NPP itself caps at 3,200 points
  * event-difficulty loss reweighting omitted
  * targets are RELATIVE in-plane offsets (neighbour - hit) in units of 10 cm, 2 per neighbour (FM4NPP:
    absolute normalized coordinates). With ~15k hits on a 10 m x 3.5 m plane the k nearest neighbours lie
    within ~1e-3 of the normalized range, so absolute targets are almost solved by copying the input
    (smoke test: loss 0.004 after 75 steps) and fall below bf16 resolution. The output head runs in fp32.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm

from qlfm import geometry as G

N_WIRES = torch.tensor([2400.0, 2400.0, 3456.0])


class NerfFixed(nn.Module):
    """NeRF positional encoding (include_input, log-sampled 2^0..2^(F-1), sin/cos) + fixed random projection."""

    def __init__(self, n_dim, d, n_freq=10):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float())
        out = n_dim * (1 + 2 * n_freq)
        self.register_buffer("proj", torch.randn(out, d))

    def forward(self, x):
        a = x[..., None] * self.freqs                         # [B, L, n, F]
        f = torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)
        return f @ self.proj


class MambaGPT(nn.Module):
    def __init__(self, d=256, n_layers=12, d_state=16, klen=30, dropout=0.1):
        super().__init__()
        self.d, self.klen = d, klen
        self.register_buffer("e_proj", torch.randn(1, d))      # fixed, as in EmbedderAdd(learnable=False)
        self.pos = NerfFixed(3, d)
        self.layers = nn.ModuleList([nn.ModuleDict(dict(norm=RMSNorm(d), mixer=Mamba2(d_model=d, d_state=d_state,
                                     d_conv=4, expand=2, headdim=64), drop=nn.Dropout(dropout)))
                                     for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.out = nn.Linear(d, klen * 2)

    def forward(self, x, return_layers=False):
        """x: [B, L, 4] = (E, wire_n, tick_n, plane_n). Returns predictions [B, L, 3k] (and per-layer features)."""
        h = x[..., :1] @ self.e_proj + self.pos(x[..., 1:4])
        feats = []
        for blk in self.layers:
            z = blk["drop"](blk["mixer"](blk["norm"](h)))
            feats.append(z)
            h = h + z
        with torch.autocast("cuda", enabled=False):
            y = self.out(self.norm(h).float())
        return (y, feats, self.norm(h)) if return_layers else y


def serialize(b, max_hits, generator=None):
    """Batch of raw hits -> padded serialized sequences.

    Returns x [B, Lmax, 4], valid [B, Lmax], plane [B, Lmax], cm [B, Lmax, 2], hit_index [B, Lmax] (into the
    flat hit arrays; -1 for padding) and hit2pos (flat hit -> flat position in the padded batch, -1 if dropped)."""
    dev = b["hit_plane"].device
    n = b["n_events"]
    plane, wire, time, q, evt = b["hit_plane"], b["hit_wire"], b["hit_time"], b["hit_q"], b["hit_evt"]
    nh = len(plane)
    keep = torch.ones(nh, dtype=torch.bool, device=dev)
    cnt = torch.bincount(evt, minlength=n)
    if cnt.max() > max_hits:
        r = torch.rand(nh, device=dev, generator=generator)
        for e in torch.nonzero(cnt > max_hits).flatten().tolist():
            idx = torch.nonzero(evt == e).flatten()
            keep[idx[torch.argsort(r[idx])[max_hits:]]] = False
    nw = N_WIRES.to(dev)[plane]
    tbin = (time / 800.0).long().clamp(0, 7)
    wbin = (wire.float() / nw * 8).long().clamp(0, 7)
    key = ((((evt * 8 + tbin) * 3 + plane) * 8 + wbin) * 4096 + wire) * 8192 + time.long().clamp(0, 8191)
    key = torch.where(keep, key, torch.full_like(key, torch.iinfo(torch.int64).max))
    order = torch.argsort(key)[: int(keep.sum())]
    e_sorted = evt[order]
    cnt_k = torch.bincount(e_sorted, minlength=n)
    L = int(cnt_k.max())
    start = torch.cumsum(cnt_k, 0) - cnt_k
    pos = torch.arange(len(order), device=dev) - start[e_sorted]
    flat = e_sorted * L + pos
    x = torch.zeros(n * L, 4, device=dev)
    x[flat, 0] = (torch.log1p(q[order].clamp(min=0) / 200.0) - 0.5) / 0.5
    x[flat, 1] = wire[order].float() / nw[order]
    x[flat, 2] = time[order] / 6400.0
    x[flat, 3] = plane[order].float() / 2.0
    valid = torch.zeros(n * L, dtype=torch.bool, device=dev); valid[flat] = True
    pl = torch.full((n * L,), -1, dtype=torch.long, device=dev); pl[flat] = plane[order]
    cm = torch.zeros(n * L, 2, device=dev)
    cm[flat, 0] = wire[order].float() * G.WIRE_PITCH_CM
    cm[flat, 1] = time[order] * G.TICK_TO_CM
    hit2pos = torch.full((nh,), -1, dtype=torch.long, device=dev); hit2pos[order] = flat
    return dict(x=x.view(n, L, 4), valid=valid.view(n, L), plane=pl.view(n, L), cm=cm.view(n, L, 2),
                hit2pos=hit2pos, L=L)


@torch.no_grad()
def knn_next_targets(s, k, unit_cm=10.0):
    """For each position i: in-plane offsets (cm / unit_cm) of its k nearest same-plane hits later in the sequence."""
    x, valid, plane, cm = s["x"], s["valid"], s["plane"], s["cm"]
    B, L, _ = x.shape
    tgt = torch.full((B, L, k, 2), -100.0, device=x.device)
    for bi in range(B):
        for p in range(3):
            idx = torch.nonzero(valid[bi] & (plane[bi] == p)).flatten()     # already in sequence order
            m = len(idx)
            if m < 2:
                continue
            c = cm[bi, idx]
            for r0 in range(0, m, 2048):
                r1 = min(m, r0 + 2048)
                dist = torch.cdist(c[r0:r1], c)                              # [rows, m]
                ar = torch.arange(r0, r1, device=x.device)[:, None]
                dist = dist.masked_fill(torch.arange(m, device=x.device)[None, :] <= ar, float("inf"))
                kk = min(k, m)
                dval, didx = torch.topk(dist, kk, dim=1, largest=False)
                t = (c[didx] - c[r0:r1, None, :]) / unit_cm                  # [rows, kk, 2]
                t = torch.where(torch.isinf(dval)[..., None], torch.full_like(t, -100.0), t)
                tgt[bi, idx[r0:r1], :kk] = t
    return tgt.view(B, L, k * 2)


class FM4NPPPretrain(nn.Module):
    """k-NNN objective wrapper used by train_pretrain (method=fm4npp)."""

    def __init__(self, net: MambaGPT, max_hits=16384):
        super().__init__()
        self.net, self.max_hits = net, max_hits

    def forward(self, b):
        s = serialize(b, self.max_hits)
        tgt = knn_next_targets(s, self.net.klen)
        pred = self.net(s["x"]).float()
        m = (tgt != -100.0) & s["valid"][..., None]
        loss = ((pred - tgt) ** 2)[m].mean()
        return dict(loss=loss, loss_c=loss.detach(), n_tok=torch.tensor(float(s["valid"].sum()), device=loss.device))


class FM4NPPBackbone(nn.Module):
    """Frozen/fine-tunable FM4NPP features for the common heads. Like FM4NPP's adapters, features are a learned
    softmax-weighted mix of the 12 layer outputs (parameter `adapter_mix`, trained even when frozen) + LayerNorm."""

    def __init__(self, net: MambaGPT, max_hits=16384):
        super().__init__()
        self.net, self.max_hits = net, max_hits
        self.d, self.hit_dim, self.n_layers = net.d, 0, len(net.layers) + 1
        self.adapter_mix = nn.Parameter(torch.zeros(len(net.layers)))
        self.adapter_norm = nn.LayerNorm(net.d)

    def layer_id(self, name):
        if name.startswith("adapter_"):
            return self.n_layers
        if name.startswith("net.norm."):                  # final RMSNorm sits after the last layer (audit #11)
            return self.n_layers - 1
        return int(name.split("layers.")[1].split(".")[0]) + 1 if "layers." in name else 0

    def forward(self, b):
        s = serialize(b, self.max_hits)
        _, feats, _ = self.net(s["x"], return_layers=True)
        w = torch.softmax(self.adapter_mix, 0)
        z = sum(wi * f for wi, f in zip(w, feats))
        z = self.adapter_norm(z.float()).reshape(-1, self.d)
        n, L = s["valid"].shape
        flat_valid = s["valid"].reshape(-1)
        keep_pos = torch.nonzero(flat_valid).flatten()
        newidx = torch.full((n * L,), -1, dtype=torch.long, device=z.device)
        newidx[keep_pos] = torch.arange(len(keep_pos), device=z.device)
        evt = keep_pos // L
        hit2tok = torch.where(s["hit2pos"] >= 0, newidx[s["hit2pos"].clamp(min=0)], s["hit2pos"])
        return dict(z=z[keep_pos], evt=evt, h_hit=None, hit2tok=hit2tok)


def build(name, args, ckpt):
    if ckpt is None:
        net = MambaGPT(d=getattr(args, "d", 256), n_layers=getattr(args, "depth", 12))
        return FM4NPPBackbone(net, getattr(args, "max_hits", None) or 16384)
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = s["args"]
    net = MambaGPT(d=a["d"], n_layers=a["depth"], d_state=a.get("d_state", 16), klen=a.get("klen", 30))
    net.load_state_dict(s["fm"])
    cap = getattr(args, "max_hits", None)                 # downstream override (round 3: no cap)
    return FM4NPPBackbone(net, cap if cap else a.get("max_hits", 16384))
