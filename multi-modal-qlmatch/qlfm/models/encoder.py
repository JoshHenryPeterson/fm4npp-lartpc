"""Token embedders and a packed (variable-length) Transformer encoder using FlashAttention-2."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_qkvpacked_func

from qlfm import geometry as G
from qlfm.tokenize import ChargeTokens, LightTokens, pointnet_pool


class Fourier(nn.Module):
    """Fourier features of low-dimensional coordinates followed by a linear map."""

    def __init__(self, in_dim, out_dim, n_freq=16, scale=(1.0,)):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(n_freq) * math.pi)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.lin = nn.Linear(in_dim * (2 * n_freq + 1), out_dim)

    def forward(self, x):
        x = x / self.scale                                   # roughly [0, 1]
        a = x[..., None] * self.freq                         # [N, in, F]
        f = torch.cat([x[..., None], a.sin(), a.cos()], -1).flatten(1)
        return self.lin(f)


def mlp(dims, act=nn.GELU, last_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


class ChargeEmbed(nn.Module):
    def __init__(self, d, hit_dim=128, pos_mode="fourier", n_freq=16, n_cells=(512, 512)):
        super().__init__()
        self.pos_mode = pos_mode
        self.hit_mlp = mlp([4, 64, hit_dim], last_act=True)
        self.proj = mlp([2 * hit_dim + 2, d, d])
        if pos_mode in ("fourier", "both", "centroid"):
            self.pos = Fourier(2, d, n_freq=n_freq, scale=(300.0, 1100.0))   # (drift x cm, wire coord cm)
        if pos_mode in ("centroid", "cellrel"):
            self.crel = mlp([2, 64, d])                      # centroid offset within the cell
        if pos_mode in ("grid", "both", "cellrel"):
            self.wcell = nn.Embedding(n_cells[0], d)       # learned per-plane wire-cell index embedding
            self.tcell = nn.Embedding(n_cells[1], d)       # learned time-cell index embedding (shared)
        self.plane = nn.Embedding(3, d)
        self.hit_dim = hit_dim

    def forward(self, ct: ChargeTokens):
        h = self.hit_mlp(ct.feat_hit)                        # [Nh, hit_dim], reused by hit heads
        nt = ct.tok_evt.numel()
        pooled = pointnet_pool(h, ct.hit2tok, nt)
        x = self.proj(torch.cat([pooled, ct.tok_stat], 1))
        return x + self.pos_embed(ct), h

    def pos_embed(self, ct: ChargeTokens):
        e = self.plane(ct.tok_plane)
        if self.pos_mode in ("fourier", "both"):
            e = e + self.pos(ct.tok_pos)
        if self.pos_mode == "centroid":
            e = e + self.pos(ct.tok_cpos)
        if self.pos_mode in ("centroid", "cellrel"):
            e = e + self.crel(ct.tok_crel)
        if self.pos_mode in ("grid", "both", "cellrel"):
            nw = self.wcell.num_embeddings // 3
            e = e + self.wcell(ct.tok_plane * nw + ct.tok_cell[:, 0].clamp(max=nw - 1)) \
                  + self.tcell(ct.tok_cell[:, 1].clamp(max=self.tcell.num_embeddings - 1))
        return e


class LightEmbed(nn.Module):
    def __init__(self, d, n_feat):
        super().__init__()
        self.proj = mlp([n_feat, d, d])
        self.pos = Fourier(2, d, scale=(130.0, 1100.0))     # PMT (y + G.PMT_Y_SHIFT, z)
        self.pmt = nn.Embedding(G.N_PMT, d)
        self.register_buffer("pmt_yz", torch.from_numpy(G.PMT_YZ.copy()) + torch.tensor([G.PMT_Y_SHIFT, 0.0]))

    def forward(self, lt: LightTokens):
        return self.proj(lt.feat) + self.pos_embed(lt)

    def pos_embed(self, lt: LightTokens):
        return self.pos(self.pmt_yz[lt.pmt]) + self.pmt(lt.pmt)


class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = mlp([d, int(d * mlp_ratio), d])
        self.heads = heads
        self.drop_path = drop_path

    def _dp(self, x):
        if not self.training or self.drop_path == 0:
            return x
        keep = torch.rand(1, device=x.device) > self.drop_path   # per-batch stochastic depth
        return x * keep / (1 - self.drop_path)

    def forward(self, x, cu, maxlen):
        n, d = x.shape
        qkv = self.qkv(self.n1(x)).view(n, 3, self.heads, d // self.heads)
        dt = torch.bfloat16
        a = flash_attn_varlen_qkvpacked_func(qkv.to(dt), cu, maxlen).reshape(n, d).to(x.dtype)
        x = x + self._dp(self.proj(a))
        return x + self._dp(self.mlp(self.n2(x)))


class Encoder(nn.Module):
    def __init__(self, d=256, depth=8, heads=8, drop_path=0.0, final_norm=False):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, heads, drop_path=drop_path * i / max(1, depth - 1))
                                     for i in range(depth)])
        self.norm = nn.LayerNorm(d) if final_norm else nn.Identity()

    def forward(self, x, cu, maxlen, return_layers=()):
        outs = []
        for i, b in enumerate(self.blocks):
            x = b(x, cu, maxlen)
            if i in return_layers:
                outs.append(x)
        x = self.norm(x)
        return (x, outs) if return_layers else x


def pack(evt_list, n_events):
    """Order tokens from several sources so each event is contiguous.

    evt_list: list of [N_i] event-index tensors. Returns (perm, cu_seqlens, maxlen, evt_sorted)
    where perm indexes into torch.cat(sources)."""
    evt = torch.cat(evt_list)
    perm = torch.argsort(evt, stable=True)
    cnt = torch.bincount(evt, minlength=n_events)
    cu = F.pad(torch.cumsum(cnt, 0), (1, 0)).to(torch.int32)
    return perm, cu, int(cnt.max()), evt[perm]
