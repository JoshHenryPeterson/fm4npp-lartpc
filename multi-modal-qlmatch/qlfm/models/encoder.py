"""Token embedders and a packed (variable-length) Transformer encoder using FlashAttention-2."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_qkvpacked_func

from qlfm import geometry as G
from torch_scatter import scatter_mean

from qlfm.tokenize import BlobTokens, ChargeTokens, FlashTokens, LightTokens, pointnet_pool


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


class SparseStem(nn.Module):
    """Charge tokenizer stem (#16): submanifold sparse 2D convolutions on the wire x (tick / t_bin) image of each
    (event, TPC, plane); returns a per-hit feature that is added to the per-hit MLP before the per-cell pooling.
    Needs spconv (the pimm venv). Runs in float32."""

    def __init__(self, out_dim, ch=(32, 64, 64), t_bin=4):
        super().__init__()
        import spconv.pytorch as spconv
        self.spconv, self.t_bin = spconv, t_bin
        self.inp = nn.Linear(2, ch[0])
        mods, cin = [], ch[0]
        for c in ch:
            mods += [spconv.SubMConv2d(cin, c, 3, padding=1, bias=False, indice_key="stem"), nn.LayerNorm(c), nn.GELU()]
            cin = c
        self.net = spconv.SparseSequential(*mods)
        self.out = nn.Linear(cin, out_dim)
        nn.init.zeros_(self.out.weight)                 # starts as a no-op on top of the per-hit MLP
        nn.init.zeros_(self.out.bias)

    def forward(self, img, wire, tick, lq, rms):
        with torch.autocast("cuda", enabled=False):
            t = torch.div(tick.float(), self.t_bin, rounding_mode="floor").long().clamp(0, 8191)
            key = (img * 4096 + wire.clamp(0, 4095)) * 8192 + t
            uk, inv = torch.unique(key, return_inverse=True)
            f = scatter_mean(torch.stack([lq.float(), rms.float()], 1), inv, dim=0, dim_size=uk.numel())
            idx = torch.stack([torch.div(uk, 4096 * 8192, rounding_mode="floor"),
                               torch.div(uk, 8192, rounding_mode="floor") % 4096, uk % 8192], 1).int()
            x = self.spconv.SparseConvTensor(self.inp(f), idx, [4096, 8192], int(img.max()) + 1)
            y = self.net(x).features
            return self.out(y)[inv]


class ChargeEmbed(nn.Module):
    def __init__(self, d, hit_dim=128, pos_mode="fourier", n_freq=16, n_cells=(512, 512), stem="mlp", pos3d=False):
        super().__init__()
        self.pos_mode = pos_mode
        self.hit_mlp = mlp([4, 64, hit_dim], last_act=True)
        self.stem = SparseStem(hit_dim) if stem == "cnn" else None
        if pos3d:                                           # 3D position of the token's hits from Wire-Cell blobs
            self.pos3 = Fourier(3, d, n_freq=n_freq, scale=G.POS3D_SCALE)
            self.no3d = nn.Parameter(torch.zeros(d))
            self.register_buffer("pos3_min", torch.tensor(G.POS3D_MIN, dtype=torch.float32))
        else:
            self.pos3 = None
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

    def forward(self, ct: ChargeTokens, stem_in=None):
        h = self.hit_mlp(ct.feat_hit)                        # [Nh, hit_dim], reused by hit heads
        if self.stem is not None:
            h = h + self.stem(*stem_in).to(h.dtype)
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
        if self.pos3 is not None and ct.tok_xyz is not None:
            p3 = self.pos3(ct.tok_xyz - self.pos3_min)
            e = e + torch.where(ct.tok_has3d[:, None], p3, self.no3d.to(p3.dtype))
        return e


class LightEmbed(nn.Module):
    """Per-PMT light tokens. pmt_pos "v1": Fourier(y, z) with the MicroBooNE scales + PMT id (all checkpoints before
    #16); "v2": scales from the detector geometry and a PMT-plane (TPC side) embedding, so facing SBND PMTs differ."""

    def __init__(self, d, n_feat, pmt_pos="v1"):
        super().__init__()
        self.proj = mlp([n_feat, d, d])
        self.pos = Fourier(2, d, scale=(130.0, 1100.0) if pmt_pos == "v1" else G.PMT_POS_SCALE)
        self.pmt = nn.Embedding(G.N_PMT, d)
        self.register_buffer("pmt_yz", torch.from_numpy(G.PMT_YZ.copy()) + torch.tensor([G.PMT_Y_SHIFT, 0.0]))
        if pmt_pos == "v2":
            self.side = nn.Embedding(2, d)
            self.register_buffer("pmt_side", torch.from_numpy(G.PMT_SIDE.copy()))
        else:
            self.side = None

    def forward(self, lt):
        return self.proj(lt.feat) + self.pos_embed(lt)

    def pos_embed(self, lt):
        e = self.pos(self.pmt_yz[lt.pmt]) + self.pmt(lt.pmt)
        if self.side is not None:
            e = e + self.side(self.pmt_side[lt.pmt])
        return e


class FlashEmbed(nn.Module):
    """One token per event from the in-beam OpFlash (tokenize_flash mode "flash")."""

    def __init__(self, d, n_feat):
        super().__init__()
        self.proj = mlp([n_feat, d, d])
        self.tok = nn.Parameter(torch.zeros(d))

    def forward(self, ft: FlashTokens):
        return self.proj(ft.feat) + self.pos_embed(ft)

    def pos_embed(self, ft: FlashTokens):
        return self.tok.expand(ft.feat.shape[0], -1)


class BlobEmbed(nn.Module):
    """3D Wire-Cell blob voxel tokens (#16): per-blob MLP + max/mean pool per voxel, 3D Fourier position of the
    voxel's charge centroid, TPC embedding, and ("nexus") the pooled per-hit embeddings of the hits on the voxel's
    blobs, which ties each 3D token to its 2D evidence."""

    def __init__(self, d, hit_dim, blob_dim=128, n_freq=16, n_feat=8):
        super().__init__()
        self.blob_mlp = mlp([n_feat, 64, blob_dim], last_act=True)
        self.proj = mlp([2 * blob_dim + 2, d, d])
        self.pos = Fourier(3, d, n_freq=n_freq, scale=G.POS3D_SCALE)
        self.register_buffer("pos_min", torch.tensor(G.POS3D_MIN, dtype=torch.float32))
        self.tpc = nn.Embedding(max(G.N_TPC, 2), d)
        self.nexus = mlp([2 * hit_dim, d, d])
        nn.init.zeros_(self.nexus[-1].weight)              # starts as a no-op
        nn.init.zeros_(self.nexus[-1].bias)

    def forward(self, bt: BlobTokens, h_hit=None, hb_hit=None, hb_blob=None):
        nt = bt.tok_evt.numel()
        pooled = pointnet_pool(self.blob_mlp(bt.feat_blob), bt.blob2tok, nt)
        x = self.proj(torch.cat([pooled, bt.tok_stat], 1)) + self.pos_embed(bt)
        if h_hit is not None and hb_hit is not None and hb_hit.numel():
            btok = bt.blob2tok[hb_blob]
            x = x + self.nexus(pointnet_pool(h_hit[hb_hit], btok, nt)).to(x.dtype)
        return x

    def pos_embed(self, bt: BlobTokens):
        return self.pos(bt.tok_cpos - self.pos_min) + self.tpc(bt.tok_tpc)


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
