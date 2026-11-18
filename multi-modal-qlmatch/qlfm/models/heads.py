"""Small downstream heads on top of (frozen or trainable) FM token features."""
import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max, scatter_mean

from qlfm.models.encoder import mlp


class AttnPool(nn.Module):
    """Attentive probe: n_q learned queries cross-attend over all tokens of each event."""

    def __init__(self, d, n_q=4, heads=8, out_dim=1, hidden=256):
        super().__init__()
        self.n_q, self.h, self.dh = n_q, heads, d // heads
        self.norm = nn.LayerNorm(d)
        self.q = nn.Parameter(torch.randn(n_q, heads, self.dh) * 0.02)
        self.kv = nn.Linear(d, 2 * d)
        self.out = mlp([n_q * d, hidden, out_dim])

    def forward(self, z, evt, n_events):
        n, d = z.shape
        k, v = self.kv(self.norm(z)).float().view(n, 2, self.h, self.dh).unbind(1)
        s = torch.einsum("nhd,qhd->nqh", k, self.q.float()) / self.dh ** 0.5      # [N, Q, H]
        smax = scatter_max(s, evt, dim=0, dim_size=n_events)[0]
        e = torch.exp(s - smax[evt])
        a = e / scatter_add(e, evt, dim=0, dim_size=n_events)[evt].clamp(min=1e-20)
        pooled = scatter_add(a[..., None] * v[:, None], evt, dim=0, dim_size=n_events)  # [B, Q, H, Dh]
        return self.out(pooled.reshape(n_events, self.n_q * d))


class MeanLinear(nn.Module):
    """Linear probe on mean-pooled token features (diagnostic)."""

    def __init__(self, d, out_dim=1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lin = nn.Linear(d, out_dim)

    def forward(self, z, evt, n_events):
        return self.lin(self.norm(scatter_mean(z.float(), evt, dim=0, dim_size=n_events)))


class HitHead(nn.Module):
    """Per-hit logit from the hit embedding (if the backbone provides one), its token's contextual
    feature, and (optionally) a per-event extra vector such as pooled late-fused light."""

    def __init__(self, d, hit_dim=128, hidden=128, extra_dim=0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = mlp([hit_dim + d + extra_dim, hidden, hidden, 1])

    def forward(self, h_hit, z_tok, hit2tok, extra=None):
        m = hit2tok >= 0
        parts = [self.norm(z_tok.float())[hit2tok.clamp(min=0)]]
        if h_hit is not None:
            parts.insert(0, h_hit.float())
        if extra is not None:
            parts.append(extra.float())
        logit = self.net(torch.cat(parts, 1)).squeeze(1)
        return torch.where(m, logit, torch.full_like(logit, -20.0))


class LateLight(nn.Module):
    """Late-fused PMT light, identical for every backbone: 32 per-PMT tokens (time-profile MLP + PMT
    position/ID embedding) trained with the head. Returns tokens with event index and a per-event mean."""

    def __init__(self, d, tok_cfg=None):
        super().__init__()
        from qlfm.models.encoder import LightEmbed
        from qlfm.tokenize import TokCfg
        self.cfg = tok_cfg or TokCfg()
        self.embed = LightEmbed(d, len(self.cfg.light_edges))
        self.modality = nn.Parameter(torch.zeros(d))

    def forward(self, b):
        from qlfm.tokenize import tokenize_light
        lt = tokenize_light(b["oph_ch"], b["oph_t"], b["oph_pe"], b["oph_evt"], b["n_events"], self.cfg)
        z = self.embed(lt) + self.modality
        pooled = scatter_mean(z.float(), lt.evt, dim=0, dim_size=b["n_events"])
        return z, lt.evt, pooled


class XFusion(nn.Module):
    """Strong late fusion (round 3): project every backbone's tokens to a common width, then n_layers of
    [charge tokens cross-attend to the event's 32 PMT tokens + MLP]. Light tokens come from LateLight at the
    same width. Identical for every backbone, so the head size no longer depends on the backbone width."""

    def __init__(self, d_in, d=256, n_layers=2, heads=8, mlp_ratio=2):
        super().__init__()
        self.d, self.h = d, heads
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.layers = nn.ModuleList(nn.ModuleDict(dict(
            nq=nn.LayerNorm(d), nkv=nn.LayerNorm(d), q=nn.Linear(d, d), kv=nn.Linear(d, 2 * d), o=nn.Linear(d, d),
            n2=nn.LayerNorm(d), mlp=mlp([d, mlp_ratio * d, d]))) for _ in range(n_layers))

    def forward(self, z, evt, zl, n_events):
        import torch.nn.functional as F
        x = self.proj(z.float())
        n_pmt = zl.shape[0] // n_events
        lt = zl.float().view(n_events, n_pmt, self.d)              # LateLight emits n_pmt tokens per event, in order
        order = torch.argsort(evt, stable=True)
        es = evt[order]
        cnt = torch.bincount(evt, minlength=n_events)
        start = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(len(es), device=z.device) - start[es]
        X = x.new_zeros(n_events, int(cnt.max()), self.d)
        X[es, pos] = x[order]
        dh = self.d // self.h
        for L in self.layers:
            q = L["q"](L["nq"](X)).view(n_events, -1, self.h, dh).transpose(1, 2)
            k, v = L["kv"](L["nkv"](lt)).view(n_events, n_pmt, 2, self.h, dh).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(n_events, -1, self.d)
            X = X + L["o"](a)
            X = X + L["mlp"](L["n2"](X))
        out = torch.empty_like(x)
        out[order] = X[es, pos]
        return out


class RawHit(nn.Module):
    """Common trainable per-hit adapter (round 3, T1 fairness): plane one-hot, normalized wire and tick,
    log charge and rms -> 64-d. Gives every backbone the same per-hit raw information."""

    def __init__(self, out=64):
        super().__init__()
        self.out = out
        self.net = mlp([7, out, out], last_act=True)

    def forward(self, b):
        from qlfm import geometry as G
        p = b["hit_plane"]
        nw = torch.tensor(G.N_WIRES, device=p.device, dtype=torch.float32)[p]
        f = torch.cat([nn.functional.one_hot(p, 3).float(), (b["hit_wire"].float() / nw)[:, None],
                       (b["hit_time"].float() / G.N_TICKS)[:, None], torch.log1p(b["hit_q"].clamp(min=0) / 200)[:, None],
                       (b["hit_rms"].float() / 5)[:, None]], 1)
        return self.net(f)


class LightHead(nn.Module):
    """T5 (round 4): charge -> light. 32 learned PMT queries cross-attend (n_layers x [cross-attention + MLP]) to the
    event's charge tokens, projected to a common width as in XFusion; each query outputs the PMT's 17 light-token
    features (log1p PE in 16 time bins + log1p total). Identical for every backbone."""

    def __init__(self, d_in, d=256, n_layers=2, heads=8, mlp_ratio=2, n_pmt=None, n_feat=17):
        super().__init__()
        from qlfm import geometry as G
        n_pmt = n_pmt or G.N_PMT
        self.d, self.h, self.n_pmt = d, heads, n_pmt
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.query = nn.Parameter(torch.randn(n_pmt, d) * 0.02)
        self.layers = nn.ModuleList(nn.ModuleDict(dict(
            nq=nn.LayerNorm(d), nkv=nn.LayerNorm(d), q=nn.Linear(d, d), kv=nn.Linear(d, 2 * d), o=nn.Linear(d, d),
            n2=nn.LayerNorm(d), mlp=mlp([d, mlp_ratio * d, d]))) for _ in range(n_layers))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_feat))

    def forward(self, z, evt, n_events):
        import torch.nn.functional as F
        x = self.proj(z.float())
        order = torch.argsort(evt, stable=True)
        es = evt[order]
        cnt = torch.bincount(evt, minlength=n_events)
        start = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(len(es), device=z.device) - start[es]
        L = max(int(cnt.max()), 1)
        K = x.new_zeros(n_events, L, self.d)
        K[es, pos] = x[order]
        keep = torch.zeros(n_events, L, dtype=torch.bool, device=z.device)
        keep[es, pos] = True
        keep[:, 0] |= cnt == 0                                     # an event without charge attends to one zero token
        mask = keep[:, None, None, :]
        Q = self.query[None].expand(n_events, -1, -1)
        dh = self.d // self.h
        for Ly in self.layers:
            q = Ly["q"](Ly["nq"](Q)).view(n_events, self.n_pmt, self.h, dh).transpose(1, 2)
            k, v = Ly["kv"](Ly["nkv"](K)).view(n_events, L, 2, self.h, dh).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(n_events, self.n_pmt, self.d)
            Q = Q + Ly["o"](a)
            Q = Q + Ly["mlp"](Ly["n2"](Q))
        return self.out(Q)                                         # [n_events, n_pmt, n_feat]
