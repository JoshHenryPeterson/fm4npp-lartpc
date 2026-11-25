"""GPU tokenizers: charge hits -> occupied wire-time cell tokens; ophits -> per-PMT tokens.

All operations act on a packed batch (hits of all events concatenated, with an event index),
so the cell size can change without re-preprocessing the cache.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch_scatter import scatter_add, scatter_max, scatter_mean

from qlfm import geometry as G


@dataclass
class TokCfg:
    cell_wires: int = 16
    cell_ticks: int = 64
    sub_w: int = 4             # MAE target sub-grid inside a cell (wires)
    sub_t: int = 8             # MAE target sub-grid inside a cell (ticks)
    q_scale: float = 200.0     # integral normalisation (ADC*tick)
    rms_scale: float = 5.0
    max_tokens: int = 4096     # per-event charge-token cap (random subsample above)
    # 3D Wire-Cell blob tokens (#16): blobs grouped into cubic voxels per TPC
    blob_voxel: float = 10.0   # voxel edge (cm)
    blob_sub: int = 2          # MAE target sub-grid per axis inside a voxel (blob_sub**3 cells)
    blob_q_scale: float = 5000.0   # blob charge normalisation (electrons)
    max_blob_tokens: int = 4096
    # PMT time bins (us, beam-window clock); fine around the expected neutrino flash
    light_edges: tuple = G.LIGHT_EDGES      # detector-dependent (qlfm.geometry)


@dataclass
class ChargeTokens:
    feat_hit: torch.Tensor     # [Nh, 4] per-hit in-cell features
    hit2tok: torch.Tensor      # [Nh] token index of each hit (-1 if dropped by the cap)
    tok_evt: torch.Tensor      # [Nt] event index
    tok_plane: torch.Tensor    # [Nt]
    tok_pos: torch.Tensor      # [Nt, 2] (drift x cm, wire coordinate cm) charge-weighted
    tok_stat: torch.Tensor     # [Nt, 2] (log n_hits, log sum q)
    tok_target: torch.Tensor   # [Nt, sub_w*sub_t] log1p sub-grid charge (MAE target)
    tok_time: torch.Tensor     # [Nt] mean tick (for strip masking)
    tok_cell: torch.Tensor     # [Nt, 2] (wire-cell index, time-cell index)
    tok_cpos: torch.Tensor     # [Nt, 2] charge-weighted centroid (drift x cm, wire coordinate cm)
    tok_crel: torch.Tensor     # [Nt, 2] centroid offset inside its cell, in [-0.5, 0.5]
    hit_keep: torch.Tensor     # [Nh] bool, hit belongs to a kept token
    # optional 3D information (#16), filled when the batch carries it
    tok_xyz: Optional[torch.Tensor] = None     # [Nt, 3] charge-weighted 3D position of the token's hits (cm), from
                                               # the Wire-Cell blobs they belong to (hit-blob association)
    tok_has3d: Optional[torch.Tensor] = None   # [Nt] bool, some hit of the token lies on a blob
    tok_txyz: Optional[torch.Tensor] = None    # [Nt, 3] charge-weighted truth 3D position (simulation, P3a target)
    tok_t3ok: Optional[torch.Tensor] = None    # [Nt] bool, truth position available


@dataclass
class LightTokens:
    feat: torch.Tensor         # [B*32, n_bins + 1] log1p(PE per time bin), log1p(total PE)
    evt: torch.Tensor          # [B*32]
    pmt: torch.Tensor          # [B*32] channel id


def tokenize_charge(plane, wire, time, q, rms, evt, n_events, cfg: TokCfg, generator=None, tpc=None,
                    xyz_hit=None, truth_xyz_hit=None):
    """plane/wire/evt long [Nh]; time/q/rms float [Nh] (time in ticks); tpc long [Nh] or None (one TPC).
    Tokens never mix TPCs: the cell key is (event, tpc, plane, wire cell, time cell).
    xyz_hit / truth_xyz_hit: optional [Nh, 3] per-hit 3D positions (NaN = none), averaged per token with the hit
    charge as weight (tok_xyz/tok_has3d and tok_txyz/tok_t3ok)."""
    dev = plane.device
    cw, ct = cfg.cell_wires, cfg.cell_ticks
    wc = torch.div(wire, cw, rounding_mode="floor")
    tc = torch.div(time, ct, rounding_mode="floor").long()
    n_wc = 3456 // cw + 1
    n_tc = 8192 // ct + 1
    n_tpc = G.N_TPC
    et = evt * n_tpc + (tpc if tpc is not None and n_tpc > 1 else 0)
    key = ((et * 3 + plane) * n_wc + wc) * n_tc + tc
    ukey, inv = torch.unique(key, return_inverse=True)            # sorted -> grouped by event
    nt = ukey.numel()
    tok_evt = torch.div(ukey, n_tpc * 3 * n_wc * n_tc, rounding_mode="floor")
    tok_plane = torch.div(ukey, n_wc * n_tc, rounding_mode="floor") % 3

    # in-cell coordinates and per-hit features
    rw = (wire - wc * cw).float() / cw
    rt = (time - tc * ct).float() / ct
    lq = torch.log1p(q.clamp(min=0) / cfg.q_scale)
    feat_hit = torch.stack([rw - 0.5, rt - 0.5, lq, rms / cfg.rms_scale], 1)

    # cell centre (used by the "fourier"/"grid" position modes)
    tok_wc = torch.div(ukey, n_tc, rounding_mode="floor") % n_wc
    tok_tc = ukey % n_tc
    ctr_t = (tok_tc.float() + 0.5) * ct
    ctr_w = (tok_wc.float() + 0.5) * cw
    tok_pos = torch.stack([(ctr_t - G.TRIG_TICK) * G.TICK_TO_CM, ctr_w * G.WIRE_PITCH_CM], 1)
    mt = ctr_t
    w = q.clamp(min=1e-3)
    wsum = scatter_add(w, inv, dim=0, dim_size=nt)
    # charge-weighted centroid (Point-MAE-style patch centre: lies on the track)
    cw_ = scatter_add(w * wire.float(), inv, dim=0, dim_size=nt) / wsum
    ctt = scatter_add(w * time, inv, dim=0, dim_size=nt) / wsum
    tok_cpos = torch.stack([(ctt - G.TRIG_TICK) * G.TICK_TO_CM, cw_ * G.WIRE_PITCH_CM], 1)
    tok_crel = torch.stack([(cw_ - tok_wc * cw) / cw - 0.5, (ctt - tok_tc * ct) / ct - 0.5], 1)
    nh = scatter_add(torch.ones_like(q), inv, dim=0, dim_size=nt)
    tok_stat = torch.stack([torch.log(nh), torch.log1p(wsum / cfg.q_scale)], 1)

    sw = (rw * cfg.sub_w).long().clamp(0, cfg.sub_w - 1)
    st = (rt * cfg.sub_t).long().clamp(0, cfg.sub_t - 1)
    nsub = cfg.sub_w * cfg.sub_t
    tgt = scatter_add(q.clamp(min=0) / cfg.q_scale, inv * nsub + sw * cfg.sub_t + st,
                      dim=0, dim_size=nt * nsub).view(nt, nsub)
    tok_target = torch.log1p(tgt)

    # per-event token cap (random subsample, deterministic under `generator`)
    counts = torch.bincount(tok_evt, minlength=n_events)
    keep_tok = torch.ones(nt, dtype=torch.bool, device=dev)
    if counts.max() > cfg.max_tokens:
        r = torch.rand(nt, device=dev, generator=generator)
        for b in torch.nonzero(counts > cfg.max_tokens).flatten().tolist():
            idx = torch.nonzero(tok_evt == b).flatten()
            drop = idx[torch.argsort(r[idx])[cfg.max_tokens:]]
            keep_tok[drop] = False
    new_idx = torch.cumsum(keep_tok.long(), 0) - 1
    hit_keep = keep_tok[inv]
    hit2tok = torch.where(hit_keep, new_idx[inv], torch.full_like(inv, -1))
    k = keep_tok
    extra = {}
    for name, ok_name, xyz in (("tok_xyz", "tok_has3d", xyz_hit), ("tok_txyz", "tok_t3ok", truth_xyz_hit)):
        if xyz is not None:
            ok = ~torch.isnan(xyz[:, 0])
            wq = torch.where(ok, w, torch.zeros_like(w))
            num = scatter_add(wq[:, None] * torch.nan_to_num(xyz.float()), inv, dim=0, dim_size=nt)
            den = scatter_add(wq, inv, dim=0, dim_size=nt)
            extra[name] = (num / den.clamp(min=1e-6)[:, None])[k]
            extra[ok_name] = (den > 0)[k]
    return ChargeTokens(**extra, feat_hit=feat_hit, hit2tok=hit2tok, tok_evt=tok_evt[k], tok_plane=tok_plane[k],
                        tok_pos=tok_pos[k], tok_stat=tok_stat[k], tok_target=tok_target[k],
                        tok_time=mt[k], tok_cell=torch.stack([tok_wc, tok_tc], 1)[k], tok_cpos=tok_cpos[k], tok_crel=tok_crel[k], hit_keep=hit_keep)


def tokenize_light(ch, t, pe, evt, n_events, cfg: TokCfg):
    """ophits -> one token per PMT (G.N_PMT: 32 MicroBooNE, 120 SBND) with a log1p(PE) time profile + log1p total."""
    dev = ch.device
    edges = torch.tensor(cfg.light_edges, device=dev, dtype=torch.float32)
    nb = len(cfg.light_edges) - 1
    b = (torch.bucketize(t.float(), edges) - 1).clamp(0, nb - 1)
    # ~1% of ophits lie outside the beam window (t < 0 or t >> 23.4 us): drop them
    inwin = (t.float() >= edges[0]) & (t.float() < edges[-1])
    pe = torch.where(inwin, pe.float().clamp(min=0), torch.zeros_like(pe, dtype=torch.float32))
    idx = (evt * G.N_PMT + ch) * nb + b
    hist = scatter_add(pe, idx, dim=0, dim_size=n_events * G.N_PMT * nb).view(n_events * G.N_PMT, nb)
    tot = hist.sum(1, keepdim=True)
    feat = torch.cat([torch.log1p(hist), torch.log1p(tot)], 1)
    evt_t = torch.arange(n_events, device=dev).repeat_interleave(G.N_PMT)
    pmt = torch.arange(G.N_PMT, device=dev).repeat(n_events)
    return LightTokens(feat=feat, evt=evt_t, pmt=pmt)


def pointnet_pool(h, hit2tok, nt):
    """Max+mean pool per-hit embeddings into tokens (hits with hit2tok<0 ignored)."""
    m = hit2tok >= 0
    idx = hit2tok[m]
    hm = scatter_max(h[m], idx, dim=0, dim_size=nt)[0]
    ha = scatter_mean(h[m], idx, dim=0, dim_size=nt)
    return torch.cat([hm, ha], 1)


# ---------------- in-beam flash light tokens (#16) ----------------
@dataclass
class FlashTokens:
    """One token per event from the in-beam (brightest in-window) OpFlash, or per-PMT tokens carrying that flash."""
    feat: torch.Tensor         # flash: [B, N_PMT + 3]; pmt_flash: [B*N_PMT, 3]
    evt: torch.Tensor
    pmt: Optional[torch.Tensor] = None     # pmt_flash only


def tokenize_flash(fl_pmt, fl_t, fl_pe, mode="flash"):
    """fl_pmt [B, N_PMT] per-PMT PE of the event's brightest in-window flash (cache ev_fl_pmt), fl_t [B] its time
    (us, beam clock; < -1e3 when there is none), fl_pe [B] its total PE.
    mode "flash": one token per event = (log1p PE per PMT, time / 3 us, log1p total, has-flash);
    mode "pmt_flash": one token per PMT = (log1p PE of that PMT in the flash, log1p total, has-flash)."""
    B = fl_pmt.shape[0]
    dev = fl_pmt.device
    has = (fl_t > -1e3).float()
    t = torch.where(has > 0, fl_t.float(), torch.zeros_like(fl_t, dtype=torch.float32)) / 3.0
    lp = torch.log1p(fl_pmt.float().clamp(min=0))
    lt = torch.log1p(fl_pe.float().clamp(min=0))
    if mode == "flash":
        feat = torch.cat([lp, t[:, None], lt[:, None], has[:, None]], 1)
        return FlashTokens(feat=feat, evt=torch.arange(B, device=dev))
    n = fl_pmt.shape[1]
    feat = torch.stack([lp.flatten(), lt.repeat_interleave(n), has.repeat_interleave(n)], 1)
    return FlashTokens(feat=feat, evt=torch.arange(B, device=dev).repeat_interleave(n),
                       pmt=torch.arange(n, device=dev).repeat(B))


# ---------------- 3D Wire-Cell blob tokens (#16) ----------------
@dataclass
class BlobTokens:
    feat_blob: torch.Tensor    # [Nb, 8] per-blob features inside its voxel
    blob2tok: torch.Tensor     # [Nb] token index of each blob (-1 if dropped by the cap)
    tok_evt: torch.Tensor      # [Nt]
    tok_tpc: torch.Tensor      # [Nt]
    tok_cpos: torch.Tensor     # [Nt, 3] charge-weighted blob-centre centroid (cm)
    tok_stat: torch.Tensor     # [Nt, 2] (log n_blobs, log1p sum q)
    tok_target: torch.Tensor   # [Nt, blob_sub**3] log1p sub-voxel charge (MAE target)


def tokenize_blobs(xyz, q, ext, tpc, evt, n_events, cfg: TokCfg, generator=None):
    """Blobs -> occupied voxel tokens. xyz [Nb, 3] blob centre (cm), q [Nb] solved charge (electrons),
    ext [Nb, 4] blob extent (cm: drift span, U/V/W wire spans), tpc/evt long [Nb]."""
    dev = xyz.device
    v = cfg.blob_voxel
    lo = torch.tensor(G.POS3D_MIN, device=dev, dtype=torch.float32)
    span = torch.tensor(G.POS3D_SCALE, device=dev, dtype=torch.float32)
    nvox = (span / v).ceil().long() + 2
    rel = (xyz.float() - lo) / v + 1.0                      # +1: room for blobs slightly outside the AV
    ijk = rel.floor().long()
    ijk = torch.minimum(torch.maximum(ijk, torch.zeros_like(ijk)), nvox - 1)
    n_tpc = G.N_TPC
    et = evt * n_tpc + (tpc if n_tpc > 1 else 0)
    key = ((et * nvox[0] + ijk[:, 0]) * nvox[1] + ijk[:, 1]) * nvox[2] + ijk[:, 2]
    ukey, inv = torch.unique(key, return_inverse=True)
    nt = ukey.numel()
    per_tok = nvox[0] * nvox[1] * nvox[2]
    tok_et = torch.div(ukey, per_tok, rounding_mode="floor")
    tok_evt = torch.div(tok_et, n_tpc, rounding_mode="floor")
    tok_tpc = tok_et % n_tpc
    inside = (rel - ijk.float()).clamp(0, 1)               # position inside the voxel, [0, 1]
    lq = torch.log1p(q.float().clamp(min=0) / cfg.blob_q_scale)
    feat_blob = torch.cat([inside - 0.5, lq[:, None], ext.float() / v], 1)
    w = q.float().clamp(min=1.0)
    wsum = scatter_add(w, inv, dim=0, dim_size=nt)
    cpos = scatter_add(w[:, None] * xyz.float(), inv, dim=0, dim_size=nt) / wsum[:, None]
    nb = scatter_add(torch.ones_like(w), inv, dim=0, dim_size=nt)
    stat = torch.stack([torch.log(nb), torch.log1p(wsum / cfg.blob_q_scale)], 1)
    sb = cfg.blob_sub
    sub = (inside * sb).long().clamp(0, sb - 1)
    sidx = (sub[:, 0] * sb + sub[:, 1]) * sb + sub[:, 2]
    tgt = scatter_add(q.float().clamp(min=0) / cfg.blob_q_scale, inv * sb ** 3 + sidx, dim=0,
                      dim_size=nt * sb ** 3).view(nt, sb ** 3)
    # per-event token cap (random subsample)
    keep = torch.ones(nt, dtype=torch.bool, device=dev)
    counts = torch.bincount(tok_evt, minlength=n_events)
    if counts.max() > cfg.max_blob_tokens:
        r = torch.rand(nt, device=dev, generator=generator)
        for b in torch.nonzero(counts > cfg.max_blob_tokens).flatten().tolist():
            idx = torch.nonzero(tok_evt == b).flatten()
            keep[idx[torch.argsort(r[idx])[cfg.max_blob_tokens:]]] = False
    new_idx = torch.cumsum(keep.long(), 0) - 1
    blob2tok = torch.where(keep[inv], new_idx[inv], torch.full_like(inv, -1))
    return BlobTokens(feat_blob=feat_blob, blob2tok=blob2tok, tok_evt=tok_evt[keep], tok_tpc=tok_tpc[keep],
                      tok_cpos=cpos[keep], tok_stat=stat[keep], tok_target=torch.log1p(tgt)[keep])


def hit_xyz_from_blobs(n_hits, hb_hit, hb_blob, blob_xyz, blob_q):
    """Per-hit 3D position = blob-charge-weighted mean centre of the blobs the hit belongs to (NaN if none).
    hb_hit / hb_blob: [Np] global hit / blob indices of the hit-blob pairs."""
    w = blob_q.float().clamp(min=1.0)[hb_blob]
    num = scatter_add(w[:, None] * blob_xyz.float()[hb_blob], hb_hit, dim=0, dim_size=n_hits)
    den = scatter_add(w, hb_hit, dim=0, dim_size=n_hits)
    out = num / den.clamp(min=1e-6)[:, None]
    out[den <= 0] = float("nan")
    return out
