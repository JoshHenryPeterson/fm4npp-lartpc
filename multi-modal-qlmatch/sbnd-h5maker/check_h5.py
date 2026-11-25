#!/usr/bin/env python3
"""Sanity summary of an sbnd_h5maker output file (needs only h5py + numpy).

    python check_h5.py FILE.h5 [--json OUT.json]

Checks the hep_hpc layout (event_id / event_id.seq_cnt per table, hit_id == row index within the event),
and prints the distributions qlfm cares about: hits per plane and TPC, wire / tick ranges, nu-truth hits,
optical-hit channels and times, flashes, and the Wire-Cell nu-candidate labels against the truth
(efficiency / purity of WC nu-cluster hits for hits with a beam-neutrino deposit).
"""
import argparse
import json

import h5py
import numpy as np

# data mode: minimum number of events with a WC nu candidate before the footprint-fraction numbering check applies
MIN_CAND_EVENTS_FOOTPRINT = 10

def offsets(f, table, n_ev):
    sc = f[table + "/event_id.seq_cnt"][:]
    cnt = np.zeros(n_ev, np.int64)
    cnt[sc[:, 0]] = sc[:, 1]
    start = np.zeros(n_ev, np.int64)
    start[1:] = np.cumsum(cnt)[:-1]
    return start, cnt


def col(f, t, c):
    a = f[t + "/" + c][:]
    return a[:, 0] if a.ndim == 2 and a.shape[1] == 1 else a


def q(a, ps=(0, 1, 50, 99, 100)):
    return [round(float(v), 3) for v in np.percentile(a, ps)] if len(a) else []


def check_blobs(f, n_ev, problems, is_data):
    """0.6 blob tables: hit-blob pairs inside the blob ranges, cluster ids consistent with wc_hit_table and
    wc_cluster_table, blob charge vs the charge of its hits, and (sim) trackid coverage and dominant share."""
    B = {}
    bs, bc = offsets(f, "wc_blob_table", n_ev)
    pb_s, pb_c = offsets(f, "wc_hit_blob_table", n_ev)
    hs, hc = offsets(f, "hit_table", n_ev)
    cs, cc = offsets(f, "wc_cluster_table", n_ev)
    blob = {c: col(f, "wc_blob_table", c) for c in f["wc_blob_table"] if not c.startswith("event_id")}
    nb = len(blob["blob_id"])
    B["blobs"] = int(nb)
    B["blobs_per_event_q"] = q(bc)
    st = col(f, "event_table", "wc_blob_status")
    B["events_with_blob_record"] = int((st == 1).sum())
    if (st == 0).any():
        problems.append("blobs: %d events have no blob record" % int((st == 0).sum()))
    bev = np.repeat(np.arange(n_ev), bc)
    if not np.array_equal(blob["blob_id"], np.arange(nb) - bs[bev]):
        problems.append("wc_blob_table: blob_id != row index within the event")
    # cluster ids: every blob cluster is a wc_cluster_table row of its event
    ccl = col(f, "wc_cluster_table", "cluster_id").astype(np.int64)
    cev = np.repeat(np.arange(n_ev), cc)
    ckey = set((cev * 100000 + ccl).tolist())
    bkey = bev * 100000 + blob["cluster_id"].astype(np.int64)
    miss = int((~np.isin(bkey, list(ckey))).sum())
    B["blobs_with_cluster_row"] = int(nb - miss)
    if miss:
        problems.append("blobs: %d blobs whose cluster_id has no wc_cluster_table row" % miss)
    # hit-blob pairs
    hid = col(f, "wc_hit_blob_table", "hit_id").astype(np.int64)
    bid = col(f, "wc_hit_blob_table", "blob_id").astype(np.int64)
    pev = np.repeat(np.arange(n_ev), pb_c)
    gh = hs[pev] + hid               # global hit row
    gb = bs[pev] + bid               # global blob row
    wi = col(f, "hit_table", "local_wire").astype(np.int64)
    tick = col(f, "hit_table", "local_time")
    plane = col(f, "hit_table", "local_plane").astype(np.int64)
    tpc = col(f, "hit_table", "tpc").astype(np.int64)
    integ = col(f, "hit_table", "integral")
    off = float(f.attrs.get("wc_tick_offset", 0))
    t_ = np.floor(tick + off).astype(np.int64)
    pn = ("u", "v", "w")
    wmin = np.stack([blob[p + "_wire_index_min"] for p in pn]).astype(np.int64)
    wmax = np.stack([blob[p + "_wire_index_max"] for p in pn]).astype(np.int64)
    ok = ((tpc[gh] == blob["tpc"][gb]) & (t_[gh] >= blob["slice_index_min"][gb]) & (t_[gh] < blob["slice_index_max"][gb])
          & (wi[gh] >= wmin[plane[gh], gb]) & (wi[gh] < wmax[plane[gh], gb]))
    B["hit_blob_pairs"] = int(len(gh))
    B["hit_blob_pairs_outside_ranges"] = int((~ok).sum())
    if (~ok).any():
        problems.append("wc_hit_blob_table: %d pairs outside the blob ranges" % int((~ok).sum()))
    inb = np.zeros(len(wi), bool)
    inb[gh] = True
    B["hit_charge_fraction_in_blobs"] = {"tpc%d_plane%d" % (t, p): round(float(integ[(tpc == t) & (plane == p) & inb].sum()
                                         / max(1e-9, integ[(tpc == t) & (plane == p)].sum())), 4)
                                         for t in range(2) for p in range(3)}
    # wc_hit_table cluster labels vs the clusters of the blobs holding the hit
    wcid = col(f, "wc_hit_table", "wc_cluster_id").astype(np.int64)
    lab = wcid >= 0
    hb_cl = blob["cluster_id"][gb].astype(np.int64)
    agree = np.zeros(len(wi), bool)
    agree[gh[hb_cl == wcid[gh]]] = True
    B["wc_labelled_hits"] = int(lab.sum())
    B["wc_labelled_hits_in_a_blob_of_that_cluster"] = int((lab & agree).sum())
    B["wc_labelled_hits_in_any_blob"] = int((lab & inb).sum())
    # cluster charge vs the charge of its hits, per plane: sum of its blobs' solved charge against the summed
    # integral of the distinct hits inside its blobs (a blob is one 4-tick slice, so single blobs are too noisy)
    B["cluster_vs_hit_charge"] = {}
    ckey_b = bev * 100000 + blob["cluster_id"].astype(np.int64)
    uk, binv = np.unique(ckey_b, return_inverse=True)
    qc = np.bincount(binv, blob["charge"], minlength=len(uk))
    for p in range(3):
        m = plane[gh] == p
        pair = np.unique(np.stack([binv[gb[m]], gh[m]]), axis=1)        # distinct (cluster, hit)
        hq = np.bincount(pair[0], integ[pair[1]], minlength=len(uk))
        sel = (hq > 0) & (qc > 0) & (np.bincount(binv, minlength=len(uk)) >= 10)
        if sel.sum() > 10:
            r = np.corrcoef(np.log(hq[sel]), np.log(qc[sel]))[0, 1]
            B["cluster_vs_hit_charge"][pn[p]] = {"clusters": int(sel.sum()), "log_corr": round(float(r), 3),
                                                 "median_charge_per_adc": round(float(np.median(qc[sel] / hq[sel])), 2)}
            if r < 0.8:
                problems.append("blobs: plane %s cluster charge vs hit charge log-correlation %.2f < 0.8" % (pn[p], r))
    if is_data:
        if (blob["trackid"] != -1).any():
            problems.append("data: blob trackid set")
        return B
    # truth: trackid coverage (charge-weighted), dominant share, track_table coverage
    tid, qb = blob["trackid"].astype(np.int64), blob["charge"]
    B["blob_charge_with_trackid"] = round(float(qb[tid >= 0].sum() / max(1e-9, qb.sum())), 4)
    B["blobs_with_trackid"] = round(float((tid >= 0).mean()) if nb else 0.0, 4)
    tq, tqd = blob["truth_q"], blob["truth_q_dom"]
    m = tq > 0
    B["dominant_share_chargeweighted"] = round(float((qb[m] * tqd[m] / tq[m]).sum() / max(1e-9, qb[m].sum())), 4)
    B["dominant_share_q"] = q(tqd[m] / tq[m], (5, 25, 50, 75))
    if "track_table" in f:
        ts, tc = offsets(f, "track_table", n_ev)
        tt = col(f, "track_table", "trackid").astype(np.int64)
        tev = np.repeat(np.arange(n_ev), tc)
        tk = set((tev * 100000000 + tt).tolist())
        bk = bev[tid >= 0] * 100000000 + tid[tid >= 0]
        found = np.isin(bk, list(tk))
        B["blob_trackids_in_track_table"] = round(float(found.mean()) if len(found) else 1.0, 4)
        if len(found) and found.mean() < 0.999:
            problems.append("blobs: %.3f of blob trackids are missing from track_table" % (1 - found.mean()))
        nu = col(f, "track_table", "nu_idx")
        B["tracks"] = int(len(tt))
        B["tracks_nu"] = int((nu >= 0).sum())
        B["tracks_with_kinematics"] = int(np.isfinite(col(f, "track_table", "start_px")).sum())
    if "hit_truth_table" in f:
        e = col(f, "hit_truth_table", "energy")
        x = col(f, "hit_truth_table", "true_x")
        B["hits_with_truth_xyz"] = int(np.isfinite(x).sum())
        if (np.isfinite(x) != (e > 0)).any():
            problems.append("hit_truth_table: true_x finite != energy > 0")
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--json")
    a = ap.parse_args()
    f = h5py.File(a.h5, "r")
    R = {"file": a.h5, "tables": {}}
    n_ev = f["event_table/event_id"].shape[0]
    R["events"] = n_ev
    problems = []
    for t in f.keys():
        g = f[t]
        n = g["event_id"].shape[0]
        R["tables"][t] = {"rows": int(n), "columns": {c: list(g[c].shape[1:]) for c in g if not c.startswith("event_id.")}}
        s, c = offsets(f, t, n_ev)
        if int(c.sum()) != n:
            problems.append("%s: seq_cnt sum %d != rows %d" % (t, c.sum(), n))
        ev = g["event_id"][:]
        evt = np.repeat(np.arange(n_ev), c)
        ref = f["event_table/event_id"][:][evt]
        if not np.array_equal(ev, ref):
            problems.append("%s: event_id rows do not follow the event order" % t)
    for t in ("hit_table", "ophit_table", "wc_hit_table"):
        s, c = offsets(f, t, n_ev)
        hid = col(f, t, "hit_id")
        evt = np.repeat(np.arange(n_ev), c)
        if not np.array_equal(hid, np.arange(len(hid)) - s[evt]):
            problems.append("%s: hit_id != row index within the event" % t)
    hs, hc = offsets(f, "hit_table", n_ev)
    ws, wcnt = offsets(f, "wc_hit_table", n_ev)
    if not np.array_equal(hc, wcnt):
        problems.append("wc_hit_table rows != hit_table rows per event")
    R["problems"] = problems

    # ---- events
    ev = {c: col(f, "event_table", c) for c in ("nu_pdg", "is_cc", "nu_energy", "n_nu", "wc_status", "beam_gate_start_ns", "n_mcp_no_truth")}
    vtx = col(f, "event_table", "nu_vtx")
    av = (np.abs(vtx[:, 0]) < 200) & (np.abs(vtx[:, 1]) < 200) & (vtx[:, 2] > 0) & (vtx[:, 2] < 500)
    R["event"] = {"nu_pdg": {int(k): int(v) for k, v in zip(*np.unique(ev["nu_pdg"], return_counts=True))},
                  "is_cc": int((ev["is_cc"] == 1).sum()), "nu_in_av": int(av.sum()),
                  "nu_energy_q": q(ev["nu_energy"]), "n_nu": {int(k): int(v) for k, v in zip(*np.unique(ev["n_nu"], return_counts=True))},
                  "wc_status_1": int((ev["wc_status"] == 1).sum()), "beam_gate_start_ns": q(ev["beam_gate_start_ns"]),
                  "mcp_without_mctruth": int(ev["n_mcp_no_truth"].sum())}

    # ---- hits
    tpc, pl, wire, tick = (col(f, "hit_table", c) for c in ("tpc", "local_plane", "local_wire", "local_time"))
    integ, rms = col(f, "hit_table", "integral"), col(f, "hit_table", "rms")
    evt = np.repeat(np.arange(n_ev), hc)
    H = {"per_event_q": q(hc), "by_tpc_plane": {}}
    for t in np.unique(tpc):
        for p in np.unique(pl):
            m = (tpc == t) & (pl == p)
            H["by_tpc_plane"]["tpc%d_plane%d" % (t, p)] = {
                "hits": int(m.sum()), "wire_max": int(wire[m].max()) if m.any() else -1,
                "tick_q": q(tick[m]), "integral_q": q(integ[m]), "rms_q": q(rms[m])}
    R["hits"] = H

    # ---- truth per hit
    nu_e = np.zeros(len(tick))
    for t in ("edep_table", "edep_cosmic_table"):
        s, c = offsets(f, t, n_ev)
        e_evt = np.repeat(np.arange(n_ev), c)
        row = hs[e_evt] + col(f, t, "hit_id")
        e = np.zeros(len(tick))
        np.add.at(e, row, col(f, t, "energy"))
        if t == "edep_table":
            nu_e = e
        else:
            cos_e = e
    nu_hit = nu_e > 0.05
    R["truth"] = {"hits_nu_gt_0.05MeV": int(nu_hit.sum()), "hits_cosmic_gt_0.05MeV": int((cos_e > 0.05).sum()),
                  "hits_no_deposit": int(((nu_e + cos_e) <= 0).sum()),
                  "nu_collection_hits_per_event_q": q(np.bincount(evt[nu_hit & (pl == 2)], minlength=n_ev)),
                  "particle_category": {int(k): int(v) for k, v in zip(*np.unique(col(f, "particle_table", "category"), return_counts=True))},
                  "particle_origin": {int(k): int(v) for k, v in zip(*np.unique(col(f, "particle_table", "origin"), return_counts=True))}}

    # ---- light
    os_, oc = offsets(f, "ophit_table", n_ev)
    ch, t, pe = (col(f, "ophit_table", c) for c in ("hit_channel", "peaktime", "pe"))
    R["light"] = {"ophits_per_event_q": q(oc), "channels_used": int(len(np.unique(ch))), "channel_range": [int(ch.min()), int(ch.max())] if len(ch) else [],
                  "peaktime_q": q(t), "pe_q": q(pe),
                  "flashes_per_event_q": q(offsets(f, "opflash_table", n_ev)[1]),
                  "flash_time_q": q(col(f, "opflash_table", "time")), "flash_pe_q": q(col(f, "opflash_table", "totalpe"))}
    tb = np.histogram(t, bins=[-1e9, -10, -2, 0, 2, 4, 6, 10, 100, 1e9])[0]
    R["light"]["peaktime_hist_edges_us"] = "-inf,-10,-2,0,2,4,6,10,100,inf"
    R["light"]["peaktime_hist"] = tb.tolist()

    # ---- Wire-Cell
    wcid, wmain = col(f, "wc_hit_table", "wc_cluster_id"), col(f, "wc_hit_table", "wc_main")
    wenu = col(f, "wc_hit_table", "wc_in_enu")
    on = wcid >= 0
    def eff_pur(sel):
        return {"selected_hits": int(sel.sum()),
                "eff_nu_hits": round(float((sel & nu_hit).sum() / max(1, nu_hit.sum())), 4),
                "purity": round(float((sel & nu_hit).sum() / max(1, sel.sum())), 4)}
    R["wc"] = {"candidates": int(f["wc_event_table/event_id"].shape[0]),
               "events_with_candidate": int(len(np.unique(f["wc_event_table/event_id"][:], axis=0))) if f["wc_event_table/event_id"].shape[0] else 0,
               "all_nu_cluster_hits": eff_pur(on), "main_cluster_hits": eff_pur(wmain == 1), "in_enu_hits": eff_pur(wenu == 1)}
    # per (tpc, plane), charge-weighted, over events with a WC nu candidate: a channel-numbering mismatch shows up
    # as one plane with efficiency ~0 while the others are fine
    cand_ev = np.zeros(n_ev, bool)
    if f["wc_event_table/event_id"].shape[0]:
        cs, cc = offsets(f, "wc_event_table", n_ev)
        cand_ev = cc > 0
    inc = cand_ev[evt]
    R["wc"]["by_tpc_plane_charge_weighted"] = {}
    for t_ in np.unique(tpc):
        for p_ in np.unique(pl):
            m = inc & (tpc == t_) & (pl == p_)
            w = integ[m]
            tr, sel = nu_hit[m], on[m]
            eff = float((w * (tr & sel)).sum() / max(1e-9, (w * tr).sum()))
            pur = float((w * (tr & sel)).sum() / max(1e-9, (w * sel).sum()))
            R["wc"]["by_tpc_plane_charge_weighted"]["tpc%d_plane%d" % (t_, p_)] = {
                "eff": round(eff, 4), "purity": round(pur, 4), "selected_hits": int(sel.sum()), "truth_hits": int(tr.sum())}
    # a numbering bug hits single planes: flag a plane far below another plane of the same TPC. A WC miss of the whole
    # neutrino lowers all three planes of a TPC together and is physics, not a bug.
    bp = R["wc"]["by_tpc_plane_charge_weighted"]
    for t_ in np.unique(tpc):
        for p_ in np.unique(pl):
            v = bp["tpc%d_plane%d" % (t_, p_)]
            best = max(bp["tpc%d_plane%d" % (t_, q_)]["eff"] for q_ in np.unique(pl))
            if v["truth_hits"] >= 50 and v["eff"] < 0.2 and best > 0.5:
                problems.append("WC hit labels: tpc%d plane%d efficiency %.3f vs %.3f on another plane (channel numbering?)"
                                % (t_, p_, v["eff"], best))
    # ---- real data (0.5): no truth, so check the labels without it
    if int(f.attrs.get("is_data", 0)):
        D = {}
        for t in ("nu_table", "edep_table", "edep_cosmic_table", "particle_table"):
            n = f[t + "/event_id"].shape[0]
            D[t + "_rows"] = int(n)
            if n:
                problems.append("data: %s has %d rows (expected 0)" % (t, n))
        hf = col(f, "event_table", "has_frameshift")
        D["events_with_frameshift"] = int((hf == 1).sum())
        D["frame_apply_at_caf_ns_q"] = q(col(f, "event_table", "frame_apply_at_caf_ns")[hf == 1]) if (hf == 1).any() else []
        D["wc_status_1"] = int((col(f, "event_table", "wc_status") == 1).sum())
        D["wc_n_bundle_cosmic_q"] = q(col(f, "event_table", "wc_n_bundle_cosmic"))
        if f["wc_event_table/event_id"].shape[0]:
            D["cosmict_flag"] = {str(k): int(v) for k, v in zip(*np.unique(col(f, "wc_event_table", "cosmict_flag"), return_counts=True))}
        # charge fraction on the WC footprint per (tpc, plane) in candidate events: a numbering bug shows as one plane
        # far below the others of the same TPC
        fr, nfp = {}, {}
        for t_ in np.unique(tpc):
            for p_ in np.unique(pl):
                m = inc & (tpc == t_) & (pl == p_)
                fr[(int(t_), int(p_))] = float(integ[m & on].sum() / max(1e-9, integ[m].sum()))
                nfp[(int(t_), int(p_))] = int((m & on).sum())
        D["footprint_charge_fraction"] = {"tpc%d_plane%d" % k: round(v, 4) for k, v in fr.items()}
        D["footprint_hits"] = {"tpc%d_plane%d" % k: v for k, v in nfp.items()}
        # a few candidates make the per-plane fractions noisy (8 of 1,058 beam-off files with 2-5 candidates fired the
        # rule, #14): apply it only with enough candidate events; below that the fractions are still reported
        n_cand_ev = int(cand_ev.sum())
        D["footprint_check_applied"] = n_cand_ev >= MIN_CAND_EVENTS_FOOTPRINT
        for (t_, p_), v in fr.items():
            best = max(w for (tt, _), w in fr.items() if tt == t_)
            if D["footprint_check_applied"] and best > 0.05 and v < 0.2 * best:
                problems.append("data WC hit labels: tpc%d plane%d footprint fraction %.3f vs %.3f (channel numbering?)"
                                % (t_, p_, v, best))
        R["data"] = D
    if "wc_blob_table" in f:
        R["blobs"] = check_blobs(f, n_ev, problems, is_data=bool(int(f.attrs.get("is_data", 0))))
    R["problems"] = problems
    print(json.dumps(R, indent=1))
    if a.json:
        json.dump(R, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
