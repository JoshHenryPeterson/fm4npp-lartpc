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
    R["problems"] = problems
    print(json.dumps(R, indent=1))
    if a.json:
        json.dump(R, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
