#!/usr/bin/env python3
"""Wire-Cell (WCP) neutrino selection as hit- and event-level labels for HDF5Maker output.

Usage:
  python wc_hit_labels.py OUT.h5 --h5 NeutrinoML_*.h5 --nusel nusel*_WCP_*.root [...] \
         [--checkout checkout_*.root] [--datatier overlay|bnb|ext]

Hit level (from the WC nusel files, samweb tier `celltree`; the T_port_2d/3d trees are not kept):
  TC holds every imaged cell (cluster_id, time_slice, wire_index_u/v/w) and T_match links each
  cluster bundle to a flash. The WC neutrino candidate is the main cluster of the in-time bundle:
  T_match rows with event_type bit 0 (flash in the beam window), preferring non-TGM (bit 3), then
  the largest measured PE. This is the same choice as `the_intime_flashid` in WCP
  prod-wire-cell-matching-nusel.cxx. A gaushit at (plane, local_wire, local_time) is matched to
  WC cells by `time_slice == (local_time + 2400) // 4` (confirmed: overlap 1% -> 95% at offset 2400).
    hit_wc/wc_cluster_id  WC cluster whose footprint contains the hit (-1 none; the main cluster
                          wins, otherwise the lowest id)
    hit_wc/wc_main        hit is in the in-time main cluster (WC neutrino candidate)
Event level (from the WC checkout file, joined on run/subrun/event; T_BDTvars is entry-aligned
with T_eval):
    event_wc/{in_nusel, has_intime_main, in_checkout, match_found, generic_nu, numu_score, nue_score,
              stm_*, match_isFC}
  generic_nu = WC generic neutrino selection: match_found & stm_eventtype != 0 & stm_lowenergy == 0
               & stm_LM == 0 & stm_TGM == 0 & stm_STM == 0 & stm_FullDead == 0 & stm_clusterlength > 15
  The final WC-nu hit label is hit_wc/wc_main & event_wc/generic_nu.

Rows of hit_wc follow hit_table row order, and rows of event_wc follow event_table, so they can be
joined by index.

Check on one overlay file (98 events; truth-nu hit = hit with an edep_table entry): WC main cluster
efficiency 0.823, purity 0.970 for truth-nu hits; 94.7% of all hits lie on some WC cluster.
"""
import argparse

import h5py
import numpy as np
import uproot

K = np.array([1 << 40, 1 << 20, 1], dtype=np.int64)
STM = ["stm_eventtype", "stm_lowenergy", "stm_LM", "stm_TGM", "stm_STM", "stm_FullDead", "stm_clusterlength"]


TC_BRANCHES = ["cluster_id", "time_slice", "wire_index_u", "wire_index_v", "wire_index_w"]


def iter_nusel(paths, wanted):
    """Yield (event key, TC arrays of that event, in-time main cluster id or None), one event at a
    time and only for keys in `wanted`. Real-data events are full of cosmic clusters, so loading
    whole files at once runs out of memory on a GPVM. Empty files (no Trun/TC) are skipped."""
    for p in paths:
        r = uproot.open(p)
        names = {k.split(";")[0] for k in r.keys()}
        if not {"Trun", "TC", "T_match"} <= names:
            print(f"skip (no WC trees): {p}")
            continue
        run = r["Trun"].arrays(["runNo", "subRunNo", "eventNo"], library="np")
        tm = r["T_match"].arrays(["tpc_cluster_id", "event_type", "pe_meas", "runNo", "subRunNo", "eventNo"], library="np")
        tkey = np.stack([tm["runNo"], tm["subRunNo"], tm["eventNo"]], 1).astype(np.int64) @ K
        pesum = tm["pe_meas"].sum(1)
        intime = (tm["event_type"] & 1) == 1
        keys = np.stack([run["runNo"], run["subRunNo"], run["eventNo"]], 1).astype(np.int64) @ K
        for i, key in enumerate(keys.tolist()):
            if key not in wanted:
                continue
            idx = np.where((tkey == key) & intime)[0]
            main = None
            if len(idx):
                ok = ((tm["event_type"][idx] >> 3) & 1) == 0
                cand = idx[ok] if ok.any() else idx
                main = int(tm["tpc_cluster_id"][cand[np.argmax(pesum[cand])]])
            tc = r["TC"].arrays(TC_BRANCHES, entry_start=i, entry_stop=i + 1, library="np")
            yield key, {k: tc[k][0] for k in TC_BRANCHES}, main


def footprint(tc, plane):
    w = tc["wire_index_" + "uvw"[plane]]
    n = np.array([len(x) for x in w], dtype=np.int64)
    wires = np.concatenate(w).astype(np.int64) if n.sum() else np.zeros(0, np.int64)
    return wires * 100000 + np.repeat(tc["time_slice"], n).astype(np.int64), np.repeat(tc["cluster_id"], n)


def load_checkout(path):
    f = uproot.open(path)["wcpselection"]
    ev = f["T_eval"].arrays(["run", "subrun", "event", "match_found", "match_isFC"] + STM, library="np")
    bdt = f["T_BDTvars"].arrays(["numu_score", "nue_score"], library="np")
    key = np.stack([ev["run"], ev["subrun"], ev["event"]], 1).astype(np.int64) @ K
    gen = (ev["match_found"] == 1) & (ev["stm_eventtype"] != 0) & (ev["stm_lowenergy"] == 0) & (ev["stm_LM"] == 0) \
        & (ev["stm_TGM"] == 0) & (ev["stm_STM"] == 0) & (ev["stm_FullDead"] == 0) & (ev["stm_clusterlength"] > 15)
    cols = dict(match_found=ev["match_found"], match_isFC=ev["match_isFC"], generic_nu=gen,
                numu_score=bdt["numu_score"], nue_score=bdt["nue_score"], **{k: ev[k] for k in STM})
    return {int(k): i for i, k in enumerate(key)}, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--nusel", nargs="+", required=True)
    ap.add_argument("--checkout")
    a = ap.parse_args()

    f = h5py.File(a.h5, "r")
    ev = f["event_table/event_id"][:].astype(np.int64)
    evkey = ev @ K
    hev = f["hit_table/event_id"][:].astype(np.int64) @ K
    pl = f["hit_table/local_plane"][:].ravel()
    wi = f["hit_table/local_wire"][:].ravel().astype(np.int64)
    ts = ((f["hit_table/local_time"][:].ravel() + 2400) // 4).astype(np.int64)
    hkey = wi * 100000 + ts

    cid = np.full(len(hev), -1, np.int32)
    is_main = np.zeros(len(hev), bool)
    in_nusel = np.zeros(len(ev), bool)
    has_main = np.zeros(len(ev), bool)
    order = np.argsort(hev, kind="stable")
    bounds = np.searchsorted(hev[order], evkey), np.searchsorted(hev[order], evkey, side="right")
    evpos = {int(k): j for j, k in enumerate(evkey)}
    for k, tc, mc in iter_nusel(a.nusel, evpos):
        j = evpos[k]
        if in_nusel[j]:
            continue                            # same event in two nusel files: keep the first
        in_nusel[j] = True
        has_main[j] = mc is not None
        rows = order[bounds[0][j]:bounds[1][j]]
        for p in range(3):
            r = rows[pl[rows] == p]
            if not len(r):
                continue
            fk, fc = footprint(tc, p)
            srt = np.lexsort((fc, fk))          # by key, then cluster id: first entry = lowest id
            fk, fc = fk[srt], fc[srt]
            pos = np.searchsorted(fk, hkey[r])
            hit = (pos < len(fk)) & (fk[np.minimum(pos, len(fk) - 1)] == hkey[r])
            cid[r[hit]] = fc[pos[hit]]
            if mc is not None:
                mk = np.isin(hkey[r], fk[fc == mc])
                is_main[r] = mk
                cid[r[mk]] = mc

    with h5py.File(a.out, "w") as o:
        g = o.create_group("hit_wc")
        g["event_id"] = f["hit_table/event_id"][:]
        g["hit_id"] = f["hit_table/hit_id"][:].ravel()
        g["wc_cluster_id"] = cid
        g["wc_main"] = is_main
        e = o.create_group("event_wc")
        e["event_id"] = ev
        e["in_nusel"] = in_nusel
        e["has_intime_main"] = has_main
        if a.checkout:
            idx, cols = load_checkout(a.checkout)
            at = np.array([idx.get(int(k), -1) for k in evkey])
            e["in_checkout"] = at >= 0
            for name, col in cols.items():
                v = np.zeros(len(ev), col.dtype)
                v[at >= 0] = col[at[at >= 0]]
                e[name] = v
    print(f"events {len(ev)}: in nusel {in_nusel.sum()}, in-time main {has_main.sum()}; "
          f"hits {len(cid)}: on a WC cluster {(cid >= 0).mean():.3f}, in main {is_main.mean():.3f}")


if __name__ == "__main__":
    main()
