#!/usr/bin/env python
"""First look at the SBND real-data caches vs the BNB CV simulation (issue #14). QLFM_DETECTOR=sbnd.

Per sample (beam-on, off-beam, simulation): hits per event; hit RMS per plane and per (plane, drift bin) with the
data/sim ratio; collection-plane hit-integral quantiles; light timing on the beam clock (PE-weighted ophit time
profile, brightest-flash time and PE); and the Wire-Cell nu-candidate rate (events with any hit_pnd_nu hit).
Every quantity is over all hits/ophits of the events read, so cosmics dominate the hit numbers in every sample;
the width comparison is therefore mostly cosmic tracks in data vs simulated (CORSIKA) cosmics, not a track-matched
measurement like #9.

Usage: python scripts/sbnd_data_firstlook.py NAME=DIR[,DIR...] ... [--max-events 20000] [--out report.json]
"""
import argparse
import glob
import json
import os

import numpy as np

from qlfm import geometry as G

DRIFT_EDGES = np.linspace(0, 2560, 7)            # ticks from the hit's own anode (full drift ~2559)
T_EDGES = np.arange(-4.0, 12.01, 0.25)           # us, beam clock
TIME_SCALE = 8
CHUNK = 2000                                     # events per read


def shards(spec):
    out = []
    for d in spec.split(","):
        out += sorted(glob.glob(os.path.join(d, "*_p0"))) if not os.path.exists(os.path.join(d, "meta.json")) else [d]
    return out


RMS_EDGES = (np.arange(0, 40 * 512 + 1) - 0.5) / 512   # float16 hit_rms in [2, 32) is exact on a 1/512 grid
Q_EDGES = np.arange(0, 20001, 1.0)


def _hquant(h, edges, qs):
    c = np.cumsum(h) / max(h.sum(), 1)
    return [float(edges[min(np.searchsorted(c, q) + 1, len(edges) - 1)] - (edges[1] - edges[0]) / 2) for q in qs]


def load(dirs, max_events, seed=0):
    """Histograms (per plane x drift bin hit RMS, Y-plane integral, PE-weighted ophit time) and per-event arrays,
    accumulated shard by shard so memory stays small."""
    rng = np.random.default_rng(seed)
    dirs = list(rng.permutation(dirs))
    nd = len(DRIFT_EDGES) - 1
    H = {"rms": np.zeros((3, nd + 1, len(RMS_EDGES) - 1)), "qy": np.zeros(len(Q_EDGES) - 1),
         "light": np.zeros(len(T_EDGES) - 1), "pe_all": 0.0, "pe_win": 0.0}
    ev = {k: [] for k in ("fl_t", "fl_pe", "nhit", "wc_any", "wc_frac")}
    n = 0
    for d in dirs:
        L = lambda k: np.load(os.path.join(d, k + ".npy"), mmap_mode="r")
        cnt = np.asarray(L("ev_hit_cnt")).astype(np.int64)
        m = min(len(cnt), max_events - n)
        if m <= 0:
            break
        hoff = np.asarray(L("ev_hit_off")).astype(np.int64)
        ooff, ocnt = (np.asarray(L(k)).astype(np.int64) for k in ("ev_oph_off", "ev_oph_cnt"))
        for e0 in range(0, m, CHUNK):              # merged shards hold ~15k events: bounded memory per chunk
            e1 = min(m, e0 + CHUNK)
            h0, h1 = hoff[e0], hoff[e1 - 1] + cnt[e1 - 1]
            t = np.asarray(L("hit_time")[h0:h1]).astype(np.float32) / TIME_SCALE
            tpc = np.asarray(L("hit_tpc")[h0:h1])
            drift = np.where(tpc == 0, t - G.TRIG_TICK, G.TPC1_FLIP_TICK - t - G.TRIG_TICK)
            dbin = np.digitize(drift, DRIFT_EDGES) - 1      # -1 / nd: outside the drift window, overall only
            plane = np.asarray(L("hit_plane")[h0:h1])
            rms = np.asarray(L("hit_rms")[h0:h1]).astype(np.float32)
            for p in range(3):
                mp = plane == p
                for i in range(nd):
                    H["rms"][p, i] += np.histogram(rms[mp & (dbin == i)], RMS_EDGES)[0]
                H["rms"][p, nd] += np.histogram(rms[mp], RMS_EDGES)[0]
            H["qy"] += np.histogram(np.asarray(L("hit_integral")[h0:h1]).astype(np.float32)[plane == 2], Q_EDGES)[0]
            o0, o1 = ooff[e0], ooff[e1 - 1] + ocnt[e1 - 1]
            ot = np.asarray(L("oph_t")[o0:o1])
            ope = np.asarray(L("oph_pe")[o0:o1]).astype(np.float32).clip(min=0)
            H["light"] += np.histogram(ot, T_EDGES, weights=ope)[0]
            H["pe_all"] += float(ope.sum())
            H["pe_win"] += float(ope[(ot >= G.BEAM_WINDOW_US[0]) & (ot < G.BEAM_WINDOW_US[1])].sum())
            pnd = np.asarray(L("hit_pnd_nu")[h0:h1]).astype(np.int64)
            wc_n = np.bincount(np.repeat(np.arange(e1 - e0), cnt[e0:e1]), pnd, e1 - e0)
            ev["wc_any"].append(wc_n > 0)
            ev["wc_frac"].append(wc_n / np.maximum(cnt[e0:e1], 1))
        ev["fl_t"].append(np.asarray(L("ev_fl_t")[:m]))
        ev["fl_pe"].append(np.asarray(L("ev_fl_pe")[:m]))
        ev["nhit"].append(cnt[:m])
        n += m
    return {"H": H, **{k: np.concatenate(v) for k, v in ev.items()}}, n


def summary(a, n):
    H = a["H"]
    nd = len(DRIFT_EDGES) - 1
    r = {"events": n, "hits_per_event": float(a["nhit"].mean())}
    for p in range(3):
        r[f"rms_med_p{p}"] = _hquant(H["rms"][p, nd], RMS_EDGES, [0.5])[0]
        r[f"rms_med_p{p}_drift"] = [_hquant(H["rms"][p, i], RMS_EDGES, [0.5])[0] if H["rms"][p, i].sum() else None
                                    for i in range(nd)]
    r["q_y_quantiles_10_50_90"] = [round(x, 1) for x in _hquant(H["qy"], Q_EDGES, [0.1, 0.5, 0.9])]
    r["light_profile_pe_per_event"] = (H["light"] / n).round(2).tolist()
    r["light_peak_us"] = float(T_EDGES[np.argmax(H["light"])] + 0.125)
    r["beam_window_pe_share"] = H["pe_win"] / max(H["pe_all"], 1e-9)
    ok = a["fl_t"] > -1e3
    r["fl_t_quantiles_10_50_90"] = np.quantile(a["fl_t"][ok], [0.1, 0.5, 0.9]).round(2).tolist() if ok.any() else None
    r["fl_pe_median"] = float(np.median(a["fl_pe"]))
    r["frac_brightest_flash_in_beam_window"] = float(((a["fl_t"] >= G.BEAM_WINDOW_US[0]) &
                                                      (a["fl_t"] < G.BEAM_WINDOW_US[1])).mean())
    r["wc_candidate_rate"] = float(a["wc_any"].mean())
    r["wc_hit_share_median_if_any"] = float(np.median(a["wc_frac"][a["wc_any"]])) if a["wc_any"].any() else None
    return r


def auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="+", help="NAME=DIR[,DIR...]")
    ap.add_argument("--max-events", type=int, default=20000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert G.DETECTOR == "sbnd", "set QLFM_DETECTOR=sbnd"
    data, rep = {}, {"drift_edges_ticks": DRIFT_EDGES.tolist(), "t_edges_us": T_EDGES.tolist(),
                     "beam_window_us": list(G.BEAM_WINDOW_US)}
    for s in a.samples:
        name, spec = s.split("=", 1)
        data[name], n = load(shards(spec), a.max_events)
        rep[name] = summary(data[name], n)
        data[name].pop("H")
        print(name, json.dumps({k: v for k, v in rep[name].items() if "profile" not in k}), flush=True)
    names = list(data)
    ref = names[-1]                                # last sample = simulation reference
    for nm in names[:-1]:
        for p in range(3):
            rep[nm][f"rms_ratio_to_{ref}_p{p}_drift"] = [
                round(x / y, 4) if x and y else None
                for x, y in zip(rep[nm][f"rms_med_p{p}_drift"], rep[ref][f"rms_med_p{p}_drift"])]
    if len(names) >= 2:                            # can light timing alone tell the first two samples apart?
        a0, a1 = data[names[0]], data[names[1]]
        rep[f"auc_fl_t_{names[0]}_vs_{names[1]}"] = auc(a0["fl_t"], a1["fl_t"])
        rep[f"auc_fl_pe_{names[0]}_vs_{names[1]}"] = auc(a0["fl_pe"], a1["fl_pe"])
    print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, dict)}), flush=True)
    for nm in names[:-1]:
        print(nm, "rms ratio vs", ref, {p: rep[nm][f"rms_ratio_to_{ref}_p{p}_drift"] for p in range(3)}, flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
