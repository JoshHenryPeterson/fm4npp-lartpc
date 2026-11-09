"""Convert one open-data HDF5 shard into a column-wise, memory-mappable cache.

Output directory layout (one per shard):
  ev_<name>.npy    per-event arrays (truth, offsets, baselines)
  hit_<name>.npy   per-hit columns, rows contiguous per event (offset ev_hit_off)
  oph_<name>.npy   per-ophit columns (offset ev_oph_off)
Truth columns are written to separate files so pretraining never reads them.

Usage: python -m qlfm.data.convert IN.h5 OUTDIR [--chunk 2000]
"""
import argparse
import json
import os
import time

import numpy as np

from qlfm.data.h5reader import OpenSampleFile
from qlfm.geometry import N_PMT

TIME_SCALE = 8  # hit time stored as uint16(round(t * 8)) -> 1/8 tick precision


def _alloc(outdir, name, shape, dtype):
    # in-RAM buffer, saved sequentially at the end (mmap writes on Lustre are very slow)
    a = np.zeros(shape, dtype)
    _alloc.files[id(a)] = os.path.join(outdir, name + ".npy")
    return a


_alloc.files = {}


def _shift(res, ev_lo):
    d, evt = res
    return d, evt - ev_lo


def convert(path, outdir, chunk=2000, part=0, nparts=1):
    """Convert events [part*n/nparts, (part+1)*n/nparts) of `path` into `outdir`."""
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    f = OpenSampleFile(path)
    ev_lo = f.n_events * part // nparts
    ev_hi = f.n_events * (part + 1) // nparts
    n = ev_hi - ev_lo
    hs, hc = f.offsets("hit_table")
    os_, oc = f.offsets("ophit_table")
    # local (part-relative) offsets
    hs, hc = hs[ev_lo:ev_hi] - hs[ev_lo], hc[ev_lo:ev_hi]
    os_, oc = os_[ev_lo:ev_hi] - os_[ev_lo], oc[ev_lo:ev_hi]
    n_hits, n_oph = int(hc.sum()), int(oc.sum())
    rd = lambda table, lo, hi, cols=None: _shift(f.read_range(table, lo + ev_lo, hi + ev_lo, cols), ev_lo)

    # ---------------- per-event truth ----------------
    ev, _ = rd("event_table", 0, n)
    out_ev = {
        "run": ev["event_id"][:, 0].astype(np.int32),
        "subrun": ev["event_id"][:, 1].astype(np.int32),
        "event": ev["event_id"][:, 2].astype(np.int32),
        "nu_pdg": ev["nu_pdg"].astype(np.int32),
        "is_cc": ev["is_cc"].astype(np.int8),
        "nu_energy": ev["nu_energy"].astype(np.float32),
        "lep_energy": ev["lep_energy"].astype(np.float32),
        "nu_vtx": ev["nu_vtx"].astype(np.float32),
        "nu_vtx_corr": ev["nu_vtx_corr"].astype(np.float32),
        "nu_time": ev["nu_vtx_wire_time"].astype(np.float32),
        "hit_off": hs, "hit_cnt": hc.astype(np.int32),
        "oph_off": os_, "oph_cnt": oc.astype(np.int32),
    }
    nu_nhits = np.zeros((n, 3), np.int32)       # per plane, hits with nu edep above 0
    nu_q = np.zeros((n, 3), np.float32)         # per plane, sum integral of nu hits
    nu_edep = np.zeros(n, np.float32)           # sum MC energy (MeV) on collection-plane hits
    pnd_score = np.full(n, -999.0, np.float32)  # Pandora max nu_score
    pnd_fm = np.full(n, 9999.0, np.float32)     # flashmatch score of that slice
    fl_n = np.zeros(n, np.int16)
    fl_t = np.full(n, -1.0, np.float32)         # time of max-PE beam flash (us)
    fl_pe = np.zeros(n, np.float32)
    fl_pmt = np.zeros((n, N_PMT), np.float32)   # per-PMT PE of max-PE beam flash

    # ---------------- per-hit / per-ophit columns ----------------
    H = {
        "plane": _alloc(outdir, "hit_plane", (n_hits,), np.uint8),
        "wire": _alloc(outdir, "hit_wire", (n_hits,), np.int16),
        "time": _alloc(outdir, "hit_time", (n_hits,), np.uint16),
        "integral": _alloc(outdir, "hit_integral", (n_hits,), np.float16),
        "rms": _alloc(outdir, "hit_rms", (n_hits,), np.float16),
        # truth / baseline (never model input)
        "nu_edep": _alloc(outdir, "hit_nu_edep", (n_hits,), np.float16),
        "cat": _alloc(outdir, "hit_cat", (n_hits,), np.int8),
        "pnd_nu": _alloc(outdir, "hit_pnd_nu", (n_hits,), np.uint8),
    }
    O = {
        "ch": _alloc(outdir, "oph_ch", (n_oph,), np.uint8),
        "t": _alloc(outdir, "oph_t", (n_oph,), np.float32),
        "width": _alloc(outdir, "oph_width", (n_oph,), np.float16),
        "pe": _alloc(outdir, "oph_pe", (n_oph,), np.float16),
    }
    stats = {"hit_id_mismatch": 0, "edep_rows": 0, "max_ophit_t": 0.0, "min_ophit_t": 1e9, "neg_hit_time": 0}

    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        r0 = hs[lo]
        h, hevt = rd("hit_table", lo, hi,
                               ["hit_id", "local_plane", "local_wire", "local_time", "integral", "rms"])
        nh = len(hevt)
        if nh:
            local = np.arange(nh) - (hs[hevt] - r0)
            stats["hit_id_mismatch"] += int((h["hit_id"] != local).sum())
            # per-hit truth and Pandora membership below are scattered by hit_id (audit #12)
            assert stats["hit_id_mismatch"] == 0, f"{path}: hit_id != row index within the event"
            stats["neg_hit_time"] += int((h["local_time"] < 0).sum())      # clipped to tick 0
            sl = slice(r0, r0 + nh)
            H["plane"][sl] = h["local_plane"]
            H["wire"][sl] = h["local_wire"]
            H["time"][sl] = np.clip(np.round(h["local_time"] * TIME_SCALE), 0, 65535)
            H["integral"][sl] = np.clip(h["integral"], 0, 65000)
            H["rms"][sl] = np.clip(h["rms"], 0, 65000)

            # ---- neutrino truth per hit (overlay cosmics are data -> no edep rows) ----
            e, eevt = rd("edep_table", lo, hi, ["hit_id", "g4_id", "energy", "energy_fraction"])
            stats["edep_rows"] += len(eevt)
            row = hs[eevt] - r0 + e["hit_id"]                       # chunk-local hit row
            edep = np.zeros(nh, np.float64)
            np.add.at(edep, row, e["energy"])
            H["nu_edep"][sl] = np.clip(edep, 0, 65000)
            # semantic category of the max-fraction particle
            cat = np.full(nh, -1, np.int8)
            if len(row):
                p, pevt = rd("particle_table", lo, hi, ["g4_id", "category"])
                pkey = pevt.astype(np.int64) << 32 | p["g4_id"].astype(np.int64) & 0xFFFFFFFF
                order = np.argsort(pkey)
                ekey = eevt.astype(np.int64) << 32 | e["g4_id"].astype(np.int64) & 0xFFFFFFFF
                idx = np.clip(np.searchsorted(pkey[order], ekey), 0, len(order) - 1)
                found = pkey[order][idx] == ekey
                ecat = np.where(found, p["category"][order][idx], 7)
                srt = np.lexsort((-e["energy_fraction"], row))
                first = np.ones(len(srt), bool)
                first[1:] = row[srt][1:] != row[srt][:-1]
                cat[row[srt][first]] = ecat[srt][first]
            H["cat"][sl] = cat
            isnu = edep > 0
            for pl in range(3):
                m = isnu & (h["local_plane"] == pl)
                np.add.at(nu_nhits[:, pl], hevt[m], 1)
                np.add.at(nu_q[:, pl], hevt[m], h["integral"][m])
            m = isnu & (h["local_plane"] == 2)
            np.add.at(nu_edep, hevt[m], edep[m])

            # ---- Pandora baseline: hits in the max-nu_score slice ----
            ph, phevt = rd("pandoraHit_table", lo, hi, ["hit_id", "slice_id"])
            pp, ppevt = rd("pandoraPrimary_table", lo, hi, ["slice_id", "nu_score", "flashmatch_score"])
            best_slice = np.full(n, -2, np.int64)
            if len(ppevt):
                srt = np.lexsort((-pp["nu_score"], ppevt))
                first = np.ones(len(srt), bool)
                first[1:] = ppevt[srt][1:] != ppevt[srt][:-1]
                bi = srt[first]
                ok = pp["nu_score"][bi] > -998
                pnd_score[ppevt[bi]] = np.where(ok, pp["nu_score"][bi], -999.0)
                pnd_fm[ppevt[bi]] = pp["flashmatch_score"][bi]
                best_slice[ppevt[bi][ok]] = pp["slice_id"][bi][ok]
            prow = hs[phevt] - r0 + ph["hit_id"]
            pnd = np.zeros(nh, np.uint8)
            pnd[prow] = (ph["slice_id"] == best_slice[phevt]) & (ph["slice_id"] >= 0)
            H["pnd_nu"][sl] = pnd

        # ---- optical hits ----
        o, oevt = rd("ophit_table", lo, hi, ["hit_channel", "peaktime", "width", "pe"])
        if len(oevt):
            # tokenize_light bins PE by evt * 32 + channel (audit #12)
            assert int(o["hit_channel"].max()) < 32, f"{path}: PMT channel >= 32"
            s0 = os_[lo]
            sl = slice(s0, s0 + len(oevt))
            O["ch"][sl] = o["hit_channel"]
            O["t"][sl] = o["peaktime"]
            O["width"][sl] = np.clip(o["width"], 0, 65000)
            O["pe"][sl] = np.clip(o["pe"], -65000, 65000)
            stats["max_ophit_t"] = max(stats["max_ophit_t"], float(o["peaktime"].max()))
            stats["min_ophit_t"] = min(stats["min_ophit_t"], float(o["peaktime"].min()))

        # ---- beam flashes (baseline / audit only) ----
        fl, flevt = rd("opflash_table", lo, hi, ["flash_id", "time", "totalpe"])
        if len(flevt):
            np.add.at(fl_n, flevt, 1)
            srt = np.lexsort((-fl["totalpe"], flevt))
            first = np.ones(len(srt), bool)
            first[1:] = flevt[srt][1:] != flevt[srt][:-1]
            bi = srt[first]
            fl_t[flevt[bi]] = fl["time"][bi]
            fl_pe[flevt[bi]] = fl["totalpe"][bi]
            best_flash = np.full(n, -2, np.int64)
            best_flash[flevt[bi]] = fl["flash_id"][bi]
            sp, spevt = rd("opflashsumpe_table", lo, hi, ["flash_id", "pmt_channel", "sumpe"])
            m = sp["flash_id"] == best_flash[spevt]
            fl_pmt[spevt[m], sp["pmt_channel"][m].astype(np.int64)] = sp["sumpe"][m]

        print(f"{os.path.basename(path)}: {hi}/{n} events, {time.time()-t0:.0f}s", flush=True)

    out_ev.update(nu_nhits=nu_nhits, nu_q=nu_q, nu_edep=nu_edep, pnd_score=pnd_score,
                  pnd_fm=pnd_fm, fl_n=fl_n, fl_t=fl_t, fl_pe=fl_pe, fl_pmt=fl_pmt)
    for k, v in out_ev.items():
        np.save(os.path.join(outdir, f"ev_{k}.npy"), v)
    for d in (H, O):
        for v in d.values():
            np.save(_alloc.files.pop(id(v)), v)
    stats.update(n_events=n, n_hits=n_hits, n_ophits=n_oph, source=os.path.abspath(path),
                 ev_lo=ev_lo, ev_hi=ev_hi,
                 seconds=time.time() - t0)
    with open(os.path.join(outdir, "meta.json"), "w") as fo:
        json.dump(stats, fo, indent=1)
    f.close()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("outdir")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    print(json.dumps(convert(a.inp, a.outdir, a.chunk, a.part, a.nparts), indent=1))


if __name__ == "__main__":
    main()
