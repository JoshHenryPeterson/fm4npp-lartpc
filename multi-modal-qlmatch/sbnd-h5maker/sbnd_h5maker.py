#!/usr/bin/env python3
"""SBND reco1 + Wire-Cell (aurora-r2) -> HDF5 in the MicroBooNE open-data layout (uboone/hdf5maker).

Run inside the SL7 container with env.sh sourced (gallery + PyROOT, h5py, uproot):

    python sbnd_h5maker.py OUT.h5 --units UNIT_DIR [UNIT_DIR ...] [--units-file F] [--max-events N]

A UNIT_DIR is an aurora-r2 production unit (sbnd-wirecell-production): `reco1_files.json` lists its reco1
files, and `pr/all/pr_evt<run>_<subrun>_<event>/tracking-pr.root` holds the Wire-Cell output per event.
`--reco1 F [F ...] --wc-dir D` is the same without a unit directory.

Real data (0.5): `--data` reads no truth product. event_table keeps its truth columns at their defaults
(nu_index -1, nu_pdg 0, nu_vtx -9999, ...); nu_table, edep_table, edep_cosmic_table and particle_table are
written with zero rows. The reco1 input should be the FrameShift output (sbnd::timing::FrameShiftInfo
"frameshift"), whose values go to event_table.frame_*_ns.

Tables (hep_hpc / ph5concat layout: one (N, k) dataset per column, `event_id` (N, 3) and
`event_id.seq_cnt` = [event sequence number, row count] for each event with rows):
  event_table         nu truth (as HDF5Maker EventInfo "nu") + n_nu, nu_mode, beam gate, wc_status, file_index
  hit_table           gaushit: hit_id, integral, rms, tpc, local_plane, local_wire, local_time (+ channel, peak_amplitude)
  edep_table          BackTracker hit -> particle energy for particles of beam-neutrino origin (MicroBooNE semantics)
  edep_cosmic_table   the same for every other origin (CORSIKA cosmics; origin 0 = no MCTruth link)
  particle_table      particles with deposits + ancestors: g4_id, g4_pdg, parent_id, category, instance, origin, ...
  ophit_table         PMT ophits (ophitpmt): hit_channel = OpChannel (0-311, PMTs only), peaktime [us], pe, ...
  opflash_table       opflashtpc0 + opflashtpc1 (tpc column), opflashsumpe_table: per-channel PE of each flash
  wc_event_table      one row per Wire-Cell neutrino candidate (T_tagger / T_kine)
  wc_hit_table        one row per hit (same order as hit_table): Wire-Cell nu-candidate footprint labels

0.6 (fm4npp #16), with --blob-dir / --blob-dump (the post-split point-cloud tree, wct/wct-blobdump.jsonnet):
  wc_blob_table       every Wire-Cell blob (all clusters): cluster_id, tpc, slice and wire ranges, centre, charge,
                      wc_nu_index, (sim) TensorSetLabeler trackid / truth_q / truth_q_dom / trackid2 / truth_q2
  wc_cluster_table    every step-2 cluster: blob census + the tracking-pr.root T_cluster flags (labels only)
  wc_hit_blob_table   hit <-> blob (many-to-many), weight = 1 / (number of blobs holding the hit)
  track_table         (sim) one row per G4 track with deposits in the readout (labeler truthtrack) + kinematics
  hit_truth_table     (sim, always) per hit: energy-weighted true (pre-SCE) and apparent (SCE-displaced) mean
                      position of its projected deposits, and their energy
"""
import argparse
import glob
import json
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = {
    "energy (nu_energy, lep_energy)": "GeV", "edep energy, edep_collection, particle momentum": "MeV / GeV (momentum)",
    "positions (nu_vtx, start/end_position, reco_vtx, y/z_center)": "cm",
    "*_ns (nu_time_ns, beam_gate_start_ns, beam_gate_width_ns, particle start_time)": "ns, G4 clock (t = 0 at the trigger)",
    "hit local_time, rms": "TPC ticks of 0.5 us from the readout start (trigger at tick 410)",
    "ophit peaktime, width, opflash time, time_width": "us, trigger clock (OpHit::PeakTime); peaktime_abs is OpHit::PeakTimeAbs",
    "integral": "ADC x tick", "pe, sumpe, totalpe": "photoelectrons",
    "wc reco_enu": "MeV", "wc flash_time_us": "us",
    "event_table time_s": "s, unix time of the event (EventAuxiliary)",
    "event_table frame_*_ns": "ns, sbnd::timing::FrameShiftInfo (real data; 0 with has_frameshift 0)",
    "wc_blob_table center_*, track_table start/end_x/y/z, hit_truth_table true_*/app_*": "cm",
    "wc_blob_table / wc_cluster_table charge, truth_q*, track_table q_*": "electrons",
    "wc_blob_table slice_index_*": "TPC ticks (0.5 us), slices of 4 ticks", "wc_cluster_table cluster_t0_ns": "ns",
    "track_table start/end_t": "ns (G4 clock)", "track_table *_px/py/pz/E": "GeV",
    "track_table e_readout, hit_truth_table energy": "MeV"}
VERSION = "sbnd-h5maker 0.6"


WC_LABEL_NOTES = (
    "Wire-Cell nu candidates (T_tagger rows, aurora-r2 / SBND pr() defaults): one per in-beam-window flash bundle "
    "(matched flash time in [0.2, 2.2) us, nu_per_bundle, length floor 15 cm). Mains already tagged cosmic upstream "
    "(TGM, STM or Q/L lm_flag > 0) are skipped, and the veto covers the whole flash bundle (nu_skip_cosmic + "
    "nu_skip_cosmic_bundle; untagged mains >= 15 cm survive), so the wc_* hit labels are charge-light matching AFTER "
    "the TGM/STM/light-mismatch veto. event_table.wc_n_bundle_cosmic counts in-window bundles dropped by that veto "
    "(T_bundle reason 2). The PR-level cosmic tagger is NOT applied: its verdict is wc_event_table.cosmict_flag "
    "(OR of its ten tests); cosmic_flag is only its flag-9 BDT input (1 = not fired or not evaluated). "
    "Further cuts (cosmict_flag, numu/nue scores, match_isFC) are for the user.")


# ----------------------------------------------------------------------------------------------- output
class Table:
    """Accumulates rows of one table, event by event."""

    def __init__(self, name, cols):
        self.name, self.cols = name, cols          # cols: {col: (dtype, k)}
        self.data = {c: [] for c in cols}
        self.evid, self.seq = [], []               # per event with rows: (run, subrun, event), (seq, count)

    def add(self, seq, rse, n, **arrays):
        if n == 0:
            return
        for c, (dt, k) in self.cols.items():
            a = np.asarray(arrays[c])
            if k == 1:
                a = a.reshape(n, 1)
            self.data[c].append(a.astype(dt, copy=False).reshape(n, k))
        self.evid.append((rse, n))
        self.seq.append((seq, n))

    def write(self, f, compression):
        g = f.create_group(self.name)
        n = sum(c for _, c in self.seq)
        ev = np.zeros((n, 3), np.int32)
        r = 0
        for rse, c in self.evid:
            ev[r:r + c] = rse
            r += c
        kw = dict(compression=compression, chunks=True) if n else {}
        g.create_dataset("event_id", data=ev, **kw)
        g.create_dataset("event_id.seq_cnt", data=np.array(self.seq, np.int64).reshape(-1, 2))
        for c, (dt, k) in self.cols.items():
            a = np.concatenate(self.data[c]) if self.data[c] else np.zeros((0, k), dt)
            g.create_dataset(c, data=a, **kw)
        return n


def i32(k=1):
    return (np.int32, k)


def f32(k=1):
    return (np.float32, k)


def f64(k=1):
    return (np.float64, k)


def schema(blobs=False):
    T = {}
    T["event_table"] = Table("event_table", {
        "is_cc": i32(), "nu_pdg": i32(), "nu_energy": f32(), "lep_energy": f32(), "nu_dir": f32(3), "nu_vtx": f32(3),
        "nu_time_ns": f32(), "lep_pdg": i32(), "nu_mode": i32(), "nu_int_type": i32(), "n_nu": i32(),
        "nu_index": i32(), "beam_gate_start_ns": f32(), "beam_gate_width_ns": f32(), "wc_status": i32(),
        "file_index": i32(), "n_mcp_no_truth": i32(),
        # 0.5: event time + FrameShiftInfo (real data), Wire-Cell in-beam bundle census (T_bundle)
        "time_s": f64(), "has_frameshift": i32(), "timing_type": i32(), "frame_apply_at_caf_ns": f32(),
        "frame_hlt_beamgate_ns": f32(), "frame_hlt_crtt1_ns": f32(), "frame_tdc_crtt1_ns": f32(),
        "frame_tdc_bes_ns": f32(), "frame_tdc_rwm_ns": f32(), "wc_n_bundle": i32(), "wc_n_bundle_cosmic": i32(),
        # 0.6: 1 = the event's blob record was found in the blob dump, 0 = not (or no --blob-dir), -1 = not asked
        "wc_blob_status": i32(), "wc_n_blobs": i32()})
    T["nu_table"] = Table("nu_table", {
        "nu_index": i32(), "nu_pdg": i32(), "is_cc": i32(), "nu_energy": f32(), "lep_energy": f32(),
        "lep_pdg": i32(), "nu_vtx": f32(3), "nu_dir": f32(3), "nu_time_ns": f32(), "nu_mode": i32(),
        "nu_int_type": i32(), "in_av": i32(), "edep_collection": f32(), "nhits_collection": i32()})
    T["hit_table"] = Table("hit_table", {
        "hit_id": i32(), "integral": f32(), "rms": f32(), "tpc": i32(), "local_plane": i32(), "local_wire": i32(),
        "local_time": f32(), "channel": i32(), "peak_amplitude": f32()})
    edep = {"hit_id": i32(), "g4_id": i32(), "energy": f32(), "energy_fraction": f32(), "is_max_ide": i32(),
            "num_electrons": f32(), "origin": i32(), "nu_index": i32()}
    T["edep_table"] = Table("edep_table", dict(edep))
    T["edep_cosmic_table"] = Table("edep_cosmic_table", dict(edep))
    # 0.6: one row per hit (sim; empty in data): energy-weighted mean of the true (pre-SCE) position of the deposits
    # projected onto the hit, the mean apparent (SCE-displaced) position -- the frame of the hits and of the
    # uncorrected blob centres, the P3a target -- and their summed energy (true_rms_* dropped on Polaris' request, #16)
    T["hit_truth_table"] = Table("hit_truth_table", {
        "hit_id": i32(), "true_x": f32(), "true_y": f32(), "true_z": f32(), "app_x": f32(), "app_y": f32(),
        "app_z": f32(), "energy": f32()})
    T["particle_table"] = Table("particle_table", {
        "g4_id": i32(), "g4_pdg": i32(), "parent_id": i32(), "category": i32(), "instance": i32(), "origin": i32(),
        "nu_index": i32(),
        "momentum": f32(), "start_position": f32(3), "end_position": f32(3), "start_time": f32(),
        "start_process": ("S24", 1), "end_process": ("S24", 1)})
    T["ophit_table"] = Table("ophit_table", {
        "hit_id": i32(), "hit_channel": i32(), "peaktime": f32(), "peaktime_abs": f32(), "width": f32(),
        "area": f32(), "amplitude": f32(), "pe": f32()})
    T["opflash_table"] = Table("opflash_table", {
        "flash_id": i32(), "tpc": i32(), "time": f32(), "time_width": f32(), "y_center": f32(), "y_width": f32(),
        "z_center": f32(), "z_width": f32(), "totalpe": f32()})
    T["opflashsumpe_table"] = Table("opflashsumpe_table", {
        "sumpe_id": i32(), "flash_id": i32(), "pmt_channel": i32(), "sumpe": f32()})
    T["wc_event_table"] = Table("wc_event_table", {
        "nu_index": i32(), "cluster_id": i32(), "has_vertex": i32(), "neutrino_type": i32(), "reco_vtx": f32(3),
        "reco_enu": f32(), "flash_time_us": f32(), "flash_pe": f32(), "flash_tpc": i32(), "match_isFC": f32(),
        "cosmic_flag": f32(), "numu_score": f32(), "nue_score": f32(), "cosmict_score": f32(),
        "n_clusters_in_pr": i32(),
        # 0.5: cosmict_flag = the cosmic tagger's verdict (OR of its ten tests); cosmic_flag above is only the
        # flag-9 BDT input (1 = not fired or never evaluated), see wc_label_notes
        "cosmict_flag": f32(), "matched_flash_gid": i32()})
    T["wc_hit_table"] = Table("wc_hit_table", {
        "hit_id": i32(), "wc_cluster_id": i32(), "wc_nu_index": i32(), "wc_main": i32(), "wc_in_enu": i32()})
    if blobs:
        T["wc_blob_table"] = Table("wc_blob_table", {
            "blob_id": i32(), "cluster_id": i32(), "tpc": i32(), "slice_index_min": i32(), "slice_index_max": i32(),
            "u_wire_index_min": i32(), "u_wire_index_max": i32(), "v_wire_index_min": i32(),
            "v_wire_index_max": i32(), "w_wire_index_min": i32(), "w_wire_index_max": i32(), "center_x": f32(),
            "center_y": f32(), "center_z": f32(), "center_x_t0cor": f32(), "charge": f32(), "npoints": i32(),
            "wc_nu_index": i32(), "trackid": i32(), "truth_q": f32(), "truth_q_dom": f32(), "trackid2": i32(),
            "truth_q2": f32()})
        T["wc_cluster_table"] = Table("wc_cluster_table", {
            "cluster_id": i32(), "n_blobs": i32(), "charge": f32(), "npoints": i32(), "length_cm": f32(),
            "in_scope": i32(), "flag_main_cluster": i32(), "flag_associated_cluster": i32(), "cluster_t0_ns": f32(),
            "flag_TGM": i32(), "flag_STM": i32(), "flag_FC": i32(), "lm_flag": i32(), "beam_flash": i32(),
            "flash_id": i32(), "matched_flash_gid": i32(), "wc_nu_index": i32(), "wc_main": i32(),
            "wc_in_enu": i32()})
        T["wc_hit_blob_table"] = Table("wc_hit_blob_table", {"hit_id": i32(), "blob_id": i32(), "weight": f32()})
        T["track_table"] = Table("track_table", {
            **{c: i32() for c in ("trackid", "pdg", "mother_trackid", "mother_pdg", "status")},
            **{c: f32() for c in ("start_x", "start_y", "start_z", "start_t", "start_px", "start_py", "start_pz",
                                  "start_E", "end_x", "end_y", "end_z", "end_t", "end_px", "end_py", "end_pz", "end_E")},
            **{c: i32() for c in ("nu_idx", "process", "ancestor_trackid", "prong_trackid")},
            **{c: f32() for c in ("q_readout", "e_readout", "q_in_blobs")}, "n_blobs": i32(), "q_blobs": f32(),
            "n_blobs_dom": i32(), "q_dom_blobs": f32()})
    return T


# ----------------------------------------------------------------------------------------------- art side
def load_gallery():
    import ROOT
    ROOT.gROOT.SetBatch(True)
    for lib in ("libcanvas", "libgallery"):
        ROOT.gSystem.Load(lib)
    ROOT.gInterpreter.AddIncludePath(HERE)
    if not ROOT.gInterpreter.Declare('#include "extract.h"'):
        sys.exit("failed to compile extract.h")
    return ROOT


def vec2np(ROOT, v, dt):
    n = v.size()
    a = np.empty(n, dt)
    if n:
        (ROOT.sbndh5.copy_i if dt == np.int32 else ROOT.sbndh5.copy_f)(v, a)
    return a


# ----------------------------------------------------------------------------------------------- Wire-Cell side
class WC:
    """Per-event Wire-Cell (aurora-r2 tracking-pr.root) event rows and hit labels."""

    def __init__(self, wc_dirs, tick_offset, ticks_per_slice, det):
        import uproot
        self.uproot = uproot
        self.dirs, self.off, self.tps = wc_dirs, tick_offset, ticks_per_slice
        self.ch_off = wc_channel_offsets(det)

    def wc_channel(self, channel, tpc, plane):
        """LArSoft channel -> Wire-Cell T_proj_data channel (plane-major numbering, see wc_channel_offsets)."""
        return np.asarray(channel, np.int64) + self.ch_off[np.asarray(tpc, np.int64), np.asarray(plane, np.int64)]

    def path(self, rse):
        for d in self.dirs:
            p = os.path.join(d, "pr_evt%d_%d_%d" % rse, "tracking-pr.root")
            if os.path.exists(p):
                return p
        return None

    def event(self, rse):
        """(status, candidate rows, footprint) ; status 1 = tracking-pr.root present, 0 = missing."""
        p = self.path(rse)
        self.census = (-1, -1)
        self.role, self.tcl = {}, None
        if p is None:
            return 0, [], None
        f = self.uproot.open(p)
        keys = {k.split(";")[0] for k in f.keys()}
        cands, role = [], {}
        if "T_tagger" in keys and f["T_tagger"].num_entries:
            tg = f["T_tagger"].arrays(["nu_index", "cluster_id", "has_vertex", "neutrino_type", "nu_x", "nu_y", "nu_z",
                                       "flash_time_us", "flash_pe", "flash_tpc", "match_isFC", "cosmic_flag",
                                       "numu_score", "nue_score", "cosmict_score", "act_cluster_id", "act_in_pr",
                                       "act_in_enu"] + [b for b in ("cosmict_flag", "matched_flash_gid")
                                                        if b in f["T_tagger"].keys()], library="ak")
            import awkward as ak
            tg = {k: (ak.to_list(tg[k]) if k.startswith("act_") else ak.to_numpy(tg[k])) for k in tg.fields}
            enu = {}
            if "T_kine" in keys and f["T_kine"].num_entries:
                kn = f["T_kine"].arrays(["nu_index", "kine_reco_Enu"], library="np")
                enu = dict(zip(kn["nu_index"].tolist(), kn["kine_reco_Enu"].tolist()))
            for i in range(len(tg["nu_index"])):
                nu = int(tg["nu_index"][i])
                main = int(tg["cluster_id"][i])
                ids, inpr, inenu = (np.asarray(tg[k][i], np.int64) for k in ("act_cluster_id", "act_in_pr", "act_in_enu"))
                for c, a, b in zip(ids.tolist(), inpr.tolist(), inenu.tolist()):
                    role.setdefault(c, (nu, int(c == main), int(b)))
                role[main] = (nu, 1, role.get(main, (nu, 1, 1))[2])
                cands.append(dict(
                    nu_index=nu, cluster_id=main, has_vertex=int(tg["has_vertex"][i]),
                    neutrino_type=int(tg["neutrino_type"][i]),
                    reco_vtx=[float(tg["nu_x"][i]), float(tg["nu_y"][i]), float(tg["nu_z"][i])],
                    reco_enu=float(enu.get(nu, -1.0)), flash_time_us=float(tg["flash_time_us"][i]),
                    flash_pe=float(tg["flash_pe"][i]), flash_tpc=int(tg["flash_tpc"][i]),
                    match_isFC=float(tg["match_isFC"][i]), cosmic_flag=float(tg["cosmic_flag"][i]),
                    numu_score=float(tg["numu_score"][i]), nue_score=float(tg["nue_score"][i]),
                    cosmict_score=float(tg["cosmict_score"][i]), n_clusters_in_pr=int(np.sum(inpr)),
                    cosmict_flag=float(tg["cosmict_flag"][i]) if "cosmict_flag" in tg else -1.0,
                    matched_flash_gid=int(tg["matched_flash_gid"][i]) if "matched_flash_gid" in tg else -1))
        fp = None
        if "T_proj_data" in keys and f["T_proj_data"].num_entries:
            import awkward as ak
            pj = f["T_proj_data"].arrays(["cluster_id", "channel", "time_slice"], library="ak")
            # cluster_id: [entry][cluster]; channel / time_slice: [entry][cluster][cell]
            ch = ak.to_numpy(ak.flatten(pj["channel"], axis=None)).astype(np.int64)
            sl = ak.to_numpy(ak.flatten(pj["time_slice"], axis=None)).astype(np.int64)
            ncell = ak.to_numpy(ak.flatten(ak.num(pj["channel"], axis=2), axis=None))
            cl = np.repeat(ak.to_numpy(ak.flatten(pj["cluster_id"], axis=None)).astype(np.int64), ncell)
            if len(ch):
                assert len(ch) == len(sl) == len(cl)
                fp = (ch, sl, cl, role)
        self.census = (-1, -1)
        self.role = role
        if "T_cluster" in keys and f["T_cluster"].num_entries:
            self.tcl = f["T_cluster"].arrays(library="np")
        if "T_bundle" in keys:
            r = f["T_bundle"].arrays(["reason"], library="np")["reason"]
            self.census = (int(len(r)), int(np.sum(r == 2)))   # NuBundleCensus kAllCosmic = 2
        return 1, cands, fp

    def hit_labels(self, fp, channel, tick, off=None):
        """Per hit: (cluster id, nu index, in main cluster, in energy clusters); -1/0 outside the footprint.

        channel must be in the Wire-Cell numbering (WC.wc_channel), not the LArSoft one."""
        n = len(channel)
        cid = np.full(n, -1, np.int32)
        nui = np.full(n, -1, np.int32)
        mn = np.zeros(n, np.int32)
        en = np.zeros(n, np.int32)
        if fp is None or n == 0:
            return cid, nui, mn, en
        fch, fsl, fcl, role = fp
        off = self.off if off is None else off
        prio = np.array([0 if role.get(int(c), (-1, 0, 0))[1] else 1 for c in fcl], np.int64)
        key = fch * 8192 + fsl
        order = np.lexsort((fcl, prio, key))
        key, fcl_s = key[order], fcl[order]
        first = np.ones(len(key), bool)
        first[1:] = key[1:] != key[:-1]
        key, fcl_s = key[first], fcl_s[first]
        hk = np.asarray(channel, np.int64) * 8192 + np.floor((np.asarray(tick) + off) / self.tps).astype(np.int64)
        idx = np.clip(np.searchsorted(key, hk), 0, len(key) - 1)
        hit = key[idx] == hk
        cid[hit] = fcl_s[idx[hit]]
        for c in np.unique(cid[hit]):
            nu, m, e = role.get(int(c), (-1, 0, 0))
            sel = cid == c
            nui[sel], mn[sel], en[sel] = nu, m, e
        return cid, nui, mn, en


def wc_channel_offsets(det):
    """Offset (tpc, plane) such that WC channel = LArSoft channel + offset.

    LArSoft numbers SBND channels TPC-major: [U0 V0 Y0][U1 V1 Y1]. The Wire-Cell PR output (T_proj_data) numbers
    them plane-major: [U0 U1][V0 V1][Y0 Y1], keeping the LArSoft wire order inside each block. Checked on test10:
    a per-(tpc, plane) channel-shift scan of the footprints against the hits peaks at exactly these offsets.
    """
    wp = {(w["tpc"], w["plane"]): w for w in det["wire_planes"]}
    tpcs = sorted({t for t, _ in wp})
    off = np.zeros((len(tpcs), 3), np.int64)
    base = 0
    for p in range(3):
        for t in tpcs:
            w = wp[(t, p)]
            first = min(w["channel0"], w["channel0"] + w["channel_step"] * (w["n_wires"] - 1))
            off[t, p] = base - first
            base += w["n_wires"]
    return off


def make_geo(ROOT, det, a):
    G = ROOT.sbndh5.Geo()
    dt = [float(x) for x in a.proj_dt.replace(":", ",").split(",")]
    wp = sorted(det["wire_planes"], key=lambda w: (w["tpc"], w["plane"]))
    assert len(wp) == 6
    for w in wp:
        G.x.push_back(w["x"])
        G.y0.push_back(w["wire0_yz"][0])
        G.z0.push_back(w["wire0_yz"][1])
        G.pdy.push_back(w["pitch_dir_yz"][0])
        G.pdz.push_back(w["pitch_dir_yz"][1])
        G.pitch.push_back(w["pitch"])
        G.n.push_back(w["n_wires"])
        G.ch0.push_back(w["channel0"])
        G.step.push_back(w["channel_step"])
        G.dt.push_back(dt[w["plane"]])
    G.drift_v = det["drift_speed_cm_per_us"]
    G.tick_us = det["tick_us"]
    G.trig_tick = det["trigger_tick"]
    av = det["active_volume"]
    for i, v in enumerate(av["x"] + av["y"] + av["z"]):
        G.av[i] = v
    G.window = a.proj_window
    G.pad = a.proj_pad
    if a.sce_map and not a.data and not ROOT.sbndh5.load_sce(G, a.sce_map):   # data: no projection
        sys.exit("cannot load SCE map " + a.sce_map)
    return G


# ----------------------------------------------------------------------------------------------- main
def units_to_inputs(units):
    files, wcdirs = [], []
    for u in units:
        rf = json.load(open(os.path.join(u, "reco1_files.json")))
        files += [r["reco1"] for r in rf if r.get("events", 1) > 0]
        wcdirs.append(os.path.join(u, "pr", "all"))
    return files, wcdirs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--units", nargs="*", default=[])
    ap.add_argument("--units-file")
    ap.add_argument("--reco1", nargs="*", default=[])
    ap.add_argument("--wc-dir", nargs="*", default=[])
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--wc-tick-offset", type=float, default=0.0,
                    help="WC time_slice = floor((gaushit PeakTime + offset) / ticks_per_slice)")
    ap.add_argument("--wc-ticks-per-slice", type=int, default=4)
    ap.add_argument("--scan-offsets", type=int, default=0,
                    help="for the first N events with a WC footprint, print the hit/footprint overlap vs tick offset")
    ap.add_argument("--compression", default="gzip")
    ap.add_argument("--tdc-offset", type=float, default=2990.0, help="SimChannel TDC = hit tick + offset")
    ap.add_argument("--scan-tdc", action="store_true", help="on the first event, print hits with IDEs vs TDC offset")
    ap.add_argument("--sce-map", default="/lus/flare/projects/neutrinoGPU/scisoft/larsoft/sbnd_data/v01_42_00/SCEoffsets/"
                    "SCEoffsets_SBND_E500_dualmap_CV_voxelTH3.root",
                    help="forward SCE map for the deposit projection ('' = no SCE)")
    ap.add_argument("--proj-dt", default="5,2,-2",
                    help="deposit projection: tick offset per plane U,V,Y (fitted on BNB CV: max hits with deposits)")
    ap.add_argument("--proj-window", type=float, default=1.0, help="deposit projection: hit window = peak +- k*RMS")
    ap.add_argument("--proj-pad", type=float, default=0.0, help="deposit projection: extra ticks on each side")
    ap.add_argument("--ophit-window", default="-10,40",
                    help="keep ophits with peaktime in [lo, hi] us ('all' = every ophit); SBND PMT readout spans +-1.5 ms")
    ap.add_argument("--scan-proj", type=int, default=0, help="on the first N events, print hits with deposits vs offset")
    ap.add_argument("--detector-json", default=os.path.join(HERE, "sbnd_detector.json"))
    ap.add_argument("--data", action="store_true",
                    help="real data: read no truth product (truth tables empty, event_table truth columns at defaults)")
    ap.add_argument("--blob-dir", default="",
                    help="0.6: directory of <unit name>.blobs.tar.gz (the wct-blobdump.jsonnet output of each unit)")
    ap.add_argument("--blob-dump", nargs="*", default=[], help="0.6: blob-dump tars, in addition to --blob-dir")
    ap.add_argument("--frameshift-label", default="frameshift",
                    help="art label of sbnd::timing::FrameShiftInfo ('' = do not read)")
    a = ap.parse_args()
    if a.data and (a.scan_tdc or a.scan_proj):
        sys.exit("--scan-tdc / --scan-proj need MC truth")

    units = list(a.units)
    if a.units_file:
        units += [l.split()[0] for l in open(a.units_file) if l.strip() and not l.startswith("#")]
    files, wcdirs = units_to_inputs(units)
    files += a.reco1
    wcdirs += a.wc_dir
    dumps = list(a.blob_dump)
    if a.blob_dir:
        for u in units:   # <unit>.blobs.tar.gz, or one <unit>.blobs.<segment>.tar.gz per step-1 segment (PARTIAL)
            g = sorted(glob.glob(os.path.join(a.blob_dir, os.path.basename(os.path.normpath(u)) + ".blobs*.tar.gz")))
            dumps += g or [os.path.join(a.blob_dir, os.path.basename(os.path.normpath(u)) + ".blobs.tar.gz")]
    blobs = None
    if dumps:
        import wc_blobs
        missing = [d for d in dumps if not os.path.exists(d)]
        if missing:
            sys.exit("missing blob dumps: %s" % missing)
        t0 = time.time()
        blobs = wc_blobs.BlobDump(dumps)
        print("blob dumps: %d files, %d events, %.0f s" % (len(dumps), len(blobs.ev), time.time() - t0), flush=True)
    if not files:
        sys.exit("no input files")

    t0 = time.time()
    ROOT = load_gallery()
    print("gallery + extract.h ready in %.0f s" % (time.time() - t0), flush=True)
    det = json.load(open(a.detector_json))
    wc = WC(wcdirs, a.wc_tick_offset, a.wc_ticks_per_slice, det)
    T = schema(blobs is not None)
    L = ROOT.sbndh5.Labels()
    L.tdc_offset = a.tdc_offset
    L.frameshift = a.frameshift_label
    G = make_geo(ROOT, det, a)
    o = ROOT.sbndh5.Out()
    vf = ROOT.std.vector("std::string")()
    for f in files:
        vf.push_back(f)
    ev = ROOT.gallery.Event(vf)
    fname2idx = {f: i for i, f in enumerate(files)}

    truth_src = 0
    seq, stats = 0, {"events": 0, "wc_missing": 0, "hits": 0, "hits_wc": 0, "edep_nu": 0, "edep_other": 0,
                     "mcp_no_truth": 0, "scan": {}, "scan_proj": {}, "sed_e_nu": 0.0, "sed_e_nu_matched": 0.0}
    scanned = 0
    while not ev.atEnd():
        t1 = time.time()
        if a.scan_tdc and seq == 0:
            offs = list(range(2990 - 400, 2990 + 401, 10)) + list(range(-200, 201, 20))
            vo = ROOT.std.vector("double")()
            for x in offs:
                vo.push_back(x)
            res = list(ROOT.sbndh5.scan_tdc(ev, L, vo))
            print("SimChannel summary [n, ch_min, ch_max, tdc_min, tdc_max, -, n_ide, sum_E]: %s"
                  % list(ROOT.sbndh5.sc_summary(ev, L)), flush=True)
            print("TDC-offset scan (offset: hits with IDEs): %s" % sorted(zip(offs, res), key=lambda kv: -kv[1])[:8], flush=True)
        if seq < a.scan_proj:
            offs = list(range(-40, 41, 1))
            vo = ROOT.std.vector("double")()
            for x in offs:
                vo.push_back(x)
            res = list(ROOT.sbndh5.scan_proj(ev, L, G, vo))
            for p in range(3):
                col = [res[i * 3 + p] for i in range(len(offs))]
                stats["scan_proj"].setdefault(p, [0] * len(offs))
                stats["scan_proj"][p] = [u + v for u, v in zip(stats["scan_proj"][p], col)]
            stats["channel_model_mismatch"] = [stats.get("channel_model_mismatch", [0, 0, 0])[p] + res[len(offs) * 3 + p]
                                               for p in range(3)]
            if seq == a.scan_proj - 1:
                for p in range(3):
                    col = stats["scan_proj"][p]
                    best = sorted(zip(offs, col), key=lambda kv: -kv[1])[:5]
                    print("projection dt scan plane %d (dt: hits with deposits): %s" % (p, best), flush=True)
                print("channel-model mismatches per plane: %s" % stats["channel_model_mismatch"], flush=True)
        if a.data:
            ROOT.sbndh5.fill_data(ev, o, L)
        else:
            ROOT.sbndh5.fill(ev, o, L, G)
        rse = (o.run, o.subrun, o.event)
        fidx = fname2idx.get(str(ev.getTFile().GetName()), -1)
        V = lambda v, dt: vec2np(ROOT, v, dt)
        nh = o.h_tpc.size()
        h_ch, h_t = V(o.h_channel, np.int32), V(o.h_time, np.float32)

        status, cands, fp = wc.event(rse)
        stats["wc_missing"] += status == 0
        brec = blobs.ev.get(rse) if blobs is not None else None
        bstat = -1 if blobs is None else int(brec is not None)
        nblob = len(brec["charge"]) if brec is not None else 0
        if blobs is not None:
            stats["blob_missing"] = stats.get("blob_missing", 0) + int(brec is None)
        T["event_table"].add(seq, rse, 1, is_cc=[o.is_cc], nu_pdg=[o.nu_pdg], nu_energy=[o.nu_energy],
                             lep_energy=[o.lep_energy], nu_dir=list(o.nu_dir), nu_vtx=list(o.nu_vtx),
                             nu_time_ns=[o.nu_time], lep_pdg=[o.lep_pdg], nu_mode=[o.nu_mode],
                             nu_int_type=[o.nu_int_type], n_nu=[o.n_nu], nu_index=[o.nu_sel],
                             beam_gate_start_ns=[o.beam_gate_start], beam_gate_width_ns=[o.beam_gate_width],
                             wc_status=[status], file_index=[fidx], n_mcp_no_truth=[o.n_mcp_no_truth],
                             time_s=[o.time_s], has_frameshift=[o.has_frameshift], timing_type=[o.timing_type],
                             frame_apply_at_caf_ns=[o.fs_apply_at_caf], frame_hlt_beamgate_ns=[o.fs_hlt_beamgate],
                             frame_hlt_crtt1_ns=[o.fs_hlt_crtt1], frame_tdc_crtt1_ns=[o.fs_tdc_crtt1],
                             frame_tdc_bes_ns=[o.fs_tdc_bes], frame_tdc_rwm_ns=[o.fs_tdc_rwm],
                             wc_n_bundle=[wc.census[0]], wc_n_bundle_cosmic=[wc.census[1]],
                             wc_blob_status=[bstat], wc_n_blobs=[nblob])
        nnu = o.v_index.size()
        T["nu_table"].add(seq, rse, nnu, nu_index=V(o.v_index, np.int32), nu_pdg=V(o.v_pdg, np.int32),
                          is_cc=V(o.v_cc, np.int32), nu_energy=V(o.v_energy, np.float32),
                          lep_energy=V(o.v_lep_energy, np.float32), lep_pdg=V(o.v_lep_pdg, np.int32),
                          nu_vtx=V(o.v_vtx, np.float32), nu_dir=V(o.v_dir, np.float32),
                          nu_time_ns=V(o.v_time, np.float32), nu_mode=V(o.v_mode, np.int32),
                          nu_int_type=V(o.v_int, np.int32), in_av=V(o.v_in_av, np.int32),
                          edep_collection=V(o.v_ecoll, np.float32), nhits_collection=V(o.v_nhits_coll, np.float32))
        T["hit_table"].add(seq, rse, nh, hit_id=np.arange(nh), integral=V(o.h_integral, np.float32),
                           rms=V(o.h_rms, np.float32), tpc=V(o.h_tpc, np.int32), local_plane=V(o.h_plane, np.int32),
                           local_wire=V(o.h_wire, np.int32), local_time=h_t, channel=h_ch,
                           peak_amplitude=V(o.h_amp, np.float32))
        e = dict(hit_id=V(o.e_hit, np.int32), g4_id=V(o.e_g4, np.int32), energy=V(o.e_energy, np.float32),
                 energy_fraction=V(o.e_frac, np.float32), is_max_ide=V(o.e_maxide, np.int32),
                 num_electrons=V(o.e_nelec, np.float32), origin=V(o.e_origin, np.int32),
                 nu_index=V(o.e_nu, np.int32))
        isnu = e["origin"] == 1
        T["edep_table"].add(seq, rse, int(isnu.sum()), **{k: v[isnu] for k, v in e.items()})
        T["edep_cosmic_table"].add(seq, rse, int((~isnu).sum()), **{k: v[~isnu] for k, v in e.items()})
        if not a.data:
            T["hit_truth_table"].add(seq, rse, nh, hit_id=np.arange(nh),
                                     **{c: V(getattr(o, "t_" + k), np.float32) for c, k in (
                                         ("true_x", "x"), ("true_y", "y"), ("true_z", "z"), ("app_x", "ax"),
                                         ("app_y", "ay"), ("app_z", "az"), ("energy", "e"))})
        npart = o.p_g4.size()
        T["particle_table"].add(seq, rse, npart, g4_id=V(o.p_g4, np.int32), g4_pdg=V(o.p_pdg, np.int32),
                                parent_id=V(o.p_parent, np.int32), category=V(o.p_cat, np.int32),
                                instance=V(o.p_inst, np.int32), origin=V(o.p_origin, np.int32),
                                nu_index=V(o.p_nu, np.int32),
                                momentum=V(o.p_mom, np.float32), start_position=V(o.p_start, np.float32),
                                end_position=V(o.p_end, np.float32), start_time=V(o.p_t, np.float32),
                                start_process=[str(s).encode()[:24] for s in o.p_sproc],
                                end_process=[str(s).encode()[:24] for s in o.p_eproc])
        oph = dict(hit_channel=V(o.o_ch, np.int32), peaktime=V(o.o_t, np.float32),
                   peaktime_abs=V(o.o_tabs, np.float32), width=V(o.o_width, np.float32),
                   area=V(o.o_area, np.float32), amplitude=V(o.o_amp, np.float32), pe=V(o.o_pe, np.float32))
        if a.ophit_window != "all":
            lo, hi = (float(x) for x in a.ophit_window.replace(":", ",").split(","))
            keep = (oph["peaktime"] >= lo) & (oph["peaktime"] <= hi)
            oph = {k: v[keep] for k, v in oph.items()}
        no = len(oph["pe"])
        T["ophit_table"].add(seq, rse, no, hit_id=np.arange(no), **oph)
        T["opflash_table"].add(seq, rse, o.f_id.size(), flash_id=V(o.f_id, np.int32), tpc=V(o.f_tpc, np.int32),
                               time=V(o.f_time, np.float32), time_width=V(o.f_twidth, np.float32),
                               y_center=V(o.f_y, np.float32), y_width=V(o.f_yw, np.float32),
                               z_center=V(o.f_z, np.float32), z_width=V(o.f_zw, np.float32),
                               totalpe=V(o.f_pe, np.float32))
        T["opflashsumpe_table"].add(seq, rse, o.s_id.size(), sumpe_id=V(o.s_id, np.int32),
                                    flash_id=V(o.s_flash, np.int32), pmt_channel=V(o.s_ch, np.int32),
                                    sumpe=V(o.s_pe, np.float32))
        if cands:
            T["wc_event_table"].add(seq, rse, len(cands), **{k: [c[k] for c in cands] for k in T["wc_event_table"].cols})
        h_wcch = wc.wc_channel(h_ch, V(o.h_tpc, np.int32), V(o.h_plane, np.int32))
        cid, nui, mn, en = wc.hit_labels(fp, h_wcch, h_t)
        T["wc_hit_table"].add(seq, rse, nh, hit_id=np.arange(nh), wc_cluster_id=cid, wc_nu_index=nui, wc_main=mn,
                              wc_in_enu=en)
        if brec is not None:
            T["wc_blob_table"].add(seq, rse, nblob, **wc_blobs.blob_rows(brec, wc.role))
            cr = wc_blobs.cluster_rows(brec, wc.tcl, wc.role)
            T["wc_cluster_table"].add(seq, rse, len(cr["cluster_id"]), **cr)
            hh, bb = blobs.hit_blob(brec, V(o.h_tpc, np.int32), V(o.h_plane, np.int32), V(o.h_wire, np.int32), h_t,
                                    a.wc_tick_offset)
            nper = np.bincount(hh, minlength=nh) if nh else np.zeros(0, np.int64)
            T["wc_hit_blob_table"].add(seq, rse, len(hh), hit_id=hh.astype(np.int32), blob_id=bb.astype(np.int32),
                                       weight=(1.0 / np.maximum(nper[hh], 1)).astype(np.float32))
            stats["blobs"] = stats.get("blobs", 0) + nblob
            stats["hit_blob_pairs"] = stats.get("hit_blob_pairs", 0) + len(hh)
            stats["hits_in_blobs"] = stats.get("hits_in_blobs", 0) + int((nper > 0).sum())
            if not a.data and "truthtrack" in brec:
                ids = ROOT.std.vector("int")()
                for t_ in brec["truthtrack"][0][:, 0].astype(np.int64).tolist():
                    ids.push_back(int(t_))
                kin, kst = ROOT.std.vector("float")(), ROOT.std.vector("int")()
                ROOT.sbndh5.track_kin(ev, L, ids, kin, kst)
                tr = wc_blobs.track_rows(brec, V(kin, np.float32), V(kst, np.int32))
                T["track_table"].add(seq, rse, len(tr["trackid"]), **tr)
        if fp is not None and scanned < a.scan_offsets:
            scanned += 1
            for off in range(-3400, 3401, 8):
                c2 = wc.hit_labels(fp, h_wcch, h_t, off=off)[0]
                stats["scan"][off] = stats["scan"].get(off, 0) + int((c2 >= 0).sum())

        stats["events"] += 1
        stats["hits"] += nh
        stats["hits_wc"] += int((cid >= 0).sum())
        stats["edep_nu"] += int(isnu.sum())
        stats["edep_other"] += int((~isnu).sum())
        stats["mcp_no_truth"] += o.n_mcp_no_truth
        stats["truth_source"] = int(o.truth_source)
        print("event %d r%d s%d e%d: %d hits, %d edep (%d nu), %d particles, %d ophits, %d flashes, wc %d "
              "(%d cand, %d hits on WC nu clusters), %.1f s; mcp %d (no MCTruth %d), assns hit-truth %d mcp-truth %d"
              % (seq, *rse, nh, len(e["origin"]), isnu.sum(), npart, no, o.f_id.size(), status, len(cands),
                 (cid >= 0).sum(), time.time() - t1, o.n_mcp, o.n_mcp_no_truth, o.n_assns_hit_truth,
                 o.n_assns_mcp_truth), flush=True)
        truth_src = o.truth_source
        stats["sed_e_nu"] += o.sed_e_nu
        stats["sed_e_nu_matched"] += o.sed_e_nu_matched
        stats["ide_unmapped"] = stats.get("ide_unmapped", 0) + o.n_ide_unmapped
        seq += 1
        if a.max_events and seq >= a.max_events:
            break
        ev.next()

    if stats["scan"]:
        best = sorted(stats["scan"].items(), key=lambda kv: -kv[1])[:5]
        print("WC tick-offset scan (offset: matched hits): %s" % best, flush=True)
    stats["scan"] = {str(k): v for k, v in stats["scan"].items()}

    tmp = a.out + ".tmp"
    with h5py.File(tmp, "w") as f:
        rows = {name: t.write(f, a.compression) for name, t in T.items()}
        f.attrs["creator"] = VERSION
        f.attrs["is_data"] = int(a.data)
        f.attrs["wc_label_notes"] = WC_LABEL_NOTES
        f.attrs["reco1_files"] = json.dumps(files)
        f.attrs["wc_dirs"] = json.dumps(wcdirs)
        f.attrs["wc_tick_offset"] = a.wc_tick_offset
        f.attrs["wc_channel_offsets"] = json.dumps(wc.ch_off.tolist())
        f.attrs["ophit_window_us"] = a.ophit_window
        f.attrs["wc_ticks_per_slice"] = a.wc_ticks_per_slice
        if blobs is not None:
            f.attrs["wc_blob_notes"] = json.dumps(wc_blobs.BLOB_ATTRS)
            f.attrs["wc_blob_dumps"] = json.dumps(dumps)
            f.attrs["wc_blob_hit_rule"] = ("hit in blob: same tpc; slice_index_min <= floor(local_time + wc_tick_offset) "
                                           "< slice_index_max; wire_index_min <= local_wire < wire_index_max on the "
                                           "hit's plane; weight = 1 / blobs holding the hit")
        f.attrs["hit_truth"] = {1: "gaushitTruthMatch assns",
                                2: "SimChannel %s backtracking, TDC = tick + %g, window PeakTime +- %g RMS"
                                   % (L.simchannel, L.tdc_offset, L.hit_time_rms),
                                0: "none (real data)" if a.data else "none",
                                3: "projection of %s deposits (SCE map %s) onto (channel, tick); window PeakTime +- %g RMS"
                                   " +- %g ticks" % (L.sed, a.sce_map or "none", a.proj_window, a.proj_pad)}.get(truth_src, "none")
        f.attrs["truth_projection"] = json.dumps({"dt_ticks_UVY": [float(x) for x in a.proj_dt.replace(":", ",").split(",")],
                                                  "window_rms": a.proj_window, "pad_ticks": a.proj_pad,
                                                  "sce_map": a.sce_map, "drift_cm_per_us": det["drift_speed_cm_per_us"],
                                                  "trigger_tick": det["trigger_tick"], "tick_us": det["tick_us"]})
        f.attrs["labels"] = json.dumps({"hit": str(L.hit), "hit_truth": str(L.hit_truth), "truth": str(L.truth),
                                        "mcparticle": str(L.mcparticle), "mcreco": str(L.mcreco),
                                        "ophit": str(L.ophit), "opflash": [str(x) for x in L.opflash],
                                        "beamgate": str(L.beamgate), "frameshift": str(L.frameshift)})
        if os.path.exists(a.detector_json):
            f.attrs["detector"] = open(a.detector_json).read()
            if "simulation" in det and not a.data:
                f.attrs["sim_config"] = json.dumps(det["simulation"])
        f.attrs["units"] = json.dumps(UNITS)
        f.attrs["nu_choice"] = "none (real data)" if a.data else ("event_table neutrino = nu_table row with the largest edep_collection (deposited energy of "
                                "that neutrino's particles on collection-plane hits); ties go to in_av = 1, then to the "
                                "lowest nu_index. event_table.nu_index names it (-1 = no neutrino).")
    os.replace(tmp, a.out)
    stats["rows"] = rows
    stats["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(stats, indent=1))
    json.dump(stats, open(a.out + ".stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
