#!/usr/bin/env python3
"""Write sbnd_detector.json: the SBND constants the qlfm code needs (data-inventory-sbnd.md section 3/4).

    python3 make_detector_json.py <sbnd_pds_mapping.json> <semi-analytical-sbnd.json> <sbnd-wires-geometry-v0206.json.bz2> > sbnd_detector.json

Sources (sbndcode v10_14_02_03, the version of the Gen2 reco1 MC):
  detectorclocks_sbnd.fcl      TriggerOffsetTPC -205 us, ClockSpeedTPC 2 MHz, FramePeriod 1300 us
  detectorproperties_sbnd.fcl  ReadOutWindowSize 3400, Efield 0.5 kV/cm, Temperature 88.4 K
  WCT sbnd/simparams.jsonnet   drift_speed 1.563 mm/us (the value the Wire-Cell sim and reco use)
  sbnd_pds_mapping.json        OpChannel -> pd_type (pmt_coated / pmt_uncoated / xarapuca_vis / xarapuca_vuv), tpc
  wire-cell-data sbnd/photodet/semi-analytical-sbnd.json   OpDet centres [cm]; OpChannel == OpDet
  wire-cell-data sbnd-wires-geometry-v0206.json.bz2         wire endpoints [mm] and channels (the geometry WCT uses):
                               every plane is linear (pitch std < 2e-3 mm), channels contiguous along the plane
"""
import bz2
import json
import sys

import numpy as np

pds = {d["channel"]: d for d in json.load(open(sys.argv[1]))}
pos = json.load(open(sys.argv[2]))
centres = pos["OpDets"]          # list indexed by OpDet == OpChannel, [cm]
g = pos["Geometry"]
assert len(centres) == len(pds) == 312

opdets = []
for ch in sorted(pds):
    c = centres[ch]
    xyz = [float(c[k]) for k in ("x", "y", "z")]
    opdets.append({"channel": ch, "type": pds[ch]["pd_type"], "tpc": pds[ch]["tpc"], "x": xyz[0], "y": xyz[1], "z": xyz[2]})

# ---- linear wire-plane model: channel = channel0 + channel_step * round(((y, z) - wire0_yz) . pitch_dir / pitch)
st = json.load(bz2.open(sys.argv[3]))["Store"]
P = np.array([[p["Point"][k] for k in "xyz"] for p in st["points"]]) / 10.0  # cm
W = st["wires"]
wire_planes = []
for tpc, an in enumerate(st["anodes"]):
    for fi in an["Anode"]["faces"]:
        for pi, pl in enumerate(st["faces"][fi]["Face"]["planes"]):
            wl = st["planes"][pl]["Plane"]["wires"]
            t = np.array([P[W[w]["Wire"]["tail"]] for w in wl])
            h = np.array([P[W[w]["Wire"]["head"]] for w in wl])
            ch = np.array([W[w]["Wire"]["channel"] for w in wl])
            c = (t + h) / 2
            d = (h[0] - t[0]) / np.linalg.norm(h[0] - t[0])
            pd = np.array([-d[2], d[1]])                  # (y, z) unit vector perpendicular to the wire
            proj = (c[:, 1:] - c[0, 1:]) @ pd
            pitch = float(np.mean(np.diff(proj)))
            dev = float(np.max(np.abs(proj - pitch * np.arange(len(wl)))))
            print("tpc %d plane %d: max deviation from the linear model %.4f cm (pitch %.5f)" % (tpc, pi, dev, pitch), file=sys.stderr)
            step = int(ch[1] - ch[0])
            assert step in (1, -1) and np.all(np.diff(ch) == step) and dev < 0.3 * abs(pitch)
            wire_planes.append({"tpc": tpc, "plane": pi, "x": round(float(c[0, 0]), 4), "n_wires": len(wl),
                                "channel0": int(ch[0]), "channel_step": step, "wire0_yz": [float(c[0, 1]), float(c[0, 2])],
                                "pitch_dir_yz": [float(pd[0]), float(pd[1])], "pitch": pitch,
                                "wire_dir": [round(float(v), 5) for v in d]})

out = {
    "detector": "SBND",
    "units": "cm, us, ticks",
    "tpcs": [
        {"tpc": 0, "drift_sign": -1, "anode_x": -202.05, "cathode_x": 0.0, "note": "anode_x = collection-plane x"},
        {"tpc": 1, "drift_sign": +1, "anode_x": +202.05, "cathode_x": 0.0},
    ],
    "active_volume": {"x": [-200.0, 200.0],
                      "y": [g["active_center_y"] - g["active_size_y"] / 2, g["active_center_y"] + g["active_size_y"] / 2],
                      "z": [g["active_center_z"] - g["active_size_z"] / 2, g["active_center_z"] + g["active_size_z"] / 2],
                      "note": "y, z from wire-cell-data semi-analytical-sbnd.json Geometry; x = +-200 cm drift (cathode at 0)"},
    "planes": [
        {"plane": 0, "name": "U", "n_wires": 1984, "pitch_cm": 0.3, "angle_deg": 60},
        {"plane": 1, "name": "V", "n_wires": 1984, "pitch_cm": 0.3, "angle_deg": -60},
        {"plane": 2, "name": "Y", "n_wires": 1670, "pitch_cm": 0.3, "angle_deg": 0},
    ],
    "channels_per_tpc": 5638,
    "n_channels": 11276,
    "wire_planes": wire_planes,
    "tick_us": 0.5,
    "readout_ticks": 3427,
    "readout_ticks_note": "Wire-Cell detsim/SP frame (WCT sbnd params daq.nticks); detectorproperties ReadOutWindowSize says 3400, the gaushits reach tick 3425",
    "trigger_tick": 410,
    "trigger_tick_note": "TriggerOffsetTPC = -205 us: the TPC readout starts 205 us (410 ticks) before the trigger; hit_table.local_time is the gaushit PeakTime in ticks from the readout start",
    "drift_speed_cm_per_us": 0.1563,
    "drift_cm_per_tick": 0.07815,
    "drift_note": "x ~ anode_x - drift_sign * (local_time - trigger_tick) * drift_cm_per_tick for charge produced at the trigger time; the per-plane time offsets fitted for the truth projection are in the HDF5 attribute truth_projection",
    "max_drift_cm": 200.0,
    "light": {
        "ophit_label": "ophitpmt (PMTs only; no X-ARAPUCA ophits in reco1)",
        "time_unit": "us, relative to the trigger (OpHit::PeakTime); peaktime_abs is OpHit::PeakTimeAbs",
        "beam_gate_note": "MC beam gate start from sim::BeamGateInfo (event_table.beam_gate_start_ns); BNB spill ~1.6 us long",
    },
    "simulation": {
        "source": "Avinay's Gen2 production (sbn_parsl wct0319 restartable, runinfo/config.toml): sbndcode v10_14_02_03 e26:prof; "
                  "gen prodgenie_corsika_proton_rockbox0p1_sbnd.fcl, g4 standard_g4_rockbox_sbnd.fcl, detsim standard_detsim_sbnd.fcl, "
                  "reco1 standard_reco1_sbnd.fcl",
        "tpc_detsim": {"module": "simtpc2d = sbnd_wcls_simsp_bothrois (wcsimsp_sbnd.fcl), Wire-Cell 2D sim + NF + SP",
                       "DL_cm2_per_s": 4.0, "DT_cm2_per_s": 8.8, "electron_lifetime_ms": 35.0, "drift_speed_mm_per_us": 1.563,
                       "fluctuate": False, "nticks": 3427, "tick0_time_us": -205,
                       "field_response": "garfield-sbnd-v1.json.bz2", "noise": "sbnd-noise-spectra-v1.json.bz2 (scaled from ProtoDUNE-SP)",
                       "enableLowROIThresholds": True, "deposits": "ionandscint (with SCE); priorSCE kept in reco1"},
        "hit_finder": {"label": "gaushit", "module": "sbnd_gaushitfinder (hitfindermodules_sbnd.fcl), morphological candidate finder per plane",
                       "input": "simtpc2d:gauss (CalDataModuleLabel, reco_sbnd.fcl)"},
        "light": {"pmt_digitizer": "opdaq (digi_pmt_sbnd.fcl)", "PMTCoatedVUVEff": 0.0392, "PMTUncoatedEff": 0.0357,
                  "PMTDarkNoiseRate_Hz": 1000.0, "PMTBaselineRMS_ADC": 2.6, "PMTChargeToADC": -25.97, "PMTSinglePEmodel": "data",
                  "ScintPreScale": 0.0392, "photon_sim": "pdfastsim (semi-analytical) G4 stage", "ophit_label": "ophitpmt"},
    },
    "opdets": opdets,
    "n_pmt": sum(1 for o in opdets if o["type"].startswith("pmt")),
}
json.dump(out, sys.stdout, indent=1)
sys.stdout.write("\n")
