"""Detector constants used by the tokenizers and label definitions: MicroBooNE (default) or SBND.

The detector is chosen at import by the environment variable QLFM_DETECTOR (uboone | sbnd); the converter refuses
an SBND file under the MicroBooNE constants.

MicroBooNE sources: uboone/OpenSamples microboone_utils.py, arXiv:2011.01375, PMT map from
NuTufts/ublightmodelnet & LArbys/lardly (keyed by OpChannel; x ~ -11 cm behind anode).

SBND (issue #11): constants from sbnd-h5maker/sbnd_detector.json (QLFM_DETECTOR_JSON overrides the path). Two TPCs
drift away from a central cathode at x = 0. Hit times are mapped to one "virtual" tick axis so that the existing
token position (tick - TRIG_TICK) * TICK_TO_CM equals x + anode distance for in-time charge:
  TPC 0 (x < 0): virtual tick = tick;   TPC 1 (x > 0): virtual tick = TPC1_FLIP_TICK - tick.
Tokens also key on the TPC (hit_tpc), so the two TPCs never share a token.
"""
import json
import os

import numpy as np

DETECTOR = os.environ.get("QLFM_DETECTOR", "uboone").lower()
N_TPC = 1
TPC1_FLIP_TICK = None

N_PLANES = 3
N_WIRES = (2400, 2400, 3456)
N_TICKS = 6400                # stored TPC window, 0.5 us / tick
TRIG_TICK = 800               # trigger (anode, t0=0) tick in the stored window
TICK_TO_CM = 0.0548965        # drift cm per tick (v_d = 0.1098 cm/us)
CATHODE_TICK = TRIG_TICK + 256.35 / TICK_TO_CM   # ~5470
WIRE_PITCH_CM = 0.3

# active volume (approx., from microboone_utils) and 3 cm fiducial margin
AV = dict(x=(0.0, 256.35), y=(-116.5, 116.5), z=(0.0, 1036.8))
FV_MARGIN_CM = 3.0

N_PMT = 32
PMT_WINDOW_US = 23.4          # beam-gate PMT readout (1500 samples @ 64 MHz)

# OpChannel -> (y, z) cm
PMT_YZ = np.array([
    (55.313, 951.861), (55.822, 911.066), (27.607, 989.712), (-0.722, 865.598),
    (-28.625, 990.356), (-56.514, 951.865), (-56.309, 911.939), (55.625, 751.884),
    (55.800, 711.073), (-0.502, 796.208), (-0.051, 664.203), (-56.284, 751.905),
    (-56.408, 711.274), (55.822, 540.929), (55.771, 500.134), (-0.549, 585.284),
    (-0.875, 453.096), (-56.205, 540.616), (-56.323, 500.221), (54.693, 328.212),
    (54.646, 287.976), (-0.706, 373.839), (-0.829, 242.014), (-57.022, 328.341),
    (-56.261, 287.639), (55.249, 128.355), (55.249, 87.7605), (27.431, 51.1015),
    (-0.303, 173.743), (-28.576, 50.4745), (-56.203, 128.179), (-56.615, 87.8695),
], dtype=np.float32)


PMT_CH = np.arange(N_PMT)      # OpChannel of each light token (index = token / PMT id)
PMT_Y_SHIFT = 65.0             # added to PMT y for the Fourier position embedding (scale 130 cm)
PMT_SIDE = np.zeros(N_PMT, dtype=np.int64)   # PMT plane: 0 for MicroBooNE; SBND 0 = TPC 0 (x < 0), 1 = TPC 1 (x > 0)
PMT_POS_SCALE = (130.0, 1100.0)              # (y + PMT_Y_SHIFT, z) extent, cm, for the "v2" PMT position embedding
# 3D extent (x, y, z), cm, for the 3D position embeddings (blob tokens, 3D-lifted charge tokens); x is the
# drift coordinate of the blob centre, y and z the AV
POS3D_SCALE = (260.0, 240.0, 1040.0)
POS3D_MIN = (0.0, -116.5, 0.0)                # AV minimum corner; 3D embeddings see (xyz - POS3D_MIN) / POS3D_SCALE
# PMT time bins (us on the beam-gate clock); fine around the expected neutrino flash (3-6 us in MicroBooNE)
LIGHT_EDGES = (0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 19.0, 24.0)
BEAM_WINDOW_US = (3.0, 6.0)
T2_PE_CUTS = (10, 20, 50, 100, 200, 500)   # beam-window PE grid for the T2 bright-flash cut (round 4)


def _load_sbnd(path):
    global N_WIRES, N_TICKS, TRIG_TICK, TICK_TO_CM, CATHODE_TICK, WIRE_PITCH_CM, AV, N_PMT, PMT_YZ, PMT_CH
    global PMT_Y_SHIFT, LIGHT_EDGES, BEAM_WINDOW_US, T2_PE_CUTS, N_TPC, TPC1_FLIP_TICK, PMT_SIDE, PMT_POS_SCALE
    global POS3D_SCALE, POS3D_MIN
    d = json.load(open(path))
    assert d["detector"] == "SBND", path
    N_TPC = len(d["tpcs"])
    N_WIRES = tuple(p["n_wires"] for p in sorted(d["planes"], key=lambda p: p["plane"]))
    WIRE_PITCH_CM = float(d["planes"][0]["pitch_cm"])
    TRIG_TICK = int(d["trigger_tick"])
    TICK_TO_CM = float(d["drift_cm_per_tick"])
    span = max(abs(t["anode_x"]) for t in d["tpcs"])                 # collection plane |x|
    TPC1_FLIP_TICK = 2 * TRIG_TICK + 2 * span / TICK_TO_CM            # ~5991
    N_TICKS = int(np.ceil(TPC1_FLIP_TICK))                            # virtual tick range
    CATHODE_TICK = TRIG_TICK + span / TICK_TO_CM
    AV = {k: tuple(v) for k, v in d["active_volume"].items() if k in "xyz"}
    pmts = sorted((o for o in d["opdets"] if o["type"].startswith("pmt")), key=lambda o: o["channel"])
    N_PMT = len(pmts)
    PMT_CH = np.array([o["channel"] for o in pmts])
    PMT_YZ = np.array([(o["y"], o["z"]) for o in pmts], dtype=np.float32)
    PMT_Y_SHIFT = -AV["y"][0]                                          # y + shift in [0, ~410] cm
    PMT_SIDE = np.array([int(o["x"] > 0) for o in pmts], dtype=np.int64)
    PMT_POS_SCALE = (AV["y"][1] - AV["y"][0], AV["z"][1] - AV["z"][0])   # ~(407, 501) cm
    POS3D_SCALE = (AV["x"][1] - AV["x"][0], AV["y"][1] - AV["y"][0], AV["z"][1] - AV["z"][0])
    POS3D_MIN = (AV["x"][0], AV["y"][0], AV["z"][0])
    # SBND OpHit::PeakTime is relative to the trigger; the BNB flash comes at ~0.3-1.7 us. The MicroBooNE bins are
    # shifted by -3 us, so the beam window (3-6 us there) becomes 0-3 us and the bin layout stays the same.
    LIGHT_EDGES = tuple(e - 3.0 for e in LIGHT_EDGES)
    BEAM_WINDOW_US = (0.0, 3.0)
    T2_PE_CUTS = (100, 200, 500, 1000, 2000, 5000)   # 120 PMTs see ~10x more light (val best cut ~2.7k PE)


if DETECTOR == "sbnd":
    _load_sbnd(os.environ.get("QLFM_DETECTOR_JSON", os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "sbnd-h5maker", "sbnd_detector.json")))
elif DETECTOR != "uboone":
    raise ValueError(f"QLFM_DETECTOR={DETECTOR}: expected uboone or sbnd")


def virtual_tick(tick, tpc):
    """Hit tick -> tick on the single token time axis (identity for MicroBooNE; TPC 1 mirrored for SBND)."""
    if TPC1_FLIP_TICK is None:
        return tick
    return np.where(np.asarray(tpc) == 1, TPC1_FLIP_TICK - np.asarray(tick), tick)


def tick_to_x(t):
    """Apparent drift coordinate (cm) assuming t0 = trigger."""
    return (np.asarray(t) - TRIG_TICK) * TICK_TO_CM


def in_volume(xyz, margin=0.0):
    """Boolean mask for points (N,3) inside the active volume shrunk by `margin` cm."""
    xyz = np.asarray(xyz)
    m = np.ones(len(xyz), bool)
    for i, k in enumerate("xyz"):
        lo, hi = AV[k]
        m &= (xyz[:, i] > lo + margin) & (xyz[:, i] < hi - margin)
    return m
