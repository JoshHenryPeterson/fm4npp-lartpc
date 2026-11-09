"""Downstream task definitions (single source of truth). Thresholds fixed after the data audit.

T1 hit     : per-hit neutrino vs cosmic/noise, on events with a visible in-AV neutrino.
T2 evt_nu  : 3 classes: 0 = essentially no neutrino charge (cosmic-only-like), 1 = visible in-AV neutrino
             interaction, 2 = neutrino interacting outside the AV that still leaves charge in the TPC.
             In-AV events with low visibility (between the thresholds) are excluded.
T3 flavor  : FV + visible: 0 = numu CC, 1 = nue CC, 2 = NC.
T4 energy  : FV + visible + CC: regress log(E_nu / GeV).
T5 light   : (round 4) all events: predict the beam-window PMT light (32 PMTs x 17 light-token features) from the
             charge alone. The target is built from the event's optical hits in the batch (no label array).
"""
import numpy as np

from qlfm.geometry import FV_MARGIN_CM, in_volume

HIT_NU_EDEP_MEV = 0.05    # hit is "neutrino" if its MC deposit exceeds this
# Two thresholds, on purpose (audit #10): the T1 per-hit target uses > HIT_NU_EDEP_MEV, while the event
# visibility counts nu_nhits (T2 classes, T1/T3/T4 eligibility) count every hit with any MC deposit (> 0,
# qlfm/data/convert.py). So a "visible" event can have a few of its counted hits labelled cosmic in T1, and a
# T2 class-0 event can hold up to VIS_NEG_HITS hits with tiny deposits. Hits with 0 < deposit <= 0.05 MeV are
# 0.12-0.15% of all test hits (qlfm/eval_perturb.py, frac_sim_below_label).
VIS_POS_Y_HITS = 30       # T2 positive: >= this many collection-plane neutrino hits
VIS_NEG_HITS = 5          # T2 negative: <= this many neutrino hits (all planes)

TASKS = ("hit", "evt_nu", "flavor", "energy", "light")
N_OUT = {"hit": 1, "evt_nu": 3, "flavor": 3, "energy": 1, "light": 17}


def event_labels(ev):
    """ev: dict of per-event arrays (from Corpus.ev_array). Returns eligibility masks + targets."""
    vtx = ev["nu_vtx"]
    in_av = in_volume(vtx)
    in_fv = in_volume(vtx, FV_MARGIN_CM)
    ny = ev["nu_nhits"][:, 2]
    ntot = ev["nu_nhits"].sum(1)
    visible = in_av & (ny >= VIS_POS_Y_HITS)
    nonu = ntot <= VIS_NEG_HITS
    out_av = ~in_av & ~nonu
    apdg = np.abs(ev["nu_pdg"])
    cc = ev["is_cc"] == 1
    flavor = np.where(~cc, 2, np.where(apdg == 12, 1, 0))
    return {
        "hit": (visible, None),
        "evt_nu": (visible | nonu | out_av, np.where(visible, 1, np.where(out_av, 2, 0)).astype(np.int64)),
        "flavor": (in_fv & visible, flavor.astype(np.int64)),
        "energy": (in_fv & visible & cc, np.log(np.clip(ev["nu_energy"], 1e-3, None)).astype(np.float32)),
        "light": (np.ones(len(vtx), bool), None),
    }


EV_KEYS = ("nu_vtx", "nu_nhits", "nu_pdg", "is_cc", "nu_energy")
