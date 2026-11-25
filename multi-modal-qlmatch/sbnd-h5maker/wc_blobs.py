"""sbnd-h5maker 0.6: Wire-Cell 3D blobs (fm4npp #16).

Input: the post-split point-cloud tree of each Wire-Cell unit, written by `wct/wct-blobdump.jsonnet` (WCT 0.38.0 step 2
truncated after protect_bundle, + TensorFileSink).  Its live tree is grouping -> clusters -> blobs, with the step-2
cluster numbering (= T_cluster / T_proj_data / wc_hit_table.wc_cluster_id) and, in simulation, the step-1
TensorSetLabeler blob truth (trackid, truth_q, truth_q_dom, trackid2, truth_q2) and the issue-41 truthtrack table.

Tables (one row per object, hep_hpc layout):
  wc_blob_table      every live blob of every cluster
  wc_cluster_table   every step-2 cluster (flags from the production tracking-pr.root T_cluster)
  wc_hit_blob_table  hit <-> blob, many-to-many (a hit is in a blob if its wire is inside the blob's wire range on the
                     hit's plane and its peak tick inside the blob's slice)
  track_table        (sim) one row per G4 track with deposits in the readout window (labeler truthtrack) + MCParticle
                     kinematics + blob census
"""
import io
import json
import re
import tarfile

import numpy as np

MM_TO_CM = 0.1


def tensor_sets(path):
    """Stream (metadata, {datapath: array}, {datapath: tensor metadata}) per tensor set of a WCT tensor tar."""
    meta, arrs, tmeta, pend = None, {}, {}, {}
    with tarfile.open(path, "r|*") as t:
        for m in t:
            n = m.name.rsplit("/", 1)[-1]
            r = re.match(r"clustering_tensorset_(\d+)_metadata\.json$", n)
            if r:
                if meta is not None:
                    yield meta, arrs, tmeta
                meta, arrs, tmeta, pend = json.load(t.extractfile(m)), {}, {}, {}
                continue
            r = re.match(r"clustering_tensor_(\d+)_(\d+)_(metadata\.json|array\.npy)$", n)
            if not r:
                continue
            if r.group(3) == "metadata.json":
                md = json.load(t.extractfile(m))
                pend[r.group(2)] = md
                if md.get("datapath"):
                    tmeta[md["datapath"]] = md
            else:
                dp = pend.get(r.group(2), {}).get("datapath")
                arrs[dp] = np.load(io.BytesIO(t.extractfile(m).read()), allow_pickle=False)
    if meta is not None:
        yield meta, arrs, tmeta


class BlobDump:
    """Per-event reduced blob records from one or more blob-dump tars, keyed by (run, subrun, event)."""

    def __init__(self, paths):
        self.ev = {}
        for p in paths:
            for meta, A, M in tensor_sets(p):
                rse = (int(meta["runNo"]), int(meta["subRunNo"]), int(meta["eventNo"]))
                self.ev[rse] = self._reduce(A, M)

    @staticmethod
    def _reduce(A, M):
        live = [k for k in A if k and re.match(r"pointtrees/\d+/live$", k)]
        if not live:
            return None
        L = live[0]
        par = A[L]
        depth = np.zeros(len(par), np.int64)
        for i in range(1, len(par)):
            depth[i] = depth[par[i]] + 1
        lp = lambda k: A[L + "/lpcmaps/arrays/" + k]
        N = L + "/pointclouds/namedpcs/"
        cl_nodes = np.where(lp("cluster_scalar") > 0)[0]
        bl_nodes = np.where(lp("scalar") > 0)[0]
        assert np.all(depth[cl_nodes] == 1) and np.all(depth[bl_nodes] == 2)
        rank = np.full(len(par), -1, np.int64)
        rank[cl_nodes] = np.arange(len(cl_nodes))
        S = lambda k: A[N + "scalar/arrays/" + k]
        C = lambda k: A[N + "cluster_scalar/arrays/" + k]
        ids = C("ident").astype(np.int64)
        r = {"cluster_ident": ids, "blob_cluster": ids[rank[par[bl_nodes]]]}
        for k in ("wpid", "slice_index_min", "slice_index_max", "u_wire_index_min", "u_wire_index_max",
                  "v_wire_index_min", "v_wire_index_max", "w_wire_index_min", "w_wire_index_max", "npoints"):
            r[k] = S(k).astype(np.int64)
        for k in ("center_x", "center_y", "center_z", "charge"):
            r[k] = S(k).astype(np.float64)
        for k, dflt in (("trackid", -1), ("trackid2", -1), ("truth_q", 0.0), ("truth_q_dom", 0.0), ("truth_q2", 0.0)):
            r[k] = S(k) if N + "scalar/arrays/" + k in A else np.full(len(bl_nodes), dflt)
        # blob t0 shift: mean of (x_t0cor - x) over the blob's 3d points (3d rows are in node order, lpcmaps = count)
        n3 = lp("3d")[bl_nodes].astype(np.int64)
        x, xt = A[N + "3d/arrays/x"], A[N + "3d/arrays/x_t0cor"]
        assert n3.sum() == len(x)
        seg = np.repeat(np.arange(len(bl_nodes)), n3)
        s = np.bincount(seg, xt - x, minlength=len(bl_nodes))
        r["dx_t0"] = np.where(n3 > 0, s / np.maximum(n3, 1), np.nan)
        r["n3d"] = n3
        tt = [k for k in A if k and re.match(r"truthtrack/\d+$", k)]
        if tt:
            r["truthtrack"] = (A[tt[0]], M[tt[0]]["columns"])
        return r

    @staticmethod
    def hit_blob(r, tpc, plane, wire, tick, tick_offset=0.0):
        """(hit_id, blob_id) pairs: blob tpc == hit tpc, slice_min <= floor(tick+off) < slice_max,
        wire_min <= hit local_wire < wire_max on the hit's plane (Wire-Cell half-open ranges).  The Wire-Cell
        blob wire index equals the LArSoft WireID.Wire (hit_table.local_wire) in both SBND TPCs (checked on
        the 0.6 test: 98-99% of the hit charge of either TPC lies in a blob; the wires-file order is reversed
        in TPC 1 and does NOT match)."""
        apa = np.asarray(tpc, np.int64)
        pl = np.asarray(plane, np.int64)
        wi = np.asarray(wire, np.int64)
        ht = np.floor(np.asarray(tick, np.float64) + tick_offset).astype(np.int64)
        bapa = wpid_apa(r["wpid"])
        nb = len(bapa)
        if nb == 0 or len(ht) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        width = int(np.max(r["slice_index_max"] - r["slice_index_min"]))
        # candidates: blobs of the same apa whose slice starts within (tick - width, tick]
        key_b = bapa * 100000 + r["slice_index_min"]
        order = np.argsort(key_b, kind="stable")
        kb = key_b[order]
        hits, blobs = [], []
        valid = apa >= 0
        for d in range(width):
            kh = apa * 100000 + (ht - d)
            lo = np.searchsorted(kb, kh, "left")
            hi = np.searchsorted(kb, kh, "right")
            cnt = np.where(valid, hi - lo, 0)
            if cnt.sum() == 0:
                continue
            h = np.repeat(np.arange(len(ht)), cnt)
            b = order[np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)]
            hits.append(h)
            blobs.append(b)
        if not hits:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        h, b = np.concatenate(hits), np.concatenate(blobs)
        inside = (ht[h] >= r["slice_index_min"][b]) & (ht[h] < r["slice_index_max"][b])
        wmin = np.stack([r["u_wire_index_min"], r["v_wire_index_min"], r["w_wire_index_min"]])
        wmax = np.stack([r["u_wire_index_max"], r["v_wire_index_max"], r["w_wire_index_max"]])
        p = pl[h]
        inside &= (wi[h] >= wmin[p, b]) & (wi[h] < wmax[p, b])
        h, b = h[inside], b[inside]
        o = np.lexsort((b, h))
        return h[o], b[o]


def wpid_apa(wpid):
    """WirePlaneId ident -> apa (bits: layer 0-2, face 3, apa 4+)."""
    return np.asarray(wpid, np.int64) >> 4


def wpid_face(wpid):
    return (np.asarray(wpid, np.int64) >> 3) & 1


BLOB_ATTRS = {
    "blob_id": "row index of the blob within the event (0..n-1); wc_hit_blob_table.blob_id refers to it",
    "cluster_id": "Wire-Cell step-2 cluster ident: the numbering of wc_hit_table.wc_cluster_id, wc_cluster_table and "
                  "the tracking-pr.root trees (T_cluster, T_proj_data)",
    "tpc": "Wire-Cell APA = TPC (0: x<0, 1: x>0); face is always 0 in SBND",
    "slice_index": "slice_index_min/max: Wire-Cell time slice range in TPC ticks (0.5 us), half-open "
                   "[min, max); slices are 4 ticks; tick 0 = LArSoft hit tick 0 (hit_table.local_time); "
                   "wc_tick_offset attribute is added to the hit tick before comparing",
    "wire_index": "u/v/w_wire_index_min/max: Wire-Cell wire index range per plane, half-open [min, max); "
                  "the index equals hit_table.local_wire (LArSoft WireID.Wire) of the same tpc and plane",
    "center": "center_x/y/z: blob centre (cm) in the Wire-Cell imaging frame: x NOT t0-corrected (drift time "
              "from the trigger), no space-charge correction (apparent positions).  center_x_t0cor = center_x + the "
              "blob's mean (x_t0cor - x) over its 3D points, i.e. x after the cluster t0 (flash) correction; equal "
              "to center_x for clusters without a matched flash",
    "charge": "solved blob charge (electrons)",
    "wc_nu_index": "T_tagger nu_index of the neutrino candidate the blob's cluster belongs to (as wc_hit_table), "
                   "else -1",
    "truth": "(sim) TensorSetLabeler: trackid = dominant G4 track (|TrackId|, the edep_table / particle_table g4_id "
             "convention) by smeared projected deposit charge inside the blob, -1 if none; truth_q = total accepted "
             "deposit weight (electrons), truth_q_dom = the dominant track's, trackid2/truth_q2 the second track",
}


def blob_rows(r, role):
    nb = len(r["charge"])
    nu = np.array([role.get(int(c), (-1, 0, 0))[0] for c in r["blob_cluster"]], np.int32) if nb else np.zeros(0, np.int32)
    return dict(
        blob_id=np.arange(nb, dtype=np.int32), cluster_id=r["blob_cluster"].astype(np.int32),
        tpc=wpid_apa(r["wpid"]).astype(np.int32),
        slice_index_min=r["slice_index_min"].astype(np.int32), slice_index_max=r["slice_index_max"].astype(np.int32),
        u_wire_index_min=r["u_wire_index_min"].astype(np.int32), u_wire_index_max=r["u_wire_index_max"].astype(np.int32),
        v_wire_index_min=r["v_wire_index_min"].astype(np.int32), v_wire_index_max=r["v_wire_index_max"].astype(np.int32),
        w_wire_index_min=r["w_wire_index_min"].astype(np.int32), w_wire_index_max=r["w_wire_index_max"].astype(np.int32),
        center_x=(r["center_x"] * MM_TO_CM).astype(np.float32), center_y=(r["center_y"] * MM_TO_CM).astype(np.float32),
        center_z=(r["center_z"] * MM_TO_CM).astype(np.float32),
        center_x_t0cor=((r["center_x"] + np.nan_to_num(r["dx_t0"])) * MM_TO_CM).astype(np.float32),
        charge=r["charge"].astype(np.float32), npoints=r["n3d"].astype(np.int32), wc_nu_index=nu,
        trackid=np.asarray(r["trackid"], np.int32), truth_q=np.asarray(r["truth_q"], np.float32),
        truth_q_dom=np.asarray(r["truth_q_dom"], np.float32), trackid2=np.asarray(r["trackid2"], np.int32),
        truth_q2=np.asarray(r["truth_q2"], np.float32))


CLUSTER_COLS = ("cluster_id", "n_blobs", "charge", "npoints", "length_cm", "in_scope", "flag_main_cluster",
                "flag_associated_cluster", "cluster_t0_ns", "flag_TGM", "flag_STM", "flag_FC", "lm_flag", "beam_flash",
                "flash_id", "matched_flash_gid", "wc_nu_index", "wc_main", "wc_in_enu")


def cluster_rows(r, tcl, role):
    """One row per step-2 cluster: the production T_cluster flags joined on cluster_id with the blob census."""
    ids = np.asarray(tcl["cluster_id"], np.int64) if tcl is not None else r["cluster_ident"]
    nb = np.zeros(len(ids), np.int32)
    q = np.zeros(len(ids), np.float32)
    pos = {int(c): i for i, c in enumerate(ids)}
    for c, ch in zip(r["blob_cluster"].tolist(), r["charge"].tolist()):
        i = pos.get(int(c))
        if i is not None:
            nb[i] += 1
            q[i] += ch
    g = (lambda k, d=-1, t=np.int32: np.asarray(tcl[k], t) if tcl is not None and k in tcl else np.full(len(ids), d, t))
    rl = [role.get(int(c), (-1, 0, 0)) for c in ids]
    return dict(cluster_id=ids.astype(np.int32), n_blobs=nb, charge=q, npoints=g("npoints"),
                length_cm=g("length_cm", np.nan, np.float32), in_scope=g("in_scope"),
                flag_main_cluster=g("is_main"), flag_associated_cluster=g("is_associated"),
                cluster_t0_ns=(g("cluster_t0_us", np.nan, np.float64) * 1000).astype(np.float32),
                flag_TGM=g("tgm"), flag_STM=g("stm"), flag_FC=g("fc"), lm_flag=g("lm"), beam_flash=g("beam_flash"),
                flash_id=g("flash_id"), matched_flash_gid=g("matched_flash_gid"),
                wc_nu_index=np.array([x[0] for x in rl], np.int32), wc_main=np.array([x[1] for x in rl], np.int32),
                wc_in_enu=np.array([x[2] for x in rl], np.int32))


TRACK_COLS = ("trackid", "pdg", "mother_trackid", "mother_pdg", "status", "start_x", "start_y", "start_z", "start_t",
              "start_px", "start_py", "start_pz", "start_E", "end_x", "end_y", "end_z", "end_t", "end_px", "end_py",
              "end_pz", "end_E", "nu_idx", "process", "ancestor_trackid", "prong_trackid", "q_readout", "e_readout",
              "q_in_blobs", "n_blobs", "q_blobs", "n_blobs_dom", "q_dom_blobs")


def track_rows(r, kin, status):
    """track_table rows from the labeler truthtrack table (+ MCParticle kinematics, + blob census)."""
    if r is None or "truthtrack" not in r:
        return None
    a, cols = r["truthtrack"]
    c = {k: a[:, i] for i, k in enumerate(cols)}
    tid = c["trackid"].astype(np.int64)
    # blob census: blobs whose dominant track is this one; their solved charge
    bt = np.asarray(r["trackid"], np.int64)
    u, inv = np.unique(bt, return_inverse=True)
    nbl = np.bincount(inv, minlength=len(u))
    qbl = np.bincount(inv, r["charge"], minlength=len(u))
    m = dict(zip(u.tolist(), zip(nbl.tolist(), qbl.tolist())))
    k = np.asarray(kin, np.float32).reshape(-1, 12) if len(tid) else np.zeros((0, 12), np.float32)
    f = lambda x: np.asarray(x, np.float32)
    i = lambda x: np.asarray(x, np.int32)
    return dict(
        trackid=i(tid), pdg=i(c["pdg"]), mother_trackid=i(c["mother_trackid"]), mother_pdg=i(c["mother_pdg"]),
        status=i(status), start_x=f(c["start_x"]), start_y=f(c["start_y"]), start_z=f(c["start_z"]),
        start_t=f(c["start_t"]), start_px=k[:, 0], start_py=k[:, 1], start_pz=k[:, 2], start_E=f(c["start_E"]),
        end_x=f(c["end_x"]), end_y=f(c["end_y"]), end_z=f(c["end_z"]), end_t=k[:, 4], end_px=k[:, 5],
        end_py=k[:, 6], end_pz=k[:, 7], end_E=k[:, 8], nu_idx=i(c["nu_idx"]), process=i(c["process"]),
        ancestor_trackid=i(c["ancestor_trackid"]), prong_trackid=i(c["prong_trackid"]),
        q_readout=f(c["q_readout"]), e_readout=f(c["e_readout"]), q_in_blobs=f(c["q_in_blobs"]),
        n_blobs=i([m.get(int(t), (0, 0.0))[0] for t in tid]), q_blobs=f([m.get(int(t), (0, 0.0))[1] for t in tid]),
        n_blobs_dom=i(c["n_blobs_dom"]), q_dom_blobs=f(c["q_dom_blobs"]))
