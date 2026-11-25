"""sbnd-h5maker 0.6 (fm4npp #16): does a blob dump split clusters as the production step 2 did?  Per event, every
cluster ident of the dump with its point count (sum of its blobs' 3D points) against tracking-pr.root T_cluster.
usage: cmp-dump.py <blobs.tar.gz> <unit pr/all dir (pr_evt<run>_<subrun>_<event>/tracking-pr.root)>"""
import sys, os, collections, numpy as np, uproot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from wc_blobs import tensor_sets
tot = collections.Counter()
for meta, A, _ in tensor_sets(sys.argv[1]):
    r, s, e = meta["runNo"], meta["subRunNo"], meta["eventNo"]
    p = os.path.join(sys.argv[2], "pr_evt%d_%d_%d" % (r, s, e), "tracking-pr.root")
    if not os.path.exists(p): tot["no_prod"] += 1; continue
    live = [k for k in A if k and k.endswith("/live") and k.startswith("pointtrees/")]
    L = live[0] + "/"
    par = A[L[:-1]]; depth = np.zeros(len(par), int)
    for i in range(1, len(par)): depth[i] = depth[par[i]] + 1
    cl_nodes = np.where(depth == 1)[0]; bl_nodes = np.where(depth == 2)[0]
    N = L + "pointclouds/namedpcs/"
    ids = A[N + "cluster_scalar/arrays/ident"] if N + "cluster_scalar/arrays/ident" in A else None
    npb = A[N + "scalar/arrays/npoints"]
    # blob rows are in node order; parent cluster node -> rank among cluster nodes
    rank = {n: i for i, n in enumerate(cl_nodes)}
    bcl = np.array([rank[par[b]] for b in bl_nodes])
    f = uproot.open(p); T = f["T_cluster"].arrays(["cluster_id", "npoints"], library="np")
    prod = dict(zip(T["cluster_id"].tolist(), T["npoints"].tolist()))
    if ids is None or len(ids) != len(cl_nodes):
        tot["ident_mismatch"] += 1; print("ident array", None if ids is None else len(ids), "clusters", len(cl_nodes)); continue
    mine = collections.Counter()
    for c, n in zip(bcl, npb): mine[int(ids[c])] += int(n)
    tot["events"] += 1
    tot["clusters_dump"] += len(mine); tot["clusters_prod"] += len(prod)
    same = sum(1 for k, v in mine.items() if prod.get(k) == v)
    tot["clusters_same_id_npoints"] += same
    if same != len(prod) or len(mine) != len(prod):
        tot["events_differ"] += 1
        if tot["events_differ"] <= 3: print("differ", r, s, e, len(mine), len(prod), same)
print(dict(tot))
