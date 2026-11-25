"""Compare two sbnd-h5maker files by values: every dataset of the reference must be equal in the new file
(NaN = NaN); datasets only in the new file are listed.  usage: cmp-h5.py REF.h5 NEW.h5"""
import sys, h5py, numpy as np
a, b = h5py.File(sys.argv[1], "r"), h5py.File(sys.argv[2], "r")
bad = n = 0; extra = []
def walk(g, h, p=""):
    global bad, n
    for k in g:
        q = p + "/" + k
        if isinstance(g[k], h5py.Group):
            walk(g[k], h[k], q); continue
        n += 1
        x, y = g[k][()], h[k][()] if k in h else None
        if y is None or x.shape != y.shape or not (np.array_equal(x, y) if x.dtype.kind in "SOiu" else np.array_equal(x, y, equal_nan=True)):
            bad += 1; print("DIFF", q, x.shape, None if y is None else y.shape)
    for k in h:
        if k not in g: extra.append(p + "/" + k)
walk(a, b)
print("datasets compared %d, differing %d; only in new: %s" % (n, bad, extra))
