#!/usr/bin/env python3
"""Dump (run, subrun, event) of every event in a list of art/ROOT files, without art.

Usage: python artroot_event_ids.py FILE_LIST OUT.txt [NPROC=4]

FILE_LIST has one local path (e.g. /pnfs/...) per line. OUT.txt gets one line per
event, "run subrun event filename", and "#ERR filename reason" for failures.

uproot mis-reads the nested, versioned art::EventID inside the unsplit EventAuxiliary
branch, so the IDs are decoded from the raw basket bytes: each EventID is preceded by
the RunID header 40 00 00 06 00 0a, followed by run, subrun, event as big-endian
uint32. The decoded count is checked against the branch entry count.

Keep NPROC small on shared GPVMs: a 16-worker run over the 24k public artroot files
was running when uboonebuild03 ran out of memory (2026-11-11).
"""
import re
import struct
import sys
from concurrent.futures import ProcessPoolExecutor

import uproot

PAT = re.compile(rb"\x40\x00\x00\x06\x00\x0a(.{4})(.{4})(.{4})", re.S)


def event_ids(path):
    try:
        br = uproot.open(path)["Events"]["EventAuxiliary"]
        out = []
        for i in range(br.num_baskets):
            data = br.basket(i).data.tobytes()
            out += [tuple(struct.unpack(">I", g)[0] for g in m.groups()) for m in PAT.finditer(data)]
        if len(out) != br.num_entries:
            return path, None, f"decoded {len(out)} ids for {br.num_entries} entries"
        return path, out, ""
    except Exception as e:  # report and keep going
        return path, None, repr(e)[:200]


def main():
    lst, outp = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    paths = [l.strip() for l in open(lst) if l.strip()]
    with ProcessPoolExecutor(nproc) as ex, open(outp, "w") as fo:
        for path, ids, err in ex.map(event_ids, paths, chunksize=4):
            name = path.rsplit("/", 1)[-1]
            if ids is None:
                fo.write(f"#ERR {name} {err}\n")
                continue
            fo.writelines(f"{r} {s} {e} {name}\n" for r, s, e in ids)


if __name__ == "__main__":
    main()
