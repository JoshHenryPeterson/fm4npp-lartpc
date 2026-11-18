"""Memory-mapped dataset over converted shards + packed collate."""
import glob
import json
import os

import numpy as np
import torch

HIT_FEAT = ("plane", "wire", "time", "integral", "rms")
HIT_TRUTH = ("nu_edep", "cat", "pnd_nu")
OPH = ("ch", "t", "pe")


class Shard:
    def __init__(self, d):
        self.dir = d
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.n = self.meta["n_events"]
        self._ev, self._hit, self._oph = {}, {}, {}
        self.has_tpc = os.path.exists(os.path.join(d, "hit_tpc.npy"))       # SBND (2 TPCs)

    def ev(self, k):
        if k not in self._ev:
            self._ev[k] = np.load(os.path.join(self.dir, f"ev_{k}.npy"), mmap_mode="r")
        return self._ev[k]

    def hit(self, k):
        if k not in self._hit:
            self._hit[k] = np.load(os.path.join(self.dir, f"hit_{k}.npy"), mmap_mode="r")
        return self._hit[k]

    def oph(self, k):
        if k not in self._oph:
            self._oph[k] = np.load(os.path.join(self.dir, f"oph_{k}.npy"), mmap_mode="r")
        return self._oph[k]


class Corpus:
    """Shards of one or more samples; global event index = (shard, local index).

    `root` may be a comma-separated list of sample directories (e.g. ".../bnb,.../nue");
    samples are concatenated in that order and `sample_id` records the origin of each event."""

    def __init__(self, root, pattern="*"):
        self.roots = root.split(",")
        self.shards, sid = [], []
        for k, r in enumerate(self.roots):
            dirs = sorted(d for d in glob.glob(os.path.join(r, pattern)) if os.path.exists(os.path.join(d, "meta.json")))
            if not dirs:
                raise FileNotFoundError(f"no converted shards under {r}/{pattern}")
            self.shards += [Shard(d) for d in dirs]
            sid += [k] * len(dirs)
        self.shard_sample = np.array(sid)
        self.sizes = np.array([s.n for s in self.shards])
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.n = int(self.sizes.sum())

    def locate(self, gidx):
        s = np.searchsorted(self.starts, gidx, side="right") - 1
        return s, gidx - self.starts[s]

    def ev_array(self, k):
        return np.concatenate([np.asarray(s.ev(k)) for s in self.shards])

    def sample_id(self):
        return np.repeat(self.shard_sample, self.sizes)


class EventDataset(torch.utils.data.Dataset):
    """Returns raw hits/ophits for one event (+ truth columns if `truth`)."""

    def __init__(self, corpus: Corpus, indices, truth=False, ev_keys=()):
        self.c = corpus
        self.idx = np.asarray(indices, dtype=np.int64)
        self.truth = truth
        self.ev_keys = ev_keys

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        g = int(self.idx[i])
        s, j = self.c.locate(g)
        sh = self.c.shards[s]
        h0, hn = int(sh.ev("hit_off")[j]), int(sh.ev("hit_cnt")[j])
        o0, on = int(sh.ev("oph_off")[j]), int(sh.ev("oph_cnt")[j])
        out = {"gidx": g}
        cols = HIT_FEAT + (("tpc",) if sh.has_tpc else ()) + (HIT_TRUTH if self.truth else ())
        for k in cols:
            out["hit_" + k] = np.asarray(sh.hit(k)[h0:h0 + hn])
        for k in OPH:
            out["oph_" + k] = np.asarray(sh.oph(k)[o0:o0 + on])
        for k in self.ev_keys:
            out["ev_" + k] = np.asarray(sh.ev(k)[j])
        return out


def collate(items):
    """Pack variable-length events into flat tensors with event indices."""
    n = len(items)
    b = {"n_events": n, "gidx": torch.tensor([it["gidx"] for it in items])}
    nh = [len(it["hit_plane"]) for it in items]
    no = [len(it["oph_ch"]) for it in items]
    b["hit_evt"] = torch.repeat_interleave(torch.arange(n), torch.tensor(nh))
    b["oph_evt"] = torch.repeat_interleave(torch.arange(n), torch.tensor(no))
    cat = lambda k: np.concatenate([it[k] for it in items])
    b["hit_plane"] = torch.from_numpy(cat("hit_plane").astype(np.int64))
    b["hit_wire"] = torch.from_numpy(cat("hit_wire").astype(np.int64))
    b["hit_time"] = torch.from_numpy(cat("hit_time").astype(np.float32) / 8.0)
    b["hit_q"] = torch.from_numpy(cat("hit_integral").astype(np.float32))
    b["hit_rms"] = torch.from_numpy(cat("hit_rms").astype(np.float32))
    if "hit_tpc" in items[0]:
        b["hit_tpc"] = torch.from_numpy(cat("hit_tpc").astype(np.int64))
    b["oph_ch"] = torch.from_numpy(cat("oph_ch").astype(np.int64))
    b["oph_t"] = torch.from_numpy(cat("oph_t").astype(np.float32))
    b["oph_pe"] = torch.from_numpy(cat("oph_pe").astype(np.float32))
    if "hit_nu_edep" in items[0]:
        b["hit_nu_edep"] = torch.from_numpy(cat("hit_nu_edep").astype(np.float32))
        b["hit_cat"] = torch.from_numpy(cat("hit_cat").astype(np.int64))
        b["hit_pnd_nu"] = torch.from_numpy(cat("hit_pnd_nu").astype(np.float32))
    for k in items[0]:
        if k.startswith("ev_"):
            b[k] = torch.from_numpy(np.stack([np.asarray(it[k]) for it in items]))
    return b


def to_device(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
