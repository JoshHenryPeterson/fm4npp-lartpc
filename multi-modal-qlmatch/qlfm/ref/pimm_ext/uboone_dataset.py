"""pimm reader for MicroBooNE open-data wire hits (our column-wise cache): one sample = one plane of one event.

Points: (wire * 0.3 cm, drift x cm, plane * 10 cm); energy: hit integral (ADC*tick). The plane offset along z
encodes plane identity (U/V/Y) without mixing planes; per-plane clouds are 2D, so only in-plane augmentations
are used in the configs. Splits/subsets come from our splits file (same as all other methods).
"""
import numpy as np
from torch.utils.data import Dataset

from pimm.datasets.builder import DATASETS
from pimm.datasets.transform import Compose

from qlfm import geometry as G
from qlfm.data.dataset import Corpus

PLANE_DZ_CM = 10.0


def plane_cloud(plane, wire, time, integral, p):
    m = plane == p
    coord = np.stack([wire[m] * G.WIRE_PITCH_CM, (time[m] - G.TRIG_TICK) * G.TICK_TO_CM,
                      np.full(m.sum(), p * PLANE_DZ_CM)], 1).astype(np.float32)
    energy = np.clip(integral[m], 1.0, None).astype(np.float32)[:, None]
    return coord, energy, np.nonzero(m)[0]


@DATASETS.register_module()
class MicroBooNEPlaneDataset(Dataset):
    def __init__(self, corpus, splits, subset="pre_f0.90", transform=None, max_len=None, min_points=64,
                 seed=0, test_mode=False, loop=1):
        self.c = Corpus(corpus)
        idx = np.load(splits)[subset]
        rng = np.random.default_rng(seed)
        samples = np.stack([np.repeat(idx, 3), np.tile(np.arange(3), len(idx))], 1)
        samples = samples[rng.permutation(len(samples))]
        if max_len is not None and max_len < len(samples):
            samples = samples[:max_len]
        self.samples = samples
        # a small subset (pretrain-size scan) is cycled up to max_len samples, so every run has the same length
        self.n = max_len if max_len is not None else len(samples)
        self.transform = Compose(transform)
        self.min_points = min_points
        self.loop = loop

    def get_data(self, i):
        g, p = self.samples[i % len(self.samples)]
        s, j = self.c.locate(g)
        sh = self.c.shards[s]
        h0, hn = int(sh.ev("hit_off")[j]), int(sh.ev("hit_cnt")[j])
        plane = np.asarray(sh.hit("plane")[h0:h0 + hn])
        coord, energy, _ = plane_cloud(plane, np.asarray(sh.hit("wire")[h0:h0 + hn]).astype(np.float32),
                                       np.asarray(sh.hit("time")[h0:h0 + hn]).astype(np.float32) / 8.0,
                                       np.asarray(sh.hit("integral")[h0:h0 + hn]).astype(np.float32), p)
        if len(coord) < self.min_points:                      # rare near-empty planes: take the next sample
            return self.get_data((i + 1) % len(self.samples))
        return dict(coord=coord, energy=energy, name=f"{g}:{p}")

    def __getitem__(self, i):
        return self.transform(self.get_data(i))

    def __len__(self):
        return self.n * self.loop


# ---------------------------------------------------------------- PTv3 + light (paper Sec. 4: "PTv3 + Sonata + L")
PMT_DZ_CM = 30.0          # PMT slab beyond the three wire-plane slabs (0, 10, 20 cm)
PMT_DY_CM = 128.0         # PMT y (-57..56 cm) shifted into the drift-coordinate range (0..256 cm)
N_LIGHT = 17


def pmt_coords():
    """32 PMT points: (z_pmt [cm] ~ Y-plane wire coordinate, y_pmt + 128 cm, 30 cm)."""
    yz = np.asarray(G.PMT_YZ, dtype=np.float32)
    return np.stack([yz[:, 1], yz[:, 0] + PMT_DY_CM, np.full(len(yz), PMT_DZ_CM, np.float32)], 1)


def light_features(ch, t, pe, edges):
    """numpy twin of qlfm.tokenize.tokenize_light for one event: [32, 17] = log1p PE per time bin, log1p total."""
    edges = np.asarray(edges, dtype=np.float32)
    nb = len(edges) - 1
    b = np.clip(np.searchsorted(edges, t, side="left") - 1, 0, nb - 1)
    inwin = (t >= edges[0]) & (t < edges[-1])
    w = np.where(inwin, np.clip(pe, 0, None), 0.0)
    hist = np.bincount(ch.astype(np.int64) * nb + b, weights=w, minlength=G.N_PMT * nb).reshape(G.N_PMT, nb)
    tot = hist.sum(1, keepdims=True)
    return np.concatenate([np.log1p(hist), np.log1p(tot)], 1).astype(np.float32), tot[:, 0].astype(np.float32)


@DATASETS.register_module()
class MicroBooNEEventLightDataset(Dataset):
    """One sample = one event: the hits of all three planes (slabs at z = 0, 10, 20 cm) plus 32 PMT points (slab at
    30 cm). Per point: coord, energy (hit integral | PMT total PE), light [17] (zeros for hits) and ispmt [1]."""

    def __init__(self, corpus, splits, subset="pre_f0.90", transform=None, max_len=None, min_points=64,
                 seed=0, test_mode=False, loop=1):
        from qlfm.tokenize import TokCfg
        self.c = Corpus(corpus)
        idx = np.load(splits)[subset]
        self.samples = idx[np.random.default_rng(seed).permutation(len(idx))]
        self.n = max_len if max_len is not None else len(self.samples)
        self.transform = Compose(transform)
        self.min_points = min_points
        self.loop = loop
        self.edges = TokCfg().light_edges
        self.pmt = pmt_coords()

    def get_data(self, i):
        g = self.samples[i % len(self.samples)]
        s, j = self.c.locate(g)
        sh = self.c.shards[s]
        h0, hn = int(sh.ev("hit_off")[j]), int(sh.ev("hit_cnt")[j])
        if hn < self.min_points:
            return self.get_data((i + 1) % len(self.samples))
        plane = np.asarray(sh.hit("plane")[h0:h0 + hn])
        wire = np.asarray(sh.hit("wire")[h0:h0 + hn]).astype(np.float32)
        time = np.asarray(sh.hit("time")[h0:h0 + hn]).astype(np.float32) / 8.0
        q = np.asarray(sh.hit("integral")[h0:h0 + hn]).astype(np.float32)
        hc = np.stack([wire * G.WIRE_PITCH_CM, (time - G.TRIG_TICK) * G.TICK_TO_CM, plane * PLANE_DZ_CM], 1)
        o0, on = int(sh.ev("oph_off")[j]), int(sh.ev("oph_cnt")[j])
        lf, tot = light_features(np.asarray(sh.oph("ch")[o0:o0 + on]), np.asarray(sh.oph("t")[o0:o0 + on]),
                                 np.asarray(sh.oph("pe")[o0:o0 + on]), self.edges)
        coord = np.concatenate([hc, self.pmt]).astype(np.float32)
        energy = np.concatenate([np.clip(q, 1.0, None), np.clip(tot, 1.0, None)])[:, None].astype(np.float32)
        light = np.concatenate([np.zeros((hn, N_LIGHT), np.float32), lf])
        ispmt = np.concatenate([np.zeros(hn, np.float32), np.ones(G.N_PMT, np.float32)])[:, None]
        return dict(coord=coord, energy=energy, light=light, ispmt=ispmt, name=str(g))

    def __getitem__(self, i):
        return self.transform(self.get_data(i))

    def __len__(self):
        return self.n * self.loop
