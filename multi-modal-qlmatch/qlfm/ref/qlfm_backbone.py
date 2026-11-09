"""qlfm (round-1 model) behind the common backbone interface."""
import torch
import torch.nn as nn

from qlfm.models.fm import FM, FMCfg
from qlfm.tokenize import TokCfg


class QLFMBackbone(nn.Module):
    def __init__(self, fm: FM):
        super().__init__()
        self.fm = fm
        self.d = fm.cfg.d
        self.hit_dim = fm.charge.hit_dim
        self.n_layers = len(fm.encoder.blocks) + 1

    @property
    def light_on(self):
        """Whether the PMT tokens are part of the input (False: charge only, e.g. for T5)."""
        return bool(self.fm.cfg.use_light and self.fm.light is not None)

    @light_on.setter
    def light_on(self, v):
        self.fm.cfg.use_light = bool(v) and self.fm.light is not None

    def layer_id(self, name):
        return int(name.split(".blocks.")[1].split(".")[0]) + 1 if ".blocks." in name else 0

    def forward(self, b):
        o = self.fm(b)
        z, evt = o["zc"], o["ct"].tok_evt
        if o["zl"] is not None:                           # charge tokens first, so hit2tok stays valid
            z, evt = torch.cat([z, o["zl"]]), torch.cat([evt, o["lt"].evt])
        return dict(z=z, evt=evt, h_hit=o["h_hit"], hit2tok=o["ct"].hit2tok)


def build(name, args, ckpt):
    if ckpt is None:
        tok = TokCfg(cell_wires=args.cell_wires, cell_ticks=args.cell_ticks, max_tokens=args.max_tokens or 8192)
        fm = FM(FMCfg(d=args.d, depth=args.depth, heads=args.heads, use_light=bool(args.light), tok=tok,
                      pos_mode=args.pos_mode, pos_freq=args.pos_freq))
    else:
        s = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = s["args"]
        # token cap as in pretraining unless overridden (audit #5: was 8192 downstream vs 4096 in pretraining)
        tok = TokCfg(cell_wires=a["cell_wires"], cell_ticks=a["cell_ticks"],
                     max_tokens=args.max_tokens or a.get("max_tokens", 4096))
        fm = FM(FMCfg(d=a["d"], depth=a["depth"], heads=a["heads"], use_light=bool(a["light"]), tok=tok,
                      pos_mode=a.get("pos_mode", "fourier"), pos_freq=a.get("pos_freq", 16)))
        fm.load_state_dict(s["fm"])
    return QLFMBackbone(fm)
