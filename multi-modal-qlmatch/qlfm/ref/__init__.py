"""Backbone interface shared by qlfm and the round-2 reference methods.

Every backbone is an nn.Module with
  d         token feature width
  hit_dim   width of the per-hit embedding it provides (0 if none)
  n_layers  number of layer-decay groups (for fine-tuning)
  layer_id(param_name) -> int in [0, n_layers]  (0 = embeddings)
  light_on  (optional) True if PMT light is part of the backbone input; settable (False = charge only)
  forward(batch) -> dict(z=[N,d] token features, evt=[N] event index,
                         h_hit=[Nh,hit_dim] or None, hit2tok=[Nh] index into z or -1)
Reference backbones are imported lazily because some live in other Python environments (pimm).
"""


def build_backbone(name, args, ckpt=None):
    if name == "qlfm":
        from qlfm.ref.qlfm_backbone import build
    elif name == "fm4npp":
        from qlfm.ref.fm4npp import build
    elif name == "fm4npp_l":
        from qlfm.ref.fm4npp_light import build
    elif name in ("panda", "panda_l", "polarmae"):
        from qlfm.ref.pimm_backbone import build
    else:
        raise ValueError(name)
    return build(name, args, ckpt)
