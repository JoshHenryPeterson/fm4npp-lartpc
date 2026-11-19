"""PoLAr-MAE pretraining on MicroBooNE per-plane 2D hit clouds (round-2 B2).

Derived from pimm configs/polarmae/pretrain-polarmae-pilarnet.py (ViT-S, C-NMS ball groups, 60 % random
masking, Chamfer + per-point energy loss). Changes forced by the data / protocol:
  * data: MicroBooNEPlaneDataset (one plane of one event; every hit is a point, no grid sampling, as PoLAr-MAE)
  * normalisation: centre (520, 130, 10) cm, scale 600 cm; log-energy range 10..5000 ADC
  * augmentation: in-plane rotation only (clouds are 2D)
  * tokenizer: ball radius 4 cm (PILArNet: 1.5 cm) and up to 1024 groups per plane -- a MicroBooNE plane
    (10 m x 3.5 m, ~26 long cosmic tracks) would need ~2k groups at 1.5 cm
  * schedule length set by the equal-node-hour budget (~4 node-hours)
"""
import os as __os   # names starting with "__" are dropped by pimm's Config loader (no module left in cfg)

_base_ = [__os.environ.get("PIMM_ROOT", __os.path.join(__os.environ.get("QLFM_WORK", "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work"), "tools/pimm")) + "/configs/_base_/default_runtime.py"]

W = __os.environ.get("QLFM_WORK", "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work")
SCALE = 600.0
batch_size = 64
num_worker = 28
batch_size_val = 32
enable_amp = False             # published PoLAr-MAE recipe trains in fp32
amp_dtype = "bfloat16"
evaluate = False
clip_grad = 3.0
find_unused_parameters = False
seed = 0
use_wandb = False
warmup_ratio = 0.05
num_samples = 1_670_000            # x loop 3 = 5.0M plane-clouds: measured 0.16 s/iter x 64 -> ~3.5 h of training
epoch = 1

_shared_transformer_kwargs = dict(postnorm=True, add_pos_at_every_layer=True, drop_rate=0.0,
                                  attn_drop_rate=0.05, drop_path_rate=0.25)
model = dict(
    type="PoLAr-MAE",
    arch="vit_small",
    num_channels=4,
    voxel_size=5,
    masking_ratio=0.6,
    masking_type="rand",
    transformer_kwargs=_shared_transformer_kwargs,
    decoder_arch="vit_small",
    decoder_kwargs=dict(transformer_kwargs=dict(depth=4, **_shared_transformer_kwargs)),
    mae_prediction="full",
    loss_weights=dict(chamfer=1.0, energy=1.0),
    tokenizer_kwargs=dict(num_groups=2048, context_length=1024, group_max_points=32, group_radius=4.0 / SCALE,
                          group_upscale_points=256, overlap_factor=0.72),
)

optimizer = dict(type="AdamW", lr=0.0004, weight_decay=0.05)
scheduler = dict(type="OneCycleLR", max_lr=0.0004, pct_start=warmup_ratio, anneal_strategy="cos",
                 div_factor=11.6, final_div_factor=11.6)

_norm = dict(type="NormalizeCoord", center=[520.0, 130.0, 10.0], scale=SCALE)
_log = dict(type="LogTransform", min_val=10.0, max_val=5000.0, log=True, keys=("energy",))
_rot_z = dict(type="RandomRotate", angle=[-1, 1], axis="z", always_apply=True, center=[0, 0, 0])

data = dict(
    num_classes=1,
    names=["hit"],
    train=dict(type="MicroBooNEPlaneDataset", corpus=f"{W}/cache/bnb,{W}/cache/nue",
               splits=f"{W}/cache/splits_all.npz", subset="pre_f0.90",
               transform=[_log, _norm, _rot_z, dict(type="ToTensor"),
                          dict(type="Collect", keys=("coord", "energy"), feat_keys=("coord", "energy"))],
               max_len=num_samples, seed=0, loop=3),
)

hooks = [
    dict(type="ParameterCounter", show_details=False, show_gradients=False, sort_by_params=True, min_params=1),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaverIteration", save_freq=200),
    dict(type="WeightDecayExclusion", exclude_bias_from_wd=True, exclude_norm_from_wd=True,
         exclude_gamma_from_wd=True, exclude_token_from_wd=True, exclude_ndim_1_from_wd=True),
    dict(type="CheckpointLoader"),
]
train = dict(type="DefaultTrainer")
