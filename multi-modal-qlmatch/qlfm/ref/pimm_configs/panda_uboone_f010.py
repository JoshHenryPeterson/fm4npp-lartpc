"""Panda (Sonata-v1m1 + PT-v3m2) pretraining on MicroBooNE per-plane 2D hit clouds (round-2 B1).
Round-4 variant: same as panda_uboone.py, pretrained on the 10 % subset (cycled to the same number of samples).

Derived from pimm configs/panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py. Changes (all forced by
the data or the compute-matched protocol):
  * data: MicroBooNEPlaneDataset (one plane of one event = one cloud; points (wire cm, drift cm, plane*10 cm),
    energy = hit integral), our pre_f0.90 split
  * normalisation: centre (520, 130, 10) cm, scale 600 cm; grid 0.3 cm (0.0005); log-energy range 10..5000 ADC
  * augmentation: in-plane only (rotation about z, flips in x/y) -- the clouds are 2D
  * mask patch sizes rescaled to the same physical size as PILArNet (2 -> 15 cm)
  * backbone size-matched to qlfm-21M: depths (2,2,2,4,2), channels (32,64,128,256,320)
  * batch 64 (16/GPU x 4), length set so the run fits the equal-node-hour budget (~4 node-hours)
"""
import os as __os   # names starting with "__" are dropped by pimm's Config loader (no module left in cfg)

_base_ = [__os.environ.get("PIMM_ROOT", __os.path.join(__os.environ.get("QLFM_WORK", "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work"), "tools/pimm")) + "/configs/_base_/default_runtime.py"]

W = __os.environ.get("QLFM_WORK", "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work")
batch_size = 64
num_worker = 28
batch_size_val = 32
mix_prob = 0
clip_grad = 3.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
find_unused_parameters = False
matmul_precision = "high"
seed = 0
use_wandb = False

SCALE = 600.0
grid_size = 0.3 / SCALE
warmup_ratio = 0.05
num_samples = 1_840_000          # plane-clouds: measured 0.434 s/iter x 64 -> ~3.5 h of training (4 x 1-h jobs)

model = dict(
    type="Sonata-v1m1",
    backbone=dict(
        type="PT-v3m2", in_channels=4, order=("hilbert", "hilbert-trans", "z", "z-trans"), stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 4, 2), enc_channels=(32, 64, 128, 256, 320), enc_num_head=(2, 4, 8, 16, 20),
        enc_patch_size=(256, 256, 256, 256, 256), mlp_ratio=4, qkv_bias=True, qk_scale=None, layer_scale=1e-5,
        attn_drop=0.0, proj_drop=0.0, drop_path=0.3, shuffle_orders=True, pre_norm=True, enable_rpe=False,
        enable_flash=True, upcast_attention=False, upcast_softmax=False, traceable=True, enc_mode=True,
        mask_token=True, cpe_first_layer_only=False, cpe_shared_weight=False,
    ),
    teacher_custom=dict(attn_drop=0.0, proj_drop=0.0, drop_path=0.0),
    head_in_channels=320 + 256 + 128,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=6,
    mask_size_start=2.0 / SCALE,
    mask_size_base=15.0 / SCALE,
    mask_size_warmup_ratio=warmup_ratio,
    mask_ratio_start=0.5,
    mask_ratio_base=0.9,
    mask_ratio_warmup_ratio=warmup_ratio,
    mask_jitter=grid_size / 2,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=warmup_ratio,
    student_temp=0.10,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1.0,
    match_max_r=2 * grid_size,
    up_cast_level=2,
)

epoch = 1
base_lr = 0.0026 * (batch_size / 48) ** 0.5
lr_decay = 0.9
base_wd = 0.04
final_wd = 0.2
dec_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(keyword=f"enc{e}.block{b}.", lr=base_lr * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1))
    for e in range(len(dec_depths)) for b in range(dec_depths[e])
]
del dec_depths
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(type="OneCycleLR", max_lr=[base_lr] + [g["lr"] for g in param_dicts], pct_start=warmup_ratio,
                 anneal_strategy="cos", div_factor=10.0, final_div_factor=1000.0)

_inplane_rot = dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8)
_view_tf = [
    dict(type="CenterShift", apply_z=False, axes=("x", "y")),
    _inplane_rot,
    dict(type="RandomFlip", p=0.5, axes=("x", "y")),
    dict(type="RandomJitter", sigma=grid_size / 4, clip=grid_size, keys=("coord",)),
]
transform = [
    dict(type="NormalizeCoord", center=[520.0, 130.0, 10.0], scale=SCALE),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", sum_keys=("energy",)),
    dict(type="LogTransform", min_val=10.0, max_val=5000.0, log=True, keys=("energy",)),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "energy"),
        global_view_num=2, global_view_scale=(0.4, 1.0),
        local_view_num=6, local_view_scale=(0.1, 0.4),
        global_shared_transform=[dict(type="MultiplicativeRandomJitter", sigma=0.05, clip=0.05, keys=("energy"), p=0.8)],
        global_transform=_view_tf,
        local_transform=_view_tf,
        max_size=30000,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=("global_origin_coord", "global_coord", "global_energy", "global_offset",
              "local_origin_coord", "local_coord", "local_energy", "local_offset", "grid_size", "name"),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_energy"),
        local_feat_keys=("local_coord", "local_energy"),
    ),
]

data = dict(
    num_classes=1,
    names=["hit"],
    train=dict(type="MicroBooNEPlaneDataset", corpus=f"{W}/cache/bnb,{W}/cache/nue",
               splits=f"{W}/cache/splits_all.npz", subset="pre_f0.10", transform=transform,
               max_len=num_samples, seed=0),
)

hooks = [   # as in the Panda config, minus wandb and the PILArNet-label PretrainEvaluator
    dict(type="ParameterCounter", show_details=False, show_gradients=False, sort_by_params=True, min_params=1),
    dict(type="WeightDecayExclusion", exclude_bias_from_wd=True, exclude_norm_from_wd=True, exclude_gamma_from_wd=True,
         exclude_token_from_wd=True, exclude_ndim_1_from_wd=True),
    dict(type="CheckpointLoader"),
    dict(type="DtypeOverrider", class_patterns=["LayerNorm"], dtype="float32", override_parameters=True,
         methods_to_override=["forward"]),
    dict(type="ModelHook"),
    dict(type="WeightDecayScheduler", base_value=base_wd, final_value=final_wd, warmup_ratio=1.0),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaverIteration", save_freq=200),
    dict(type="PrototypeUsageLogger", log_frequency=50, prefix="prototypes"),
    dict(type="FeatureStdMonitor", log_frequency=50, prefix="feature_std", monitor_student=True, monitor_teacher=True,
         track_channels=False),
]
train = dict(type="DefaultTrainer")
