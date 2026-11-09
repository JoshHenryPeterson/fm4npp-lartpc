"""PTv3 + Sonata + light (paper Sec. 4): Panda's recipe (panda_uboone.py) on event-level clouds with the PMTs.

Changes from panda_uboone.py:
  * data: MicroBooNEEventLightDataset -- one sample = one event: hits of all three planes (slabs z = 0/10/20 cm)
    + 32 PMT points (slab z = 30 cm, at (z_pmt, y_pmt + 128 cm)); per point energy (hit integral | PMT total PE),
    17 light features (zeros for hits) and a PMT flag -> in_channels = 3 + 1 + 17 + 1 = 22
  * PMT points go through the same views, crops, masks and distillation as hits (early fusion in a point cloud)
  * batch 24 events (6/GPU; ~3x the points of a single-plane cloud); length set by the 20 nh budget
"""
_base_ = ["/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/tools/pimm/configs/_base_/default_runtime.py"]

W = "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work"     # pimm configs cannot import modules
batch_size = 24
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
num_samples = 3_280_000          # events: smoke test 0.50 s per 24-event batch -> 19 h of training (20 nh budget)

model = dict(
    type="Sonata-v1m1",
    backbone=dict(
        type="PT-v3m2", in_channels=22, order=("hilbert", "hilbert-trans", "z", "z-trans"), stride=(2, 2, 2, 2),
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
    dict(type="Update", keys_dict={"index_valid_keys": ["coord", "energy", "light", "ispmt"]}),
    dict(type="NormalizeCoord", center=[520.0, 130.0, 10.0], scale=SCALE),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", sum_keys=("energy",)),
    dict(type="LogTransform", min_val=10.0, max_val=5000.0, log=True, keys=("energy",)),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "energy", "light", "ispmt"),
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
        keys=("global_origin_coord", "global_coord", "global_energy", "global_light", "global_ispmt", "global_offset",
              "local_origin_coord", "local_coord", "local_energy", "local_light", "local_ispmt", "local_offset",
              "grid_size", "name"),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_energy", "global_light", "global_ispmt"),
        local_feat_keys=("local_coord", "local_energy", "local_light", "local_ispmt"),
    ),
]

data = dict(
    num_classes=1,
    names=["hit"],
    train=dict(type="MicroBooNEEventLightDataset", corpus=f"{W}/cache/bnb,{W}/cache/nue",
               splits=f"{W}/cache/splits_all.npz", subset="pre_f0.90", transform=transform,
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
    # snapshots every 5 % of the run, kept (convergence probes at 10/25/50 %)
    dict(type="CheckpointSaverIteration", save_freq=num_samples // batch_size // 20, save_iter_checkpoints=True),
    dict(type="PrototypeUsageLogger", log_frequency=50, prefix="prototypes"),
    dict(type="FeatureStdMonitor", log_frequency=50, prefix="feature_std", monitor_student=True, monitor_teacher=True,
         track_channels=False),
]
train = dict(type="DefaultTrainer")
