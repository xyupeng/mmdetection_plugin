_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    # '../_base_/datasets/coco_detection.py',
    '../_base_/datasets/waymo_det_2d.py',  # TODO: generate info file
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmdet_plugin'],
    allow_failed_imports=False
)

ckpt = './checkpoints/debug/mask_rcnn_swin-t_coco.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=ckpt),
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=dict(
            num_classes=4,  # TODO: num_classes
        )
    )
)

# dataset
data_root = './data/waymo/ply_format'
info_format = 'ply'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        data_root=data_root,
        info_format=info_format,
        ann_file='waymo_det2d_infos_training.pkl',
        load_interval=1,
    ),
    val=dict(
        data_root=data_root,
        info_format=info_format,
        ann_file='waymo_det2d_infos_validation.pkl',
    ),
    test=dict(
        data_root=data_root,
        info_format=info_format,
        ann_file='waymo_det2d_infos_validation.pkl',
    ),
)
evaluation = dict(interval=2, metric='bbox')

# schedule
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],
)

# runtime
log_config = dict(interval=100)
checkpoint_config = dict(interval=4)
runner = dict(type='EpochBasedRunner', max_epochs=12)
