custom_imports = dict(
    imports=['mmdet_plugin'],
    allow_failed_imports=False
)

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    # '../_base_/datasets/coco_detection.py',
    '../_base_/datasets/waymo_det_2d.py',  # TODO: generate info file
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_tiny_patch4_window7.pth'  # TODO: load mask rcnn
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
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=dict(
            num_classes=4,  # TODO: num_classes
        )
    )
)

# dataset
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
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
