# High-resolution fine-tune config for Co-DETR + InternImage-L.
#
# This config is a small train-time high-res validation branch from the current
# best fresh epoch7 checkpoint.  It keeps the model/head unchanged and only
# adjusts the training/eval resize policy.  Submission remains gated by strict
# final-TXT validation + hard-val outside this config.

_base_ = ['./codetr_internimage_l_mm_config.py']

# Train-time high-resolution fine-tune.  Keep width cap at 1333 to avoid changing
# aspect-ratio behavior too aggressively; raise the short side above the prior
# 480-800 multiscale policy to test the planned Stage-3 high-res path.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [dict(
                type='Resize',
                img_scale=[(1333, 832), (1333, 864), (1333, 896)],
                multiscale_mode='value',
                keep_ratio=True,
            )],
            [
                dict(
                    type='Resize',
                    img_scale=[(1333, 768), (1333, 800), (1333, 832)],
                    multiscale_mode='value',
                    keep_ratio=True,
                ),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type='Resize',
                    img_scale=[(1333, 832), (1333, 864), (1333, 896)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 896),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            ),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

optimizer = dict(
    type='AdamW',
    lr=1e-6,
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[3, 4])
runner = dict(type='EpochBasedRunner', max_epochs=5)
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
checkpoint_config = dict(interval=1, max_keep_ckpts=6)
log_config = dict(interval=50)

fp16 = None
find_unused_parameters = True
