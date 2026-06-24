# Eval-only Co-DETR + InternImage-L config with effective short-side 832.
# For 16:9 AIC images, (1333, 832) is capped by width and becomes ~1333x750.
# Use width ~= 832 * 16 / 9 = 1479 so the short side actually reaches 832.

_base_ = ['./codetr_internimage_l_mm_config.py']

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1479, 832),
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
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

fp16 = None
find_unused_parameters = True
