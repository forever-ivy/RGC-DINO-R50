# Co-DETR + InternImage-L Stage-1 low-resolution sanity config.
#
# MMDetection 2.x / Sense-X Co-DETR compatible config.  This is the first
# InternImage-L bridge after the R50 sanity: RGB-only, 12 classes, low resource
# defaults, and no foreground training.  RGC tri-modal fusion is added after the
# detector/backbone path and weight loading are proven stable.

_base_ = [
    '../external/Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py',
]

num_classes = 12
dataset_type = 'CocoDataset'
data_root = 'outputs/codetr_coco/fold0/'
classes = (
    'person',
    'boat',
    'animal',
    'seat',
    'sign',
    'bicycle',
    'car',
    'ball',
    'light',
    'garbage can',
    'uav',
    'tricycle',
)

internimage_pretrained = '/data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth'
# Detection-side partial initializer.  The current file is a Co-DINO Swin-L
# Objects365 checkpoint, so backbone/neck/class-head mismatches are expected and
# must be audited instead of silently accepted.
load_from = '/data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        # CUDA DCNv3 op is compiled under external/Co-DETR/ops_dcnv3.
        # Fall back to 'DCNv3_pytorch' only for CPU-only metadata/debug checks.
        core_op='DCNv3',
        channels=160,
        depths=[5, 5, 22, 5],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.0,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=internimage_pretrained),
    ),
    neck=dict(in_channels=[160, 320, 640, 1280]),
    query_head=dict(num_classes=num_classes),
    roi_head=[dict(
        type='CoStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64],
            finest_scale=56),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=12.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=120.0)))],
    bbox_head=[dict(
        type='CoATSSHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=1,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=12.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=24.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0))],
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        classes=classes,
        filter_empty_gt=False,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        classes=classes,
    ),
)

optimizer = dict(
    type='AdamW',
    lr=2e-5,
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=1)
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
log_config = dict(interval=50)

# Keep disabled until DCNv3 and MMCV deformable attention half kernels are proven.
fp16 = None
find_unused_parameters = True
