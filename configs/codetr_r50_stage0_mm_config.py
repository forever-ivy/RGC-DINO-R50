# Co-DETR-R50 Stage-0 sanity config for the external Co-DETR repository.
#
# Purpose: verify the external Co-DETR code, exported COCO fold, 12-class query
# head, validation mAP, and LSF launch path before spending time on
# InternImage-L.  This is MMDetection 2.x style because Sense-X/Co-DETR is based
# on that stack.  It is RGB-only sanity; RGC tri-modal fusion is added after the
# detector path is proven stable.

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

# Override the query and auxiliary heads to the competition's 12 classes.  MMCV
# list merging replaces whole list entries, so roi_head/bbox_head are copied from
# the Co-DETR base config with only num_classes adjusted.
model = dict(
    backbone=dict(init_cfg=None),
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
    workers_per_gpu=2,
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
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=1)
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
log_config = dict(interval=20)

# Disable fp16 for the first sanity path because the available MMCV deformable
# attention op in this environment does not implement the Half forward kernel.
# Revisit AMP after the full Co-DETR stack is stable.
fp16 = None
find_unused_parameters = True
