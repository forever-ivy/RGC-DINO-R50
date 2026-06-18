# Tiny Co-DETR-R50 training smoke config.
# Runs on 4 training images and 2 validation images only to verify the external
# Co-DETR training loop starts in the isolated environment.

_base_ = ['./codetr_r50_stage0_mm_config.py']

data_root = 'outputs/codetr_coco/tiny/'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
    ),
    val=dict(
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
    ),
    test=dict(
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
    ),
)

runner = dict(type='EpochBasedRunner', max_epochs=1)
evaluation = dict(interval=1, metric='bbox')
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=1)
