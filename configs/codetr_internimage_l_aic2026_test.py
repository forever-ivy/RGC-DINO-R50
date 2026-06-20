# Test-set config for Co-DETR + InternImage-L on AIC2026 phase-1 images.

_base_ = ['./codetr_internimage_l_mm_config.py']

test_data_root = 'outputs/codetr_coco/aic2026_test/'

data = dict(
    test=dict(
        ann_file=test_data_root + 'annotations/instances_test2017.json',
        img_prefix=test_data_root + 'test2017/',
    ),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=0),
)

evaluation = dict(interval=1, metric='bbox')
