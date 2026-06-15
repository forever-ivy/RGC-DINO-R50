_base_ = ["../external/IDEA-Research-DINO/config/DINO/DINO_4scale.py"]

num_classes = 12
dn_labelbook_size = 12

# A0 smoke/default settings. Longer runs MUST override epochs/lr_drop via CLI.
# NOTE: lr_drop=1 is ONLY valid for the 1-epoch smoke. For real multi-epoch
# training, train_rgc_dino._resolve_lr_drop() auto-corrects config lr_drop<=1 to
# round(0.9*epochs) so the StepLR does not decay every epoch. Pass --lr-drop
# explicitly to override.
epochs = 1
lr_drop = 1
batch_size = 1
use_ema = False
save_checkpoint_interval = 1

# NOTE: data_aug_scales/data_aug_max_size below only affect the official COCO
# data pipeline. RGC training uses src/rgc_dino/dino_dataset.py instead, whose
# multi-scale jitter is controlled by --train-image-max-sides and horizontal
# flip by --random-horizontal-flip-prob. These two keys are inert for RGC runs.
data_aug_scales = [640]
data_aug_max_size = 960
