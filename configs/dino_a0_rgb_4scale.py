_base_ = ["../external/IDEA-Research-DINO/config/DINO/DINO_4scale.py"]

num_classes = 12
dn_labelbook_size = 12

# A0 smoke/default settings. Longer runs can override these with --options.
epochs = 1
lr_drop = 1
batch_size = 1
use_ema = False
save_checkpoint_interval = 1

# Keep the first smoke modest on RTX 3090 while preserving 4-scale DINO.
data_aug_scales = [640]
data_aug_max_size = 960
