#!/usr/bin/env bash


exp_name="eval"
ckpt_path="/scratch/bartonp/checkpoints/deeplab_adam_b4_bigvalarea.ckpt"

# Dataset hyperparameters
test_root_dir="/home/pf/pfstud/bartonp/slf_avalanches/2018"
test_ava_file="avalanches0118_endversion.shp"
dem_dir="" #"/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056_packbits_tiled.tif"
tile_size="256 256"
bands="3 4"
means="1023.9 949.9"
stds="823.4 975.5"

gpus=1
benchmark=True
log_dir="$(pwd)/lightning_logs"

# model hyperparameters
in_channels=2
model='deeplabv3+'
backbone='resnet50'

python -m evaluation.davos_gt_eval \
--exp_name $exp_name \
--ckpt_path $ckpt_path \
--test_root_dir $test_root_dir \
--test_ava_file $test_ava_file \
--dem_dir "$dem_dir" \
--tile_size $tile_size \
--bands $bands \
--means $means \
--stds $stds \
--gpus $gpus \
--benchmark $benchmark \
--log_dir $log_dir \
--model $model \
--backbone $backbone \
--in_channels $in_channels \
