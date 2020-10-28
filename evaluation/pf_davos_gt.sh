#!/usr/bin/env bash


exp_name="deeplabv3+_dem"
#ckpt_path="/scratch/bartonp/checkpoints/deeplab_pretrained.ckpt"
ckpt_path="/scratch/bartonp/checkpoints/deeplabv3+_dem_ba2_accgrads2.ckpt"

viz_diffs=True
save_dir="/scratch/bartonp/images/davos_gt_eval"

# Dataset hyperparameters
test_root_dir="/home/pf/pfstud/bartonp/slf_avalanches/2018"
test_ava_file="avalanches0118_endversion.shp"
dem_dir="/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056_packbits_tiled.tif"
tile_size="400 400"
bands="3 4"
means="1023.9 949.9"
stds="823.4 975.5"

gpus=1
benchmark=True
log_dir="$(pwd)/lightning_logs/eval"

python -m evaluation.davos_gt_eval \
--exp_name $exp_name \
--ckpt_path $ckpt_path \
--viz_diffs $viz_diffs \
--save_dir "$save_dir" \
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
