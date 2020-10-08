#!/usr/bin/env bash

checkpoint="/scratch/bartonp/checkpoints/deeplab_adam_b4_bigvalarea.ckpt"

save_dir="/scratch/bartonp/images/deeplab_adam_b4_bigvalarea"

# Dataset hyperparameters
train_root_dir="/home/pf/pfstud/bartonp/slf_avalanches/2018"
train_ava_file="avalanches0118_endversion.shp"
train_region_file="Train_area_2018.shp"
val_root_dir="$train_root_dir"
val_ava_file="$train_ava_file"
val_region_file="Val_area_2018.shp"
dem_dir="" #"/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056_packbits_tiled.tif"
tile_size="512 512"
aval_certainty=3
bands="3 4"
means="1023.9 949.9"
stds="823.4 975.5"

python -m evaluation.qualitative \
$checkpoint \
--save_dir $save_dir \
--train_root_dir $train_root_dir \
--train_ava_file $train_ava_file \
--train_region_file $train_region_file \
--val_root_dir $val_root_dir \
--val_ava_file $val_ava_file \
--val_region_file $val_region_file \
--dem_dir "$dem_dir" \
--tile_size $tile_size \
--aval_certainty $aval_certainty \
--bands $bands \
--means $means \
--stds $stds \
