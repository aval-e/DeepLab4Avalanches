#!/usr/bin/env bash


# Dataset hyperparameters
train_root_dir="/home/pf/pfstud/bartonp/slf_avalanches/2018"
train_ava_file="avalanches0118_endversion.shp"
train_region_file="Train_area_2018.shp"
val_root_dir="$train_root_dir"
val_ava_file="$train_ava_file"
val_region_file="Val_area_2018.shp"
dem_dir="" # '/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056.tif'
tile_size="256 256"
aval_certainty=1

# Training hyperparameters
seed=42
gpus=1
batch_size=2
max_epochs=10
row_log_interval=5
log_save_interval=20
default_root_dir="/scratch/bartonp"

# Model hyperparameters
lr=1e-4
in_channels=4
train_viz_interval=100
val_viz_idx=1

python -m trainer.train \
--train_root_dir $train_root_dir \
--train_ava_file $train_ava_file \
--train_region_file $train_region_file \
--val_root_dir $val_root_dir \
--val_ava_file $val_ava_file \
--val_region_file $val_region_file \
--dem_dir "$dem_dir" \
--tile_size $tile_size \
--aval_certainty $aval_certainty \
--seed $seed \
--gpus $gpus \
--batch_size $batch_size \
--max_epochs $max_epochs \
--row_log_interval $row_log_interval \
--log_save_interval $log_save_interval \
--default_root_dir $default_root_dir \
--lr $lr \
--in_channels $in_channels \
--train_viz_interval $train_viz_interval \
--val_viz_idx $val_viz_idx \
--limit_train_batches 20 \
--limit_val_batches 5 \
--profiler True \