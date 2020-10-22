#!/usr/bin/env bash

exp_name="my_experiment"

checkpoint=""
resume_training=False

# Dataset hyperparameters
train_root_dir="/home/pf/pfstud/bartonp/slf_avalanches/2018"
train_ava_file="avalanches0118_endversion.shp"
train_region_file="Train_area_2018.shp"
val_root_dir="$train_root_dir"
val_ava_file="$train_ava_file"
val_region_file="Val_area_2018.shp"
dem_dir="" #"/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056_packbits_tiled.tif"
tile_size="256 256"
aval_certainty=3
bands="3 4"
means="1023.9 949.9"
stds="823.4 975.5"

# data augmentation
hflip_p=0.5
rand_rotation=180

# Training hyperparameters
seed=42
deterministic=True
gpus=1
batch_size=2
max_epochs=10
val_check_interval=0.5
log_every_n_steps=5
flush_logs_every_n_steps=20
log_dir="/scratch/bartonp/lightning_logs"

# Model hyperparameters
model='deeplab'
optimiser="adam"
lr=5e-5
in_channels=2
train_viz_interval=20
val_viz_idx=2

python -m trainer.train \
--exp_name $exp_name \
--date "$(date +"%d.%m.%y")" \
--time "$(date +"%T")" \
--checkpoint "$checkpoint" \
--resume_training $resume_training \
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
--hflip_p $hflip_p \
--rand_rotation $rand_rotation \
--seed $seed \
--deterministic $deterministic \
--gpus $gpus \
--batch_size $batch_size \
--max_epochs $max_epochs \
--val_check_interval $val_check_interval \
--log_every_n_steps $log_every_n_steps \
--flush_logs_every_n_steps $flush_logs_every_n_steps \
--log_dir $log_dir \
--model $model \
--optimiser $optimiser \
--lr $lr \
--in_channels $in_channels \
--train_viz_interval $train_viz_interval \
--val_viz_idx $val_viz_idx \
--limit_train_batches 0.01 \
--limit_val_batches 0.02 \
--profiler True \
