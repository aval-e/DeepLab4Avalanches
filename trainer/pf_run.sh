#!/usr/bin/env bash

exp_name="avanet"

checkpoint=""
resume_training=False

# Dataset hyperparameters
train_root_dir="/home/pf/pfstud/bartonp/slf_avalanches/2018"
train_ava_file="avalanches0118_endversion.shp"
train_region_file="Train_area_2018.shp"
val_root_dir="$train_root_dir"
val_ava_file="$train_ava_file"
val_region_file="Val_area_2018.shp"
dem_dir="/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056_packbits_tiled.tif"
tile_size=256
aval_certainty=2
bands="3 4"
means="1023.9 949.9"
stds="823.4 975.5"

# data augmentation
hflip_p=0.5
rand_rotation=180

# Training hyperparameters
seed=42
deterministic=False
gpus=1
batch_size=4
batch_augm=2
accumulate_grad_batches=4
max_epochs=20
val_check_interval=0.5
log_every_n_steps=50
flush_logs_every_n_steps=100
log_dir="/scratch/bartonp/lightning_logs"

# Model hyperparameters
model='avanet'
backbone='adapted_resnet'
decoder='deeplab'
optimiser="adam"
lr=5e-5
in_channels=3
train_viz_interval=2000
val_viz_idx=4

# Avanet options
avanet_rep_stride_with_dil=True
avanet_no_blocks="3 4 4 3"
avanet_deformable=True
avanet_px_per_iter=4
avanet_grad_attention=False

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
--batch_augm $batch_augm \
--accumulate_grad_batches $accumulate_grad_batches \
--max_epochs $max_epochs \
--val_check_interval $val_check_interval \
--log_every_n_steps $log_every_n_steps \
--flush_logs_every_n_steps $flush_logs_every_n_steps \
--log_dir $log_dir \
--model $model \
--backbone $backbone \
--decoder $decoder \
--optimiser $optimiser \
--lr $lr \
--in_channels $in_channels \
--train_viz_interval $train_viz_interval \
--val_viz_idx $val_viz_idx \
--avanet_rep_stride_with_dil $avanet_rep_stride_with_dil \
--avanet_no_blocks $avanet_no_blocks \
--avanet_deformable $avanet_deformable \
--avanet_px_per_iter $avanet_px_per_iter \
--avanet_grad_attention $avanet_grad_attention \
--limit_train_batches 1.0 \
--limit_val_batches 1.0 \
--profiler True \
