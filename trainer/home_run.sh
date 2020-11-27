#!/usr/bin/env bash
# used when running on personal laptop

checkpoint=""
resume_training=False

# dataset hyperparameters
train_root_dir="/home/patrick/ecovision/data/2018"
dem_dir="/home/patrick/ecovision/data/2018/avalanches0118_endversion.tif"
train_region_file='Small_test_area.shp'
val_region_file="$train_region_file"
tile_size=128
aval_certainty=1
bands="3 4"
means="1023.9 949.9" #"986.3 1028.3 1023.9 949.9"
stds="823.4 975.5" #"1014.3 955.9 823.4 975.5"
batch_size=2
batch_augm=0
num_workers=1

# data augmentation
hflip_p=0.5
rand_rotation=20

# training hyperparameters
seed=42
gpus=1
log_every_n_steps=5
flush_logs_every_n_steps=20
max_epochs=5
train_viz_interval=5
limit_train_batches=1000
limit_val_batches=10
log_dir="$(pwd)/lightning_logs"
benchmark=True

# model hyperparameters
lr=1e-4
in_channels=3
model='deeplabv4'
backbone='avanet_leaky'


python -m trainer.train \
--date "$(date +"%d.%m.%y")" \
--time "$(date +"%T")" \
--train_root_dir $train_root_dir \
--checkpoint "$checkpoint" \
--resume_training $resume_training \
--train_region_file $train_region_file \
--val_region_file $val_region_file \
--tile_size $tile_size \
--aval_certainty $aval_certainty \
--dem_dir $dem_dir \
--bands $bands \
--means $means \
--stds $stds \
--batch_size $batch_size \
--batch_augm $batch_augm \
--num_workers $num_workers \
--hflip_p $hflip_p \
--rand_rotation $rand_rotation \
--seed $seed \
--gpus $gpus \
--log_every_n_steps $log_every_n_steps \
--flush_logs_every_n_steps $flush_logs_every_n_steps \
--max_epochs $max_epochs \
--train_viz_interval $train_viz_interval \
--lr $lr \
--model $model \
--backbone $backbone \
--in_channels $in_channels \
--limit_train_batches $limit_train_batches \
--limit_val_batches $limit_val_batches \
--log_dir $log_dir \
--benchmark $benchmark \
--profiler True \
