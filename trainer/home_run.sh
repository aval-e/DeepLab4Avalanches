#!/usr/bin/env bash
# used when running on personal laptop

checkpoint=""
resume_training=False

# dataset hyperparameters
train_root_dir="/home/patrick/ecovision/data/2018"
dem_dir="/home/patrick/ecovision/data/2018/avalanches0118_endversion.tif"
tile_size="64 64"
aval_certainty=1
bands="3 4"
means="1023.9 949.9" #"986.3 1028.3 1023.9 949.9"
stds="823.4 975.5" #"1014.3 955.9 823.4 975.5"

# data augmentation
hflip_p=0.5
rand_rotation=20

# training hyperparameters
seed=42
gpus=1
max_epochs=5
row_log_interval=3
log_save_interval=20
train_viz_interval=5
limit_train_batches=20
limit_val_batches=10
log_dir="$(pwd)/lightning_logs"
benchmark=True

# model hyperparameters
lr=1e-4
in_channels=4
model='deeplabv3+'
backbone='resnet50'


python -m trainer.train \
--date "$(date +"%d.%m.%y")" \
--time "$(date +"%T")" \
--train_root_dir $train_root_dir \
--checkpoint "$checkpoint" \
--resume_training $resume_training \
--tile_size $tile_size \
--aval_certainty $aval_certainty \
--dem_dir $dem_dir \
--bands $bands \
--means $means \
--stds $stds \
--hflip_p $hflip_p \
--rand_rotation $rand_rotation \
--seed $seed \
--gpus $gpus \
--max_epochs $max_epochs \
--row_log_interval $row_log_interval \
--log_save_interval $log_save_interval \
--train_viz_interval $train_viz_interval \
--lr $lr \
--model $model \
--backbone $backbone \
--in_channels $in_channels \
--limit_train_batches $limit_train_batches \
--limit_val_batches $limit_val_batches \
--log_dir $log_dir \
--benchmark $benchmark \

