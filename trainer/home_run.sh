#!/usr/bin/env bash
# used when running on personal laptop

# dataset hyperparameters
train_root_dir="/home/patrick/ecovision/data/2018"
aval_certainty=1
bands="1 2 3 4"

# data augmentation
hflip_p=0.5
rand_rotation=20

# training hyperparameters
seed=42
gpus=1
row_log_interval=5
log_save_interval=20
train_viz_interval=5
limit_train_batches=20
limit_val_batches=10
log_dir="$(pwd)/lightning_logs"

# model hyperparameters
lr=1e-4
in_channels=4


python -m trainer.train \
--date "$(date +"%d.%m.%y")" \
--time "$(date +"%T")" \
--train_root_dir $train_root_dir \
--aval_certainty $aval_certainty \
--bands $bands \
--hflip_p $hflip_p \
--rand_rotation $rand_rotation \
--seed $seed \
--gpus $gpus \
--row_log_interval $row_log_interval \
--log_save_interval $log_save_interval \
--train_viz_interval $train_viz_interval \
--lr $lr \
--in_channels $in_channels \
--limit_train_batches $limit_train_batches \
--limit_val_batches $limit_val_batches \
--log_dir $log_dir \

