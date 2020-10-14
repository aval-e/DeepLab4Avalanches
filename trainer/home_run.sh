#!/usr/bin/env bash
# used when running on personal laptop

# dataset hyperparameters
train_root_dir="/home/patrick/ecovision/data/2018"
tile_size="64 64"
aval_certainty=1
bands="1 2 3 4"

# data augmentation
hflip_p=0.5
rand_rotation=20

# training hyperparameters
seed=42
gpus=1
log_every_n_steps=5
flush_logs_every_n_steps=20
train_viz_interval=5
limit_train_batches=10
limit_val_batches=10
log_dir="$(pwd)/lightning_logs"

# model hyperparameters
lr=1e-4
in_channels=4
model='deeplabv3+'


python -m trainer.train \
--date "$(date +"%d.%m.%y")" \
--time "$(date +"%T")" \
--train_root_dir $train_root_dir \
--tile_size $tile_size \
--aval_certainty $aval_certainty \
--bands $bands \
--hflip_p $hflip_p \
--rand_rotation $rand_rotation \
--seed $seed \
--gpus $gpus \
--log_every_n_steps $log_every_n_steps \
--flush_logs_every_n_steps $flush_logs_every_n_steps \
--train_viz_interval $train_viz_interval \
--lr $lr \
--model $model \
--in_channels $in_channels \
--limit_train_batches $limit_train_batches \
--limit_val_batches $limit_val_batches \
--log_dir $log_dir \
--profiler True \
