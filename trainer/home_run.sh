#!/usr/bin/env bash
# used when running on personal laptop

# dataset hyperparameters
train_root_dir="/home/patrick/ecovision/data/2018"
aval_certainty=1

# training hyperparameters
seed=42
gpus=1
row_log_interval=5
log_save_interval=20
train_viz_interval=5

# model hyperparameters
lr=1e-4

python -m trainer.train \
--train_root_dir $train_root_dir \
--aval_certainty $aval_certainty \
--seed $seed \
--gpus $gpus \
--row_log_interval $row_log_interval \
--log_save_interval $log_save_interval \
--train_viz_interval $train_viz_interval \
--lr $lr \
#--overfit_batches 2 \


