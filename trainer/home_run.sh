#!/usr/bin/env bash
# used when running on personal laptop

# dataset hyperparameters
train_root_dir="/home/patrick/ecovision/data/2018"

# training hyperparameters
gpus=1
row_log_interval=5
log_save_interval=20

# model hyperparameters
lr=1e-3

python -m trainer.train \
--train_root_dir $train_root_dir \
--gpus $gpus \
--row_log_interval $row_log_interval \
--log_save_interval $log_save_interval \
--lr $lr \


