#!/usr/bin/env bash


# dataset hyperparameters
train_root_dir="/scratch/bartonp/data/2018"

# training hyperparameters
gpus=-1 # set this under BSUB command for cluster
row_log_interval=5
log_save_interval=20

#Model hyperparameters
lr=1e-3

python -m trainer.train \
--train_root_dir $train_root_dir \
--gpus $gpus \
--row_log_interval $row_log_interval \
--log_save_interval $log_save_interval \
--lr $lr \

