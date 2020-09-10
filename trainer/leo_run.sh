#!/usr/bin/env bash
# used when running on leonhard cluster

export PYTHONPATH=$PWD


# Parameters for bsub command
#BSUB -n 6
#BSUB -W 90
#BSUB -R "rusage[ngpus_excl_p=2]"

 
# dataset hyperparameters
train_root_dir="/cluster/work/igp_psr/bartonp/slf_avalanches/2018"

# training hyperparameters
gpus=-1 # set this under BSUB command for cluster
default_root_dir="/cluster/scratch//bartonp"
row_log_interval=5
log_save_interval=20
distributed_backend='ddp'

# Model hyperparameters
lr=1e-3

python -m trainer.train \
--train_root_dir $train_root_dir \
--gpus $gpus \
--default_root_dir $default_root_dir \
--row_log_interval $row_log_interval \
--log_save_interval $log_save_interval \
--lr $lr \
--distributed_backend $distributed_backend
