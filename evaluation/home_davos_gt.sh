#!/usr/bin/env bash
# used when running on personal laptop

ckpt_path='best'
save_dir="/tmp/davos_gt"

# dataset hyperparameters
test_root_dir="/home/patrick/ecovision/data/2018"
tile_size=64
bands="1 2 3 4"

# training hyperparameters
gpus=1
max_epochs=3
limit_test_batches=1000
log_dir="$(pwd)/lightning_logs"
benchmark=True

# model hyperparameters
in_channels=4
model='deeplabv3+'
backbone='resnet50'


python -m evaluation.davos_gt_eval \
--ckpt_path $ckpt_path \
--save_dir $save_dir \
--date "$(date +"%d.%m.%y")" \
--time "$(date +"%T")" \
--test_root_dir $test_root_dir \
--tile_size $tile_size \
--bands $bands \
--gpus $gpus \
--max_epochs $max_epochs \
--model $model \
--backbone $backbone \
--in_channels $in_channels \
--limit_test_batches $limit_test_batches \
--log_dir $log_dir \
--benchmark $benchmark \

