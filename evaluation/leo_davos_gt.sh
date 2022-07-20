#!/usr/bin/env bash

exp_name="nameofExperiment"
ckpt_path="/path/to/the/checkpoint/file.ckpt"

viz_diffs=False
save_dir="/cluster/scratch/bartonp/lightning_logs/eval/davos_gt_eval"

# Dataset hyperparameters
test_root_dir="/path/to/root/directory/test_year"
test_ava_file="avalanchePolygons.shp"
test_gt_file='Methodenvergleich2019.shp'
dem_dir="/path/to/DEM/DHM.tif"
tile_size=512

gpus=1
log_dir="/dir/to/eval/davos_gt_eval"

python -m evaluation.davos_gt_eval \
--exp_name $exp_name \
--ckpt_path $ckpt_path \
--viz_diffs $viz_diffs \
--save_dir "$save_dir" \
--test_root_dir $test_root_dir \
--test_ava_file $test_ava_file \
--test_gt_file $test_gt_file \
--dem_dir "$dem_dir" \
--tile_size $tile_size \
--gpus $gpus \
--log_dir $log_dir \
