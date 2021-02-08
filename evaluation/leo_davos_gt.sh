#!/usr/bin/env bash

#BSUB -n 5
#BSUB -W 239
#BSUB -R "rusage[ngpus_excl_p=1]"
#BSUB -R "rusage[mem=6000]"

exp_name="avanet_davos_gt_eval_19"
ckpt_path="/cluster/scratch/bartonp/lightning_logs/final/negoffsets/myresnet34_negoffets/myresnet34_negoffets/version_0/checkpoints/epoch=17-step=22103.ckpt"

viz_diffs=False
save_dir="/cluster/scratch/bartonp/lightning_logs/eval/davos_gt_eval"

# Dataset hyperparameters
#test_root_dir="/cluster/scratch/bartonp/slf_avalanches/2018"
#test_ava_file="avalanches0118_endversion.shp"
test_root_dir="/cluster/scratch/bartonp/slf_avalanches/2019"
test_ava_file="avalanches0119_endversion.shp"
test_gt_file='Methodenvergleich2019.shp'
dem_dir="/cluster/work/igp_psr/bartonp/dem_ch/swissalti3d_2017_ESPG2056_packbits_tiled.tif"
tile_size=512

gpus=1
log_dir="/cluster/scratch/bartonp/lightning_logs/eval/davos_gt_eval"

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
