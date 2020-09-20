#!/usr/bin/env bash
# used when running on leonhard cluster

export PYTHONPATH=$PWD


# Parameters for bsub command
#BSUB -n 8
#BSUB -W 1400
#BSUB -R "rusage[ngpus_excl_p=2]"
#BSUB -R "rusage[mem=16384]"
# #BSUB -o "8_workers_4batches"


# Dataset hyperparameters
train_root_dir="/cluster/scratch/bartonp/slf_avalanches/2018"
train_ava_file="avalanches0118_endversion.shp"
train_region_file="Train_area_2018.shp"
val_root_dir="$train_root_dir"
val_ava_file="$train_ava_file"
val_region_file="Val_area_2018.shp"
dem_dir="" #"/cluster/work/igp_psr/bartonp/dem_ch/swissalti3d_2017_ESPG2056.tif"
tile_size="256 256"
aval_certainty=1
num_workers=2

# Training hyperparameters
seed=42
deterministic=True
gpus=2
batch_size=2
max_epochs=10
row_log_interval=50
log_save_interval=100
distributed_backend="ddp"
default_root_dir="/cluster/scratch/bartonp"


# Model hyperparameters
lr=5e-5
in_channels=4
train_viz_interval=400
val_viz_idx=4

python -m trainer.train \
--date "$(date +"%d.%m.%y")" \
--time "$(date +"%T")" \
--train_root_dir $train_root_dir \
--train_ava_file $train_ava_file \
--train_region_file $train_region_file \
--val_root_dir $val_root_dir \
--val_ava_file $val_ava_file \
--val_region_file $val_region_file \
--dem_dir "$dem_dir" \
--tile_size $tile_size \
--aval_certainty $aval_certainty \
--num_workers $num_workers \
--seed $seed \
--deterministic $deterministic \
--gpus $gpus \
--batch_size $batch_size \
--max_epochs $max_epochs \
--row_log_interval $row_log_interval \
--log_save_interval $log_save_interval \
--distributed_backend $distributed_backend \
--default_root_dir $default_root_dir \
--lr $lr \
--in_channels $in_channels \
--train_viz_interval $train_viz_interval \
--val_viz_idx $val_viz_idx \
