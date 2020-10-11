#!/usr/bin/env bash
# used when running on leonhard cluster

export PYTHONPATH=$PWD
	

# Parameters for bsub command
#BSUB -n 8
#BSUB -W 230
#BSUB -R "rusage[ngpus_excl_p=2]"
#BSUB -R "rusage[mem=16384]"
# #BSUB -o "8_workers_4batches"

exp_name="deeplabv3+_adam_5e-5"

# Dataset hyperparameters
train_root_dir="/cluster/scratch/bartonp/slf_avalanches/2018"
train_ava_file="avalanches0118_endversion.shp"
train_region_file="Train_area_2018.shp"
val_root_dir="$train_root_dir"
val_ava_file="$train_ava_file"
val_region_file="Val_area_2018.shp"
dem_dir="" #"/cluster/work/igp_psr/bartonp/dem_ch/swissalti3d_2017_ESPG2056_packbits_tiled.tif"
tile_size="256 256"
aval_certainty=2
bands="3 4"
num_workers=2
means="1023.9 949.9" #"986.3 1028.3 1023.9 949.9"
stds="823.4 975.5" #"1014.3 955.9 823.4 975.5"

# Data augmentation
hflip_p=0.5
rand_rotation=180

# Training hyperparameters
seed=42
deterministic=True
gpus=2
batch_size=4
max_epochs=20
val_check_interval=0.25
row_log_interval=25
log_save_interval=100
distributed_backend="ddp"
log_dir="/cluster/scratch/bartonp/lightning_logs"


# Model hyperparameters
model='deeplabv3+'
optimiser="adam"
lr=5e-5
momentum=0.9
weight_decay=0.0
in_channels=2
train_viz_interval=400
val_viz_idx=4

python -m trainer.train \
--exp_name $exp_name \
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
--bands $bands \
--num_workers $num_workers \
--means $means \
--stds $stds \
--hflip_p $hflip_p \
--rand_rotation $rand_rotation \
--seed $seed \
--deterministic $deterministic \
--gpus $gpus \
--batch_size $batch_size \
--max_epochs $max_epochs \
--val_check_interval $val_check_interval \
--row_log_interval $row_log_interval \
--log_save_interval $log_save_interval \
--distributed_backend $distributed_backend \
--log_dir $log_dir \
--model $model \
--optimiser $optimiser \
--lr $lr \
--momentum $momentum \
--weight_decay $weight_decay \
--in_channels $in_channels \
--train_viz_interval $train_viz_interval \
--val_viz_idx $val_viz_idx \
