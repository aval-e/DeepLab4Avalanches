#!/usr/bin/env bash
# used when running on leonhard cluster

export PYTHONPATH=$PWD
	

# Parameters for bsub command
#BSUB -n 10
#BSUB -W 239
#BSUB -R "rusage[ngpus_excl_p=2]"
#BSUB -R "rusage[mem=6000]"
#BSUB -R "select[gpu_model0==GeForceGTX1080Ti]"
# #BSUB -o "lsf.resnet34"

exp_name="19_myresnet34"

checkpoint="" #"/cluster/scratch/bartonp/lightning_logs/year_comparison/resnet18_dspp/version_0/checkpoints/epoch=16.ckpt"
resume_training=False

# Dataset hyperparameters
train_root_dir="/cluster/scratch/bartonp/slf_avalanches/2019"
train_ava_file="avalanches0119_endversion.shp"
train_region_file="Train_area_2019.shp"
val_root_dir="$train_root_dir"
val_ava_file="$train_ava_file"
val_region_file="Val_area_2019.shp"
val_gt_file="Methodenvergleich2019.shp"
dem_dir="/cluster/work/igp_psr/bartonp/dem_ch/swissalti3d_2017_ESPG2056_packbits_tiled.tif"
tile_size=512
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
deterministic=False
gpus=2
batch_size=4
batch_augm=2
accumulate_grad_batches=2
max_epochs=18
val_check_interval=1.0
log_every_n_steps=200
flush_logs_every_n_steps=200
accelerator="ddp"
sync_batchnorm=True
log_dir="/cluster/scratch/bartonp/lightning_logs/presentation"
benchmark=True


# Model hyperparameters
model='avanet'
backbone='adapted_resnet34'
decoder='avanet_new'
optimiser="adam"
lr=1e-4
lr_scheduler='multistep'
scheduler_steps="10"
scheduler_gamma=0.25
momentum=0.9
weight_decay=0.0
in_channels=3
train_viz_interval=1000
val_viz_interval=1
val_viz_idx=4

# Avanet options
avanet_rep_stride_with_dil=True
avanet_no_blocks="3 4 4 3"
avanet_deformable=True
avanet_px_per_iter=4
avanet_grad_attention=False

decoder_out_ch=512
decoder_dspf_ch="64 128 256"
decoder_rates="4 8 12 16"
decoder_deformable=True

python -m trainer.train \
--exp_name $exp_name \
--date "$(date +"%d.%m.%y")" \
--time "$(date +"%T")" \
--checkpoint "$checkpoint" \
--resume_training $resume_training \
--train_root_dir $train_root_dir \
--train_ava_file $train_ava_file \
--train_region_file $train_region_file \
--val_root_dir $val_root_dir \
--val_ava_file $val_ava_file \
--val_region_file $val_region_file \
--val_gt_file $val_gt_file \
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
--batch_augm $batch_augm \
--accumulate_grad_batches $accumulate_grad_batches \
--max_epochs $max_epochs \
--val_check_interval $val_check_interval \
--log_every_n_steps $log_every_n_steps \
--flush_logs_every_n_steps $flush_logs_every_n_steps \
--accelerator $accelerator \
--sync_batchnorm $sync_batchnorm \
--log_dir $log_dir \
--benchmark $benchmark \
--model $model \
--backbone $backbone \
--decoder $decoder \
--optimiser $optimiser \
--lr $lr \
--momentum $momentum \
--weight_decay $weight_decay \
--in_channels $in_channels \
--train_viz_interval $train_viz_interval \
--val_viz_interval $val_viz_interval \
--val_viz_idx $val_viz_idx \
--scheduler_gamma $scheduler_gamma \
--scheduler_steps $scheduler_steps \
--lr_scheduler $lr_scheduler \
--avanet_rep_stride_with_dil $avanet_rep_stride_with_dil \
--avanet_no_blocks $avanet_no_blocks \
--avanet_deformable $avanet_deformable \
--avanet_px_per_iter $avanet_px_per_iter \
--avanet_grad_attention $avanet_grad_attention \
--decoder_out_ch $decoder_out_ch \
--decoder_dspf_ch $decoder_dspf_ch \
--decoder_rates $decoder_rates \
--decoder_deformable $decoder_deformable \
