#!/usr/bin/env bash


export PYTHONPATH=$PWD

exp_name="nameofExperiment"

checkpoint="" # specify path if retraining is desired "/path/to/the/checkpoint/file.ckpt"
resume_training=False

# Dataset hyperparameters
train_root_dir="/path/to/root/directory/containingTIFSandSHPS/"
train_ava_file="avalanchePolygons.shp"
train_region_file="Train_area.shp"
val_root_dir="$train_root_dir"
val_ava_file="$train_ava_file"
val_region_file="Vali_area.shp"
val_gt_file="Methodenvergleich2018.shp" # optional test with avalanche point validation data
dem_dir="/path/to/DEM/DHM.tif"
tile_size=512
aval_certainty=3
bands="3 4" # bands to be used from the optical data
num_workers=20
means="1023.9 949.9" # mean for bands specified
stds="823.4 975.5" # std for bands specified

# Data augmentation
hflip_p=0.5
rand_rotation=180

# Training hyperparameters
loss=weighted_bce # bce, weighted_bce (=weights for avalanche certainty), focal, soft_dice or bce_edges
seed=42
deterministic=False
gpus=1
batch_size=4
batch_augm=2
accumulate_grad_batches=2
max_epochs=18
val_check_interval=1.0
log_every_n_steps=200
flush_logs_every_n_steps=200
accelerator="ddp"
sync_batchnorm=True
log_dir="/path/to/experiments"
benchmark=True


# Model hyperparameters
model='avanet' # avanet or deeplabv3+ or deeplab or sa_unet or mask_rcnn
backbone='adapted_resnet34' # adapted_resnetxx or resnetxx
decoder='avanet' # avanet or deeplabv3+ or deeplab or sa_unet or mask_rcnn
optimiser="adam" # adam or sgd
lr=1e-4 #learning rate
lr_scheduler='multistep' # multistep or plateau; updates the learning rate at each step specified below
scheduler_steps="10"
scheduler_gamma=0.25
momentum=0.9
weight_decay=0.0
in_channels=3
train_viz_interval=2000
val_viz_interval=1
val_viz_idx=4

# Avanet options
avanet_rep_stride_with_dil=True
avanet_px_per_iter=4
decoder_out_ch=512
decoder_dspf_ch="64 128 256"
decoder_rates="4 8 12"

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
--loss $loss \
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
--avanet_px_per_iter $avanet_px_per_iter \
--decoder_out_ch $decoder_out_ch \
--decoder_dspf_ch $decoder_dspf_ch \
--decoder_rates $decoder_rates \
