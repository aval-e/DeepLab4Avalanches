#!/usr/bin/env bash

#BSUB -n 5
#BSUB -W 239
#BSUB -R "rusage[ngpus_excl_p=1]"
#BSUB -R "rusage[mem=6000]"
# #BSUB -R "select[gpu_model0==GeForceGTX1080Ti]"

python -m evaluation.predict_region \
--image_dir '/cluster/scratch/bartonp/slf_avalanches/2018/orthos_1-5m_RGBN_LV95_16bit_img' \
--dem_path '/cluster/work/igp_psr/bartonp/dem_ch/swissalti3d_2017_ESPG2056_packbits_tiled.tif' \
--region_file '/cluster/scratch/bartonp/slf_avalanches/2018/test_region/Test_region_2018.shp' \
--output_dir 'cluster/scratch/bartonp/predictions' \
--checkpoint '/cluster/scratch/bartonp/lightning_logs/final/negoffsets/myresnet34_negoffets/myresnet34_negoffets/version_0/checkpoints/epoch=17-step=22103.ckpt' \
--aval_path '/cluster/scratch/bartonp/slf_avalanches/2018/avalanches0118_endversion.tif'
