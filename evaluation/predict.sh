#!/usr/bin/env bash

python -m evaluation.predict_region \
--image_dir '/home/elyas/Desktop/SpotTheAvalanche/DroneData/PatchyPatch/Drone_imagery' \
--dem_path '/home/elyas/Desktop/SpotTheAvalanche/DroneData/davos_dem_25cm_lv95.tif' \
--region_file '/home/elyas/Desktop/SpotTheAvalanche/DroneData/AOI/braema_aoi.shp' \
--output_path '/home/elyas/Desktop/SpotTheAvalanche/DroneData/PatchyPatch/PatchOutput/outputCoordschangedoverlap.tif' \
--checkpoint '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Checkpoints/epoch=12-step=32850.ckpt' \
--aval_path '' \