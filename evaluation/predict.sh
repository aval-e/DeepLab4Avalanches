#!/usr/bin/env bash

python -m evaluation.predict_region \
--image_dir '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Drone_imagery' \
--dem_path '/home/elyas/Desktop/SpotTheAvalanche/DroneData/davos_dem_25cm_lv95.tif' \
--region_file '/home/elyas/Desktop/SpotTheAvalanche/DroneData/annotations/20190119_BraemaN_lv95.shx' \
--output_path '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Output/output.tif' \
--checkpoint '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Checkpoints/epoch=12-step=32850.ckpt' \
--aval_path '' \