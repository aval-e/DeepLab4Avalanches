#!/usr/bin/env bash

python -m evaluation.predict_region \
--image_dir '/path/to/the/image/directory' \
--dem_path '/path/to/DEM/DHM.tif'\
--region_file '/path/to/specify/the/region/for/prediction/region.shp' \
--output_path '/path/to/store/output.tif' \
--checkpoint '/path/to/the/checkpoint/file.ckpt' \
--aval_path '/path/for/avalanches/to/validate/against/avalanches.tif' # optional


