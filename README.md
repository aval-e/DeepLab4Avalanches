# AvaMap
Avalanche Mapping in optical satellite imagery using deep learning.

This readme only gives an overview. More documentation is provided in the form of docstrings throughout the code.

For an explanation of the concepts, ideas and evaluation, see the report.

### Structure

* **datasets**: code for reading satellite images and ground truth labels in the form of pytorch datasets
* **evaluation**: various scripts for running quantitative and qualitative evaluation
* **experiments**: pytorch lightning module for running experiments in an easily configurable way
* **modeling**: model architecture and parts
* **trainer**: contains training scripts
* **utils**: various utilities for data manipulation, visualization, etc.

### Installation

1. Create virtual environment
2. Install gdal
3. Change to avamap root directory
4. `pip install -r requirements.txt`
5. `pip install -e .`

On leonhard make sure to load the following modules in addition to the pip requirements:
  1) StdEnv            4) cuda/10.1.243   7) jpeg/9b           10) geos/3.7.1   13) eth_proxy
  2) gcc/4.8.5         5) cudnn/7.6.4     8) libpng/1.6.27     11) nano/2.6.3
  3) openblas/0.2.19   6) nccl/2.4.8-1    9) python_gpu/3.7.4  12) gdal/2.4.4

### Training

Training can be run with the `trainer/train.py` script. Training can be customised and hyperparameters set by passing the corresponding flags. All available options can be shown with:

`python trainer/train.py --help`

Some examples used, can be found in the form of bash scripts in the `trainer` directory.

### Prediction

To automatically predict avalanches on new satellite images run the `evaluation/predict_region.py` script. This creates a 1 band GeoTiff with values ranging from 0-1, where 0 corresponds to no avalanche, 0.5 is uncertain, and 1 is an avalanche. The predictions can be thresholded to get a binary map of predictions, with their certainty determined by the threshold.

The following flags need to be set:

* --image_dir: directory containing all satellite images
* --dem_path: path to the DEM if needed
* --region_file: a shapefile specifiying which region to predict over
* --output_path: the complete path and file name in which to store predictions. This will be a GeoTiff