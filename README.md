# AvaMap
Avalanche Mapping in optical satellite imagery using deep learning

This readme only gives an overview. More documentation is provided in the form of docstrings throughout the code

### Structure

* Datasets: code for reading satellite images and ground truth labels in the form of pytorch datasets
* Evaluation: various scripts for running quantitative and qualitative evaluation
* Experiments: pytorch lightning module for running experiments in an easily configurable way
* Modeling: model architecture and parts
* trainer: contains training scripts
* utils: various utilities for data manipulation, visualisation, etc.

### Installing

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

Some presets used can be found in the form of bash scripts.

