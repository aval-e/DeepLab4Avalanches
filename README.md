# AvaMap
Avalanche Mapping with deep learning

Run code as modules from the avamap directory. This will solve any problems with import errors of sibling packages.

Run training like so:
* in avamap directory
* bash trainer/run_training.sh

On leonhard make sure to load the following modules in addition to the pip requirements:
  1) StdEnv            4) cuda/10.1.243   7) jpeg/9b           10) geos/3.7.1
  2) gcc/4.8.5         5) cudnn/7.6.4     8) libpng/1.6.27     11) nano/2.6.3
  3) openblas/0.2.19   6) nccl/2.4.8-1    9) python_gpu/3.7.4  12) gdal/2.4.4
