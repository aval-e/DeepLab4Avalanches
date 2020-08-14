import os.path
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019/Spot6_Ortho_2_3_3_4_5'
file = 'ortho_2_dim_spot6_pms_201901161003537_sen_37024891010.TIF'
path = os.path.join(folder, file)

dataset = gdal.Open(path, gdal.GA_ReadOnly)
print(dataset)
rb = dataset.GetRasterBand(1)
image = rb.ReadAsArray()

plt.imshow(image)
plt.show()