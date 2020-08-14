import os.path
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
import numpy as np

tif_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019/Spot6_Ortho_2_3_3_4_5'
tif = 'ortho_2_dim_spot6_pms_201901161003537_sen_37024891010.TIF'
tif_path = os.path.join(tif_folder, tif)

dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)

bands = []
for band in range(3):
    rb = dataset.GetRasterBand(band+1)
    bands.append(rb.ReadAsArray())

image = np.stack(bands, axis=2)
plt.imshow(image)
plt.show()