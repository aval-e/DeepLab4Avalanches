import os.path
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
import data_utils

# data_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019'
data_folder = '/home/patrick/ecovision/data'
tif_folder = 'Spot6_Ortho_2_3_3_4_5'
tif = 'ortho_2_dim_spot6_pms_201901161003537_sen_37024891011.TIF'
shp = 'avalanches0119_endversion.shp'

# open satellite image
tif_path = os.path.join(os.path.join(data_folder, tif_folder), tif)
image_raster = gdal.Open(tif_path, gdal.GA_ReadOnly)

# open shapefile
shape_path = os.path.join(data_folder, shp)
shapefile = ogr.Open(shape_path)

# extract patch into numpy array
offset = (5000, 5000)  # x, y
res = (5000, 5000)  # x, y
image = data_utils.getAllBandsAsNumpy(image_raster, offset, res, bands=[1,2,3])
shp_image = data_utils.getNumpyFromShapefile(shapefile, image_raster, offset, res)

# overlay shapefile in red
image[:, :, 0] += 0.2 * shp_image
image[:, :, 1] -= 0.2 * shp_image
image[:, :, 2] -= 0.2 * shp_image

plt.figure(dpi=500)
plt.imshow(image)
plt.show()
