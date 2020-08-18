import os.path
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
import data_utils
import glob

# data_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019'
data_folder = '/home/patrick/ecovision/data'
tif_folder = 'Spot6_Ortho_2_3_3_4_5'
# tif = 'ortho_2_dim_spot6_pms_201901161003537_sen_37024891011.TIF'
shp = 'avalanches0119_endversion.shp'

# open satellite images
all_tiffs = glob.glob(os.path.join(os.path.join(data_folder, tif_folder), '*.TIF'))
vrt = gdal.BuildVRT('/tmp/myVRT.vrt', all_tiffs)

# open shapefile
shape_path = os.path.join(data_folder, shp)
shapefile = ogr.Open(shape_path)

# extract patch into numpy array
offset = (10000, 10000)  # x, y
res = (5000, 5000)  # x, y
image = data_utils.getAllBandsAsNumpy(vrt, offset, res, bands=[1,2,3])
shp_image = data_utils.getNumpyFromShapefile(shapefile, vrt, offset, res)

# overlay shapefile in red
image[:, :, 0] += 0.4 * shp_image
image[:, :, 1] -= 0.4 * shp_image
image[:, :, 2] -= 0.4 * shp_image

plt.figure(dpi=300)
plt.imshow(image)
plt.show()
