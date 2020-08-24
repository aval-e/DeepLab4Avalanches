import glob
import os.path
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
from utils import data_utils


# data_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019'
data_folder = '/home/patrick/ecovision/data'
tif_folder = 'Spot6_Ortho_2_3_3_4_5'
ava_file = 'avalanches0119_endversion.shp'
region_file = 'Region_Selection.shp'

# open satellite images
all_tiffs = glob.glob(os.path.join(os.path.join(data_folder, tif_folder), '*.TIF'))
vrt = gdal.BuildVRT('/tmp/myVRT.vrt', all_tiffs)

# open avalanche shapefile
ava_path = os.path.join(data_folder, ava_file)
avalanches = ogr.Open(ava_path)

# extract patch into numpy array
offset = (18000, 0)  # x, y
res = (5000, 5000)  # x, y
image = data_utils.get_all_bands_as_numpy(vrt, offset, res, bands=[1, 2, 3])
shp_image = data_utils.get_numpy_from_shapefile(avalanches, vrt, offset, res)

# overlay shapefile in red
image[:, :, 0] += 0.4 * shp_image
image[:, :, 1] -= 0.4 * shp_image
image[:, :, 2] -= 0.4 * shp_image

plt.figure(dpi=300)
plt.imshow(image)
plt.show()
