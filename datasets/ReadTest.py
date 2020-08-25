import os
import glob
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
from random import randint
from utils import data_utils, viz_utils


# data_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019'
data_folder = '/home/patrick/ecovision/data'
tif_folder = 'Spot6_Ortho_2_3_3_4_5'
ava_file = 'avalanches0119_endversion.shp'
region_file = 'Region_Selection.shp'

region = gpd.read_file(os.path.join(data_folder, region_file))

# generate sample points
tile_size = (2000, 2000)  # x, y in meters
points = data_utils.generate_point_grid(region, tile_size)

coord = points.iloc[randint(0, len(points)-1)]

fig, ax = plt.subplots()
region.plot(ax=ax, facecolor='red')
points.plot(ax=ax, color='b')
ax.scatter(coord.x, coord.y, color='g')
plt.show()


# open satellite images
all_tiffs = glob.glob(os.path.join(os.path.join(data_folder, tif_folder), '*.TIF'))
vrt = gdal.BuildVRT('/tmp/myVRT.vrt', all_tiffs)

# extract patch into numpy array
geo_transform = vrt.GetGeoTransform()
pixel_w = geo_transform[1]  # pixel width eg. 1 pixel => 1.5m
ulx, uly, lrx, lry = data_utils.get_raster_extent(vrt)
offset = ((coord.x - ulx)/pixel_w, (-coord.y + uly)/pixel_w)
res = (round(tile_size[0]/pixel_w), round(tile_size[1]/pixel_w))
image = data_utils.get_all_bands_as_numpy(vrt, offset, res, bands=[1, 2, 3])

# open avalanche shapefile
ava_path = os.path.join(data_folder, ava_file)
avalanches = ogr.Open(ava_path)
avalanches_l = avalanches.GetLayer()

shp_images = []
for i in range(1,4):
    avalanches_l.SetAttributeFilter("aval_shape = " + str(i))
    shp_images.append(data_utils.get_numpy_from_shapefile(avalanches, vrt, offset, res))

viz_utils.overlay_and_plot_avalanches_by_certainty(image, shp_images)