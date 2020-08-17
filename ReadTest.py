import os.path
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
import numpy as np

# data_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019'
data_folder = '/home/patrick/ecovision/data'
tif_folder = 'Spot6_Ortho_2_3_3_4_5'
tif = 'ortho_2_dim_spot6_pms_201901161003537_sen_37024891011.TIF'
shp = 'avalanches0119_endversion.shp'

# get satellite image
tif_path = os.path.join(os.path.join(data_folder, tif_folder), tif)
dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)
bands = []
for band in range(3):
    rb = dataset.GetRasterBand(band+1)
    bands.append(rb.ReadAsArray(5000, 5000, 5000, 5000))

# plot image
image = np.stack(bands, axis=2)
# image = (image / image.max())
plt.imshow(image)
plt.show()


# get shapefile in raster format
geo_transform = dataset.GetGeoTransform()
x_min = geo_transform[0]
y_max = geo_transform[3]
x_max = x_min + geo_transform[1] * dataset.RasterXSize
y_min = y_max + geo_transform[5] * dataset.RasterYSize
x_res = dataset.RasterXSize
y_res = dataset.RasterYSize

shape_path = os.path.join(data_folder, shp)
mb_v = ogr.Open(shape_path)
mb_l = mb_v.GetLayer()
pixel_width = geo_transform[1]

shp_raster = '/tmp/shp_raster.tif'
target_ds = gdal.GetDriverByName('GTiff').Create(shp_raster, x_res, y_res, 1, gdal.GDT_Byte)
target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, pixel_width))
gdal.RasterizeLayer(target_ds, [1], mb_l)
target_ds = None

# plot shapefile over satellite image
raster = gdal.Open(shp_raster, gdal.GA_ReadOnly)
shp_image = raster.GetRasterBand(1).ReadAsArray()
image = np.concatenate([image, np.expand_dims(shp_image, axis=2)], axis=2)
# image /= image.max()
print('about tot diividie')
image = np.divide(image, 65535.0)
print('did divide')
plt.imshow(image)

plt.show()
print('reached the end')
